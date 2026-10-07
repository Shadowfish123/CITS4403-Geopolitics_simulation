"""
Data pipeline tests: filtering, signs, inertia scaling, anonymisation, loading.

Run from the project root:  py -m pytest tests -v
"""

import csv
import math
import os
import sys

import networkx as nx
import pytest

import helpers  # noqa: F401  (puts src/ on the path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

import data_pipeline as dp
from anonymise_data import anonymise_csv


def row(a, b, tone, mentions):
    return {"country_a": a, "country_b": b, "avg_tone": tone, "num_mentions": mentions}


ROWS = [row("A", "B", -1.0, 100), row("A", "C", -2.0, 50), row("B", "C", -3.0, 10),
        row("C", "D", -4.0, 1), row("A", "D", -5.0, 1)]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_a", "country_b", "avg_tone", "num_mentions", "num_events"])
        for r in rows:
            w.writerow([r["country_a"], r["country_b"], r["avg_tone"], r["num_mentions"], 1])


# ----------------------------------------------------------------- filtering

def test_top_countries_are_chosen_by_total_mentions_and_only_their_pairs_kept():
    top, kept = dp.select_top_countries(ROWS, n=3)
    assert set(top) == {"A", "B", "C"}                    # D has only 2 mentions
    assert {(r["country_a"], r["country_b"]) for r in kept} == {("A", "B"), ("A", "C"), ("B", "C")}


def test_asking_for_more_countries_than_exist_returns_all():
    top, kept = dp.select_top_countries(ROWS, n=50)
    assert set(top) == {"A", "B", "C", "D"} and len(kept) == len(ROWS)


# ----------------------------------------------------------------- threshold

def test_median_threshold_even_and_odd():
    assert dp.tone_threshold([row("a", "b", t, 1) for t in (1, 2, 3, 4)]) == 2.5
    assert dp.tone_threshold([row("a", "b", t, 1) for t in (1, 2, 3)]) == 2


def test_percentile_threshold():
    rows = [row("a", "b", t, 1) for t in (1, 2, 3, 4, 5)]
    assert dp.tone_threshold(rows, 0) == 1
    assert dp.tone_threshold(rows, 25) == 2
    assert dp.tone_threshold(rows, 75) == 4
    assert dp.tone_threshold(rows, 100) == 5


def test_threshold_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        dp.tone_threshold([])


# --------------------------------------------------------------------- signs

def test_sign_is_positive_only_when_strictly_above_the_median():
    top, kept = dp.select_top_countries(ROWS, n=3)        # tones -1, -2, -3; median -2
    G, mapping = dp.build_network(kept, top)
    assert G[mapping["A"]][mapping["B"]]["sign"] == 1     # -1 > -2
    assert G[mapping["A"]][mapping["C"]]["sign"] == -1    # equal to the median -> rival
    assert G[mapping["B"]][mapping["C"]]["sign"] == -1


def test_even_number_of_distinct_tones_splits_exactly_in_half():
    rows = [row(f"X{i}", f"Y{i}", float(i), 10) for i in range(10)]
    top = sorted({c for r in rows for c in (r["country_a"], r["country_b"])})
    G, _ = dp.build_network(rows, top)
    assert dp.summarise(G)["positive_edges"] == 5


def test_higher_percentile_threshold_means_fewer_cooperative_pairs():
    rows = [row(f"X{i}", f"Y{i}", float(i), 10) for i in range(20)]
    top = sorted({c for r in rows for c in (r["country_a"], r["country_b"])})
    p50 = dp.summarise(dp.build_network(rows, top, percentile=50)[0])["positive_edges"]
    p75 = dp.summarise(dp.build_network(rows, top, percentile=75)[0])["positive_edges"]
    assert p75 < p50


# ------------------------------------------------------------------- inertia

def test_inertia_is_bounded_and_increases_with_mentions():
    assert dp.scale_inertia(0, 100) == 0
    assert dp.scale_inertia(100, 100) == pytest.approx(0.9)
    values = [dp.scale_inertia(m, 1000) for m in (1, 10, 100, 1000)]
    assert values == sorted(values) and all(0 <= v <= 0.9 for v in values)


def test_log_scaling_keeps_small_counts_visible_where_linear_does_not():
    assert dp.scale_inertia(10, 1000, "log") > 10 * dp.scale_inertia(10, 1000, "linear")


def test_unknown_scaling_is_an_error():
    with pytest.raises(ValueError):
        dp.scale_inertia(5, 10, scaling="sqrt")


def test_edges_carry_sign_and_inertia_in_range():
    top, kept = dp.select_top_countries(ROWS, n=3)
    G, _ = dp.build_network(kept, top)
    for _, _, d in G.edges(data=True):
        assert d["sign"] in (1, -1) and 0 <= d["inertia"] <= 0.9


# ------------------------------------------------------------- anonymisation

def test_labels_are_nation_n_and_real_codes_never_appear():
    top, kept = dp.select_top_countries(ROWS, n=3)
    G, mapping = dp.build_network(kept, top)
    assert set(G.nodes) == {"Nation_1", "Nation_2", "Nation_3"}
    assert not (set(G.nodes) & set(mapping))


def test_anonymisation_is_deterministic_for_a_seed_and_shuffled():
    codes = [f"K{i}" for i in range(10)]
    assert dp.anonymise(codes, seed=3) == dp.anonymise(codes, seed=3)
    in_rank_order = {c: f"Nation_{i + 1}" for i, c in enumerate(codes)}
    assert dp.anonymise(codes, seed=0) != in_rank_order     # rank must not leak


def test_missing_pairs_have_no_edge_but_every_country_is_a_node():
    top, kept = dp.select_top_countries(ROWS, n=4)
    G, mapping = dp.build_network([r for r in kept if {r["country_a"], r["country_b"]} != {"A", "D"}], top)
    assert G.number_of_nodes() == 4
    assert not G.has_edge(mapping["A"], mapping["D"])


def test_no_rows_gives_a_network_with_nodes_and_no_edges():
    G, _ = dp.build_network([], ["A", "B"])
    assert G.number_of_nodes() == 2 and G.number_of_edges() == 0


# ------------------------------------------------------------------- loading

def test_load_and_build_end_to_end(tmp_path):
    path = tmp_path / "rel.csv"
    write_csv(path, ROWS)
    G, _ = dp.load_and_build(str(path), n_countries=3)
    assert G.number_of_nodes() == 3 and G.number_of_edges() == 3
    s = dp.summarise(G)
    assert s["triangles"] == 1 and s["density"] == 1.0


def test_load_relationships_converts_types(tmp_path):
    path = tmp_path / "rel.csv"
    write_csv(path, ROWS)
    loaded = dp.load_relationships(str(path))
    assert isinstance(loaded[0]["avg_tone"], float) and isinstance(loaded[0]["num_mentions"], int)


def test_summarise_known_network():
    G = nx.Graph()
    G.add_edge("a", "b", sign=1, inertia=0.2)
    G.add_edge("b", "c", sign=-1, inertia=0.4)
    G.add_edge("a", "c", sign=-1, inertia=0.6)
    s = dp.summarise(G)
    assert (s["nodes"], s["edges"], s["triangles"]) == (3, 3, 1)
    assert (s["positive_edges"], s["negative_edges"]) == (1, 2)
    assert s["inertia_min"] == 0.2 and s["inertia_max"] == 0.6


# ------------------------------------------------- anonymise_data.py (the file)

def test_anonymised_csv_hides_codes_and_keeps_numbers(tmp_path):
    raw, out = tmp_path / "raw.csv", tmp_path / "anon.csv"
    write_csv(raw, ROWS)
    n_rows, n_countries = anonymise_csv(str(raw), str(out), seed=0)
    text = out.read_text()
    assert (n_rows, n_countries) == (5, 4)
    assert not any(code in text.split("\n", 1)[1] for code in ("A,", "B,", "C,", "D,"))
    anon = list(csv.DictReader(open(out)))
    assert [r["avg_tone"] for r in anon] == [str(r["avg_tone"]) for r in ROWS]
    assert [r["num_mentions"] for r in anon] == [str(r["num_mentions"]) for r in ROWS]
