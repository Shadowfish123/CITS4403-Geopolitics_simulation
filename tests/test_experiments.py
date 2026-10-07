"""
Experiment helper tests: statistics, tipping points, sub-sampling, permutation test.

Run from the project root:  py -m pytest tests -v
"""

import math
import random

import networkx as nx
import pytest

from helpers import triangle
import run_experiments as ex


def balanced_two_bloc_network():
    """Two groups of three; + inside a group, - between groups: fully balanced."""
    G = nx.complete_graph(6)
    for u, v in G.edges:
        G[u][v]["sign"] = 1 if (u < 3) == (v < 3) else -1
        G[u][v]["inertia"] = 0.0
    return G


# ---------------------------------------------------------------- statistics

def test_wilson_interval_bounds_and_known_values():
    lo, hi = ex.wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = ex.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.8
    lo, hi = ex.wilson(15, 30)
    assert lo < 0.5 < hi and hi - lo < 0.4
    assert all(math.isnan(x) for x in ex.wilson(0, 0))


def test_wilson_interval_narrows_with_more_runs():
    narrow = ex.wilson(150, 300)
    wide = ex.wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_mean_ci():
    m, half = ex.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and half > 0
    assert ex.mean_ci([5.0]) == (5.0, 0.0)
    assert math.isnan(ex.mean_ci([])[0])


def test_crossing_interpolates_between_points():
    assert ex.crossing([0, 1, 2], [1.0, 0.75, 0.25]) == pytest.approx(1.5)
    assert ex.crossing([0, 1, 2], [1.0, 0.9, 0.8]) is None       # never falls below
    assert ex.crossing([0, 1, 2], [0.3, 0.2, 0.1]) == 0           # starts below


def test_tipping_points_per_inertia_setting():
    rows = [{"noise": 0.0, "inertia": False, "conv_rate": 1.0},
            {"noise": 0.02, "inertia": False, "conv_rate": 0.0},
            {"noise": 0.0, "inertia": True, "conv_rate": 1.0},
            {"noise": 0.02, "inertia": True, "conv_rate": 0.5001}]
    points = {t["inertia"]: t["noise_at_50pct_settling"] for t in ex.tipping_points(rows)}
    assert points[False] == pytest.approx(0.01)
    assert points[True] is None or points[True] > 0.0199         # barely above 50%


# -------------------------------------------------------------- sub-sampling

def test_subsample_keeps_requested_fraction_and_all_nodes():
    G = balanced_two_bloc_network()                               # 15 edges
    H = ex.subsample_edges(G, 0.6, random.Random(0))
    assert H.number_of_nodes() == 6 and H.number_of_edges() == 9
    assert G.number_of_edges() == 15                              # original untouched
    assert ex.subsample_edges(G, 1.0, random.Random(0)).number_of_edges() == 15


def test_subsample_is_reproducible():
    G = balanced_two_bloc_network()
    a = sorted(ex.subsample_edges(G, 0.5, random.Random(7)).edges())
    b = sorted(ex.subsample_edges(G, 0.5, random.Random(7)).edges())
    assert a == b


# ------------------------------------------------------------------ one run

def test_run_one_returns_expected_fields_and_is_reproducible():
    G0 = balanced_two_bloc_network()
    task = (G0, 0.0, False, 3, 2000, 200)
    r1, r2 = ex.run_one(task), ex.run_one(task)
    assert r1 == r2
    assert r1["converged"] and r1["consistent"] and r1["n_blocs"] == 2
    assert r1["signs_retained"] == 1.0                            # already balanced: nothing flips
    assert r1["triangles"] == 20
    assert G0[0][1]["sign"] == 1                                  # input network not modified


def test_run_one_on_an_unbalanced_network_settles_and_changes_signs():
    G0 = nx.complete_graph(5)
    for u, v in G0.edges:
        G0[u][v]["sign"] = -1                                     # all hostile: unbalanced
        G0[u][v]["inertia"] = 0.0
    r = ex.run_one((G0, 0.0, False, 1, 5000, 200))
    assert r["converged"] and r["signs_retained"] < 1.0


# --------------------------------------------------------- summarise results

def test_summarise_condition_counts_and_intervals():
    results = [{"converged": True, "consistent": True, "steps": 100, "signs_retained": 0.5,
                "final_balance": 1.0, "triangles": 10, "n_components": 1},
               {"converged": True, "consistent": False, "steps": 300, "signs_retained": 0.7,
                "final_balance": 1.0, "triangles": 10, "n_components": 1},
               {"converged": False, "consistent": False, "steps": 40000, "signs_retained": 0.6,
                "final_balance": 0.8, "triangles": 10, "n_components": 1}]
    s = ex.summarise_condition(results)
    assert s["repeats"] == 3 and s["conv_rate"] == pytest.approx(2 / 3)
    assert s["consistent_rate"] == 0.5                             # 1 of the 2 settled runs
    assert s["mean_steps_converged"] == 200
    assert s["conv_lo"] < s["conv_rate"] < s["conv_hi"]


# ------------------------------------------------------------ permutation test

def test_a_fully_balanced_network_is_more_balanced_than_shuffles():
    r = ex.initial_balance_test(balanced_two_bloc_network(), n_perm=2000, seed=1)
    assert r["observed_balance"] == 1.0
    assert r["shuffled_mean_balance"] < 0.8
    assert r["p_more_balanced"] < 0.05


def test_permutation_test_preserves_the_edge_signs_of_the_input():
    G = balanced_two_bloc_network()
    before = sorted(G.edges(data="sign"))
    ex.initial_balance_test(G, n_perm=50, seed=0)
    assert sorted(G.edges(data="sign")) == before
