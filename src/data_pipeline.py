"""
Data pipeline: GDELT snapshot -> anonymised signed network
============================================================
Reads data/country_relationships.csv (written by utils/gdelt_fetch.py) and
builds the signed network the model runs on.

Steps
-----
1. FILTER      keep the top-N countries by total mention volume, and only
               the pairs among them.
2. SIGN        a pair is +1 ("more cooperative coverage") if its average tone
               is above a threshold, otherwise -1. The threshold is a
               percentile of the pair tones AMONG THE SELECTED COUNTRIES
               (default 50 = their median). A fixed cut-off of 0 would make
               almost every pair a rivalry, because news tone skews negative.
3. INERTIA     each edge gets an inertia in [0, inertia_max] from its mention
               count. Mention counts are heavy-tailed, so the default is log
               scaling; linear scaling would leave most edges near zero.
4. ANONYMISE   countries are relabelled Nation_1..Nation_N in a random but
               seeded order. The mapping is returned but must never be printed
               or written to any output.
"""

import csv
import math
import random
import statistics

import networkx as nx


def load_relationships(csv_path="data/country_relationships.csv"):
    """Load the aggregated GDELT pair data as a list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "country_a": r["country_a"],
                "country_b": r["country_b"],
                "avg_tone": float(r["avg_tone"]),
                "num_mentions": int(r["num_mentions"]),
            })
    return rows


def select_top_countries(rows, n=8):
    """Return (top country codes, rows whose BOTH countries are in the top n)."""
    activity = {}
    for r in rows:
        for c in (r["country_a"], r["country_b"]):
            activity[c] = activity.get(c, 0) + r["num_mentions"]
    top = sorted(activity, key=activity.get, reverse=True)[:n]
    top_set = set(top)
    kept = [r for r in rows
            if r["country_a"] in top_set and r["country_b"] in top_set]
    return top, kept


def tone_threshold(rows, percentile=50):
    """Tone value at the given percentile (50 = median) of the given rows."""
    tones = sorted(r["avg_tone"] for r in rows)
    if not tones:
        raise ValueError("no rows to compute a threshold from")
    if percentile == 50:
        return statistics.median(tones)
    k = (len(tones) - 1) * percentile / 100
    lo, hi = math.floor(k), math.ceil(k)
    return tones[lo] + (tones[hi] - tones[lo]) * (k - lo)


def scale_inertia(mentions, max_mentions, scaling="log", inertia_max=0.9):
    """Map a mention count to an inertia in [0, inertia_max]."""
    if max_mentions <= 0:
        return 0.0
    if scaling == "log":
        frac = math.log1p(mentions) / math.log1p(max_mentions)
    elif scaling == "linear":
        frac = mentions / max_mentions
    else:
        raise ValueError("scaling must be 'log' or 'linear'")
    return inertia_max * frac


def anonymise(top_countries, seed=0):
    """Map country codes to Nation_1.. in a seeded random order (private)."""
    order = list(top_countries)
    random.Random(seed).shuffle(order)
    return {code: f"Nation_{i + 1}" for i, code in enumerate(order)}


def build_network(rows, top_countries, percentile=50, scaling="log",
                  inertia_max=0.9, seed=0):
    """
    Build the anonymised signed network from the filtered rows.
    Returns (G, mapping). Each edge has 'sign' (+1/-1) and 'inertia'.
    Pairs with no data simply have no edge (this is the network's density).
    """
    mapping = anonymise(top_countries, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(mapping.values())
    if not rows:
        return G, mapping

    threshold = tone_threshold(rows, percentile)
    max_mentions = max(r["num_mentions"] for r in rows)
    for r in rows:
        sign = 1 if r["avg_tone"] > threshold else -1   # strictly above = +1
        inertia = scale_inertia(r["num_mentions"], max_mentions,
                                scaling, inertia_max)
        G.add_edge(mapping[r["country_a"]], mapping[r["country_b"]],
                   sign=sign, inertia=inertia)
    return G, mapping


def summarise(G):
    """Counts for the report. Safe to print: contains no country identities."""
    n_edges = G.number_of_edges()
    n_pos = sum(1 for _, _, d in G.edges(data=True) if d["sign"] == 1)
    triangles = sum(nx.triangles(G).values()) // 3
    possible = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
    inertias = [d["inertia"] for _, _, d in G.edges(data=True)]
    return {
        "nodes": G.number_of_nodes(),
        "edges": n_edges,
        "density": n_edges / possible if possible else 0.0,
        "positive_edges": n_pos,
        "negative_edges": n_edges - n_pos,
        "triangles": triangles,
        "inertia_min": min(inertias) if inertias else 0.0,
        "inertia_median": statistics.median(inertias) if inertias else 0.0,
        "inertia_max": max(inertias) if inertias else 0.0,
    }


def load_and_build(csv_path="data/country_relationships.csv", n_countries=8,
                   percentile=50, scaling="log", inertia_max=0.9, seed=0):
    """Convenience wrapper: load -> filter -> sign -> inertia -> anonymise."""
    rows = load_relationships(csv_path)
    top, kept = select_top_countries(rows, n=n_countries)
    return build_network(kept, top, percentile, scaling, inertia_max, seed)
