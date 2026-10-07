"""
Bloc detection tests: does the two-colouring check report the right blocs?

Run from the project root:  py -m pytest tests -v
"""

import math
import random

import networkx as nx
import pytest

from helpers import triangle
import balance_model as bm
from blocs import analyse_blocs, detect_blocs


def signed_cycle(signs):
    """A cycle of len(signs) countries; signs[i] is the edge from i to i+1."""
    G = nx.Graph()
    n = len(signs)
    for i, s in enumerate(signs):
        G.add_edge(i, (i + 1) % n, sign=s)
    return G


def test_all_positive_network_is_one_bloc():
    G = nx.complete_graph(5)
    for u, v in G.edges:
        G[u][v]["sign"] = 1
    r = analyse_blocs(G)
    assert r["consistent"] and r["n_blocs"] == 1 and r["n_components"] == 1


def test_two_cliques_with_negative_edges_between_are_two_blocs():
    G = nx.complete_graph(6)
    group = {0: "x", 1: "x", 2: "x", 3: "y", 4: "y", 5: "y"}
    for u, v in G.edges:
        G[u][v]["sign"] = 1 if group[u] == group[v] else -1
    r = analyse_blocs(G)
    assert r["consistent"] and r["n_blocs"] == 2
    f = r["faction"]
    assert f[0] == f[1] == f[2] and f[3] == f[4] == f[5] and f[0] != f[3]


def test_unbalanced_triangle_is_not_consistent():
    r = analyse_blocs(triangle(1, 1, -1))
    assert r["consistent"] is False and r["n_blocs"] is None


def test_unbalanced_four_cycle_has_no_triangles_but_is_not_consistent():
    """The case triangle-balance misses: a chordless unbalanced loop."""
    G = signed_cycle([1, 1, 1, -1])
    assert bm.get_triads(G) == []
    assert bm.balance_ratio(G) == 1.0            # looks balanced by triangles
    assert analyse_blocs(G)["consistent"] is False


@pytest.mark.parametrize("signs, expected_blocs", [([1, 1, 1, 1], 1), ([1, -1, 1, -1], 2)])
def test_balanced_four_cycles(signs, expected_blocs):
    r = analyse_blocs(signed_cycle(signs))
    assert r["consistent"] and r["n_blocs"] == expected_blocs


def test_disconnected_network_counts_components():
    G = nx.Graph()
    G.add_edge("a", "b", sign=1)
    G.add_edge("c", "d", sign=1)
    r = analyse_blocs(G)
    assert r["n_components"] == 2 and r["n_blocs"] == 2 and r["consistent"]
    assert r["faction"]["a"] == r["faction"]["b"] != r["faction"]["c"]


def test_empty_and_single_node_networks():
    r = analyse_blocs(nx.Graph())
    assert (r["n_components"], r["n_blocs"], r["consistent"]) == (0, 0, True)
    G = nx.Graph()
    G.add_node("solo")
    assert analyse_blocs(G)["n_blocs"] == 1


def test_detect_blocs_wrapper():
    _, n = detect_blocs(signed_cycle([1, -1, 1, -1]))
    assert n == 2
    _, n_bad = detect_blocs(triangle(1, 1, -1))
    assert math.isnan(n_bad)


@pytest.mark.parametrize("seed", range(8))
def test_converged_complete_network_is_globally_consistent(seed):
    """On a complete network, triangle-balance implies global balance."""
    G, _, converged = bm.run_simulation(n_nations=7, density=1.0, noise=0.0,
                                        max_steps=20000, converge_patience=300,
                                        seed=seed)
    assert converged
    r = analyse_blocs(G)
    assert r["consistent"] and r["n_blocs"] in (1, 2)


@pytest.mark.parametrize("seed", range(20))
def test_consistent_implies_every_triangle_balanced(seed):
    """Random sparse networks: whenever the check says consistent, all triangles are balanced."""
    G = bm.create_signed_network(8, density=0.5, seed=seed)
    if analyse_blocs(G)["consistent"]:
        assert bm.balance_ratio(G) == 1.0
