"""
Triad tests: the core structural balance rule.

Run from the project root:  py -m pytest tests -v
"""

import itertools
import random

import networkx as nx
import pytest

from helpers import triangle
import balance_model as bm


@pytest.mark.parametrize("signs", list(itertools.product([1, -1], repeat=3)))
def test_triad_balanced_iff_even_number_of_negative_edges(signs):
    """+++ and +-- are balanced; ++- and --- are not (all edge placements)."""
    G = triangle(*signs)
    n_negative = sum(1 for s in signs if s == -1)
    assert bm.is_balanced(G, ("a", "b", "c")) == (n_negative % 2 == 0)


@pytest.mark.parametrize("signs", [s for s in itertools.product([1, -1], repeat=3)
                                    if s[0] * s[1] * s[2] == -1])
@pytest.mark.parametrize("edge", [("a", "b"), ("b", "c"), ("a", "c")])
def test_flipping_any_one_edge_of_an_unbalanced_triad_restores_balance(signs, edge):
    G = triangle(*signs)
    assert not bm.is_balanced(G, ("a", "b", "c"))
    G[edge[0]][edge[1]]["sign"] *= -1
    assert bm.is_balanced(G, ("a", "b", "c"))


def test_get_triads_counts_closed_triangles_only():
    assert len(bm.get_triads(nx.complete_graph(5))) == 10      # C(5,3)
    assert bm.get_triads(nx.path_graph(5)) == []               # no triangles


def test_balance_ratio_known_value():
    # 4 countries, all pairs connected: triangles abc (+++), abd, acd, bcd
    G = nx.complete_graph(4)
    for u, v in G.edges:
        G[u][v]["sign"] = 1
    assert bm.balance_ratio(G) == 1.0
    G[0][1]["sign"] = -1                  # breaks triangles 012 and 013
    assert bm.balance_ratio(G) == pytest.approx(2 / 4)


def test_step_repairs_an_unbalanced_triad_when_noise_is_zero():
    G = triangle(1, 1, -1)
    triads = bm.get_triads(G)
    assert bm.step(G, triads, random.Random(0), noise=0.0) is True
    assert bm.is_balanced(G, triads[0])


def test_step_does_nothing_to_a_balanced_triad_when_noise_is_zero():
    G = triangle(1, -1, -1)
    before = {tuple(sorted(e[:2])): e[2]["sign"] for e in G.edges(data=True)}
    assert bm.step(G, bm.get_triads(G), random.Random(0), noise=0.0) is False
    after = {tuple(sorted(e[:2])): e[2]["sign"] for e in G.edges(data=True)}
    assert before == after


