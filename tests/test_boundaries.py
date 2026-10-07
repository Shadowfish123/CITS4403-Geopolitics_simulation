"""
Boundary tests: empty, tiny and degenerate networks, and extreme noise.

Run from the project root:  py -m pytest tests -v
"""

import random

import networkx as nx
import pytest

from helpers import triangle
import balance_model as bm


def test_empty_graph():
    G = nx.Graph()
    assert bm.get_triads(G) == []
    assert bm.balance_ratio(G) == 1.0            # vacuously balanced
    _, n_factions = bm.detect_blocs(G)
    assert n_factions == 0


def test_graph_with_no_triangles_has_nothing_to_balance():
    G = nx.path_graph(6)
    for u, v in G.edges:
        G[u][v]["sign"] = random.Random(u).choice([1, -1])
    before = {tuple(sorted(e[:2])): e[2]["sign"] for e in G.edges(data=True)}
    _, history, converged = bm.run_simulation(G=G, max_steps=3000,
                                              converge_patience=500, seed=1)
    after = {tuple(sorted(e[:2])): e[2]["sign"] for e in G.edges(data=True)}
    assert history[-1][1] == 1.0
    assert converged
    assert before == after


def test_single_triad_network_converges_to_balance():
    G = triangle(-1, -1, -1)
    G2, history, converged = bm.run_simulation(G=G, max_steps=2000,
                                               converge_patience=200, seed=3)
    assert converged
    assert bm.balance_ratio(G2) == 1.0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_creating_tiny_networks_does_not_crash(n):
    G = bm.create_signed_network(n, density=1.0, seed=0)
    assert G.number_of_nodes() == n
    assert G.number_of_edges() <= n * (n - 1) // 2


def test_density_controls_edge_count():
    full = bm.create_signed_network(8, density=1.0, seed=0)
    assert full.number_of_edges() == 28
    half = bm.create_signed_network(8, density=0.5, seed=0)
    assert half.number_of_edges() == 14


def test_zero_steps_does_not_crash():
    G = bm.create_signed_network(5, density=1.0, seed=0)
    _, history, converged = bm.run_simulation(G=G, max_steps=0, seed=0)
    assert converged is False
    assert len(history) >= 1


def test_noise_one_flips_an_edge_every_step():
    G = triangle(1, -1, -1)                       # balanced
    triads = bm.get_triads(G)
    rng = random.Random(0)
    assert all(bm.step(G, triads, rng, noise=1.0) for _ in range(20))


def test_noise_zero_never_unbalances_a_balanced_network():
    G = triangle(1, 1, 1)
    triads = bm.get_triads(G)
    rng = random.Random(0)
    for _ in range(200):
        bm.step(G, triads, rng, noise=0.0)
    assert bm.balance_ratio(G, triads) == 1.0


def test_same_seed_gives_identical_results():
    runs = []
    for _ in range(2):
        G, history, converged = bm.run_simulation(n_nations=6, density=1.0, noise=0.0,
                                                  max_steps=5000, converge_patience=300,
                                                  seed=11)
        runs.append((history, converged, sorted(G.edges(data="sign"))))
    assert runs[0] == runs[1]
