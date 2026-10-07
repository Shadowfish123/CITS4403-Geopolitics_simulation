"""
Inertia tests: established relationships resist flipping.

Run from the project root:  py -m pytest tests -v
"""

import random

import networkx as nx

from helpers import triangle
import balance_model as bm


def with_inertia(G, value):
    for u, v in G.edges:
        G[u][v]["inertia"] = value
    return G


def signs(G):
    return {tuple(sorted(e[:2])): e[2]["sign"] for e in G.edges(data=True)}


def test_missing_or_zero_inertia_never_resists_and_draws_no_random_number():
    G = triangle(1, 1, -1)
    rng = random.Random(0)
    state = rng.getstate()
    assert bm.edge_resists(G, ("a", "b"), rng) is False          # no attribute
    with_inertia(G, 0.0)
    assert bm.edge_resists(G, ("a", "b"), rng) is False          # zero inertia
    assert rng.getstate() == state                               # RNG untouched


def test_full_inertia_freezes_the_network_even_under_maximum_noise():
    G = with_inertia(triangle(1, 1, -1), 1.0)
    before = signs(G)
    triads = bm.get_triads(G)
    rng = random.Random(1)
    for _ in range(200):
        assert bm.step(G, triads, rng, noise=1.0, use_inertia=True) is False
        assert bm.step(G, triads, rng, noise=0.0, use_inertia=True) is False
    assert signs(G) == before


def test_inertia_is_ignored_when_use_inertia_is_false():
    G = with_inertia(triangle(1, 1, -1), 1.0)
    assert bm.step(G, bm.get_triads(G), random.Random(0), noise=0.0,
                   use_inertia=False) is True                     # repaired anyway


def test_resist_probability_matches_the_inertia_value():
    flips = 0
    trials = 2000
    rng = random.Random(42)
    for _ in range(trials):
        G = with_inertia(triangle(1, 1, -1), 0.7)
        flips += bm.step(G, bm.get_triads(G), rng, noise=0.0, use_inertia=True)
    assert abs(flips / trials - 0.3) < 0.05                       # 1 - 0.7


def test_zero_inertia_reproduces_the_baseline_run_exactly():
    base = bm.create_signed_network(7, density=1.0, seed=5)
    plain = base.copy()
    zero = with_inertia(base.copy(), 0.0)
    _, h1, c1 = bm.run_simulation(G=plain, noise=0.005, max_steps=8000,
                                  converge_patience=300, seed=9, use_inertia=False)
    _, h2, c2 = bm.run_simulation(G=zero, noise=0.005, max_steps=8000,
                                  converge_patience=300, seed=9, use_inertia=True)
    assert (h1, c1) == (h2, c2)


def test_inertia_run_terminates_and_reports_history():
    G = with_inertia(bm.create_signed_network(6, density=1.0, seed=2), 0.5)
    G2, history, converged = bm.run_simulation(G=G, noise=0.0, max_steps=20000,
                                               converge_patience=300, seed=3,
                                               use_inertia=True)
    assert converged
    assert history[-1][1] == 1.0
