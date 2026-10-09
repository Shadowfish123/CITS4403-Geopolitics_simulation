"""
Geography tests: positions, economic tiers, the gate, and compatibility
=======================================================================
The last two tests matter most. They check that the gate can only ever remove
relationships and that, when it keeps everything, the model run is *identical*
to the run without a geography layer. Geography is therefore an addition to the
substrate, not a change to the model.

Run from the project root, either way:
    py -m pytest tests -v
    py tests/test_geography.py
"""

import os
import random
import sys
import math

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import balance_model as bm
import geography as geo


NODES = ["Nation_%d" % (i + 1) for i in range(8)]


def signed_ring(nodes=NODES, positive_fraction=0.5, seed=0):
    """A small signed network to gate: every pair present, random signs."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            G.add_edge(u, v, sign=(1 if rng.random() < positive_fraction else -1),
                       inertia=round(rng.random() * 0.9, 3))
    return G


# ------------------------------------------------------------ positions

def test_positions_are_inside_the_unit_square_and_reproducible():
    first = geo.assign_positions(NODES, seed=3)
    second = geo.assign_positions(NODES, seed=3)
    different = geo.assign_positions(NODES, seed=4)
    assert first == second
    assert first != different
    for x, y in first.values():
        assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0


def test_torus_distance_is_never_larger_than_plane_distance():
    positions = geo.assign_positions(NODES, seed=1)
    plane = geo.pair_distances(positions, "plane")
    torus = geo.pair_distances(positions, "torus")
    assert set(plane) == set(torus)
    assert all(torus[pair] <= plane[pair] + 1e-12 for pair in plane)
    assert any(torus[pair] < plane[pair] - 1e-9 for pair in plane)


def test_unknown_space_is_an_error():
    try:
        geo.pair_distances(geo.assign_positions(NODES, seed=0), "sphere")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown space")


# ------------------------------------------------------- economic tiers

def test_fixed_economy_has_exactly_two_strong_nations():
    economy = geo.assign_economy(NODES, seed=0)
    values = list(economy.values())
    assert values.count(geo.STRONG_ECONOMY) == geo.DEFAULT_N_STRONG
    assert values.count(geo.WEAK_ECONOMY) == len(NODES) - geo.DEFAULT_N_STRONG
    assert economy == geo.assign_economy(NODES, seed=0)
    assert economy != geo.assign_economy(NODES, seed=1)


def test_random_economy_is_positive_and_differs_between_draws():
    first = geo.assign_economy(NODES, seed=0, scaling="random")
    second = geo.assign_economy(NODES, seed=1, scaling="random")
    assert all(value > 0 for value in first.values())
    assert first != second
    assert any(abs(first[node] - second[node]) > 1e-9 for node in NODES)


def test_unknown_economy_scaling_is_an_error():
    try:
        geo.assign_economy(NODES, seed=0, scaling="uniform")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown scaling")


# --------------------------------------------------------------- the gate

def test_boundary_distance_is_admissible():
    """A pair exactly at the sum of the two reaches counts as reachable."""
    positions = {"a": (0.0, 0.0), "b": (0.5, 0.0)}
    distances = geo.pair_distances(positions, "plane")
    reach = {"a": 0.2, "b": 0.3}
    assert geo.is_admissible("a", "b", distances, reach)
    assert not geo.is_admissible("a", "b", distances, {"a": 0.2, "b": 0.29})


def test_reach_grows_with_the_economic_tier():
    economy = {"strong": geo.STRONG_ECONOMY, "weak": geo.WEAK_ECONOMY}
    reach = geo.reach_by_node(economy, r_min=0.0, beta_econ=1.0)
    assert reach["strong"] > reach["weak"]
    assert reach["strong"] == geo.STRONG_ECONOMY


def test_gate_keeps_everything_when_the_reach_is_large():
    G = signed_ring()
    positions = geo.assign_positions(G.nodes(), seed=0)
    economy = geo.assign_economy(G.nodes(), seed=0)
    H = geo.hard_gate(G, positions, economy, r_min=5.0)
    assert H.number_of_edges() == G.number_of_edges()
    assert sorted(H.edges(data=True)) == sorted(G.edges(data=True))


def test_gate_only_ever_removes_relationships():
    G = signed_ring()
    for seed in range(5):
        positions = geo.assign_positions(G.nodes(), seed=seed)
        economy = geo.assign_economy(G.nodes(), seed=seed)
        H = geo.hard_gate(G, positions, economy, r_min=0.0)
        assert set(H.edges()) <= set(G.edges())
        assert H.number_of_nodes() == G.number_of_nodes()


def test_gate_preserves_signs_and_inertia_of_surviving_edges():
    G = signed_ring()
    positions = geo.assign_positions(G.nodes(), seed=2)
    economy = geo.assign_economy(G.nodes(), seed=2)
    H = geo.hard_gate(G, positions, economy, r_min=0.0)
    for u, v, data in H.edges(data=True):
        assert data['sign'] == G[u][v]['sign']
        assert data['inertia'] == G[u][v]['inertia']


def test_strong_nations_keep_more_relationships_than_weak_ones():
    """The gate is not a random thinning: it depends on who the pair are."""
    G = signed_ring()
    strong_kept = weak_kept = 0
    strong_total = weak_total = 0
    for seed in range(20):
        positions = geo.assign_positions(G.nodes(), seed=seed)
        economy = geo.assign_economy(G.nodes(), seed=seed)
        H = geo.hard_gate(G, positions, economy, r_min=0.0)
        strong = [node for node in G.nodes() if economy[node] == geo.STRONG_ECONOMY]
        for u, v in G.edges():
            pair = (u in strong) + (v in strong)
            if pair == 2:
                strong_total += 1
                strong_kept += H.has_edge(u, v)
            elif pair == 0:
                weak_total += 1
                weak_kept += H.has_edge(u, v)
    assert strong_kept / strong_total > weak_kept / weak_total


def test_calibration_hits_the_requested_share():
    G = signed_ring()
    r_min = geo.calibrate_r_min(G, target_keep=0.75, probes=10, seed=0)
    shares = []
    for probe in range(10):
        positions = geo.assign_positions(G.nodes(), seed=100 + probe)
        economy = geo.assign_economy(G.nodes(), seed=100 + probe)
        shares.append(geo.kept_fraction(G, positions, economy, r_min))
    assert abs(sum(shares) / len(shares) - 0.75) < 0.12


def test_soft_gate_keeps_the_requested_number_and_is_reproducible():
    G = signed_ring()
    positions = geo.assign_positions(G.nodes(), seed=0)
    first = geo.soft_gate(G, positions, seed=5, keep=18)
    second = geo.soft_gate(G, positions, seed=5, keep=18)
    assert first.number_of_edges() == 18
    assert sorted(first.edges()) == sorted(second.edges())
    assert set(first.edges()) <= set(G.edges())


def test_soft_gate_prefers_short_relationships():
    G = signed_ring()
    positions = geo.assign_positions(G.nodes(), seed=0)
    H = geo.soft_gate(G, positions, seed=0, keep=10, scale=0.2)
    assert geo.mean_edge_length(H, positions) < geo.mean_edge_length(G, positions)


# --------------------------------------------------------- describing

def test_describe_counts_match_a_hand_built_network():
    G = nx.Graph()
    G.add_edge("a", "b", sign=1, inertia=0.0)
    G.add_edge("b", "c", sign=-1, inertia=0.0)
    G.add_edge("a", "c", sign=1, inertia=0.0)
    G.add_node("d")
    positions = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (1.0, 1.0), "d": (0.0, 1.0)}
    summary = geo.describe(G, positions)
    assert summary["nodes"] == 4 and summary["edges"] == 3
    assert summary["triangles"] == 1 and summary["isolated_nodes"] == 1
    assert summary["components"] == 2 and summary["positive_edges"] == 2
    assert summary["max_degree"] == 2


# ------------------------------------------------- compatibility with the model

def test_model_run_is_identical_when_the_gate_keeps_everything():
    """The geography layer must not change the model when nothing is removed."""
    G = signed_ring(seed=7)
    positions = geo.assign_positions(G.nodes(), seed=0)
    economy = geo.assign_economy(G.nodes(), seed=0)
    H = geo.hard_gate(G, positions, economy, r_min=5.0)

    _, plain_history, plain_converged = bm.run_simulation(
        G=G.copy(), noise=0.01, max_steps=4000, converge_patience=200, seed=11)
    _, gated_history, gated_converged = bm.run_simulation(
        G=H.copy(), noise=0.01, max_steps=4000, converge_patience=200, seed=11)

    assert plain_history == gated_history
    assert plain_converged == gated_converged
    assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))


def test_the_model_runs_on_a_gated_network():
    G = signed_ring(seed=3)
    positions = geo.assign_positions(G.nodes(), seed=3)
    economy = geo.assign_economy(G.nodes(), seed=3)
    H = geo.hard_gate(G, positions, economy, r_min=0.0)
    model, history, converged = bm.run_simulation(
        G=H.copy(), noise=0.0, max_steps=20000, converge_patience=400, seed=5)
    assert history[-1][1] == 1.0
    assert converged
    assert set(model.edges()) <= set(G.edges())


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print("ok    %s" % name)
        except Exception as error:                       # noqa: BLE001
            failures += 1
            print("FAIL  %s: %s: %s" % (name, type(error).__name__, error))
    print("\n%d test(s) failed" % failures)
    sys.exit(1 if failures else 0)
