"""
Geography Layer for the Structural Balance Model
=================================================
CITS4403 Research Project

The main model treats every pair of nations as able to hold a relationship.
This module adds the layer underneath it: which pairs can hold a direct
relationship *at all* is decided by where the nations are and how strong their
economies are.

Rules
-----
* Every nation sits at a random point on a flat map, ``(x, y)`` in ``[0, 1]``.
* Every nation carries an economic tier: two "strong" and six "weak" by
  default, assigned to the eight nations in a seeded random order.
* A nation reaches ``reach_i = r_min + beta_econ * economy_i`` map units.
* A pair is ADMISSIBLE when ``distance(i, j) <= reach_i + reach_j``. Two distant
  weak economies can therefore never hold a direct relationship, while two
  strong economies can reach across the map. This is the conditional hard gate:
  impossible below the threshold, possible above it.
* A soft gate is provided as a control arm. It keeps each relationship with
  probability ``exp(-distance / scale)``, so long-range ties are unlikely
  instead of impossible.

The module only ever REMOVES relationships from a network built by
data_pipeline. Signs and inertia are never touched and no edge is ever added,
so the model in balance_model.py, the bloc test in blocs.py and the experiment
helpers in run_experiments.py all run on the result unchanged.

Usage (from the project root):
    import geography as geo
    positions = geo.assign_positions(G.nodes(), seed=0)
    economy = geo.assign_economy(G.nodes(), seed=0)
    r_min = geo.calibrate_r_min(G, target_keep=0.75)
    H = geo.hard_gate(G, positions, economy, r_min)
"""

import math
import random

# ---------------------------------------------------------------------------
# 1. COUNTRY POSITIONS AND ECONOMIC TIERS
# ---------------------------------------------------------------------------

DEFAULT_N_STRONG = 2          # of eight nations
STRONG_ECONOMY = 1.0
WEAK_ECONOMY = 0.25
DEFAULT_R_MIN = 0.0           # map units; calibrated by calibrate_r_min
DEFAULT_BETA_ECON = 1.0       # extra reach per unit of economic tier
DEFAULT_TARGET_KEEP = 0.75    # share of the real relationships that survive
DEFAULT_SOFT_SCALE = 0.35     # map units for the soft gate
DEFAULT_ECON_SIGMA = 0.7      # lognormal spread for the random-economy arm


def assign_positions(nodes, seed=None):
    """
    Place every nation at a random point on the unit square.

    The map is flat: coordinates are drawn from ``[0, 1]`` in both directions.
    Distance on this map is what the hard and soft gates use.

    Parameters
    ----------
    nodes : iterable
        Nation labels.
    seed : int or None
        RNG seed, so a "position draw" is reproducible.

    Returns
    -------
    positions : dict
        {node: (x, y)}
    """
    rng = random.Random(seed)
    return {node: (rng.random(), rng.random()) for node in sorted(nodes)}


def assign_economy(nodes, seed=None, n_strong=DEFAULT_N_STRONG,
                   scaling="fixed", sigma=DEFAULT_ECON_SIGMA):
    """
    Give every nation an economic tier.

    Parameters
    ----------
    nodes : iterable
        Nation labels.
    seed : int or None
        RNG seed for which nations are the strong ones.
    n_strong : int
        How many nations are strong. Two of eight is the default.
    scaling : str
        'fixed'  -> exactly n_strong nations at STRONG_ECONOMY, the rest at
                    WEAK_ECONOMY. Small variance between draws, which is why it
                    is the default.
        'random' -> a lognormal economy per nation, so the strength ordering
                    itself changes between draws. Used as a sensitivity arm.
    sigma : float
        Spread of the lognormal draw when scaling='random'.

    Returns
    -------
    economy : dict
        {node: economy}
    """
    if scaling == "fixed":
        order = list(nodes)
        random.Random(seed).shuffle(order)
        strong = set(order[:n_strong])
        return {node: (STRONG_ECONOMY if node in strong else WEAK_ECONOMY)
                for node in sorted(nodes)}
    if scaling == "random":
        rng = random.Random(seed)
        return {node: math.exp(rng.gauss(0.0, sigma)) for node in sorted(nodes)}
    raise ValueError("scaling must be 'fixed' or 'random'")


def reach_by_node(economy, r_min=DEFAULT_R_MIN, beta_econ=DEFAULT_BETA_ECON):
    """
    How far each nation can sustain a direct relationship, in map units.

    A strong economy reaches further, so it is the *pair* of economies that
    decides whether a distant relationship is possible at all.
    """
    return {node: r_min + beta_econ * value for node, value in economy.items()}


def pair_distance(a, b, space="plane"):
    """
    Distance between two points.

    'plane' is the flat map the default experiments use. 'torus' wraps the map
    around both edges so that no point sits in a corner; it is kept as a
    robustness check against the boundary effect of the flat map.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if space == "torus":
        dx = min(dx, 1.0 - dx)
        dy = min(dy, 1.0 - dy)
    elif space != "plane":
        raise ValueError("space must be 'plane' or 'torus'")
    return math.hypot(dx, dy)


def pair_distances(positions, space="plane"):
    """Distances for every pair of nations: {(a, b): distance} with a < b."""
    nodes = sorted(positions)
    return {tuple(sorted((a, b))): pair_distance(positions[a], positions[b], space)
            for a in nodes for b in nodes if a < b}


# ---------------------------------------------------------------------------
# 2. THE GATE: WHICH PAIRS CAN HOLD A RELATIONSHIP
# ---------------------------------------------------------------------------

def is_admissible(a, b, distances, reach, tolerance=0.0):
    """
    True when the pair is close enough for its combined reach.

    A pair exactly at the sum of the two reaches counts as admissible, so the
    boundary is inclusive and does not depend on floating point luck.
    """
    distance = distances[tuple(sorted((a, b)))]
    return distance <= reach[a] + reach[b] + tolerance


def kept_fraction(G, positions, economy, r_min, beta_econ=DEFAULT_BETA_ECON,
                  space="plane"):
    """Share of the existing relationships that survive the hard gate."""
    if G.number_of_edges() == 0:
        return 0.0
    distances = pair_distances(positions, space)
    reach = reach_by_node(economy, r_min, beta_econ)
    kept = sum(1 for u, v in G.edges()
               if is_admissible(u, v, distances, reach))
    return kept / G.number_of_edges()


def calibrate_r_min(G, target_keep=DEFAULT_TARGET_KEEP, probes=30, seed=0,
                    beta_econ=DEFAULT_BETA_ECON, n_strong=DEFAULT_N_STRONG,
                    space="plane", economy_scaling="fixed"):
    """
    Find the r_min that keeps the target share of relationships on average.

    Fixing r_min by hand would make the comparison unfair, because the same
    radius means a different density on every map and every snapshot. Here
    r_min is bisected until the average kept share over a set of position draws
    matches target_keep, so a gated network and a randomly thinned network can
    be compared at the same number of relationships.

    Returns
    -------
    r_min : float
    """
    low, high = -1.0, 2.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        shares = []
        for probe in range(probes):
            positions = assign_positions(G.nodes(), seed=seed + probe)
            economy = assign_economy(G.nodes(), seed=seed + probe,
                                     n_strong=n_strong,
                                     scaling=economy_scaling)
            shares.append(kept_fraction(G, positions, economy, mid,
                                        beta_econ, space))
        if sum(shares) / len(shares) < target_keep:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def hard_gate(G, positions, economy, r_min=DEFAULT_R_MIN,
              beta_econ=DEFAULT_BETA_ECON, space="plane"):
    """
    Copy of G keeping only the relationships geography allows.

    Nodes are never removed, so a nation with no surviving relationship stays
    in the network as an isolated node. Signs and inertia of the surviving
    relationships are copied unchanged.
    """
    H = G.copy()
    distances = pair_distances(positions, space)
    reach = reach_by_node(economy, r_min, beta_econ)
    for u, v in list(H.edges()):
        if not is_admissible(u, v, distances, reach):
            H.remove_edge(u, v)
    return H


def soft_gate(G, positions, seed=None, scale=DEFAULT_SOFT_SCALE, keep=None,
              space="plane"):
    """
    Copy of G keeping a distance-weighted random sample of relationships.

    Each relationship survives with probability proportional to
    ``exp(-distance / scale)``, so long-range ties are unlikely rather than
    impossible. The number of survivors is fixed to `keep` when it is given,
    which keeps the arm comparable with the hard gate at the same density.
    """
    H = G.copy()
    edges = sorted(H.edges())
    if keep is None:
        keep = round(DEFAULT_TARGET_KEEP * len(edges))
    if keep >= len(edges):
        return H
    distances = pair_distances(positions, space)
    rng = random.Random(seed)
    remaining = list(edges)
    chosen = set()
    while len(chosen) < keep and remaining:
        weights = [math.exp(-distances[tuple(sorted(edge))] / scale)
                   for edge in remaining]
        picked = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.add(picked)
        remaining.remove(picked)
    for u, v in edges:
        if (u, v) not in chosen:
            H.remove_edge(u, v)
    return H


# ---------------------------------------------------------------------------
# 3. DESCRIBING A GATED NETWORK
# ---------------------------------------------------------------------------

def mean_edge_length(G, positions, space="plane"):
    """Average map distance of the relationships that survive, in map units."""
    if G.number_of_edges() == 0:
        return 0.0
    distances = pair_distances(positions, space)
    return sum(distances[tuple(sorted(edge))] for edge in G.edges()) / G.number_of_edges()


def describe(G, positions, space="plane"):
    """
    Identity-free summary of a gated network, for the report and the CSVs.

    'triangles' is the number of closed triads, which is the quantity the
    dynamics actually work on: the balance rule and the noise both act inside a
    triad, so two networks with the same number of relationships but different
    triangle counts are not the same experiment.
    """
    import networkx as nx

    degrees = [degree for _, degree in G.degree()]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "triangles": sum(nx.triangles(G).values()) // 3,
        "components": nx.number_connected_components(G),
        "isolated_nodes": sum(1 for degree in degrees if degree == 0),
        "max_degree": max(degrees) if degrees else 0,
        "positive_edges": sum(1 for *_, d in G.edges(data=True) if d['sign'] == 1),
        "mean_edge_length": mean_edge_length(G, positions, space),
    }
