"""
Structural Balance Model of Geopolitical Bloc Formation
==========================================================
CITS4403 Research Project

Models a network of N nations as a signed graph:
    +1 edge = alliance
    -1 edge = rivalry

Update rule (local triad balance, Antal-Krapivsky-Redner style):
    At each timestep, pick a random UNBALANCED triad (a triple of nodes
    whose edge signs multiply to -1) and flip ONE of its edges so the
    triad becomes balanced (product of signs = +1).

    A small `noise` probability lets a country act "irrationally" and
    flip an edge away from balance instead, preventing the system from
    freezing into a trivial state and letting us study whether/when
    stable bipolar blocs still emerge.

Core research question:
    Under what conditions does the network settle into two stable
    opposing blocs (vs. one bloc, vs. never stabilizing)?
"""

import random
import itertools
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# 1. NETWORK INITIALISATION
# ---------------------------------------------------------------------------

def create_signed_network(n_nations, density=1.0, seed=None):
    """
    Create a random signed network of nations.

    Parameters
    ----------
    n_nations : int
        Number of countries (nodes).
    density : float in (0, 1]
        Fraction of all possible pairs that have an edge (relationship).
        density=1.0 -> fully connected (every pair has an opinion of
        each other, the classic structural-balance setting).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    G : networkx.Graph
        Nodes are countries (0..n_nations-1).
        Each edge has attribute 'sign' in {+1, -1}.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nations))

    all_pairs = list(itertools.combinations(range(n_nations), 2))
    n_edges = max(1, int(density * len(all_pairs)))
    chosen_pairs = rng.sample(all_pairs, n_edges)

    for (i, j) in chosen_pairs:
        sign = rng.choice([1, -1])
        G.add_edge(i, j, sign=sign)

    return G


# ---------------------------------------------------------------------------
# 2. BALANCE CHECKING
# ---------------------------------------------------------------------------

def get_triads(G):
    """Return all closed triads (triangles) in G as (i, j, k) tuples."""
    triads = []
    for i, j, k in itertools.combinations(G.nodes(), 3):
        if G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k):
            triads.append((i, j, k))
    return triads


def triad_sign_product(G, triad):
    """Product of the three edge signs in a triad. +1 = balanced."""
    i, j, k = triad
    return (G[i][j]['sign'] * G[j][k]['sign'] * G[i][k]['sign'])


def is_balanced(G, triad):
    return triad_sign_product(G, triad) == 1


def balance_ratio(G, triads=None):
    """Fraction of closed triads that are currently balanced."""
    if triads is None:
        triads = get_triads(G)
    if not triads:
        return 1.0  # vacuously balanced (e.g. sparse/no triangles)
    balanced = sum(1 for t in triads if is_balanced(G, t))
    return balanced / len(triads)


# ---------------------------------------------------------------------------
# 3. UPDATE RULE (local triad dynamics with noise)
# ---------------------------------------------------------------------------

def step(G, triads, rng, noise=0.0):
    """
    Perform one update step:
    - Pick a random closed triad.
    - If unbalanced, flip a random edge in it towards balance
      (with probability 1-noise), or flip it AWAY from balance /
      randomly with probability `noise` (models irrational/random
      geopolitical shocks).
    - If balanced, do nothing this step (no forced churn).

    Returns True if an edge was flipped, else False.
    """
    if not triads:
        return False

    i, j, k = rng.choice(triads)

    if rng.random() < noise:
        # random shock: flip a random edge of the triad regardless of balance
        edge = rng.choice([(i, j), (j, k), (i, k)])
        G[edge[0]][edge[1]]['sign'] *= -1
        return True

    if is_balanced(G, (i, j, k)):
        return False  # nothing to resolve

    # Unbalanced triad -> flip ONE edge to restore balance.
    # Flipping any single edge in an unbalanced triad makes it balanced
    # (since the sign product must become +1), so pick one at random.
    edge = rng.choice([(i, j), (j, k), (i, k)])
    G[edge[0]][edge[1]]['sign'] *= -1
    return True


# ---------------------------------------------------------------------------
# 4. SIMULATION LOOP
# ---------------------------------------------------------------------------

def run_simulation(n_nations=20, density=1.0, noise=0.0, max_steps=20000,
                    check_every=50, seed=None, converge_patience=2000):
    """
    Run the structural balance simulation and track balance ratio over time.

    Returns
    -------
    G : final network
    history : list of (step, balance_ratio) samples
    converged : bool, whether the system reached balance_ratio == 1.0
                and stayed there for `converge_patience` steps
    """
    rng = random.Random(seed)
    G = create_signed_network(n_nations, density=density, seed=seed)
    triads = get_triads(G)

    history = []
    steps_fully_balanced = 0
    converged = False

    for t in range(max_steps):
        step(G, triads, rng, noise=noise)

        if t % check_every == 0:
            br = balance_ratio(G, triads)
            history.append((t, br))

            if br == 1.0:
                steps_fully_balanced += check_every
            else:
                steps_fully_balanced = 0

            if steps_fully_balanced >= converge_patience:
                converged = True
                break

    history.append((t, balance_ratio(G, triads)))
    return G, history, converged


# ---------------------------------------------------------------------------
# 5. BLOC DETECTION (after convergence, how many blocs formed?)
# ---------------------------------------------------------------------------

def detect_blocs(G):
    """
    Once a network is (fully or mostly) balanced, it should split into
    at most 2 factions such that all intra-faction edges are + and all
    inter-faction edges are -. We detect this via a simple BFS/coloring
    over the POSITIVE edges only: nodes connected by a + edge share a
    faction; a - edge means they should be in different factions.

    Returns a dict {node: faction_id} and the number of distinct factions
    found. This is a heuristic — if the network never fully balanced,
    the split may be inconsistent (which is itself an interesting result).
    """
    faction = {}
    next_id = 0

    for node in G.nodes():
        if node in faction:
            continue
        # BFS assigning factions based on edge sign
        faction[node] = next_id
        queue = [node]
        while queue:
            cur = queue.pop()
            for nbr in G.neighbors(cur):
                sign = G[cur][nbr]['sign']
                target_faction = faction[cur] if sign == 1 else _other_faction(faction[cur], next_id)
                if nbr not in faction:
                    faction[nbr] = target_faction
                    queue.append(nbr)
        next_id += 1

    n_factions = len(set(faction.values()))
    return faction, n_factions


def _other_faction(fid, max_id):
    """Helper: crude 2-faction toggle (0<->1) within a connected component."""
    return 1 - fid if fid in (0, 1) else fid + 1


if __name__ == "__main__":
    # Quick smoke test
    G, history, converged = run_simulation(n_nations=15, density=1.0,
                                            noise=0.0, seed=42)
    print(f"Converged: {converged}")
    print(f"Final balance ratio: {history[-1][1]:.3f}")
    factions, n_factions = detect_blocs(G)
    print(f"Number of blocs detected: {n_factions}")
