"""
Bloc detection for signed networks
===================================
A signed network is *balanced* (Harary) exactly when its countries can be split
into at most two groups so that every positive edge lies inside a group and
every negative edge lies between the two groups.

We test this directly by trying to 2-colour each connected component with a
breadth-first search: a positive edge forces equal colours, a negative edge
forces opposite colours. If a conflict appears, that component is frustrated.

This matters because the simulation only balances closed TRIANGLES. On a
complete network, "no unbalanced triangle" is equivalent to global balance.
On a network with missing relationships it is not: an unbalanced loop of four
or more countries with no chords contains no triangle at all. So a run can end
triangle-balanced yet still fail this test, and we report that case separately.
"""

from collections import deque


def analyse_blocs(G):
    """
    Return a dict describing the blocs in G:
        consistent   True if every connected component can be 2-coloured
        n_components number of connected components
        n_blocs      total number of blocs (1 or 2 per component), or None if
                     the network is not consistent
        faction      {node: faction_id}; ids are unique across components.
                     Only meaningful when consistent is True.
    """
    colour = {}
    faction = {}
    consistent = True
    n_components = 0
    n_blocs = 0

    for start in G.nodes():
        if start in colour:
            continue
        component_id = n_components
        n_components += 1
        colour[start] = 0
        queue = deque([start])
        members = [start]
        component_ok = True

        while queue:
            u = queue.popleft()
            for v in G.neighbors(u):
                wanted = colour[u] if G[u][v]["sign"] == 1 else 1 - colour[u]
                if v not in colour:
                    colour[v] = wanted
                    members.append(v)
                    queue.append(v)
                elif colour[v] != wanted:
                    component_ok = False

        for node in members:
            faction[node] = 2 * component_id + colour[node]
        if component_ok:
            n_blocs += len({colour[node] for node in members})
        else:
            consistent = False

    return {
        "consistent": consistent,
        "n_components": n_components,
        "n_blocs": n_blocs if consistent else None,
        "faction": faction,
    }


def detect_blocs(G):
    """
    Backwards-compatible wrapper: returns (faction, n_blocs).
    n_blocs is NaN when the network is not consistent (not 2-colourable),
    so averages over runs do not silently include meaningless counts.
    """
    result = analyse_blocs(G)
    n_blocs = result["n_blocs"] if result["consistent"] else float("nan")
    return result["faction"], n_blocs
