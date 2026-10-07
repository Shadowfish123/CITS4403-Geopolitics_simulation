"""Shared helpers for the test files."""
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def triangle(s_ab, s_bc, s_ac):
    """A 3-country network with the given edge signs."""
    G = nx.Graph()
    G.add_edge("a", "b", sign=s_ab)
    G.add_edge("b", "c", sign=s_bc)
    G.add_edge("a", "c", sign=s_ac)
    return G
