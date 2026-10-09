"""
Geographically Gated Experiments
=================================
CITS4403 Research Project

Runs the noise experiment of run_experiments.py on networks that geography has
thinned, and adds the control arms needed to tell a *geography* effect apart
from a plain reduction in the number of relationships.

Substrates
----------
real     the snapshot as it is
random   the snapshot thinned uniformly at random to the same number of
         relationships; this is exactly the control the density experiment in
         run_experiments.py already uses
geo      the snapshot thinned by the geography gate

Every arm is repeated over many POSITION DRAWS, because the map itself is a
random draw. A single draw is one sample of the mechanism, not a result, so the
tables report both the pooled rate and the spread across draws.

Everything statistical (Wilson intervals, tipping points, the run scorer) is
imported from run_experiments.py, so the new arms are measured the same way as
the original ones.

Usage (from the project root):
    py src/run_geo_experiments.py --quick
    py src/run_geo_experiments.py --draws 30 --repeats 5 --workers 4
"""

import math
import os
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")                      # draw to files only; no window needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geography as geo                     # noqa: E402
import run_experiments as ex                # noqa: E402

# ---------------------------------------------------------------------------
# 1. CONSTANTS AND SUBSTRATE CONSTRUCTION
# ---------------------------------------------------------------------------

SUBSTRATES = ["real", "random", "geo"]
# The same grid as run_experiments.py, so the gated arms can be put next to the
# published curves without rescaling anything. The thinned arms need the top of
# the range: with fewer triangles they absorb far more noise before they stop
# settling, and a grid that stops at 0.02 would censor every one of them.
NOISE_LEVELS = list(ex.NOISE_LEVELS)
GATE_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03]
SPACE_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03]
ECONOMY_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03]
KEEP_LEVELS = [0.9, 0.75, 0.6, 0.45]
KEEP_NOISE_LEVELS = [0.01, 0.02]
DEFAULT_KEEP = geo.DEFAULT_TARGET_KEEP
CALIBRATION_PROBES = 30
MIN_DRAWS_FOR_BAND = 5


def write_csv(path, rows):
    """
    Delegate to run_experiments so both sets of tables share one format.

    Kept next to the constants because the driver writes tables from the first
    experiment onwards, before any of the plotting helpers exist.
    """
    ex.write_csv(path, rows)


def tipping_points(rows, group_key):
    """Noise level at which each arm falls below 50% settling."""
    return ex.tipping_points(rows, group_key=group_key)


def build_substrate(G0, kind, draw_seed, keep=DEFAULT_KEEP, space="plane",
                    gate="hard", economy_scaling="fixed", r_min=None,
                    soft_scale=geo.DEFAULT_SOFT_SCALE):
    """
    One position draw of one substrate.

    Parameters
    ----------
    G0 : networkx.Graph
        The real-data snapshot network.
    kind : str
        'real', 'random' or 'geo'.
    draw_seed : int
        Seed for the position draw and for which nations are the strong ones.
    keep : float
        Share of the real relationships the thinned substrates keep.
    space : str
        'plane' (default) or 'torus', passed to geography.pair_distance.
    gate : str
        'hard' for the reach threshold, 'soft' for the distance-weighted
        sample. Only used when kind='geo'.
    economy_scaling : str
        'fixed' (two strong, six weak) or 'random' (lognormal per nation).
    r_min : float or None
        Reach intercept. None calibrates it for this draw; passing one value in
        from the caller keeps every draw on the same gate.

    Returns
    -------
    (H, positions, economy, r_min)
    """
    nodes = sorted(G0.nodes())
    positions = geo.assign_positions(nodes, seed=draw_seed)
    economy = geo.assign_economy(nodes, seed=draw_seed, scaling=economy_scaling)

    if kind == "real":
        return G0.copy(), positions, economy, r_min

    if r_min is None:
        r_min = geo.calibrate_r_min(G0, target_keep=keep, probes=CALIBRATION_PROBES,
                                    seed=draw_seed, space=space,
                                    economy_scaling=economy_scaling)

    if kind == "random":
        H = ex.subsample_edges(G0, keep, random.Random(draw_seed))
    elif kind == "geo":
        if gate == "hard":
            H = geo.hard_gate(G0, positions, economy, r_min=r_min, space=space)
        else:
            H = geo.soft_gate(G0, positions, seed=draw_seed, scale=soft_scale,
                              keep=round(keep * G0.number_of_edges()), space=space)
    else:
        raise ValueError("kind must be 'real', 'random' or 'geo'")
    return H, positions, economy, r_min


def structure_of(networks):
    """
    Average structure of the substrates over the position draws.

    Edges, triangles and components are reported because two substrates can
    hold the same number of relationships and still be different experiments:
    the balance rule and the noise both act inside a closed triad, so the
    triangle count, not the edge count, sets how much dynamics happens.
    """
    summaries = []
    for H, positions, economy, _ in networks:
        summaries.append(geo.describe(H, positions))
    keys = [key for key in summaries[0]] if summaries else []
    out = {}
    for key in keys:
        values = [s[key] for s in summaries]
        out["mean_" + key] = statistics.fmean(values)
    return out


# ---------------------------------------------------------------------------
# 2. AGGREGATING OVER POSITION DRAWS
# ---------------------------------------------------------------------------

def percentile(values, fraction):
    """Value at a percentile of a list, without pulling in numpy."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise_over_draws(per_draw):
    """
    Pool the runs of one condition and add the spread across position draws.

    per_draw : list of lists
        One list of run records per position draw.
    """
    flat = [record for runs in per_draw for record in runs]
    summary = ex.summarise_condition(flat)
    rates = [sum(1 for record in runs if record["converged"]) / len(runs)
             for runs in per_draw]
    summary["draws"] = len(per_draw)
    summary["draw_rate_lo"] = percentile(rates, 2.5)
    summary["draw_rate_hi"] = percentile(rates, 97.5)
    summary["draw_rate_min"] = min(rates)
    summary["draw_rate_max"] = max(rates)
    return summary


# ---------------------------------------------------------------------------
# 3. ONE SUBSTRATE ACROSS NOISE AND POSITION DRAWS
# ---------------------------------------------------------------------------

def substrate_sweep(G0, kind, noise_levels, draws, repeats, workers, max_steps,
                    patience, base_seed, keep=DEFAULT_KEEP, space="plane",
                    gate="hard", economy_scaling="fixed", r_min=None,
                    use_inertia=False, extra=None):
    """
    Settling behaviour of one substrate: noise x position draw x dynamics seed.

    The position draws are built once and reused for every noise level, so the
    comparison between noise levels is paired on the map.

    Returns
    -------
    rows : list of dict
        One row per noise level, ready for a CSV.
    raw : list of dict
        One row per individual run.
    """
    rows, raw = [], []
    networks = [
        build_substrate(G0, kind, draw_seed=base_seed + draw, keep=keep,
                        space=space, gate=gate,
                        economy_scaling=economy_scaling, r_min=r_min)
        for draw in range(draws)
    ]
    structure = structure_of(networks)
    used_r_min = networks[0][3]

    for noise in noise_levels:
        per_draw = []
        for draw, (H, positions, economy, _) in enumerate(networks):
            tasks = [(H, noise, use_inertia, base_seed + 1000 * draw + i,
                      max_steps, patience) for i in range(repeats)]
            runs = ex.run_many(tasks, workers)
            per_draw.append(runs)
            raw += [{"substrate": kind, "noise": noise, "draw": draw,
                     "inertia": use_inertia, "seed": base_seed + 1000 * draw + i,
                     **(extra or {}), **record}
                    for i, record in enumerate(runs)]
        rows.append({"substrate": kind, "noise": noise, "inertia": use_inertia,
                     "keep": keep, "space": space, "gate": gate,
                     "economy": economy_scaling, "r_min": used_r_min,
                     **(extra or {}), **structure, **summarise_over_draws(per_draw)})
    return rows, raw


# ---------------------------------------------------------------------------
# 4. THE MAIN ARM: SUBSTRATES ACROSS NOISE
# ---------------------------------------------------------------------------

def noise_experiment(G0, noise_levels, draws, repeats, workers, max_steps,
                     patience, base_seed, keep=DEFAULT_KEEP, space="plane",
                     gate="hard", economy_scaling="fixed", r_min=None):
    """
    Settling rate against noise for the real, randomly thinned and gated
    networks, with and without relationship inertia.

    The random arm is the control: it answers "is it geography, or is it just
    fewer relationships?". The inertia arms answer "does inertia still raise
    the noise the network can absorb once geography has thinned it?".
    """
    rows, raw = [], []
    for use_inertia in (False, True):
        for kind in SUBSTRATES:
            r, w = substrate_sweep(
                G0, kind, noise_levels, draws, repeats, workers, max_steps,
                patience, base_seed, keep=keep, space=space, gate=gate,
                economy_scaling=economy_scaling, r_min=r_min,
                use_inertia=use_inertia)
            rows += r
            raw += w
    return rows, raw
