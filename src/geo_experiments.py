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


# ---------------------------------------------------------------------------
# 5. CONTROL ARMS: GATE, SPACE AND ECONOMY
# ---------------------------------------------------------------------------

def gate_experiment(G0, noise_levels, draws, repeats, workers, max_steps,
                    patience, base_seed, keep=DEFAULT_KEEP, space="plane",
                    economy_scaling="fixed", r_min=None):
    """
    Hard gate against soft gate at the same number of relationships.

    The hard gate makes a far relationship impossible; the soft gate only makes
    it unlikely. If the two agree, the result is about how many short-range
    relationships survive; if they disagree, long-range bridges matter.
    """
    rows, raw = [], []
    for gate in ("hard", "soft"):
        r, w = substrate_sweep(
            G0, "geo", noise_levels, draws, repeats, workers, max_steps,
            patience, base_seed, keep=keep, space=space, gate=gate,
            economy_scaling=economy_scaling, r_min=r_min)
        rows += r
        raw += w
    return rows, raw


def space_experiment(G0, noise_levels, draws, repeats, workers, max_steps,
                     patience, base_seed, keep=DEFAULT_KEEP, gate="hard",
                     economy_scaling="fixed", r_min=None):
    """
    Flat map against a map with no corners.

    On a flat map the nations nearest the corners have fewer neighbours within
    any radius, which is a property of the map rather than of geography in
    general. The torus arm removes that boundary effect, so any difference is
    the boundary, not the mechanism.
    """
    rows, raw = [], []
    for space in ("plane", "torus"):
        r, w = substrate_sweep(
            G0, "geo", noise_levels, draws, repeats, workers, max_steps,
            patience, base_seed, keep=keep, space=space, gate=gate,
            economy_scaling=economy_scaling, r_min=r_min)
        rows += r
        raw += w
    return rows, raw


def economy_experiment(G0, noise_levels, draws, repeats, workers, max_steps,
                       patience, base_seed, keep=DEFAULT_KEEP, space="plane",
                       gate="hard", r_min=None):
    """
    Two strong economies against a lognormal economy for every nation.

    With two strong nations out of eight there is only one strong-strong pair,
    so the fixed tiers keep the draws comparable; the random arm checks whether
    the conclusion depends on that choice.
    """
    rows, raw = [], []
    for scaling in ("fixed", "random"):
        r, w = substrate_sweep(
            G0, "geo", noise_levels, draws, repeats, workers, max_steps,
            patience, base_seed, keep=keep, space=space, gate=gate,
            economy_scaling=scaling, r_min=r_min)
        rows += r
        raw += w
    return rows, raw


# ---------------------------------------------------------------------------
# 6. HOW FAR THE GATE IS CLOSED: THE KEEP SWEEP
# ---------------------------------------------------------------------------

def keep_experiment(G0, noise_levels, draws, repeats, workers, max_steps,
                    patience, base_seed, keeps=None, space="plane", gate="hard",
                    economy_scaling="fixed"):
    """
    Close the gate further and further and compare with random thinning.

    This is the arm that separates "the thinning is spatial" from "there are
    simply fewer relationships". At a mild gate the two substrates can look
    identical because both keep a similar number of closed triads; the question
    is whether they separate as more relationships are removed. The real
    snapshot is included at keep = 1.0 as the reference point.
    """
    keeps = list(keeps) if keeps is not None else list(KEEP_LEVELS)
    rows, raw = [], []
    for keep in keeps + [1.0]:
        kinds = ("real",) if keep >= 1.0 else ("random", "geo")
        r_min = None
        if keep < 1.0:
            r_min = geo.calibrate_r_min(G0, target_keep=keep,
                                        probes=CALIBRATION_PROBES,
                                        seed=base_seed, space=space,
                                        economy_scaling=economy_scaling)
        for kind in kinds:
            r, w = substrate_sweep(
                G0, kind, noise_levels, draws, repeats, workers, max_steps,
                patience, base_seed, keep=keep, space=space, gate=gate,
                economy_scaling=economy_scaling, r_min=r_min)
            rows += r
            raw += w
        print("      keep %.2f done (r_min %s)" % (
            keep, "n/a" if r_min is None else "%.3f" % r_min))
    return rows, raw


# ---------------------------------------------------------------------------
# 7. PLOTTING AND OUTPUT
# ---------------------------------------------------------------------------

def _band(ax, rows, label, colour, style="-"):
    """Draw one arm: pooled settling rate with the spread over draws."""
    ordered = sorted(rows, key=lambda r: r["noise"])
    xs = [r["noise"] for r in ordered]
    ys = [r["conv_rate"] for r in ordered]
    if len(ordered) >= MIN_DRAWS_FOR_BAND:
        ax.fill_between(xs, [r["draw_rate_lo"] for r in ordered],
                        [r["draw_rate_hi"] for r in ordered],
                        color=colour, alpha=0.15)
    band = " (shaded: spread over position draws)" if len(ordered) >= MIN_DRAWS_FOR_BAND else ""
    ax.plot(xs, ys, style, marker="o", color=colour, label=label + band)


def plot_by_key(rows, key, title, path, use_inertia=None):
    """Settling rate against noise, one line per value of `key`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    values = sorted({r[key] for r in rows}, key=str)
    for index, value in enumerate(values):
        subset = [r for r in rows if r[key] == value and
                  (use_inertia is None or r["inertia"] == use_inertia)]
        if subset:
            _band(ax, subset, str(value), colours[index % len(colours)])
    ax.set(xlabel="Noise level", ylabel="Fraction of runs that settled",
           title=title, ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inertia(rows, title, path):
    """Inertia on and off, for the real and the gated substrate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {(False, "real"): ("tab:blue", "--"), (True, "real"): ("tab:blue", "-"),
              (False, "geo"): ("tab:orange", "--"), (True, "geo"): ("tab:orange", "-")}
    for (inertia, substrate), (colour, style) in styles.items():
        subset = [r for r in rows if r["inertia"] == inertia
                  and r["substrate"] == substrate]
        if subset:
            _band(ax, subset, "%s, %s" % (substrate, "inertia" if inertia else "no inertia"),
                  colour, style)
    ax.set(xlabel="Noise level", ylabel="Fraction of runs that settled",
           title=title, ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_structure(rows, key, title, path):
    """Edges, triangles and components of each arm, as grouped bars."""
    values = sorted({r[key] for r in rows}, key=str)
    first = [next(r for r in rows if r[key] == value) for value in values]
    metrics = ["mean_edges", "mean_triangles", "mean_components"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.8 / len(metrics)
    for index, metric in enumerate(metrics):
        positions = [i + index * width for i in range(len(values))]
        ax.bar(positions, [row[metric] for row in first], width=width,
               label=metric.replace("mean_", ""))
    ax.set_xticks([i + width for i in range(len(values))], [str(v) for v in values])
    ax.set(ylabel="count (mean over position draws)", title=title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_keep_sweep(rows, title, path):
    """
    What the gate keeps, and what that does to noise tolerance.

    Left: closed triads as more relationships are removed. Right: settling rate
    at each noise level, with the spread over position draws shaded.
    """
    keeps = sorted({r["keep"] for r in rows})
    noises = sorted({r["noise"] for r in rows})
    colours = {"real": "tab:grey", "random": "tab:blue", "geo": "tab:orange"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for kind in ("real", "random", "geo"):
        triads = []
        xs = []
        for keep in keeps:
            match = [r for r in rows if r["keep"] == keep and r["substrate"] == kind]
            if match:
                xs.append(keep)
                triads.append(match[0]["mean_triangles"])
        if xs:
            ax1.plot(xs, triads, "o-", color=colours[kind], label=kind)
    ax1.set(xlabel="share of relationships kept",
            ylabel="closed triads (mean over draws)", title="What the gate keeps")

    for kind in ("random", "geo"):
        for index, noise in enumerate(noises):
            xs, ys, lows, highs = [], [], [], []
            for keep in keeps:
                match = [r for r in rows if r["keep"] == keep
                         and r["substrate"] == kind and r["noise"] == noise]
                if match:
                    row = match[0]
                    xs.append(keep)
                    ys.append(row["conv_rate"])
                    lows.append(row["draw_rate_lo"])
                    highs.append(row["draw_rate_hi"])
            if xs:
                style = "--" if index else "-"
                ax2.plot(xs, ys, style, marker="o", color=colours[kind],
                         label="%s, noise %g" % (kind, noise))
                ax2.fill_between(xs, lows, highs, color=colours[kind], alpha=0.12)
    ax2.set(xlabel="share of relationships kept",
            ylabel="fraction of runs that settled", ylim=(-0.05, 1.05),
            title="Noise tolerance as the gate closes")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

