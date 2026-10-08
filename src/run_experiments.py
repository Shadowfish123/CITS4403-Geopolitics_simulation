"""
Experiments on the real-data networks
======================================
Runs the structural balance model on networks built from the GDELT snapshots
and writes tables (CSV) and figures (PNG) to a results folder.

Experiments
-----------
noise      Does the network settle into balance as random shocks (noise)
           increase, with and without relationship inertia? Also records how
           long settling takes and whether the final network really splits
           into at most two blocs.
initial    Permutation test: are the real-data signs more (or less) balanced
           than the same relationships with the signs shuffled at random?
density    Remove a fraction of the relationships and check how often a run that
           settles (no unbalanced triangle) is also globally consistent.
threshold  Repeat the noise experiment with the cooperative/hostile sign
           threshold at the 40th, 50th and 60th percentile of tone.

Design notes
------------
* Each condition is repeated with many seeds. The SAME seeds are used with and
  without inertia, so the comparison is paired.
* Proportions (settling rate, consistency rate) get Wilson 95% intervals;
  means get normal 95% intervals.
* The starting network is fixed for each snapshot, so results describe that
  snapshot, not geopolitics in general.
* Only anonymised labels are ever used; no country identities appear.

Usage (from the project root):
    py src/run_experiments.py --quick                 # fast smoke test
    py src/run_experiments.py                         # full run, both days, N=8
    py src/run_experiments.py --n 8 10 --workers 4    # more networks, parallel
"""

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")                      # draw to files only; no window needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import balance_model as bm                  # noqa: E402
import data_pipeline as dp                  # noqa: E402
from blocs import analyse_blocs             # noqa: E402

NOISE_LEVELS = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
THRESHOLD_NOISE_LEVELS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
DENSITY_FRACTIONS = [1.0, 0.85, 0.7, 0.55, 0.4]
PERCENTILES = [40, 50, 60]
MIN_SETTLED_FOR_TIME_PLOT = 10    # hide mean times based on fewer settled runs than this


# ------------------------------------------------------------ small statistics

def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n (stays inside 0 to 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values, z=1.96):
    """(mean, half-width of a normal 95% interval); NaN if no values."""
    if not values:
        return (float("nan"), float("nan"))
    m = statistics.fmean(values)
    if len(values) < 2:
        return (m, 0.0)
    return (m, z * statistics.stdev(values) / math.sqrt(len(values)))


def crossing(xs, rates, level=0.5):
    """
    The x value at which the rate first falls below `level`, by linear
    interpolation. Returns xs[0] if it starts below, or None if it never falls.
    """
    if rates[0] < level:
        return xs[0]
    for i in range(1, len(xs)):
        if rates[i] < level:
            x0, x1, y0, y1 = xs[i - 1], xs[i], rates[i - 1], rates[i]
            return x0 + (x1 - x0) * (y0 - level) / (y0 - y1)
    return None


# -------------------------------------------------------------------- networks

def network_from_csv(csv_path, n_countries, percentile=50):
    """Anonymised signed network from a snapshot. The label mapping is dropped."""
    rows = dp.load_relationships(csv_path)
    top, kept = dp.select_top_countries(rows, n=n_countries)
    G, _ = dp.build_network(kept, top, percentile=percentile)
    return G


def subsample_edges(G, fraction, rng):
    """Copy of G keeping a random `fraction` of its edges (all nodes kept)."""
    H = G.copy()
    edges = sorted(H.edges())
    n_remove = len(edges) - round(fraction * len(edges))
    for u, v in rng.sample(edges, n_remove):
        H.remove_edge(u, v)
    return H


# ------------------------------------------------------------------ single run

def run_one(task):
    """
    One simulation. task = (G0, noise, use_inertia, seed, max_steps, patience).
    Defined at module level so it can be sent to worker processes.
    """
    G0, noise, use_inertia, seed, max_steps, patience = task
    G, history, converged = bm.run_simulation(
        G=G0.copy(), noise=noise, max_steps=max_steps, converge_patience=patience,
        seed=seed, use_inertia=use_inertia)
    blocs = analyse_blocs(G)
    n_edges = max(1, G0.number_of_edges())
    kept = sum(1 for u, v, d in G.edges(data=True) if d["sign"] == G0[u][v]["sign"])
    return {
        "converged": bool(converged),
        "steps": history[-1][0],
        "final_balance": history[-1][1],
        "consistent": bool(blocs["consistent"]),
        "n_blocs": blocs["n_blocs"] if blocs["n_blocs"] is not None else "",
        "n_components": blocs["n_components"],
        "signs_retained": kept / n_edges,
        "triangles": len(bm.get_triads(G0)),
    }


def run_many(tasks, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(run_one, tasks, chunksize=4))
    return [run_one(t) for t in tasks]


def summarise_condition(results):
    """Collapse the repeats of one condition into proportions and means."""
    n = len(results)
    conv = [r for r in results if r["converged"]]
    k_conv = len(conv)
    lo, hi = wilson(k_conv, n)
    k_cons = sum(1 for r in conv if r["consistent"])
    cons_lo, cons_hi = wilson(k_cons, k_conv)
    ret, ret_ci = mean_ci([r["signs_retained"] for r in results])
    steps = [r["steps"] for r in conv]
    return {
        "repeats": n,
        "conv_rate": k_conv / n if n else float("nan"),
        "conv_lo": lo, "conv_hi": hi,
        "consistent_rate": k_cons / k_conv if k_conv else float("nan"),
        "consistent_lo": cons_lo, "consistent_hi": cons_hi,
        "signs_retained": ret, "signs_retained_ci": ret_ci,
        "mean_steps_converged": statistics.fmean(steps) if steps else float("nan"),
        "mean_final_balance": statistics.fmean([r["final_balance"] for r in results]) if n else float("nan"),
        "mean_triangles": statistics.fmean([r["triangles"] for r in results]) if n else float("nan"),
        "mean_components": statistics.fmean([r["n_components"] for r in results]) if n else float("nan"),
    }


# ----------------------------------------------------------------- experiments

def noise_experiment(G0, noise_levels, repeats, workers, max_steps, patience, base_seed, extra=None):
    """Settling rate vs noise, inertia off and on, with paired seeds."""
    rows, raw = [], []
    for noise in noise_levels:
        for use_inertia in (False, True):
            tasks = [(G0, noise, use_inertia, base_seed + i, max_steps, patience)
                     for i in range(repeats)]
            results = run_many(tasks, workers)
            row = {"noise": noise, "inertia": use_inertia, **(extra or {}),
                   **summarise_condition(results)}
            rows.append(row)
            raw += [{"noise": noise, "inertia": use_inertia, "seed": base_seed + i,
                     **(extra or {}), **r} for i, r in enumerate(results)]
    return rows, raw


def density_experiment(G0, fractions, repeats, workers, max_steps, patience, base_seed):
    """No noise: how often is a settled network also globally consistent?"""
    rows, raw = [], []
    base_edges = max(1, G0.number_of_edges())
    for frac in fractions:
        tasks = []
        for i in range(repeats):
            H = subsample_edges(G0, frac, random.Random(base_seed + 1000 + i))
            tasks.append((H, 0.0, False, base_seed + i, max_steps, patience))
        results = run_many(tasks, workers)
        density = statistics.fmean(t[0].number_of_edges() for t in tasks) / (
            G0.number_of_nodes() * (G0.number_of_nodes() - 1) / 2)
        rows.append({"edge_fraction_kept": frac, "density": density,
                     **summarise_condition(results)})
        raw += [{"edge_fraction_kept": frac, "seed": base_seed + i, **r}
                for i, r in enumerate(results)]
    return rows, raw


def threshold_experiment(csv_path, n_countries, percentiles, noise_levels, repeats,
                         workers, max_steps, patience, base_seed):
    """Noise experiment repeated at several sign thresholds."""
    rows, raw = [], []
    for p in percentiles:
        G0 = network_from_csv(csv_path, n_countries, percentile=p)
        r, w = noise_experiment(G0, noise_levels, repeats, workers, max_steps,
                                patience, base_seed, extra={"percentile": p,
                                "positive_edges": sum(1 for *_, d in G0.edges(data=True) if d["sign"] == 1),
                                "edges": G0.number_of_edges()})
        rows += r
        raw += w
    return rows, raw


def fmt_point(x):
    """Readable tipping point: a number, or a note if the rate never fell below 50%."""
    return "never fell below 50% in the tested range" if x is None else f"{x:.4f}"


def initial_balance_test(G0, n_perm=10000, seed=0):
    """
    Are the real-data signs more balanced than chance?
    Keeps the network's structure and its number of positive edges, shuffles
    which edges are positive, and compares the fraction of balanced triangles
    with the real one. p_more_balanced is the share of shuffles at least as
    balanced as the real network; p_less_balanced is the share at most as balanced.
    """
    triads = bm.get_triads(G0)
    observed = bm.balance_ratio(G0, triads)
    edges = list(G0.edges())
    signs = [G0[u][v]["sign"] for u, v in edges]
    rng = random.Random(seed)
    H = G0.copy()
    ratios = []
    for _ in range(n_perm):
        rng.shuffle(signs)
        for (u, v), sign in zip(edges, signs):
            H[u][v]["sign"] = sign
        ratios.append(bm.balance_ratio(H, triads))
    return {
        "triangles": len(triads),
        "observed_balance": observed,
        "shuffled_mean_balance": statistics.fmean(ratios),
        "p_more_balanced": (1 + sum(r >= observed for r in ratios)) / (n_perm + 1),
        "p_less_balanced": (1 + sum(r <= observed for r in ratios)) / (n_perm + 1),
        "permutations": n_perm,
    }


def tipping_points(rows, group_key=None):
    """Noise level where the settling rate falls below 50%, per inertia setting."""
    out = []
    groups = sorted({r.get(group_key) for r in rows}, key=str) if group_key else [None]
    for g in groups:
        for inertia in (False, True):
            sub = sorted((r for r in rows if r["inertia"] == inertia and
                          (group_key is None or r.get(group_key) == g)),
                         key=lambda r: r["noise"])
            xs = [r["noise"] for r in sub]
            point = crossing(xs, [r["conv_rate"] for r in sub])
            entry = {"inertia": inertia, "noise_at_50pct_settling": point}
            if group_key:
                entry[group_key] = g
            out.append(entry)
    return out


# -------------------------------------------------------------------- plotting

def _errbar(ax, xs, ys, lo, hi, **kw):
    ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                marker="o", capsize=3, **kw)


def plot_noise(rows, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for inertia, colour, label in [(False, "tab:blue", "Without inertia"),
                                   (True, "tab:orange", "With inertia")]:
        sub = sorted((r for r in rows if r["inertia"] == inertia), key=lambda r: r["noise"])
        xs = [r["noise"] for r in sub]
        _errbar(ax1, xs, [r["conv_rate"] for r in sub], [r["conv_lo"] for r in sub],
                [r["conv_hi"] for r in sub], color=colour, label=label)
        pts = [(r["noise"], r["mean_steps_converged"]) for r in sub
               if r["conv_rate"] * r["repeats"] >= MIN_SETTLED_FOR_TIME_PLOT]
        if pts:
            ax2.plot([x for x, _ in pts], [y for _, y in pts], marker="o", color=colour, label=label)
    ax1.set(xlabel="Noise level", ylabel="Fraction of runs that settled",
            title="Settling rate (95% CI)", ylim=(-0.05, 1.05))
    ax2.set(xlabel="Noise level", ylabel="Mean steps to settle (settled runs only)",
            title=f"Time to settle (points with at least {MIN_SETTLED_FOR_TIME_PLOT} settled runs)",
            yscale="log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_density(rows, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = [r["density"] for r in rows]
    _errbar(ax, xs, [r["conv_rate"] for r in rows], [r["conv_lo"] for r in rows],
            [r["conv_hi"] for r in rows], color="tab:blue", label="Settled (no unbalanced triangle)")
    ok = [r for r in rows if not math.isnan(r["consistent_rate"])]
    if ok:
        _errbar(ax, [r["density"] for r in ok], [r["consistent_rate"] for r in ok],
                [r["consistent_lo"] for r in ok], [r["consistent_hi"] for r in ok],
                color="tab:red", label="Settled AND globally consistent")
    ax.set(xlabel="Network density", ylabel="Fraction of runs", title=title, ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold(rows, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colours = {40: "tab:green", 50: "tab:blue", 60: "tab:purple"}
    for p in sorted({r["percentile"] for r in rows}):
        for inertia, style in [(False, "--"), (True, "-")]:
            sub = sorted((r for r in rows if r["percentile"] == p and r["inertia"] == inertia),
                         key=lambda r: r["noise"])
            ax.plot([r["noise"] for r in sub], [r["conv_rate"] for r in sub], style, marker="o",
                    color=colours.get(p, "gray"),
                    label=f"{p}th percentile, {'inertia' if inertia else 'no inertia'}")
    ax.set(xlabel="Noise level", ylabel="Fraction of runs that settled", title=title,
           ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- IO

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the real-data experiments.")
    parser.add_argument("--days", nargs="+",
                        default=["data/country_relationships.csv",
                                 "data/country_relationships_day2.csv"])
    parser.add_argument("--n", type=int, nargs="+", default=[8],
                        help="number of top countries per network")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default="results")
    parser.add_argument("--only", nargs="+", choices=["noise", "density", "threshold", "initial"],
                        default=["noise", "density", "threshold", "initial"])
    parser.add_argument("--quick", action="store_true",
                        help="small settings to check everything runs")
    args = parser.parse_args()

    noise_levels, thr_noise = NOISE_LEVELS, THRESHOLD_NOISE_LEVELS
    fractions, percentiles, repeats = DENSITY_FRACTIONS, PERCENTILES, args.repeats
    if args.quick:
        noise_levels, thr_noise = [0.0, 0.01, 0.02], [0.0, 0.01]
        fractions, percentiles, repeats = [1.0, 0.6], [50], 5

    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    summary = []

    for csv_path in args.days:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        day = "day2" if stem.endswith("day2") else ("day1" if stem == "country_relationships" else stem)
        for n in args.n:
            tag = f"{day}_n{n}"
            G0 = network_from_csv(csv_path, n)
            s = dp.summarise(G0)
            print(f"\n== {tag}: {s['nodes']} nations, {s['edges']} relationships, "
                  f"{s['triangles']} triangles ==")

            if "initial" in args.only:
                init_rows = []
                for pct in percentiles:
                    Gp = network_from_csv(csv_path, n, percentile=pct)
                    t = initial_balance_test(Gp, n_perm=1000 if args.quick else 10000, seed=args.seed)
                    init_rows.append({"percentile": pct, **t})
                    print(f"   initial balance, percentile {pct}: real {t['observed_balance']:.3f} vs "
                          f"shuffled {t['shuffled_mean_balance']:.3f} "
                          f"(p more balanced {t['p_more_balanced']:.4f}, p less balanced {t['p_less_balanced']:.4f})")
                write_csv(os.path.join(args.out, f"initial_balance_{tag}.csv"), init_rows)

            if "noise" in args.only:
                rows, raw = noise_experiment(G0, noise_levels, repeats, args.workers,
                                             args.max_steps, args.patience, args.seed)
                write_csv(os.path.join(args.out, f"noise_{tag}.csv"), rows)
                write_csv(os.path.join(args.out, f"noise_{tag}_runs.csv"), raw)
                plot_noise(rows, f"Noise and inertia ({tag}, {repeats} repeats)",
                           os.path.join(args.out, f"noise_{tag}.png"))
                for t in tipping_points(rows):
                    print(f"   noise at 50% settling, inertia={t['inertia']}: {fmt_point(t['noise_at_50pct_settling'])}")
                    summary.append({"experiment": "noise", "network": tag, **t})
                print(f"   [{time.time() - start:.0f}s] noise done")

            if "density" in args.only:
                rows, raw = density_experiment(G0, fractions, repeats, args.workers,
                                               args.max_steps, args.patience, args.seed)
                write_csv(os.path.join(args.out, f"density_{tag}.csv"), rows)
                plot_density(rows, f"Density and global consistency ({tag})",
                             os.path.join(args.out, f"density_{tag}.png"))
                print(f"   [{time.time() - start:.0f}s] density done")

            if "threshold" in args.only:
                rows, raw = threshold_experiment(csv_path, n, percentiles, thr_noise, repeats,
                                                 args.workers, args.max_steps, args.patience, args.seed)
                write_csv(os.path.join(args.out, f"threshold_{tag}.csv"), rows)
                plot_threshold(rows, f"Sign threshold sensitivity ({tag})",
                               os.path.join(args.out, f"threshold_{tag}.png"))
                for t in tipping_points(rows, group_key="percentile"):
                    print(f"   percentile {t['percentile']}, inertia={t['inertia']}: "
                          f"noise at 50% settling {fmt_point(t['noise_at_50pct_settling'])}")
                    summary.append({"experiment": "threshold", "network": tag, **t})
                print(f"   [{time.time() - start:.0f}s] threshold done")

    write_csv(os.path.join(args.out, "tipping_points.csv"), summary)
    print(f"\nFinished in {time.time() - start:.0f}s. Files are in {args.out}/")


if __name__ == "__main__":
    main()
