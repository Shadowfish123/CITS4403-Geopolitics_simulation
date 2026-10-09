"""
Geographically Gated Experiments on the Real-Data Networks
===========================================================
CITS4403 Research Project

Driver for the geography layer. It builds the same eight-nation snapshot
networks as run_experiments.py, thins them with a geography gate, and then runs
the noise experiment on real, randomly thinned and gated networks so the three
can be compared at the same number of relationships.

The map is a random draw, so every arm is repeated over many position draws and
the tables report both the pooled rate and the spread across draws.

Usage (from the project root):
    py src/run_geo_experiments.py --quick
    py src/run_geo_experiments.py --draws 30 --repeats 5 --workers 4
    py src/run_geo_experiments.py --only noise gate --n 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_pipeline as dp                  # noqa: E402
import geography as geo                     # noqa: E402
import geo_experiments as gex               # noqa: E402
import run_experiments as ex                # noqa: E402

EXPERIMENT_NAMES = ("noise", "gate", "space", "economy", "keep")


def save_figure(function_name, rows, *args, **kwargs):
    """
    Draw a figure if this version of geo_experiments.py provides the helper.

    The driver is committed before all of the figures exist, so a missing
    plotting function skips the picture instead of stopping the run.
    """
    function = getattr(gex, function_name, None)
    if function is None:
        print("      (%s not available in this version, figure skipped)" % function_name)
        return
    function(rows, *args, **kwargs)


def describe_snapshot(G0, r_min, keep, base_seed, space="plane"):
    """Print what the gate actually does to one draw of the network."""
    print("   substrate   edges  triangles  components  isolated  mean distance")
    for kind in gex.SUBSTRATES:
        H, positions, economy, _ = gex.build_substrate(
            G0, kind, draw_seed=base_seed, keep=keep, space=space, r_min=r_min)
        summary = geo.describe(H, positions)
        print("   %-10s %5d %10d %11d %9d %14.3f" % (
            kind, summary["edges"], summary["triangles"],
            summary["components"], summary["isolated_nodes"],
            summary["mean_edge_length"]))


def main():
    parser = argparse.ArgumentParser(description="Run the geography experiments.")
    parser.add_argument("--days", nargs="+",
                        default=["data/country_relationships.csv",
                                 "data/country_relationships_day2.csv"])
    parser.add_argument("--n", type=int, nargs="+", default=[8],
                        help="number of top countries per network")
    parser.add_argument("--draws", type=int, default=30,
                        help="position draws per condition")
    parser.add_argument("--repeats", type=int, default=5,
                        help="dynamics seeds inside each position draw")
    parser.add_argument("--keep", type=float, default=gex.DEFAULT_KEEP,
                        help="share of relationships the thinned arms keep")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default="results_geo")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENT_NAMES),
                        default=list(EXPERIMENT_NAMES))
    parser.add_argument("--quick", action="store_true",
                        help="small settings to check everything runs")
    args = parser.parse_args()

    draws, repeats = args.draws, args.repeats
    noise_levels = gex.NOISE_LEVELS
    if args.quick:
        draws, repeats = 5, 3
        noise_levels = [0.0, 0.01, 0.02]

    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    summary = []

    for csv_path in args.days:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        day = "day2" if stem.endswith("day2") else (
            "day1" if stem == "country_relationships" else stem)
        for n in args.n:
            tag = "%s_n%d" % (day, n)
            G0 = ex.network_from_csv(csv_path, n)
            s = dp.summarise(G0)
            print("\n== %s: %d nations, %d relationships, %d triangles ==" % (
                tag, s["nodes"], s["edges"], s["triangles"]))

            r_min = geo.calibrate_r_min(G0, target_keep=args.keep,
                                        probes=gex.CALIBRATION_PROBES,
                                        seed=args.seed)
            print("   calibrated reach intercept r_min = %.3f "
                  "(target: keep %.0f%% of relationships)" % (r_min, 100 * args.keep))
            describe_snapshot(G0, r_min, args.keep, args.seed)

            for name in args.only:
                function = getattr(gex, name + "_experiment", None)
                if function is None:
                    print("   [%s] not available in this version, skipped" % name)
                    continue

                if name == "noise":
                    rows, raw = function(
                        G0, noise_levels, draws, repeats, args.workers,
                        args.max_steps, args.patience, args.seed,
                        keep=args.keep, r_min=r_min)
                    gex.write_csv(os.path.join(args.out, "noise_%s.csv" % tag), rows)
                    gex.write_csv(os.path.join(args.out, "noise_%s_runs.csv" % tag), raw)
                    save_figure("plot_by_key", rows, "substrate",
                                "Settling rate: real vs randomly thinned vs gated (%s, "
                                "%d draws)" % (tag, draws),
                                os.path.join(args.out, "noise_%s.png" % tag),
                                use_inertia=False)
                    save_figure("plot_inertia", rows,
                                "Inertia and geography (%s, %d draws)" % (tag, draws),
                                os.path.join(args.out, "noise_inertia_%s.png" % tag))
                    save_figure("plot_structure", rows, "substrate",
                                "What the gate does to the network (%s, %d draws)"
                                % (tag, draws),
                                os.path.join(args.out, "structure_%s.png" % tag))
                    for point in gex.tipping_points(rows, group_key="substrate"):
                        print("   %s, inertia=%s: noise at 50%% settling %s" % (
                            point["substrate"], point["inertia"],
                            ex.fmt_point(point["noise_at_50pct_settling"])))
                        summary.append({"experiment": "noise", "network": tag, **point})

                elif name == "gate":
                    rows, raw = function(
                        G0, gex.GATE_NOISE_LEVELS, draws, repeats, args.workers,
                        args.max_steps, args.patience, args.seed,
                        keep=args.keep, r_min=None)
                    gex.write_csv(os.path.join(args.out, "gate_%s.csv" % tag), rows)
                    save_figure("plot_by_key", rows, "gate",
                                "Hard gate vs soft gate (%s, %d draws)" % (tag, draws),
                                os.path.join(args.out, "gate_%s.png" % tag))

                elif name == "space":
                    rows, raw = function(
                        G0, gex.SPACE_NOISE_LEVELS, draws, repeats, args.workers,
                        args.max_steps, args.patience, args.seed,
                        keep=args.keep, r_min=None)
                    gex.write_csv(os.path.join(args.out, "space_%s.csv" % tag), rows)
                    save_figure("plot_by_key", rows, "space",
                                "Flat map vs map without corners (%s, %d draws)"
                                % (tag, draws),
                                os.path.join(args.out, "space_%s.png" % tag))

                elif name == "economy":
                    rows, raw = function(
                        G0, gex.ECONOMY_NOISE_LEVELS, draws, repeats, args.workers,
                        args.max_steps, args.patience, args.seed,
                        keep=args.keep, r_min=None)
                    gex.write_csv(os.path.join(args.out, "economy_%s.csv" % tag), rows)
                    save_figure("plot_by_key", rows, "economy",
                                "Fixed economic tiers vs random economies (%s, %d draws)"
                                % (tag, draws),
                                os.path.join(args.out, "economy_%s.png" % tag))

                else:
                    rows, raw = function(
                        G0, gex.KEEP_NOISE_LEVELS, draws, repeats, args.workers,
                        args.max_steps, args.patience, args.seed)
                    gex.write_csv(os.path.join(args.out, "keep_%s.csv" % tag), rows)
                    save_figure("plot_keep_sweep", rows,
                                "Closing the gate (%s, %d draws)" % (tag, draws),
                                os.path.join(args.out, "keep_%s.png" % tag))

                print("   [%s] done  [%.0fs]" % (name, time.time() - start))

    if summary:
        gex.write_csv(os.path.join(args.out, "tipping_points_geo.csv"), summary)
    print("\nFinished in %.0fs. Files are in %s/" % (time.time() - start, args.out))


if __name__ == "__main__":
    main()
