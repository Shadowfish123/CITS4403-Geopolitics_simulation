"""
Identity-free summary of the data snapshot, for the report
===========================================================
Prints counts only (no country labels), so the output is safe to paste into
the report, notebook or demo.

Usage (from the project root):
    py utils/check_snapshot.py
    py utils/check_snapshot.py --n 8 10 --csv data/country_relationships.csv
"""

import argparse
import os
import statistics
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data_pipeline as dp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Summarise the data snapshot.")
    parser.add_argument("--csv", default="data/country_relationships.csv")
    parser.add_argument("--n", type=int, nargs="+", default=[8, 10],
                        help="numbers of top countries to summarise")
    args = parser.parse_args()

    rows = dp.load_relationships(args.csv)
    countries = {r["country_a"] for r in rows} | {r["country_b"] for r in rows}
    print(f"Snapshot: {len(rows)} country pairs across {len(countries)} countries")

    for n in args.n:
        top, kept = dp.select_top_countries(rows, n=n)
        possible = n * (n - 1) // 2
        threshold = dp.tone_threshold(kept, 50)
        G, _ = dp.build_network(kept, top)
        s = dp.summarise(G)
        mentions = sorted(r["num_mentions"] for r in kept)
        print(f"\nTop {n} countries by mention volume")
        print(f"  pairs with data: {len(kept)} of {possible} ({len(kept) / possible:.0%})")
        print(f"  median tone (sign threshold): {threshold:.2f}")
        print(f"  signs: {s['positive_edges']} positive, {s['negative_edges']} negative")
        print(f"  closed triangles: {s['triangles']}")
        print(f"  connected components: {nx.number_connected_components(G)}")
        print(f"  mentions min/median/max: {mentions[0]} / "
              f"{int(statistics.median(mentions))} / {mentions[-1]}")
        print(f"  inertia min/median/max: {s['inertia_min']:.2f} / "
              f"{s['inertia_median']:.2f} / {s['inertia_max']:.2f}")


if __name__ == "__main__":
    main()
