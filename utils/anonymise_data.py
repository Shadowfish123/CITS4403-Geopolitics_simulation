"""
Anonymise a raw GDELT snapshot before it is committed
======================================================
The raw snapshot written by utils/gdelt_fetch.py contains real country codes.
Project rule: no real country identities in anything we commit or show.
This script relabels every country as C001, C002, ... in a seeded random order
and writes a copy with identical numbers (tone, mentions, events).

The code-to-label mapping is never printed or saved.

Usage (from the project root):
    py utils/gdelt_fetch.py --out data/raw_gdelt_snapshot.csv
    py utils/anonymise_data.py
Raw files named data/raw_*.csv are listed in .gitignore and stay on your laptop.
"""

import argparse
import csv
import os
import random


def anonymise_csv(raw_path, out_path, seed=0):
    """Write an anonymised copy of raw_path to out_path. Returns (rows, countries)."""
    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))

    codes = sorted({r["country_a"] for r in rows} | {r["country_b"] for r in rows})
    random.Random(seed).shuffle(codes)
    label = {code: f"C{i + 1:03d}" for i, code in enumerate(codes)}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country_a", "country_b", "avg_tone",
                         "num_mentions", "num_events"])
        for r in rows:
            writer.writerow([label[r["country_a"]], label[r["country_b"]],
                             r["avg_tone"], r["num_mentions"],
                             r.get("num_events", "")])
    return len(rows), len(codes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymise a raw GDELT snapshot.")
    parser.add_argument("--raw", default="data/raw_gdelt_snapshot.csv")
    parser.add_argument("--out", default="data/country_relationships.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_rows, n_countries = anonymise_csv(args.raw, args.out, args.seed)
    print(f"Wrote {n_rows} pairs across {n_countries} countries to {args.out}")
