"""
GDELT Data Fetcher (v2: aggregates many files)
================================================
Downloads several GDELT 2.0 event exports for one day, aggregates them
into country-pair relationship data, and saves a CSV for the model.

RUN THIS ON YOUR OWN LAPTOP (it needs access to data.gdeltproject.org).

Usage (from the project root):
    python utils/gdelt_fetch.py
    python utils/gdelt_fetch.py --date 2026-10-01 --every 4

Why many files? One 15-minute export only covers a handful of country
pairs. GDELT publishes 96 exports per day; --every 4 takes one per hour
(24 files), which gives a much denser pair network.

Output columns: country_a, country_b, avg_tone, num_mentions, num_events
    avg_tone     = mention-weighted average of GDELT AvgTone for the pair
    num_mentions = total NumMentions for the pair (used for FILTERING and
                   as the basis of relationship "inertia")
    num_events   = number of event rows contributing (for reporting)
"""

import argparse
import csv
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

BASE_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.export.CSV.zip"

# 0-based column positions in the GDELT 2.0 events export (61 columns)
COL_ACTOR1_COUNTRY = 7
COL_ACTOR2_COUNTRY = 17
COL_NUM_MENTIONS = 31
COL_AVG_TONE = 34


def build_timestamps(day, every=4):
    """Timestamps of the 96 daily 15-minute exports; keep every `every`-th."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    slots = [start + timedelta(minutes=15 * i) for i in range(96)]
    return [s.strftime("%Y%m%d%H%M%S") for s in slots[::every]]


def is_country_code(code):
    """GDELT country codes are 3 uppercase letters (e.g. 'USA')."""
    return len(code) == 3 and code.isalpha() and code.isupper()


def parse_export_bytes(zip_bytes, pair_data):
    """
    Add one export file's rows into pair_data, in place.
    pair_data[(a, b)] = [weighted_tone_sum, mention_sum, event_count]
    Returns the number of rows used.
    """
    used = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            reader = csv.reader(
                io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                try:
                    a1 = row[COL_ACTOR1_COUNTRY].strip()
                    a2 = row[COL_ACTOR2_COUNTRY].strip()
                    if a1 == a2 or not (is_country_code(a1) and is_country_code(a2)):
                        continue
                    mentions = int(row[COL_NUM_MENTIONS])
                    tone = float(row[COL_AVG_TONE])
                except (IndexError, ValueError):
                    continue
                pair = tuple(sorted((a1, a2)))   # A-B and B-A merge
                entry = pair_data[pair]
                entry[0] += tone * mentions
                entry[1] += mentions
                entry[2] += 1
                used += 1
    return used


def download(url, retries=2, timeout=60):
    """Return file bytes, or None if missing/unreachable."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # that 15-min slot simply doesn't exist
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return None


def save_to_csv(pair_data, out_path):
    rows = []
    for (a, b), (tone_sum, mention_sum, n_events) in pair_data.items():
        if mention_sum > 0:
            rows.append((a, b, round(tone_sum / mention_sum, 4), mention_sum, n_events))
    rows.sort(key=lambda r: r[3], reverse=True)   # most-mentioned pairs first

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country_a", "country_b", "avg_tone", "num_mentions", "num_events"])
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch and aggregate GDELT events.")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    parser.add_argument("--date", default=yesterday.isoformat(),
                        help="UTC day to fetch, YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--every", type=int, default=4,
                        help="take every Nth 15-min file (4 = hourly, 1 = all 96)")
    parser.add_argument("--out", default="data/country_relationships.csv")
    args = parser.parse_args()

    day = datetime.strptime(args.date, "%Y-%m-%d")
    timestamps = build_timestamps(day, every=args.every)
    print(f"Fetching up to {len(timestamps)} files for {args.date} ...")

    pair_data = defaultdict(lambda: [0.0, 0, 0])
    fetched = 0
    for i, ts in enumerate(timestamps, 1):
        data = download(BASE_URL.format(ts=ts))
        if data is None:
            print(f"  [{i}/{len(timestamps)}] {ts}: not available, skipped")
            continue
        used = parse_export_bytes(data, pair_data)
        fetched += 1
        print(f"  [{i}/{len(timestamps)}] {ts}: {used} usable rows")

    if fetched == 0:
        raise SystemExit("No files could be downloaded. Check your internet "
                         "connection or try a different --date.")

    rows = save_to_csv(pair_data, args.out)
    countries = {c for r in rows for c in r[:2]}
    print(f"\nDone: {fetched} files, {len(rows)} country pairs, "
          f"{len(countries)} countries -> {args.out}")


if __name__ == "__main__":
    main()
