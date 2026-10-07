"""
GDELT fetch tests: parsing and aggregating export files, with no internet needed.

Run from the project root:  py -m pytest tests -v
"""

import io
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
import gdelt_fetch as gf


def export_row(a1, a2, mentions, tone):
    """One line of a GDELT 2.0 events export (61 tab-separated columns)."""
    cols = [""] * 61
    cols[gf.COL_ACTOR1_COUNTRY] = a1
    cols[gf.COL_ACTOR2_COUNTRY] = a2
    cols[gf.COL_NUM_MENTIONS] = str(mentions)
    cols[gf.COL_AVG_TONE] = str(tone)
    cols[6] = 'Name with "quotes" inside'
    return "\t".join(cols)


def make_zip(lines):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("export.CSV", "\n".join(lines))
    return buf.getvalue()


def parse(*files):
    data = defaultdict(lambda: [0.0, 0, 0])
    used = [gf.parse_export_bytes(f, data) for f in files]
    return data, used


@pytest.mark.parametrize("code, ok", [("USA", True), ("us", False), ("usa", False),
                                       ("", False), ("US1", False), ("ABCD", False)])
def test_country_code_validation(code, ok):
    assert gf.is_country_code(code) is ok


def test_pair_order_does_not_matter_and_tone_is_mention_weighted():
    data, _ = parse(make_zip([export_row("AAA", "BBB", 10, -2.0),
                              export_row("BBB", "AAA", 30, 2.0)]))
    tone_sum, mentions, events = data[("AAA", "BBB")]
    assert (mentions, events) == (40, 2)
    assert tone_sum / mentions == pytest.approx(1.0)       # (10*-2 + 30*2) / 40


def test_invalid_rows_are_skipped():
    data, used = parse(make_zip([export_row("AAA", "AAA", 5, 1.0),     # same country
                                 export_row("", "BBB", 5, 1.0),         # missing code
                                 export_row("ab", "BBB", 5, 1.0),       # bad code
                                 "too\tshort",                           # malformed
                                 export_row("AAA", "CCC", 4, -5.0)]))   # the only good one
    assert used == [1] and list(data) == [("AAA", "CCC")]


def test_aggregation_across_several_files():
    f1 = make_zip([export_row("AAA", "BBB", 10, -2.0)])
    f2 = make_zip([export_row("AAA", "BBB", 10, -4.0)])
    data, _ = parse(f1, f2)
    tone_sum, mentions, _ = data[("AAA", "BBB")]
    assert mentions == 20 and tone_sum / mentions == pytest.approx(-3.0)


def test_timestamps_cover_the_day():
    hourly = gf.build_timestamps(datetime(2026, 10, 1), every=4)
    assert len(hourly) == 24
    assert hourly[0] == "20261001000000" and hourly[-1] == "20261001230000"
    assert len(gf.build_timestamps(datetime(2026, 10, 1), every=1)) == 96


def test_saved_csv_is_sorted_by_mentions_and_has_a_header(tmp_path):
    data, _ = parse(make_zip([export_row("AAA", "BBB", 5, -1.0), export_row("AAA", "CCC", 50, -2.0)]))
    out = tmp_path / "out" / "rel.csv"
    rows = gf.save_to_csv(data, str(out))
    assert rows[0][3] == 50
    assert out.read_text().splitlines()[0] == "country_a,country_b,avg_tone,num_mentions,num_events"
