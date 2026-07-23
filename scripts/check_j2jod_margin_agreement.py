"""Cross-source check: LED tool J2JOD margins vs the LEHD flat file.

Reproduces the corrected margin caveat in
``data/external/employer_firm_target_sources.md`` entry 6.

Two independent publications of the same quantity — the national
J2JOD firm-size margins — disagree, and the size of that
disagreement is the point. E11's constraints after 2016Q1 are
margins-only (the origin x destination detail is suppressed from
2016Q2), so cross-source wobble on the margins is a noise datum for
the E11 floor build rather than a provenance footnote.

The earlier version of the caveat attributed the gap to the flat
file's inclusion of public-sector ("N" firm size) flows. That
explanation predicts the tool's margin sits *below* the flat file in
every quarter. It sits above in 37 of 41. The deviations run both
ways, consistent with independent noise infusion applied to the two
tabulations.

Run from the repository root::

    .venv/bin/python scripts/check_j2jod_margin_agreement.py
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "data" / "external" / "j2jod_us_firmsize_od_2015on.csv"

#: The one-sided-margin flat file, release-stamped (not
#: ``latest_release``, which is a moving alias).
FLAT_URL = (
    "https://lehd.ces.census.gov/data/j2j/R2026Q1/us/j2jod/"
    "j2jod_us_d_fs_gn_ns_oslp_u.csv.gz"
)

#: The all-demographics / all-industry / all-firm-age national cell.
#: Both the destination-side and the ``_orig``-side dimensions must
#: be pinned: the flat file carries sector-crossed rows on each side,
#: and filtering only one side silently multiplies the row count.
MARGIN_FILTER = {
    "ind_level": "A",
    "industry": "00",
    "ind_level_orig": "A",
    "industry_orig": "00",
    "sex": 0,
    "agegrp": "A00",
    "race": "A0",
    "ethnicity": "A0",
    "education": "E0",
    "firmage": "0",
    "firmage_orig": "0",
    "ownercode": "A00",
    "ownercode_orig": "A00",
}

FIRST_YEAR = 2015


def load_flat() -> pd.DataFrame:
    """Download and filter the flat file to the national margin."""
    print(f"downloading {FLAT_URL}")
    with urllib.request.urlopen(FLAT_URL, timeout=300) as resp:
        payload = gzip.decompress(resp.read())
    frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
    mask = frame["year"] >= FIRST_YEAR
    for column, value in MARGIN_FILTER.items():
        mask &= frame[column].astype(type(value)) == value
    out = frame[mask].copy()
    out["fo"] = out["firmsize_orig"].astype(str)
    out["fd"] = out["firmsize"].astype(str)
    return out


def compare(flat: pd.DataFrame, tool: pd.DataFrame, fo: str, fd: str):
    """Percent deviation of the tool's EE from the flat file's."""
    a = flat[(flat.fo == fo) & (flat.fd == fd)][
        ["year", "quarter", "EE"]
    ].rename(columns={"EE": "flat"})
    b = tool[(tool.fo == fo) & (tool.fd == fd)][
        ["year", "quarter", "EE"]
    ].rename(columns={"EE": "tool"})
    merged = a.merge(b, on=["year", "quarter"]).dropna()
    merged = merged[merged["flat"] > 0]
    if merged.empty:
        return None
    merged["pct"] = (merged.tool - merged.flat) / merged.flat * 100
    return merged


def main() -> None:
    tool = pd.read_csv(EXTRACT)
    tool["fo"] = tool["firmsize_orig"].astype(str)
    tool["fd"] = tool["firmsize"].astype(str)
    flat = load_flat()

    rows = [("all-size EE margin", "0", "0")]
    rows += [(f"dest margin size {c}", "0", c) for c in "12345"]
    rows += [(f"orig margin size {c}", c, "0") for c in "12345"]

    print(
        f"\n{'comparison':24s} {'n':>3} {'above':>8} "
        f"{'min%':>7} {'max%':>7} {'mean%':>7}"
    )
    lo: list[float] = []
    hi: list[float] = []
    headline = None
    for label, fo, fd in rows:
        merged = compare(flat, tool, fo, fd)
        if merged is None:
            continue
        n = len(merged)
        above = int((merged.pct > 0).sum())
        print(
            f"{label:24s} {n:3d} {above:4d}/{n:<3d} "
            f"{merged.pct.min():7.2f} {merged.pct.max():7.2f} "
            f"{merged.pct.mean():7.2f}"
        )
        if label.startswith("all-size"):
            headline = (n, above)
        else:
            lo.append(merged.pct.min())
            hi.append(merged.pct.max())

    if headline is None or not lo:
        sys.exit("no comparable cells; the flat-file layout changed")

    n, above = headline
    print(
        f"\nall-size margin: tool above flat in {above}/{n} quarters"
        f"\nper-size envelope: {min(lo):.2f}% to {max(hi):.2f}%"
    )
    # A pure size-N exclusion would put the tool below the flat file
    # in every quarter. State the refutation rather than leaving the
    # reader to infer it from the table.
    if above > n / 2:
        print(
            "\nThe tool's margin is above the flat file in a majority "
            "of quarters, so the gap is not the flat file's extra "
            "public-sector ('N') flows: that would bias the tool's "
            "margin downward everywhere. Two-sided deviations of "
            "this size are consistent with independent noise "
            "infusion on the two tabulations."
        )


if __name__ == "__main__":
    main()
