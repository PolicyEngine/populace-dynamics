"""Canonical firm-size banding — interface contract C2.

**Semantics (review finding F5).** The canonical variable means
*administrative enterprise size*: the total employment of the legal
enterprise across all its locations, as the Census Bureau counts it in
SUSB (March-12 payroll headcount). Every survey label is a *noisy
measure* of that quantity, not an alternative definition of it:

* CPS ASEC firm size is worker-reported firm size at all locations,
  for the longest job held in the *preceding calendar year*. **Two
  distinct codings measure it and they are not interchangeable:** the
  raw Census ASEC person file carries ``NOEMP`` (codes 1-6), while
  the IPUMS-CPS harmonised extract carries ``FIRMSIZE`` (a different,
  wider code set). The same integer means different bands in each —
  e.g. code 2 is *10-49* in NOEMP (2011-2018) but *10-24* in IPUMS
  ``FIRMSIZE`` — so the mapper takes an explicit ``coding`` argument
  and refuses to guess (see :func:`cps_firmsize_to_canonical` and
  :func:`noemp_to_canonical`);
* SIPP 2014+ ``EJB1_EMPSIZE`` is worker-reported size *at the
  worker's location* (establishment size — the redesign dropped the
  all-locations question), so it is a proxy-chain input, biased
  toward small bands for multi-establishment firms;
* LEHD QWI/J2J firm-size categories are administrative (derived from
  UI/BLS records), national March employment of the employer.

Bands are **headcount** bands. Policy thresholds stated in FTEs (the
ACA applicable-large-employer cut is 50 *full-time equivalents* at 30
hours/week, not headcount) are handled by a person-side hours join and
are out of C2 scope.

**Canonical bands.** Five bands with edges at 10 / 50 / 100 / 500::

    LT10        1-9
    B10_49     10-49
    B50_99     50-99
    B100_499  100-499
    B500_PLUS 500+

Rationale: 50 must be a band edge (ACA/state-mandate thresholds; QWI's
20-49/50-249 cut and the detailed SUSB classes support it), and these
edges exactly nest the SUSB detailed enterprise-size classes and the
2011-2018 ASEC bands — the only ASEC vintage whose bands resolve 50.

**Mappings are total but not always exact.** Every raw source code
maps to exactly one :class:`BandSpan` — a contiguous run of canonical
bands. ``exact=True`` means the source band lies inside a single
canonical band. Source bands that straddle a canonical edge (e.g. the
2019+ ASEC "25 to 99", BDS "20 to 99", QWI "0-19", most SIPP bands)
return a multi-band span: the ambiguity is carried explicitly rather
than resolved by a hidden convention (issue #192 review, point 3).
Down-stream code must either work at a coarseness where the span
collapses to one band or model the within-span allocation.

Boundary convention: canonical bands partition the positive integers;
a source label "X to Y" is read as the integer interval [X, Y]. SIPP's
inclusive upper bounds ("26 to 50", "201 to 500", ...) therefore
straddle canonical edges by a single integer and are returned as
(conservative) multi-band spans.

Verified sources for every raw code set (retrieved 2026-07-14):

* CPS/IPUMS ``FIRMSIZE`` codes and the 2011-2018 vs 2019+ band break:
  https://cps.ipums.org/cps-action/variables/FIRMSIZE
* SIPP ``EJB1_EMPSIZE`` categories:
  https://api.census.gov/data/2023/sipp/variables/EJB1_EMPSIZE.json
* SUSB detailed enterprise-size classes (``ENTRSIZE``): the committed
  ``data/external/susb_us_sector_size_2022.csv`` labels;
* LEHD firm-size labels:
  https://lehd.ces.census.gov/data/schema/latest/label_firmsize.csv
* BDS ``fsize`` categories: the committed
  ``data/external/bds_us_firm_size_1978_2022.csv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CanonicalBand",
    "BandSpan",
    "CANONICAL_BANDS",
    "band_of_count",
    "cps_firmsize_to_canonical",
    "noemp_to_canonical",
    "sipp_empsize_to_canonical",
    "susb_entrsize_to_canonical",
    "lehd_firmsize_to_canonical",
    "bds_fsize_to_canonical",
    "CPS_FIRMSIZE_INTERVALS_2011_2018",
    "CPS_FIRMSIZE_INTERVALS_STANDARD",
    "CPS_FIRMSIZE_VALID_CODES",
    "CPS_NOEMP_INTERVALS_2011_2018",
    "CPS_NOEMP_INTERVALS_2019_PLUS",
    "SIPP_EMPSIZE_INTERVALS",
    "SUSB_ENTRSIZE_INTERVALS",
    "SUSB_SUBTOTAL_CODES",
    "LEHD_FIRMSIZE_INTERVALS",
    "BDS_FSIZE_INTERVALS",
]


class CanonicalBand(Enum):
    """Canonical administrative enterprise-size bands (headcount)."""

    LT10 = (1, 9)
    B10_49 = (10, 49)
    B50_99 = (50, 99)
    B100_499 = (100, 499)
    B500_PLUS = (500, math.inf)

    @property
    def lo(self) -> int:
        return self.value[0]

    @property
    def hi(self) -> float:
        return self.value[1]


CANONICAL_BANDS: tuple[CanonicalBand, ...] = tuple(CanonicalBand)


@dataclass(frozen=True)
class BandSpan:
    """The canonical band(s) covering one raw source category.

    ``bands`` is a contiguous, ordered run of canonical bands.
    ``exact`` is True iff the source interval lies within a single
    canonical band (``len(bands) == 1``).
    """

    bands: tuple[CanonicalBand, ...]

    @property
    def exact(self) -> bool:
        return len(self.bands) == 1

    @property
    def band(self) -> CanonicalBand:
        """The single canonical band; raises unless :attr:`exact`."""
        if not self.exact:
            raise ValueError(
                f"Span {self.bands} is ambiguous; no single canonical "
                "band. Handle the ambiguity explicitly."
            )
        return self.bands[0]


def band_of_count(n: int) -> CanonicalBand:
    """Canonical band containing an exact employment count ``n >= 1``.

    ``n`` must be a whole number (headcount); a non-integral value such
    as ``9.5`` is a caller error and raises ``ValueError`` rather than
    falling through the partition.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        if not (isinstance(n, float) and n.is_integer()):
            raise ValueError(
                f"Employment count must be a whole number, got {n!r}."
            )
        n = int(n)
    if n < 1:
        raise ValueError(f"Employment count must be >= 1, got {n}.")
    for band in CANONICAL_BANDS:
        if band.lo <= n <= band.hi:
            return band
    raise AssertionError("unreachable: bands partition [1, inf)")


def _span(lo: int, hi: float) -> BandSpan:
    """The (contiguous) canonical bands intersecting [lo, hi]."""
    if hi < lo:
        raise ValueError(f"Inverted interval [{lo}, {hi}].")
    bands = tuple(b for b in CANONICAL_BANDS if b.lo <= hi and b.hi >= lo)
    if not bands:
        raise ValueError(f"Empty interval [{lo}, {hi}].")
    return BandSpan(bands)


# ---------------------------------------------------------------------
# CPS ASEC NOEMP / IPUMS FIRMSIZE
# ---------------------------------------------------------------------

#: IPUMS FIRMSIZE code -> integer interval, 2011-2018 ASEC vintages
#: (bands: under 10 / 10-49 / 50-99 / 100-499 / 500-999 / 1000+).
CPS_FIRMSIZE_INTERVALS_2011_2018: dict[int, tuple[int, float]] = {
    1: (1, 9),  # Under 10
    4: (10, 49),  # 10 to 49
    6: (50, 99),  # 50 to 99
    7: (100, 499),  # 100 to 499
    8: (500, 999),  # 500 to 999
    9: (1000, math.inf),  # 1000+
}

#: IPUMS FIRMSIZE code -> interval for 1992-2010 and 2019+ vintages
#: (bands: under 10 / 10-24 / 25-99 / 100-499 / 500-999 / 1000+); the
#: pre-2011 format returned in 2019. Code 3 ("Under 25") is the
#: 1988-1991 grouping, kept for completeness.
CPS_FIRMSIZE_INTERVALS_STANDARD: dict[int, tuple[int, float]] = {
    1: (1, 9),  # Under 10
    2: (10, 24),  # 10 to 24
    # Code 3 overlaps codes 1 and 2 by value but never co-occurs with
    # them: it is exclusive to the 1988-1991 vintage. The non-overlap
    # invariant is therefore checked on this table with code 3 removed.
    3: (1, 24),  # Under 25 (1988-1991)
    5: (25, 99),  # 25 to 99 -- straddles the 50 edge
    7: (100, 499),
    8: (500, 999),
    9: (1000, math.inf),
}

#: FIRMSIZE codes that are not firm-size reports (NIU).
CPS_FIRMSIZE_NIU = {0}

#: IPUMS ``FIRMSIZE`` codes that actually occur, per survey-year
#: regime (verified against the IPUMS-CPS availability grid). A code
#: outside its regime's set is vintage-impossible and must raise
#: rather than borrow an interval from another vintage: e.g. code 3
#: ("Under 25") is exclusive to 1988-1991, and codes 2/4 flip meaning
#: across the 2011-2018 break. Only the 2011+ ASEC firm-size era is
#: served here (matching the loader side, #194); pre-2011 vintages
#: and the 1988-1991 "Under 25" grouping are documented in the
#: interval tables but are not dispatchable.
CPS_FIRMSIZE_VALID_CODES: dict[str, frozenset[int]] = {
    "2011_2018": frozenset({1, 4, 6, 7, 8, 9}),
    "standard": frozenset({1, 2, 5, 7, 8, 9}),
}


def _cps_firmsize_span(code: int, year: int) -> BandSpan | None:
    """IPUMS ``FIRMSIZE`` leg of :func:`cps_firmsize_to_canonical`."""
    if code in CPS_FIRMSIZE_NIU:
        return None
    if 2011 <= year <= 2018:
        table, valid = CPS_FIRMSIZE_INTERVALS_2011_2018, "2011_2018"
    elif year >= 1992:
        table, valid = CPS_FIRMSIZE_INTERVALS_STANDARD, "standard"
    else:
        raise ValueError(
            f"No verified IPUMS FIRMSIZE code set for ASEC {year}; only "
            "1992+ is served (the 1988-1991 'Under 25' vintage is "
            "documented but not dispatchable)."
        )
    if code not in CPS_FIRMSIZE_VALID_CODES[valid]:
        raise KeyError(
            f"IPUMS FIRMSIZE code {code} does not occur in ASEC {year} "
            f"(valid: {sorted(CPS_FIRMSIZE_VALID_CODES[valid])}). If this "
            "is a raw Census NOEMP code, pass coding='census_noemp'."
        )
    return _span(*table[code])


def cps_firmsize_to_canonical(
    code: int, year: int, *, coding: str = "ipums_firmsize"
) -> BandSpan | None:
    """Map a CPS ASEC firm-size code to canonical bands.

    ``coding`` selects the source code set and is **required to be
    explicit about which one is in hand**, because the two codings
    share integers with different meanings (module docstring):

    * ``"ipums_firmsize"`` (default) — IPUMS-CPS harmonised
      ``FIRMSIZE``;
    * ``"census_noemp"`` — the raw Census ASEC person-file ``NOEMP``
      (delegates to :func:`noemp_to_canonical`).

    ``year`` is the ASEC survey year (the bands changed for the
    2011-2018 ASECs and reverted from 2019 on). Returns ``None`` for
    NIU. Vintage-impossible codes raise ``KeyError`` rather than
    silently borrowing another vintage's interval. Note the 2019+
    code 5 ("25 to 99") straddles the canonical 50 edge and returns an
    inexact two-band span; identification of the 50 cut from 2019+
    ASEC alone is impossible (ADR 0003).
    """
    if coding == "census_noemp":
        return noemp_to_canonical(code, year)
    if coding != "ipums_firmsize":
        raise ValueError(
            f"Unknown coding {coding!r}; expected 'ipums_firmsize' or "
            "'census_noemp'."
        )
    return _cps_firmsize_span(code, year)


# ---------------------------------------------------------------------
# Raw Census ASEC person-file NOEMP (distinct numbering from FIRMSIZE)
# ---------------------------------------------------------------------

#: NOEMP code -> interval, 2011-2018 ASEC (code 0 is NIU). Bands:
#: under 10 / 10-49 / 50-99 / 100-499 / 500-999 / 1000+. This is the
#: raw Census coding emitted by the ASEC firm-size loader (#194); the
#: integers do **not** line up with IPUMS FIRMSIZE.
CPS_NOEMP_INTERVALS_2011_2018: dict[int, tuple[int, float]] = {
    1: (1, 9),  # under_10
    2: (10, 49),  # 10_49
    3: (50, 99),  # 50_99
    4: (100, 499),  # 100_499
    5: (500, 999),  # 500_999
    6: (1000, math.inf),  # 1000_plus
}

#: NOEMP code -> interval, 2019+ ASEC (code 0 is NIU). Bands:
#: under 10 / 10-24 / 25-99 / 100-499 / 500-999 / 1000+.
CPS_NOEMP_INTERVALS_2019_PLUS: dict[int, tuple[int, float]] = {
    1: (1, 9),  # under_10
    2: (10, 24),  # 10_24
    3: (25, 99),  # 25_99 -- straddles the 50 edge
    4: (100, 499),  # 100_499
    5: (500, 999),  # 500_999
    6: (1000, math.inf),  # 1000_plus
}

#: NOEMP codes that are not firm-size reports (NIU).
CPS_NOEMP_NIU = {0}


def noemp_to_canonical(code: int, year: int) -> BandSpan | None:
    """Map a raw Census ASEC ``NOEMP`` code to canonical bands.

    The counterpart to :func:`cps_firmsize_to_canonical` for the raw
    person-file coding used by the ASEC firm-size loader (#194).
    ``year`` selects the 2011-2018 vs 2019+ band regime; only the 2011+
    firm-size era is defined. Returns ``None`` for NIU (code 0). The
    2019+ code 3 ("25 to 99") straddles the canonical 50 edge and
    returns an inexact two-band span.
    """
    if code in CPS_NOEMP_NIU:
        return None
    if 2011 <= year <= 2018:
        table = CPS_NOEMP_INTERVALS_2011_2018
    elif year >= 2019:
        table = CPS_NOEMP_INTERVALS_2019_PLUS
    else:
        raise ValueError(
            f"No verified NOEMP band regime for ASEC {year}; the raw "
            "ASEC firm-size series starts in 2011."
        )
    if code not in table:
        raise KeyError(
            f"NOEMP code {code} is not valid for ASEC {year} "
            f"(valid: {sorted(table)})."
        )
    return _span(*table[code])


# ---------------------------------------------------------------------
# SIPP 2014+ EJB1_EMPSIZE (establishment size — a proxy, not firm size)
# ---------------------------------------------------------------------

#: EJB1_EMPSIZE code -> interval (Census SIPP variable metadata).
#: Inclusive upper bounds ("10 to 25", "26 to 50", ...) straddle the
#: canonical edges by one integer; spans are conservative.
SIPP_EMPSIZE_INTERVALS: dict[int, tuple[int, float]] = {
    1: (1, 9),  # Less than 10
    2: (10, 25),  # 10 to 25
    3: (26, 50),  # 26 to 50
    4: (51, 100),  # 51 to 100
    5: (101, 200),  # 101 to 200
    6: (201, 500),  # 201 to 500
    7: (501, 1000),  # 501 to 1000
    8: (1001, math.inf),  # Greater than 1000
}

SIPP_EMPSIZE_MISSING = {-9}


def sipp_empsize_to_canonical(code: int) -> BandSpan | None:
    """Map a SIPP ``EJB1_EMPSIZE`` code to canonical bands.

    Returns ``None`` for missing (-9). Remember the measured concept
    is *establishment* size; even an exact band mapping remains a
    noisy proxy for enterprise size (module docstring).
    """
    if code in SIPP_EMPSIZE_MISSING:
        return None
    if code not in SIPP_EMPSIZE_INTERVALS:
        raise KeyError(f"EJB1_EMPSIZE code {code} is not valid.")
    return _span(*SIPP_EMPSIZE_INTERVALS[code])


# ---------------------------------------------------------------------
# SUSB detailed enterprise-size classes (ENTRSIZE)
# ---------------------------------------------------------------------

#: Detail ENTRSIZE code (zero-padded string, 2022 vintage) -> interval.
SUSB_ENTRSIZE_INTERVALS: dict[str, tuple[int, float]] = {
    "02": (1, 4),  # <5
    "03": (5, 9),
    "04": (10, 14),
    "05": (15, 19),
    "06": (20, 24),
    "07": (25, 29),
    "08": (30, 34),
    "09": (35, 39),
    "10": (40, 49),
    "11": (50, 74),
    "12": (75, 99),
    "13": (100, 149),
    "14": (150, 199),
    "15": (200, 299),
    "16": (300, 399),
    "17": (400, 499),
    "18": (500, 749),
    "19": (750, 999),
    # Census skips codes 20/21; "31" really is the 1,000-1,499 class
    # (verified against the raw 2022 SUSB detailed-sizes file).
    "31": (1000, 1499),
    "22": (1500, 1999),
    "23": (2000, 2499),
    "24": (2500, 4999),
    "25": (5000, math.inf),
}

#: Total/subtotal ENTRSIZE codes (excluded from the detail mapping):
#: 01 = Total, 33 = <20, 37 = <500.
SUSB_SUBTOTAL_CODES = {"01", "33", "37"}


def susb_entrsize_to_canonical(code: str) -> BandSpan | None:
    """Map a SUSB detailed ``ENTRSIZE`` code to canonical bands.

    Returns ``None`` for total/subtotal codes. Every detail class
    nests exactly in one canonical band (the canonical edges were
    chosen for this).
    """
    key = str(code).zfill(2)
    if key in SUSB_SUBTOTAL_CODES:
        return None
    if key not in SUSB_ENTRSIZE_INTERVALS:
        raise KeyError(f"SUSB ENTRSIZE code {code!r} is not valid.")
    return _span(*SUSB_ENTRSIZE_INTERVALS[key])


# ---------------------------------------------------------------------
# LEHD QWI / J2J firm-size categories
# ---------------------------------------------------------------------

#: LEHD firmsize code -> interval (label_firmsize.csv). Code 1 is
#: "0-19": zero-employment quarters occur in administrative data; the
#: interval is clipped to start at 1 for banding purposes.
LEHD_FIRMSIZE_INTERVALS: dict[int, tuple[int, float]] = {
    1: (1, 19),  # 0-19 Employees
    2: (20, 49),
    3: (50, 249),
    4: (250, 499),
    5: (500, math.inf),
}

#: Code 0 is the all-sizes margin; "N" is public-sector not-available.
LEHD_FIRMSIZE_NON_DETAIL = {"0", "N"}


def lehd_firmsize_to_canonical(code: int | str) -> BandSpan | None:
    """Map a QWI/J2J ``firmsize`` code to canonical bands.

    Returns ``None`` for the all-sizes margin (0) and the
    public-sector "N" category.
    """
    if str(code) in LEHD_FIRMSIZE_NON_DETAIL:
        return None
    try:
        key = int(code)
    except (TypeError, ValueError):
        raise KeyError(f"LEHD firmsize code {code!r} is not valid.") from None
    if key not in LEHD_FIRMSIZE_INTERVALS:
        raise KeyError(f"LEHD firmsize code {code!r} is not valid.")
    return _span(*LEHD_FIRMSIZE_INTERVALS[key])


# ---------------------------------------------------------------------
# BDS firm-size categories (fsize)
# ---------------------------------------------------------------------

#: BDS ``fsize`` label -> interval (bds2022_fz.csv categories).
BDS_FSIZE_INTERVALS: dict[str, tuple[int, float]] = {
    "a) 1 to 4": (1, 4),
    "b) 5 to 9": (5, 9),
    "c) 10 to 19": (10, 19),
    "d) 20 to 99": (20, 99),  # straddles the 50 edge
    "e) 100 to 499": (100, 499),
    "f) 500 to 999": (500, 999),
    "g) 1000 to 2499": (1000, 2499),
    "h) 2500 to 4999": (2500, 4999),
    "i) 5000 to 9999": (5000, 9999),
    "j) 10000+": (10000, math.inf),
}


def bds_fsize_to_canonical(fsize: str) -> BandSpan:
    """Map a BDS ``fsize`` category label to canonical bands."""
    if fsize not in BDS_FSIZE_INTERVALS:
        raise KeyError(f"BDS fsize category {fsize!r} is not valid.")
    return _span(*BDS_FSIZE_INTERVALS[fsize])
