"""Build DRAFT aggregate-side noise floors for gates E1/E2/E6/E7/E11
(workstream B, issue #192).

REPORTED ANCHOR, NOT A GATE RUN — and explicitly a DRAFT: C3 (the
employer gate block) has not locked, no thresholds are proposed here,
and nothing below is ratified. This is the firm-side counterpart to
the workstream-A floor battery (#212): it commits the floor-building
method for the aggregate-reference gates before any candidate model
exists (issue #192 protocol: floors -> thresholds -> referee round ->
one-shot runs).

Unlike the survey-side floors, the references here are published
administrative aggregates, so person-disjoint half-splits are not
available. The floors are instead **temporal-stability floors**:

* **QWI/J2J (E2/E6/E7/E11 proxies):** for every published firm-size x
  sector cell, the year-over-year same-quarter absolute log ratio of
  each rate/level, 2015Q1 on. Same-quarter comparison absorbs
  seasonality (the extracts are not seasonally adjusted). The
  variation deliberately **includes true business-cycle signal** —
  most visibly the 2020-2021 pandemic years — so each floor is
  reported both on all pairs and excluding pairs touching 2020/2021,
  with both on the record rather than one adjusted away.
* **SUSB (E1):** the committed extract is a single 2022 cross-section
  with no temporal replicate, so a same-source stability floor is
  degenerate. E1 instead carries (a) the published SUSB noise flags
  (G/H/J), converted to the flag-implied relative-sd upper bound per
  cell, and (b) a BDS 2012-2022 year-over-year stability floor for
  the national firm-size *margin* (BDS has no sector axis; the
  sector-axis stability is not derivable from committed extracts —
  recorded as a method finding, not patched).

All band semantics come from
:mod:`populace_dynamics.firms.banding` — bands are never re-derived
here. Cells whose minimum denominator over the window is below
``THIN_JOBS`` (a draft choice, recorded) are flagged thin; national
cells are all thick in practice, and the flag is carried so the
state-level C3 cells inherit the convention.

Usage::

    python scripts/build_employer_firm_floors.py

writes ``runs/employer_firm_floors_draft_v0.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from populace_dynamics.firms import banding, targets  # noqa: E402

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/employer_firm_floors_draft_v0.json"
)

#: Pandemic years: YoY pairs touching these are reported separately
#: (never dropped from the full-sample figures).
PANDEMIC_YEARS = frozenset({2020, 2021})

#: Draft thin-cell threshold on the minimum cell denominator (jobs).
THIN_JOBS = 10_000

#: BDS stability window (recent decade ending at the extract's last
#: year); the full 1978-2022 series is in the committed extract if
#: the referee round prefers a different window.
BDS_WINDOW = (2012, 2022)

#: SUSB employment noise flags -> published relative-sd upper bound.
#: G: CV < 2%; H: 2-5%; J: >= 5% (no upper bound published -> null).
SUSB_FLAG_CV_BOUND = {"G": 0.02, "H": 0.05, "J": None}
SUSB_FLAG_SEVERITY = {"G": 0, "H": 1, "J": 2}

#: BDS fsize categories grouped to the coarsest partition that the
#: canonical bands can express without splitting a source category.
#: "20 to 99" straddles the canonical 50 edge (see banding) and is
#: carried as its own inexact group rather than allocated.
BDS_GROUPS: dict[str, tuple[str, ...]] = {
    "1_9": ("a) 1 to 4", "b) 5 to 9"),
    "10_19": ("c) 10 to 19",),
    "20_99": ("d) 20 to 99",),
    "100_499": ("e) 100 to 499",),
    "500_plus": (
        "f) 500 to 999",
        "g) 1000 to 2499",
        "h) 2500 to 4999",
        "i) 5000 to 9999",
        "j) 10000+",
    ),
}


def _round(x: float | None, nd: int = 5) -> float | None:
    return None if x is None else round(float(x), nd)


def _span_record(span: banding.BandSpan) -> dict:
    return {
        "canonical_bands": [b.name for b in span.bands],
        "exact": span.exact,
    }


def _gap_summary(gaps: list[tuple[int, float]]) -> dict:
    """Mean/sd of |log YoY ratio|, full and excluding pandemic pairs.

    ``gaps`` holds (later_year, gap); a pair touches the pandemic if
    either the later year or the year before it is in PANDEMIC_YEARS.
    """
    full = [g for _, g in gaps]
    ex = [
        g
        for y, g in gaps
        if y not in PANDEMIC_YEARS and (y - 1) not in PANDEMIC_YEARS
    ]
    # ddof=1. n is 8-11 pairs here, where the population sd
    # understates by ~5% -- and these sds feed a `mean + k * sd`
    # threshold policy, so understating them biases thresholds tight
    # (against the model) rather than harmlessly. It also matches the
    # A-side batteries' across-seed sds, which the referee round will
    # read alongside these. A single-pair cell has no dispersion and
    # yields None rather than 0.0, which would read as a perfect
    # floor (see the E11 detail window).
    out = {
        "floor_abs_log_ratio_mean": _round(np.mean(full)) if full else None,
        "floor_abs_log_ratio_sd": (
            _round(np.std(full, ddof=1)) if len(full) > 1 else None
        ),
        "n_pairs": len(full),
        "ex_pandemic_mean": _round(np.mean(ex)) if ex else None,
        "ex_pandemic_sd": (
            _round(np.std(ex, ddof=1)) if len(ex) > 1 else None
        ),
        "n_pairs_ex_pandemic": len(ex),
    }
    return out


def _yoy_gaps(cell: pd.DataFrame, col: str) -> list[tuple[int, float]]:
    """Same-quarter year-over-year |log ratio| pairs for one cell."""
    s = cell.set_index(["year", "quarter"])[col]
    gaps = []
    for (year, quarter), value in s.items():
        prev = s.get((year - 1, quarter))
        if prev is None or pd.isna(prev) or pd.isna(value):
            continue
        if prev <= 0 or value <= 0:
            continue
        gaps.append((int(year), abs(math.log(value / prev))))
    return gaps


def _lehd_cell_block(
    frame: pd.DataFrame,
    rate_cols: dict[str, str],
    denom_col: str,
) -> tuple[dict, dict]:
    """Per-firm-size detail at the all-industry margin + a cross-cell
    summary over every sector x firm-size cell.

    ``rate_cols`` maps output name -> frame column.
    """
    detail: dict = {}
    margin = frame[frame["industry"] == "00"]
    for code, cell in margin.groupby("firmsize"):
        label = cell["firmsize_label"].iloc[0]
        span = banding.lehd_firmsize_to_canonical(int(code))
        min_denom = int(cell[denom_col].min())
        rec: dict = {
            "firmsize_label": label,
            **_span_record(span),
            "min_denominator_jobs": min_denom,
            "thin": min_denom < THIN_JOBS,
        }
        for name, col in rate_cols.items():
            rec[name] = {
                "level_mean": _round(cell[col].mean()),
                **_gap_summary(_yoy_gaps(cell, col)),
            }
        detail[f"firmsize{int(code)}"] = rec

    sector_cells = frame[frame["industry"] != "00"]
    summary: dict = {
        "n_cells": int(sector_cells.groupby(["industry", "firmsize"]).ngroups),
    }
    for name, col in rate_cols.items():
        means = []
        n_thin = 0
        for _, cell in sector_cells.groupby(["industry", "firmsize"]):
            gaps = _yoy_gaps(cell, col)
            if not gaps:
                continue
            means.append(float(np.mean([g for _, g in gaps])))
            if int(cell[denom_col].min()) < THIN_JOBS:
                n_thin += 1
        summary[name] = {
            "cell_floor_median": _round(np.median(means)),
            "cell_floor_p90": _round(np.quantile(means, 0.9)),
            "cell_floor_max": _round(np.max(means)),
            "n_cells_with_pairs": len(means),
            "n_thin_cells": n_thin,
        }
    return detail, summary


# ---------------------------------------------------------------------
# E1 — SUSB employment share by firm-size band x sector
# ---------------------------------------------------------------------


def e1_block() -> dict:
    susb = targets.load_susb_sector_size()
    detail = susb[
        (susb["naics_sector"] != "--")
        & (~susb["entrsize_code"].isin(banding.SUSB_SUBTOTAL_CODES))
        & (susb["entrsize_code"] != "01")
    ].copy()
    detail["band"] = detail["entrsize_code"].map(
        lambda c: banding.susb_entrsize_to_canonical(c).band.name
    )
    by_sector: dict = {}
    for sector, grp in detail.groupby("naics_sector"):
        total = grp["employment"].sum()
        bands: dict = {}
        for band, cell in grp.groupby("band"):
            worst = max(
                cell["employment_noise_flag"],
                key=lambda f: SUSB_FLAG_SEVERITY[f],
            )
            bands[band] = {
                "employment": int(cell["employment"].sum()),
                "share": _round(cell["employment"].sum() / total),
                "noise_flag_worst": worst,
                "cv_upper_bound": SUSB_FLAG_CV_BOUND[worst],
            }
        by_sector[sector] = bands

    bds = targets.load_bds_firm_size()
    lo, hi = BDS_WINDOW
    bds = bds[(bds["year"] >= lo - 1) & (bds["year"] <= hi)]
    group_of = {
        label: group
        for group, labels in BDS_GROUPS.items()
        for label in labels
    }
    bds = bds.assign(group=bds["fsize"].map(group_of))
    emp = bds.groupby(["year", "group"])["emp"].sum().unstack()
    shares = emp.div(emp.sum(axis=1), axis=0)
    bds_margin: dict = {}
    for group, labels in BDS_GROUPS.items():
        spans = [banding.bds_fsize_to_canonical(lb) for lb in labels]
        union = tuple(dict.fromkeys(b for span in spans for b in span.bands))
        gaps = [
            (int(year), abs(math.log(shares.loc[year, group] / prev)))
            for year, prev in zip(
                shares.index[1:],
                shares[group].to_numpy()[:-1],
                strict=True,
            )
        ]
        bds_margin[group] = {
            "bds_fsize_categories": list(labels),
            "canonical_bands": [b.name for b in union],
            "exact": len(union) == 1,
            "share_2022": _round(shares.loc[hi, group]),
            **_gap_summary(gaps),
        }
    return {
        "susb_2022_share_by_sector_band": by_sector,
        "bds_size_margin_yoy_stability": {
            "window": list(BDS_WINDOW),
            "groups": bds_margin,
        },
    }


# ---------------------------------------------------------------------
# QWI (E6/E7) and J2J (E2 proxy, E11 margins)
# ---------------------------------------------------------------------


def e6_e7_block() -> dict:
    qwi = targets.load_qwi_firmsize_sector()
    detail, summary = _lehd_cell_block(
        qwi,
        {
            "e6_hire_rate": "hire_rate",
            "e6_separation_rate": "separation_rate",
            "e7_earns_mean": "EarnS",
        },
        "EmpTotal",
    )
    # EarnS is nominal: its raw YoY variation embeds aggregate nominal
    # wage growth (a trend, not sampling noise). Record the
    # aggregate-relative variant alongside, both on the record.
    margin = qwi[qwi["industry"] == "00"].copy()
    agg = (
        margin.assign(w=margin["EarnS"] * margin["EmpS"])
        .groupby(["year", "quarter"])
        .agg(w=("w", "sum"), emps=("EmpS", "sum"))
    )
    agg_earns = (agg["w"] / agg["emps"]).rename("agg_earns").reset_index()
    rel = margin.merge(agg_earns, on=["year", "quarter"])
    rel["earns_rel"] = rel["EarnS"] / rel["agg_earns"]
    for code, cell in rel.groupby("firmsize"):
        detail[f"firmsize{int(code)}"]["e7_earns_rel_to_aggregate"] = (
            _gap_summary(_yoy_gaps(cell, "earns_rel"))
        )
    return {"by_firmsize_all_industry": detail, "sector_cells": summary}


def e2_e11_block() -> tuple[dict, dict]:
    j2j = targets.load_j2j_firmsize_sector()
    j2j = j2j.copy()
    base = j2j["MainB"].where(j2j["MainB"] > 0)
    j2j["main_hire_rate"] = j2j["MHire"] / base
    j2j["main_separation_rate"] = j2j["MSep"] / base
    j2j["ee_hire_rate"] = j2j["EEHire"] / base
    j2j["ee_separation_rate"] = j2j["EESep"] / base
    detail, summary = _lehd_cell_block(
        j2j,
        {
            "hire_rate": "main_hire_rate",
            "separation_rate": "main_separation_rate",
            "j2j_hire_rate": "j2j_hire_rate",
            "j2j_separation_rate": "j2j_separation_rate",
            "ee_hire_rate": "ee_hire_rate",
            "ee_separation_rate": "ee_separation_rate",
        },
        "MainB",
    )
    e2 = {
        "by_firmsize_all_industry": detail,
        "sector_cells": summary,
        "by_sex_age": e2_sexage_block(),
    }
    e11 = e11_block()
    return e2, e11


#: The rate families carried on the sex x age axis. Same |log YoY
#: ratio| machinery as the firm-size axis, so the two E2 axes are on
#: one footing for the referee round.
SEXAGE_RATE_COLS = {
    "hire_rate": "hire_rate",
    "separation_rate": "separation_rate",
    "j2j_hire_rate": "j2j_hire_rate",
    "j2j_separation_rate": "j2j_separation_rate",
}


def e2_sexage_block() -> dict:
    """E2's registered sex x age gate axis (#192; the #228 extract).

    The floor the earlier draft deferred: E2 is registered by sex x
    age, and until the ``sa`` extract landed the committed
    tabulations carried no such axis. Cells are the 3 x 9 sex x age
    grid at the all-industry margin (margins included, so the
    all-sexes and all-ages rows are floorable too), 2015Q1-2025Q1.
    """
    frame = targets.load_j2j_sexage()
    cells: dict = {}
    for (sex, agegrp), cell in frame.groupby(["sex", "agegrp"]):
        min_denom = int(cell["MainB"].min())
        rec: dict = {
            "sex": int(sex),
            "sex_label": cell["sex_label"].iloc[0],
            "agegrp": agegrp,
            "agegrp_label": cell["agegrp_label"].iloc[0],
            "min_denominator_jobs": min_denom,
            "thin": min_denom < THIN_JOBS,
        }
        for name, col in SEXAGE_RATE_COLS.items():
            rec[name] = {
                "level_mean": _round(cell[col].mean()),
                **_gap_summary(_yoy_gaps(cell, col)),
            }
        cells[f"sex{int(sex)}_{agegrp}"] = rec

    # Cross-cell summary over the 2 x 8 non-margin cells only: the
    # margins are aggregates of them, so pooling both would double
    # count and pull the median toward the (much more stable)
    # aggregate rows.
    detail_cells = frame[(frame["sex"] != 0) & (frame["agegrp"] != "A00")]
    summary: dict = {
        "n_cells": int(detail_cells.groupby(["sex", "agegrp"]).ngroups),
        "note": (
            "non-margin cells only (sex in 1,2 x agegrp A01-A08); the "
            "all-sexes / all-ages rows are aggregates of these and "
            "are reported per-cell above, not pooled here"
        ),
    }
    for name, col in SEXAGE_RATE_COLS.items():
        means = []
        n_thin = 0
        for _, cell in detail_cells.groupby(["sex", "agegrp"]):
            gaps = _yoy_gaps(cell, col)
            if not gaps:
                continue
            means.append(float(np.mean([g for _, g in gaps])))
            if int(cell["MainB"].min()) < THIN_JOBS:
                n_thin += 1
        summary[name] = {
            "cell_floor_median": _round(np.median(means)),
            "cell_floor_p90": _round(np.quantile(means, 0.9)),
            "cell_floor_max": _round(np.max(means)),
            "n_cells_with_pairs": len(means),
            "n_thin_cells": n_thin,
        }
    return {"cells": cells, "cross_cell": summary}


def e11_block() -> dict:
    """E11's disposition, restated now that the OD extract exists.

    Not "no extract" any more (#228 commits the full 6 x 6 origin x
    destination grid) but still not a floorable cross: the national
    detail is published only for 2015Q1-2016Q1, which yields exactly
    one same-quarter year-over-year pair per detail cell (2016Q1 vs
    2015Q1). One pair gives a gap but no dispersion, so no
    ``mean + k * sd`` floor exists on the detail. The margins run
    through 2025Q1 and are floorable.
    """
    od = targets.load_j2jod_firmsize()
    detail = od[(od["firmsize"] > 0) & (od["firmsize_orig"] > 0)]
    observed = detail.dropna(subset=["EE"])
    quarters = (
        observed[["year", "quarter"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])
    )
    periods = [(int(y), int(q)) for y, q in quarters.to_numpy()]
    pairs_per_cell = {
        f"{int(o)}to{int(d)}": len(_yoy_gaps(cell, "EE"))
        for (o, d), cell in observed.groupby(["firmsize_orig", "firmsize"])
    }
    max_pairs = max(pairs_per_cell.values()) if pairs_per_cell else 0

    # EE margins are counts, so their YoY variation carries aggregate
    # flow growth (a trend, not noise) exactly as raw EarnS carries
    # nominal wage growth. Both variants are committed, matching the
    # e7_nominal_trend treatment: `ee` is the raw count and `ee_rel`
    # is the share of that quarter's all-size EE total, which divides
    # the common trend out.
    total = (
        od[(od["firmsize_orig"] == 0) & (od["firmsize"] == 0)]
        .set_index(["year", "quarter"])["EE"]
        .rename("ee_total")
    )
    margin = od[(od["firmsize_orig"] == 0) & (od["firmsize"] > 0)].copy()
    margin = margin.join(total, on=["year", "quarter"])
    margin["ee_rel"] = margin["EE"] / margin["ee_total"].where(
        margin["ee_total"] > 0
    )
    margin_gaps: dict = {}
    for code, cell in margin.groupby("firmsize"):
        observed_cell = cell.dropna(subset=["EE"])
        margin_gaps[f"firmsize{int(code)}"] = {
            "firmsize_label": cell["firmsize_label"].iloc[0],
            "ee": _gap_summary(_yoy_gaps(observed_cell, "EE")),
            "ee_rel": _gap_summary(_yoy_gaps(observed_cell, "ee_rel")),
        }

    return {
        "status": (
            "detail floor NOT derivable (one YoY pair per cell); "
            "destination-size margin floor derivable"
        ),
        "detail_window": {
            "observed_quarters": [f"{y}Q{q}" for y, q in periods],
            "n_quarters": len(periods),
            "max_yoy_pairs_per_cell": max_pairs,
            "why_not_floorable": (
                "the national origin x destination cross is published "
                "only for 2015Q1-2016Q1 (from 2016Q2 every detail "
                "cell carries status flag 11); same-quarter "
                "year-over-year pairing therefore yields at most one "
                "pair per detail cell, which gives a gap but no "
                "dispersion, so no mean + k*sd floor exists on the "
                "cross. This is a different disposition from the "
                "earlier draft's 'no extract committed': the extract "
                "exists (#228), the temporal replicate does not"
            ),
        },
        "destination_size_margin": margin_gaps,
        "cross_source_margin_disagreement": {
            "all_size_ee_tool_above_flat_file": "37 of 41 quarters",
            "all_size_ee_deviation_range_pct": [-1.00, 2.02],
            "per_size_deviation_range_pct": [-3.20, 3.67],
            "note": (
                "the LED Extraction Tool's margins and the LEHD flat "
                "file's d_fs margins are independent publications of "
                "the same quantity and disagree by up to ~3% in "
                "either direction (provenance entry 6; reproduce with "
                "scripts/check_j2jod_margin_agreement.py). Since "
                "E11's post-2016Q1 constraints are margins-only, this "
                "cross-source wobble bounds how tight any E11 margin "
                "threshold can be, independently of the temporal "
                "floor above"
            ),
        },
    }


def build() -> dict:
    e2, e11 = e2_e11_block()
    return {
        "artifact": "employer_firm_floors",
        "version": "draft_v0.1",
        "status": "DRAFT - NOT RATIFIED; C3 not locked; no thresholds",
        "issue": "192",
        "workstream": "B",
        "sources": {
            "susb": "data/external/susb_us_sector_size_2022.csv",
            "bds": "data/external/bds_us_firm_size_1978_2022.csv",
            "qwi": "data/external/qwi_us_firmsize_sector_2015on.csv",
            "j2j": "data/external/j2j_us_firmsize_sector_2015on.csv",
            "j2j_sexage": "data/external/j2j_us_sexage_2015on.csv",
            "j2jod": "data/external/j2jod_us_firmsize_od_2015on.csv",
            "provenance": ("data/external/employer_firm_target_sources.md"),
        },
        "method": (
            "temporal-stability floors on published administrative "
            "aggregates: year-over-year same-quarter |log ratio| per "
            "cell (QWI/J2J, 2015Q1 on; same-quarter comparison "
            "absorbs seasonality in the not-seasonally-adjusted "
            "extracts), year-over-year |log share ratio| for the BDS "
            "size margin (2012-2022), and published noise-flag CV "
            "bounds for the single-vintage SUSB table; banding via "
            "populace_dynamics.firms.banding only; thin flag at "
            f"minimum cell denominator < {THIN_JOBS} jobs (draft "
            "choice)"
        ),
        "unit_rules": [
            "QWI/J2J cells count jobs, not persons (ADR 0003): the "
            "job-to-person adjustment (~ multiple-jobholding rate, "
            "~5%) is a pre-registered C3 item",
            "QWI EarnS is MEAN monthly earnings of full-quarter "
            "employees; QWI never publishes medians; E7 is stated on "
            "means",
            "J2J extract ownership is oslp (state/local + private) "
            "while QWI is private-only (op) and SUSB excludes "
            "government; NAICS 92 is dropped from the J2J extract "
            "but state/local employment embedded in other sectors "
            "(esp. 61, 62) remains — E2/E11 cells must restate on a "
            "private-comparable basis or carry this scope caveat "
            "(locks with C3)",
        ],
        "method_findings": {
            "e1_no_sector_replicate": (
                "the committed SUSB extract is a single 2022 "
                "cross-section: a same-source temporal or resampling "
                "floor on the size x sector cells is degenerate. The "
                "E1 floor is therefore composed of the published "
                "SUSB noise-flag CV bounds per cell plus a BDS "
                "year-over-year stability floor that exists only for "
                "the national size margin — the sector axis has no "
                "stability floor derivable from committed extracts"
            ),
            "e1_bds_straddle": (
                "the BDS '20 to 99' category straddles the canonical "
                "50 edge (banding.bds_fsize_to_canonical is inexact "
                "there), so the BDS margin floor is stated on a "
                "coarsened partition (20_99 kept whole), not on the "
                "five canonical bands"
            ),
            "e2_sex_age_axis_built": (
                "SUPERSEDES draft_v0's 'e2_no_age_sex_axis'. E2's "
                "registered sex x age axis is now floored from the "
                "committed J2J sex x age extract (#228): the full "
                "3 x 9 grid at the all-industry margin, 2015Q1-"
                "2025Q1, same |log YoY ratio| machinery as the "
                "firm-size axis, reported per cell and pooled over "
                "the 2 x 8 non-margin cells. The firm-size x sector "
                "floors remain the aggregate-side references they "
                "always were; the two E2 axes are now on one "
                "footing. Naming correction carried from the earlier "
                "draft: LEHD's sex x age tabulation is 'sa'; 'se' is "
                "sex x EDUCATION, and the draft_v0 finding named the "
                "wrong one"
            ),
            "e11_extract_committed_but_no_temporal_replicate": (
                "SUPERSEDES draft_v0's 'e11_no_od_extract', which is "
                "now factually stale: the origin x destination "
                "firm-size cross IS committed (#228, the full 6 x 6 "
                "grid). The obstacle is temporal, not availability. "
                "The national detail is published only for "
                "2015Q1-2016Q1 (status flag 11 from 2016Q2), so "
                "same-quarter year-over-year pairing yields at most "
                "ONE pair per detail cell — a gap with no "
                "dispersion, hence no mean + k*sd floor on the "
                "cross. The destination-size margins run through "
                "2025Q1 and are floored in the e11 block. A second, "
                "independent bound on any margin threshold comes "
                "from cross-source disagreement: the LED tool's "
                "margins and the LEHD flat file's differ by up to "
                "~3% in either direction (e11.cross_source_margin_"
                "disagreement)"
            ),
            "release_revision_noise_unfloored": (
                "a third floorable concept, recorded and NOT built: "
                "vintage-to-vintage revision noise. LEHD revises "
                "across releases, and none of the floors here see "
                "that — every extract is a single release (R2026Q1). "
                "Observed during the #228 review: LEHD rotated to "
                "R2026Q2 mid-round and the J2JOD values for "
                "2015Q1-2025Q1 were unchanged across the rotation "
                "(all 1,476 rows), which is one datum, on one "
                "series, over one rotation — suggestive that "
                "revision noise is small for these aggregates, not "
                "evidence that it is zero. Building it needs two "
                "release-stamped vintages of the same series "
                "committed; the C3 referee round should decide "
                "whether E1/E2/E6/E7/E11 thresholds must carry a "
                "revision allowance on top of the temporal floor"
            ),
            "e12_deferred": (
                "E12 (AKM moments) has no committed extract: AKM "
                "variance decompositions require linked "
                "employer-employee microdata, and the published "
                "decompositions are research outputs rather than a "
                "recurring aggregate release. No floor is buildable; "
                "E12 is recorded as deferred pending a committed, "
                "provenance-pinned reference extract, and must not "
                "lock with C3 without one"
            ),
            "cycle_signal_in_floors": (
                "temporal-stability floors on published aggregates "
                "include true business-cycle variation (2020-2021 "
                "most visibly) as well as source noise; both the "
                "full-sample and ex-pandemic figures are committed "
                "rather than choosing one — the C3 referee round "
                "picks the formulation with both on the record"
            ),
            "floors_not_monotone_in_disaggregation": (
                "an empirical finding from the sex x age build, and "
                "a trap for the threshold policy: the temporal "
                "floor is NOT monotone in disaggregation. Of the 26 "
                "non-aggregate sex x age cells, the number whose "
                "ex-pandemic floor is TIGHTER than the all-sexes "
                "all-ages cell is 13 (hire), 10 (separation), 6 "
                "(j2j hire), 7 (j2j separation). The pattern is "
                "interpretable -- the 45-99 age cells are the most "
                "stable and the 19-34 cells the least, while the "
                "aggregate carries compositional shift the older "
                "cells do not -- but the consequence is procedural: "
                "a floor measured on a margin CANNOT be used as a "
                "conservative bound for the cells beneath it. Every "
                "gated cell needs its own floor, or the threshold "
                "policy must say explicitly which cell's floor "
                "governs (C3 open question 1)"
            ),
            "e11_margin_trend": (
                "the E11 destination-size margins are EE flow "
                "COUNTS, so their year-over-year variation carries "
                "aggregate flow growth (a trend, not noise) exactly "
                "as raw EarnS carries nominal wage growth. Both are "
                "committed: 'ee' (raw counts) and 'ee_rel' (share of "
                "the quarter's all-size EE total, which divides the "
                "common trend out). The relative variant runs roughly "
                "half the raw one; the C3 referee round picks the "
                "formulation, as for E7"
            ),
            "e7_nominal_trend": (
                "raw EarnS YoY variation embeds aggregate nominal "
                "wage growth (a trend, not noise); the "
                "aggregate-relative EarnS floor is committed "
                "alongside the raw one, both on the record"
            ),
        },
        "e1": e1_block(),
        "e2": e2,
        "e6_e7": e6_e7_block(),
        "e11": e11,
        "e12": {"status": "deferred - no committed extract"},
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")


if __name__ == "__main__":
    main()
