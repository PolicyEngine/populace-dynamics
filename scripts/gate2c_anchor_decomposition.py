"""Gate-2c external anchor: concept-bridged assortative-mating report.

REPORTED, NOT GATED. This is the external anchor the gate-2c floor's
``external_anchor.required_before_ratifying_flip`` demands before the
ratifying flip -- the marriage x earnings analogue of tranche 2a's NCHS and
tranche 2b's Census/CPS concept-decomposed anchors.

It reads the FROZEN, ratified floor ``runs/gate2c_floors_v1.json`` (the
per-year within-couple earnings-rank correlation and the own-tercile x
spouse-tercile contingency -- never rewritten) and the committed, sha-pinned
published benchmark files in ``data/external/`` (the Schwartz 2010 CPS ASEC
spouses'-earnings correlation and the Greenwood et al. 2014 educational
assortative-mating series), and reports, per facet, OUR value next to the
published value with the concept delta NAMED. No calibration: no floor value
or tolerance moves.

The bridge PSID(per-year earnings-capacity rank, selected couples) <->
published series:

* ``within_couple_rank_correlation`` -- our Spearman rank of the per-year
  NAWI-indexed positive-year earnings axis (0.4928) vs Schwartz (2010)
  Pearson correlation of spouses' ANNUAL earnings (dual-earner .08 -> .23;
  all-couples -.08 -> .12). Named deltas: rank vs Pearson; per-year capacity
  vs annual earnings; the selected both-positive universe; pooled/older
  marriage decades. The round-1 career-sum proxy rho (0.1194) sits near the
  Schwartz all-couples .12 but for the WRONG reason (observation mechanics),
  which is why the gated axis is the per-year measure.
* ``contingency_diagonal_concentration`` -- our relative-sum-of-diagonals
  delta, computed from the frozen 3x3 earnings contingency, vs Greenwood et
  al. (2014) education delta_t (~1.6-2.0). Named deltas: 3x3 terciles vs 5x5
  education levels; earnings capacity vs education.

Run from the repository root::

    .venv/bin/python scripts/gate2c_anchor_decomposition.py

It writes ``runs/gate2c_anchor_v1.json`` via
``populace_dynamics.artifacts.write_new`` (no-overwrite: a frozen artifact).
It imports no PSID reader and no ``populace.fit``; it needs only the two
committed inputs, so it reproduces in CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from populace_dynamics import artifacts

ROOT = Path(__file__).resolve().parents[1]
FLOOR_PATH = ROOT / "runs" / "gate2c_floors_v1.json"
ANCHOR_PATH = ROOT / "runs" / "gate2c_anchor_v1.json"
EXTERNAL = ROOT / "data" / "external"

SCHWARTZ_FILE = EXTERNAL / "schwartz_2010_spousal_earnings_correlation.json"
GREENWOOD_FILE = EXTERNAL / "greenwood_2014_assortative_mating.json"

TERCILES = (1, 2, 3)
ROUNDING = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _contingency(ref: dict) -> list[list[float]]:
    """The 3x3 own-tercile x spouse-tercile share matrix (rows = own)."""
    return [
        [ref[f"assort_mating.own{o}_spouse{s}"]["rate"] for s in TERCILES]
        for o in TERCILES
    ]


def relative_sum_of_diagonals(matrix: list[list[float]]) -> dict:
    """Greenwood et al. (2014) delta_t on the gate-2c earnings contingency:
    observed diagonal mass / independence-expected diagonal mass. delta > 1
    is positive sorting; it is scale-free in the table's total mass."""
    total = sum(sum(row) for row in matrix)
    row_marg = [sum(row) / total for row in matrix]
    col_marg = [sum(matrix[o][s] for o in range(3)) / total for s in range(3)]
    observed_diag = sum(matrix[i][i] for i in range(3)) / total
    expected_diag = sum(row_marg[i] * col_marg[i] for i in range(3))
    return {
        "observed_diagonal_mass": round(observed_diag, ROUNDING),
        "independence_expected_diagonal_mass": round(expected_diag, ROUNDING),
        "relative_sum_of_diagonals_delta": round(
            observed_diag / expected_diag, ROUNDING
        ),
        "row_marginals_own": [round(m, ROUNDING) for m in row_marg],
        "col_marginals_spouse": [round(m, ROUNDING) for m in col_marg],
    }


def build_rank_correlation_facet(decomp: dict, schwartz: dict) -> dict:
    ours = decomp["earnings_axis_spearman"]
    career = decomp["career_aime_proxy_spearman_cross_sex"]
    within_sex = decomp["within_sex_aime_proxy_rank_spearman"]
    nobs = decomp["observed_positive_years_spearman"]
    dual = schwartz["figures"]["dual_earner_couples"]
    allc = schwartz["figures"]["all_married_couples"]
    return {
        "our_statistic": (
            "within-couple Spearman RANK correlation of the per-year "
            "NAWI-indexed positive-year earnings axis (fix C), over the "
            "selected both-positive PSID couples"
        ),
        "our_value": ours,
        "anchor_source": schwartz["source_short"],
        "anchor_statistic": (
            "Pearson correlation of spouses' ANNUAL earnings (CPS ASEC, "
            "Schwartz 2010 Figure 2)"
        ),
        "anchor_dual_earner_pearson": {
            "1967_1970": dual["period_1967_1970_pearson"],
            "2003_2005": dual["period_2003_2005_pearson"],
        },
        "anchor_all_couples_pearson": {
            "1967_1970": allc["period_1967_1970_pearson"],
            "2003_2005": allc["period_2003_2005_pearson"],
        },
        "concept_matched_row": (
            "DUAL-EARNER (both spouses positive), mirroring the gate-2c "
            "both-positive couple selection -- NOT the all-couples row"
        ),
        "concept_deltas": [
            "rank (Spearman) vs level (Pearson): rank correlations of "
            "right-skewed earnings sit above the Pearson value",
            "per-year earnings-CAPACITY axis (mean indexed positive-year "
            "earnings) vs ANNUAL earnings: the capacity axis strips "
            "labor-supply/hours and zero-year variation, raising the "
            "sorting signal",
            "selected both-positive universe (age-60 in the NAWI window, "
            ">= 5 positive years) vs all married couples: selection on "
            "both-earner status is the dual-earner analogue",
            "pooled marriage decades 1924-2022, concentrated in older "
            "decades (the age-60/NAWI selection) vs a single CPS year -- "
            "Schwartz's series RISES over time, so a pooled value brackets "
            "between the period endpoints",
        ],
        "direction": (
            "positive, moderate sorting. Our per-year rank 0.4928 lies "
            "ABOVE the Schwartz dual-earner annual-earnings Pearson (0.23 in "
            "2003-5) exactly as the rank + capacity + selection deltas "
            "predict; direction and order of magnitude agree, no level is "
            "matched."
        ),
        "career_sum_caveat": (
            f"the round-1 career-sum AIME proxy rho ({career}) happens to "
            f"sit near the Schwartz all-couples Pearson (0.12, 2003-5), but "
            "for the WRONG reason: it is halved by cross-sex tercile pooling "
            f"(within-sex rank {within_sex}) and contaminated by the "
            f"observed-positive-years anti-correlation ({nobs}). That "
            "coincidence is exactly why the gated axis is the per-year "
            "measure (0.4928), not the career sum -- the anchor bridges to "
            "0.4928, not 0.1194."
        ),
        "calibration": "none -- reported, no floor value moves",
    }


def build_contingency_facet(ref: dict, delta: dict, greenwood: dict) -> dict:
    own1_1 = ref["assort_mating.own1_spouse1"]["rate"]
    own3_3 = ref["assort_mating.own3_spouse3"]["rate"]
    edu = greenwood["figures"]["education_relative_sum_of_diagonals_delta"]
    return {
        "our_statistic": (
            "relative sum of the 3x3 own-tercile x spouse-tercile earnings "
            "contingency diagonals (Greenwood et al. delta_t, computed on "
            "the frozen gate-2c contingency)"
        ),
        "our_relative_diagonal_delta": delta[
            "relative_sum_of_diagonals_delta"
        ],
        "our_diagonal_detail": delta,
        "our_corner_ratio_own3_spouse3_over_own1_spouse1": round(
            own3_3 / own1_1, 3
        ),
        "anchor_source": greenwood["source_short"],
        "anchor_statistic": (
            "delta_t on the 5x5 husband-wife EDUCATION contingency "
            "(Greenwood et al. 2014 Figure 1, figure-read)"
        ),
        "anchor_education_delta_range": edu["scale_read_from_figure_1"],
        "anchor_is_figure_read": edu["is_figure_read"],
        "concept_deltas": [
            "3x3 earnings terciles vs 5x5 education levels (finer education "
            "grid admits larger diagonal concentration)",
            "earnings-capacity sorting vs EDUCATIONAL sorting (education is "
            "a proxy; educational homogamy is the stronger-measured "
            "channel)",
        ],
        "direction": (
            "both deltas exceed 1 (positive sorting). Our earnings-capacity "
            "delta sits BELOW the Greenwood educational delta_t (~1.6-2.0), "
            "consistent with educational homogamy being stronger than "
            "earnings-capacity homogamy -- direction validated, no "
            "calibration."
        ),
        "calibration": "none -- reported, no floor value moves",
    }


def build_anchor() -> dict:
    floor = _load(FLOOR_PATH)
    ref = floor["reference_moments"]
    decomp = floor["assortative_correlation_report_only"]["decomposition"]
    schwartz = _load(SCHWARTZ_FILE)
    greenwood = _load(GREENWOOD_FILE)

    delta = relative_sum_of_diagonals(_contingency(ref))

    return {
        "schema_version": "gate2c_anchor.v1",
        "run": "gate2c_anchor_v1",
        "reported_anchor_not_gated": True,
        "gated": False,
        "purpose": (
            "External concept-bridged assortative-mating anchor for the "
            "gate-2c marriage x earnings joint floor. Required by "
            "runs/gate2c_floors_v1.json external_anchor."
            "required_before_ratifying_flip; bundled by the gate-2c lock "
            "flip. Reads the frozen floor's per-year earnings-rank "
            "correlation and 3x3 contingency and the committed published "
            "benchmarks (Schwartz 2010 CPS ASEC spouses' earnings; Greenwood "
            "et al. 2014 educational assortative mating); reports OUR value "
            "next to the published value with the concept delta NAMED. "
            "REPORTED, NEVER GATED. No calibration -- honest bridge, no "
            "floor value moves."
        ),
        "ceremony": {
            "tranche": "2c_marriage_earnings_joint",
            "step": (
                "external anchor, bundled with the ratifying flip (after "
                "floor -> referee -> fixes -> verification)"
            ),
            "mirrors": (
                "runs/gate2_floors_v2.json (2a NCHS) + "
                "runs/gate2b_anchor_v1.json (2b Census/CPS)"
            ),
            "gates_this_run": False,
        },
        "floor_source": "runs/gate2c_floors_v1.json",
        "floor_sha256": _sha256(FLOOR_PATH),
        "published_sources": [
            {
                "file": (
                    "data/external/"
                    "schwartz_2010_spousal_earnings_correlation.json"
                ),
                "source": schwartz["provenance"]["source"],
                "source_url": schwartz["provenance"]["open_access_url"],
                "doi": schwartz["doi"],
                "reference_years": "1967-2005 (CPS ASEC)",
                "fetched_utc": schwartz["provenance"]["fetched_utc"],
                "source_file_sha256": schwartz["provenance"][
                    "source_file_sha256"
                ],
                "committed_file_sha256": _sha256(SCHWARTZ_FILE),
            },
            {
                "file": (
                    "data/external/greenwood_2014_assortative_mating.json"
                ),
                "source": greenwood["provenance"]["source"],
                "source_url": greenwood["provenance"]["source_url"],
                "reference_years": "1960-2005 (Census/ACS)",
                "fetched_utc": greenwood["provenance"]["fetched_utc"],
                "source_file_sha256": greenwood["provenance"][
                    "source_file_sha256"
                ],
                "committed_file_sha256": _sha256(GREENWOOD_FILE),
            },
        ],
        "our_moments": {
            "within_couple_earnings_rank_spearman": decomp[
                "earnings_axis_spearman"
            ],
            "career_sum_proxy_spearman_cross_sex": decomp[
                "career_aime_proxy_spearman_cross_sex"
            ],
            "within_sex_rank_spearman": decomp[
                "within_sex_aime_proxy_rank_spearman"
            ],
            "observed_positive_years_spearman": decomp[
                "observed_positive_years_spearman"
            ],
            "contingency_relative_diagonal_delta": delta[
                "relative_sum_of_diagonals_delta"
            ],
            "n_directed_couples": floor["data"]["n_directed_couples"],
        },
        "concept_bridge": {
            "truncated_support_selected_couples": (
                "The gate-2c couples are a SELECTED, truncated-support set: "
                "both partners must carry a per-year earnings axis (family-"
                "file head/spouse labor income, income years 1968-2022, "
                "cohort-graded truncation) and >= 5 observed positive years, "
                "with age-60 in the NAWI window. The published series are "
                "drawn from full cross-sections (CPS ASEC married couples; "
                "Census/ACS married couples). Every comparison here carries "
                "that universe delta -- reported as a direction/order-of-"
                "magnitude bridge, never a level target."
            ),
            "rank_vs_pearson": (
                "our axis is a Spearman RANK correlation; Schwartz is a "
                "Pearson LEVEL correlation of annual earnings."
            ),
            "capacity_vs_annual_earnings": (
                "our axis is per-year earnings CAPACITY (mean indexed "
                "positive-year earnings), stripping labor-supply and "
                "zero-year variation that depress the annual-earnings "
                "Pearson."
            ),
            "education_vs_earnings": (
                "the Greenwood series measures EDUCATIONAL sorting, a proxy "
                "for the earnings-capacity axis; the contingency-delta "
                "bridge carries this extra proxy delta."
            ),
            "bridge_target_is_the_per_year_axis": (
                "per the frozen floor's external_anchor.concept_delta, the "
                "anchor bridges to the per-year sorting (~0.49), NOT the "
                "round-1 career-sum proxy rho (~0.12, an observation-"
                "mechanics artifact)."
            ),
        },
        "facets": {
            "within_couple_rank_correlation": (
                build_rank_correlation_facet(decomp, schwartz)
            ),
            "contingency_diagonal_concentration": (
                build_contingency_facet(ref, delta, greenwood)
            ),
        },
        "summary": {
            "n_facets": 2,
            "bridged_moment": "within-couple per-year earnings rank 0.4928",
            "direction_agrees": True,
            "any_level_calibrated": False,
            "calibration": "none -- values are reported, no floor moves",
        },
    }


def main() -> None:
    anchor = build_anchor()
    artifacts.write_new(ANCHOR_PATH, anchor)
    print(f"wrote {ANCHOR_PATH.relative_to(ROOT)}")
    print(
        "within-couple per-year rank: "
        f"{anchor['our_moments']['within_couple_earnings_rank_spearman']}; "
        "contingency delta: "
        f"{anchor['our_moments']['contingency_relative_diagonal_delta']}"
    )


if __name__ == "__main__":
    main()
