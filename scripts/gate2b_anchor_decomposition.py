"""Gate-2b external anchor: concept-decomposed Census/CPS shape/ratio report.

REPORTED, NOT GATED. This is the external anchor the gate-2b floor's
``external_anchor.required_before_ratifying_flip`` (round-1 referee finding
F) demands before the ratifying flip -- the household-composition analogue
of tranche 2a's concept-decomposed NCHS marriage/divorce and ASFR anchors.

It reads the FROZEN, ratified floor ``runs/gate2b_floors_v1.json``
(the PSID ``reference_moments`` side -- never rewritten) and the committed,
sha-pinned Census/CPS benchmark files in ``data/external/`` (the Census
side), and reports, per anchorable cell, the raw PSID/Census ratio with the
concept delta NAMED -- exactly like 2a's "raw 2.44x = person-vs-couple (x2)
x denominator (x1.22) -> residual ~1.0" decomposition. No calibration: the
ratios are honest and move no floor value.

The bridge PSID(family unit, person-weighted) <-> Census(household):
* ``hh_size``           -- household-vs-person weighting, then family-unit
                           fragmentation (Census HH-4).
* ``coresident_spouse`` -- PSID codes 20/22 = spouse OR partner, so the
                           concept-matched anchor adds CPS AD-3
                           living_with_partner to living_with_spouse.
* ``coresident_parent`` -- CPS AD-3 child_of_householder, with the 15-24
                           age-floor delta (PSID includes 15-17) and the
                           35-64 band-aggregation delta.
* ``multigen``          -- Census B11017 (>=3 generations), a HOUSEHOLD
                           share vs the PSID person share (concept gap
                           named, not differenced).
* ``coresident_grandchild`` / ``coresident_child`` -- grandparent-side /
                           householder-side denominators with no clean
                           published per-age level (direction only).
* transitions           -- flows with no cross-sectional Census level
                           (flip-note 4).

Run from the repository root::

    .venv/bin/python scripts/gate2b_anchor_decomposition.py

It writes ``runs/gate2b_anchor_v1.json`` via
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
FLOOR_PATH = ROOT / "runs" / "gate2b_floors_v1.json"
ANCHOR_PATH = ROOT / "runs" / "gate2b_anchor_v1.json"
EXTERNAL = ROOT / "data" / "external"

HH_SIZE_FILE = EXTERNAL / "census_household_size_2023.json"
LIVING_FILE = EXTERNAL / "census_living_arrangements_2023.json"
MULTIGEN_FILE = EXTERNAL / "census_multigenerational_2020.json"

# PSID gate band -> CPS AD-3 band (see census_living_arrangements_2023.json
# band_mapping_to_psid). 35-44/45-54/55-64 all share the single CPS 35-64
# band (a named band-aggregation delta); 15-24 maps to CPS 18-24 (a named
# age-floor delta: PSID keeps ages 15-17).
PSID_TO_CPS_BAND = {
    "15-24": "18-24",
    "25-34": "25-34",
    "35-44": "35-64",
    "45-54": "35-64",
    "55-64": "35-64",
    "65-74": "65-74",
    "75+": "75+",
}

RATIO_ROUNDING = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _split_cell(cell: str) -> tuple[str, str, str]:
    """``coresident_spouse.25-34|female`` -> (family, band, sex)."""
    family, rest = cell.split(".", 1)
    band, sex = rest.split("|", 1)
    return family, band, sex


def spouse_partner_anchor(living: dict, band: str, sex: str) -> float:
    """Concept-matched coresident-spouse anchor: CPS living_with_spouse +
    living_with_partner (PSID codes 20 AND 22), as a fraction."""
    cell = living["bands"][PSID_TO_CPS_BAND[band]][sex]
    return (cell["living_with_spouse"] + cell["living_with_partner"]) / 100.0


def spouse_only_anchor(living: dict, band: str, sex: str) -> float:
    cell = living["bands"][PSID_TO_CPS_BAND[band]][sex]
    return cell["living_with_spouse"] / 100.0


def child_of_householder_anchor(
    living: dict, band: str, sex: str
) -> float | None:
    """CPS child_of_householder (living in a parent's home) as a fraction;
    None where the CPS band has no such column (65-74, 75+)."""
    cell = living["bands"][PSID_TO_CPS_BAND[band]][sex]
    value = cell.get("child_of_householder")
    return None if value is None else value / 100.0


def _ratio(psid: float, anchor: float) -> float:
    return round(psid / anchor, RATIO_ROUNDING)


def build_hh_size(ref: dict, hh: dict) -> dict:
    """Person-level household-size distribution: PSID family-unit size vs the
    Census person-level household-size share (household counts reweighted to
    persons). Named deltas: household-vs-person weighting, then family-unit
    fragmentation."""
    person = hh["derived"]["person_level_share"]
    household = hh["derived"]["household_level_share"]
    cells = []
    for size in ["1", "2", "3", "4", "5+"]:
        psid = ref[f"hh_size.{size}"]["rate"]
        cells.append(
            {
                "cell": f"hh_size.{size}",
                "psid_ref": round(psid, RATIO_ROUNDING),
                "census_household_level": household[size],
                "census_person_level": person[size],
                "ratio_vs_person_level": _ratio(psid, person[size]),
                "ratio_vs_household_level": _ratio(psid, household[size]),
            }
        )
    return {
        "statistic": "person-level household-size share (PSID family unit)",
        "census_table": hh["table"],
        "census_reference_year": hh["reference_year"],
        "concept_delta": (
            "TWO named deltas. (1) household-vs-person weighting: HH-4 is a "
            "share of HOUSEHOLDS; the PSID hh_size is a share of PERSONS, so "
            "the concept-matched Census column is census_person_level "
            "(household counts x size, renormalised). (2) family-unit "
            "fragmentation: the PSID family unit splits subfamilies out of "
            "the Census household, shifting person mass from large units to "
            "1-2 person units -- so the residual ratio rises above 1 at "
            "size 1-2 and falls below 1 at size 4/5+, a monotonic concept "
            "signature, not noise."
        ),
        "cells": cells,
    }


def build_coresident_spouse(ref: dict, living: dict) -> dict:
    cells = []
    for cell_key in sorted(
        k for k in ref if k.startswith("coresident_spouse.")
    ):
        _, band, sex = _split_cell(cell_key)
        psid = ref[cell_key]["rate"]
        spouse_only = spouse_only_anchor(living, band, sex)
        matched = spouse_partner_anchor(living, band, sex)
        cells.append(
            {
                "cell": cell_key,
                "psid_ref": round(psid, RATIO_ROUNDING),
                "cps_band": PSID_TO_CPS_BAND[band],
                "census_spouse_only": round(spouse_only, RATIO_ROUNDING),
                "census_spouse_plus_partner": round(matched, RATIO_ROUNDING),
                "raw_ratio_vs_spouse_only": _ratio(psid, spouse_only),
                "residual_vs_spouse_plus_partner": _ratio(psid, matched),
            }
        )
    return {
        "statistic": "coresident spouse/partner share (PSID codes 20/22)",
        "census_table": living["table"],
        "census_reference_year": living["reference_year"],
        "concept_delta": (
            "PSID coresident_spouse counts a coresident spouse OR partner "
            "(MX8 codes 20 and 22); CPS living_with_spouse is married-"
            "spouse-present only. The concept factor is the partner-"
            "inclusion: adding CPS living_with_partner gives the matched "
            "anchor, and residual_vs_spouse_plus_partner collapses toward 1 "
            "(the 2a person-vs-couple pattern). Age-band deltas remain: PSID "
            "15-24 includes ages 15-17 (CPS 18-24) and PSID 35-44/45-54/"
            "55-64 each map to the single CPS 35-64 band, so those residuals "
            "carry a band-composition term, not a data disagreement."
        ),
        "cells": cells,
    }


def build_coresident_parent(ref: dict, living: dict) -> dict:
    cells = []
    concept_gap = []
    for cell_key in sorted(
        k for k in ref if k.startswith("coresident_parent.")
    ):
        _, band, sex = _split_cell(cell_key)
        if band not in PSID_TO_CPS_BAND:
            continue
        anchor = child_of_householder_anchor(living, band, sex)
        psid = ref[cell_key]["rate"]
        if anchor is None:
            concept_gap.append(cell_key)
            continue
        cells.append(
            {
                "cell": cell_key,
                "psid_ref": round(psid, RATIO_ROUNDING),
                "cps_band": PSID_TO_CPS_BAND[band],
                "census_child_of_householder": round(anchor, RATIO_ROUNDING),
                "raw_ratio": _ratio(psid, anchor),
            }
        )
    return {
        "statistic": "share living with a coresident parent (PSID codes "
        "30/33/35/38 as ego)",
        "census_table": living["table"],
        "census_reference_year": living["reference_year"],
        "concept_delta": (
            "CPS child_of_householder counts an adult living in a parent's "
            "home as the householder's child; PSID coresident_parent counts "
            "any coresident parent in the family unit (broader). At 15-24 "
            "the PSID ratio exceeds 1 chiefly through the age-floor delta "
            "(PSID keeps ages 15-17, ~universally with a parent; CPS band is "
            "18-24). At 25-34 the ratio sits below 1 (child-of-householder "
            "is a stricter arrangement than any coresident parent). CPS has "
            "no child_of_householder column at 65-74/75+: concept gap."
        ),
        "cells": cells,
        "no_census_level_cells": concept_gap,
    }


def build_multigen(ref: dict, multigen: dict) -> dict:
    fig = multigen["figures"]
    household_share = fig["share_of_all_households_2020_pct"] / 100.0
    gated_person_cells = [
        "multigen.15-24|female",
        "multigen.25-34|female",
        "multigen.45-54|female",
    ]
    cells = []
    for cell_key in gated_person_cells:
        psid = ref[cell_key]["rate"]
        cells.append(
            {
                "cell": cell_key,
                "psid_ref": round(psid, RATIO_ROUNDING),
                "census_household_share_2020": round(household_share, 4),
                "raw_ratio_person_over_household": _ratio(
                    psid, household_share
                ),
                "concept_gap": True,
            }
        )
    return {
        "statistic": "person share in a >=3-generation (B11017) family unit",
        "census_table": multigen["table"],
        "census_reference_year": multigen["reference_year"],
        "concept_delta": (
            "The B11017 concept now MATCHES (both are >=3 distinct "
            "generations after fixes B/C -- the span>=2 predecessor did "
            "not). But the anchor is a HOUSEHOLD share (4.7% of households, "
            "7.2% of family households, 2020) while the PSID cells are a "
            "PERSON share by age x sex; persons-in-multigen-households "
            "exceed the multigen-household share (larger households), so a "
            "level difference is a unit artifact, reported not differenced. "
            "The grandchild-side corroborator (8.4% of children under 18 in "
            "a grandparent's home, 2020) fixes the DIRECTION."
        ),
        "person_vs_household_note": (
            "raw_ratio_person_over_household is person-share / "
            "household-share and is expected to exceed 1 by the unit delta "
            "alone; it is NOT a residual and is not expected near 1."
        ),
        "cells": cells,
    }


def build_concept_gap_families(ref: dict, multigen: dict) -> dict:
    fig = multigen["figures"]
    return {
        "coresident_grandchild": {
            "statistic": "grandparent-side share with a coresident "
            "grandchild",
            "census_direction_anchor": {
                "children_under_18_in_grandparents_home_2020_pct": fig[
                    "children_under_18_in_grandparents_home_2020_pct"
                ],
                "note": "This is a GRANDCHILD-side share (children living "
                "with a grandparent); the PSID cells are GRANDPARENT-side "
                "(adults with a coresident grandchild) -- a different "
                "denominator. Direction only, no per-cell ratio.",
            },
            "gated_cells": [
                "coresident_grandchild.45-54|female",
                "coresident_grandchild.55+|female",
            ],
        },
        "coresident_child": {
            "statistic": "share living with a coresident child",
            "census_level": "none",
            "note": "No clean published per-age x sex Census level for "
            "'lives with an own/step/social child' that matches the PSID "
            "coresidence concept; the ACS own-children-in-household tables "
            "are householder-keyed and family-typed. Reported as a concept "
            "gap; the 35-44 female PSID rate (~0.77) is consistent in "
            "direction with peak-childrearing family-household shares.",
        },
        "transitions": {
            "statistic": "wave-to-wave household-composition transitions",
            "families": [
                "parental_home_exit",
                "spousal_loss",
                "multigen_entry",
                "multigen_exit",
            ],
            "census_level": "none",
            "note": "Transitions are FLOWS; there is no cross-sectional "
            "Census stock to anchor a level against (flip-note 4). Per-era "
            "PSID transition rates could be added at a later anchor round; "
            "round 1 did not require them.",
        },
    }


def build_anchor() -> dict:
    floor = _load(FLOOR_PATH)
    ref = floor["reference_moments"]
    hh = _load(HH_SIZE_FILE)
    living = _load(LIVING_FILE)
    multigen = _load(MULTIGEN_FILE)

    families = {
        "hh_size": build_hh_size(ref, hh),
        "coresident_spouse": build_coresident_spouse(ref, living),
        "coresident_parent": build_coresident_parent(ref, living),
        "multigen": build_multigen(ref, multigen),
    }
    families.update(build_concept_gap_families(ref, multigen))

    anchored_cells = sum(
        len(fam["cells"])
        for fam in (
            families["hh_size"],
            families["coresident_spouse"],
            families["coresident_parent"],
            families["multigen"],
        )
    )

    return {
        "schema_version": "gate2b_anchor.v1",
        "run": "gate2b_anchor_v1",
        "reported_anchor_not_gated": True,
        "gated": False,
        "purpose": (
            "External concept-decomposed Census/CPS shape/ratio anchor for "
            "the gate-2b household-composition floor. Required by "
            "runs/gate2b_floors_v1.json external_anchor."
            "required_before_ratifying_flip (round-1 referee finding F); "
            "bundled by the gate-2b lock flip. Reads the frozen floor's "
            "PSID reference_moments and the committed Census/CPS benchmarks; "
            "reports per-cell PSID/Census ratios with the family-unit-vs-"
            "household concept delta NAMED. REPORTED, NEVER GATED. No "
            "calibration -- honest ratios, no floor value moves."
        ),
        "ceremony": {
            "tranche": "2b_relationship_household",
            "step": "external anchor, bundled with the ratifying flip "
            "(after floor -> referee -> fixes -> verification)",
            "mirrors_tranche_2a": "runs/gate2_floors_v2.json external_anchor "
            "(concept-decomposed NCHS marriage/divorce + ASFR)",
            "gates_this_run": False,
        },
        "floor_source": "runs/gate2b_floors_v1.json",
        "floor_sha256": _sha256(FLOOR_PATH),
        "census_sources": [
            {
                "file": "data/external/census_household_size_2023.json",
                "table_id": hh["table"],
                "source_url": hh["provenance"]["source_url"],
                "reference_year": hh["reference_year"],
                "fetched_utc": hh["provenance"]["fetched_utc"],
                "sha256": _sha256(HH_SIZE_FILE),
            },
            {
                "file": "data/external/census_living_arrangements_2023.json",
                "table_id": living["table"],
                "source_url": living["provenance"]["source_url"],
                "reference_year": living["reference_year"],
                "fetched_utc": living["provenance"]["fetched_utc"],
                "sha256": _sha256(LIVING_FILE),
            },
            {
                "file": "data/external/census_multigenerational_2020.json",
                "table_id": multigen["table"],
                "source_url": multigen["provenance"]["source_url"],
                "reference_year": multigen["reference_year"],
                "fetched_utc": multigen["provenance"]["fetched_utc"],
                "sha256": _sha256(MULTIGEN_FILE),
            },
        ],
        "concept_bridge": {
            "family_unit_vs_household": (
                "The PSID FAMILY UNIT is narrower than the Census HOUSEHOLD "
                "(PSID splits subfamilies and applies its own cohabitor/"
                "roomer rules); every ratio here carries this delta, so the "
                "anchor is a shape/ratio report, never a level gate -- the "
                "household analogue of 2a's person-vs-couple x denominator "
                "decomposition."
            ),
            "person_vs_household_weighting": (
                "PSID stocks are person-weighted; Census hh_size and B11017 "
                "are household-weighted. hh_size is bridged by reweighting "
                "Census household counts to persons; multigen names the unit "
                "delta rather than differencing."
            ),
            "spouse_or_partner_inclusion": (
                "PSID coresident_spouse = codes 20 (spouse) and 22 "
                "(partner); the concept-matched Census anchor adds CPS "
                "living_with_partner to living_with_spouse."
            ),
            "multigen_b11017_bridge": multigen["provenance"]["definition"],
            "in_law_and_greatgrand_note": (
                "flip-note 3: the PSID MX7 generation map places lineal "
                "in-laws in the generation count (37 child-in-law -1, 57 "
                "parent-in-law +1, 67/69 +2) while the MX8 coresidence sets "
                "EXCLUDE in-law-by-marriage links; and great-grand codes "
                "lump at +/-2 (65, 68/69) -- a strict +/-3 map would move "
                "811 person-waves (0.09% of the panel). ACS relationship "
                "coding also lumps grandchild+, so the committed choice is "
                "concept-consistent with B11017."
            ),
            "grandparent_vs_grandchild_side": (
                "coresident_grandchild is grandparent-side; the only "
                "published Census person share (8.4% of children in a "
                "grandparent's home) is grandchild-side -- direction only."
            ),
            "age_band_deltas": (
                "PSID 15-24 includes ages 15-17 (CPS bands start at 18); "
                "PSID 35-44/45-54/55-64 each map to the single CPS 35-64 "
                "band."
            ),
            "transitions_no_cross_sectional_anchor": (
                "parental_home_exit / spousal_loss / multigen_entry / "
                "multigen_exit are flows; no cross-sectional Census level "
                "(flip-note 4)."
            ),
        },
        "families": families,
        "summary": {
            "n_cells_with_census_ratio": anchored_cells,
            "families_with_clean_residual": [
                "hh_size",
                "coresident_spouse",
                "coresident_parent",
            ],
            "families_reported_as_concept_gap": [
                "multigen",
                "coresident_grandchild",
                "coresident_child",
                "transitions",
            ],
            "coresident_spouse_residual_note": (
                "After the partner-inclusion concept factor, the 14 "
                "coresident_spouse residuals sit near 1 for the clean-band "
                "cells (25-34, 65-74, 75+), the 2a-style bullseye; the "
                "15-24 and 35-64-mapped cells carry a named band term."
            ),
            "calibration": "none -- ratios are reported, no floor moves",
        },
    }


def main() -> None:
    anchor = build_anchor()
    artifacts.write_new(ANCHOR_PATH, anchor)
    print(f"wrote {ANCHOR_PATH.relative_to(ROOT)}")
    print(
        "cells with a Census ratio: "
        f"{anchor['summary']['n_cells_with_census_ratio']}"
    )


if __name__ == "__main__":
    main()
