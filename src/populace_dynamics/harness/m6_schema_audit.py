"""Static column-contract audit for the M6 real-frame input path.

The registered `gate_m6` run's second crash was an integration defect: a phase
(`build_realized_population`) read a column (`sex`) from a frame
(`demographic_panel`) that the real loader never produces, and the only test
over the path used a synthetic fixture that happened to carry it (grading #42
comment 4972045579).  The unit suite passed while the real-frame path could not.

This module closes that gap with a *static* contract, computing no scored
statistic: an explicit in-code manifest of every column each real-frame-reading
phase requires from each certified loader / factory frame, checkable against a
frame-name -> schema mapping.  Two suites consume it:

* the unit suite (`tests/test_m6_schema_audit.py`) checks the manifest against
  :data:`COMMITTED_FRAME_SCHEMAS`, the schemas verified from the loader source,
  so the defect class is caught with no staged data;
* the integration_psid suite (`tests/test_m6_schema_audit_psid.py`) checks it
  against the **real** loaders' output columns, the definitive contact test.

Scope: the pre-scoring real-frame contact surface -- input assembly
(`m6_inputs.assemble_m6_inputs`), the truth-table builders (`m6_cells`), and the
realized-population builder (`m6_population.build_realized_population`).  These
are the phases that read the raw PSID/native-panel frames directly; the scoring,
pre-flight, and reporting phases operate on the assembled population and native
panels, not on the loader frames, so they carry no loader-schema contract.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

__all__ = [
    "COMMITTED_FRAME_SCHEMAS",
    "PHASE_FRAME_COLUMN_READS",
    "audited_frame_names",
    "audit_columns",
]

#: Certified loader / factory output schemas, each verified from the source of
#: the loader that produces it.  ``demographic_panel`` deliberately carries **no**
#: ``sex`` -- that is the entire correction (§2.8.3f): person sex is on
#: ``death_records``, not the demo frame.
COMMITTED_FRAME_SCHEMAS: dict[str, frozenset[str]] = {
    # data.panels.demographic_panel (panels.py:252-253)
    "demographic_panel": frozenset(
        {
            "person_id",
            "period",
            "age",
            "sequence",
            "relationship",
            "weight",
            "interview",
        }
    ),
    # data.deaths.read_death_records (deaths.py:181-194)
    "death_records": frozenset(
        {
            "person_id",
            "sex_code",
            "sex",
            "death_code",
            "death_status",
            "death_year",
            "death_year_lo",
            "death_year_hi",
        }
    ),
    # data.family.family_earnings_panel (family.py:798-841)
    "earnings_panel": frozenset(
        {
            "person_id",
            "period",
            "earnings",
            "role",
            "age",
            "weight",
            "earnings_acc",
        }
    ),
    # data.disability.read_disability_status (raw, pre-attach_sex)
    "disability_status": frozenset(
        {
            "person_id",
            "period",
            "age",
            "weight",
            "status_code",
            "disabled",
            "retired",
        }
    ),
    # data.disability.build_disability_panel(...).person_years (disability.py:511-524)
    "disability_panel.person_years": frozenset(
        {
            "person_id",
            "period",
            "age",
            "sex",
            "weight",
            "status_code",
            "disabled",
            "retired",
        }
    ),
    # data.transitions.build_marital_panel(...).person_years (transitions.py:288-291)
    "marital.person_years": frozenset(
        {
            "person_id",
            "year",
            "age",
            "sex",
            "weight",
            "marital_state",
            "marriage_duration",
            "years_since_dissolution",
        }
    ),
    # data.transitions.build_marital_panel(...).events (transitions.py:293-297)
    "marital.events": frozenset(
        {
            "person_id",
            "year",
            "age",
            "sex",
            "weight",
            "transition",
            "marriage_duration",
            "years_since_dissolution",
        }
    ),
    # data.household_composition.build_household_panel(...).person_waves
    # (household_composition.py:412-424 + _add_transitions)
    "household.person_waves": frozenset(
        {
            "person_id",
            "year",
            "age",
            "band",
            "sex",
            "weight",
            "hh_size",
            "coresident_spouse",
            "coresident_parent",
            "coresident_child",
            "coresident_grandchild",
            "multigen",
            "has_next",
            "next_coresident_parent",
            "next_coresident_spouse",
            "next_multigen",
        }
    ),
    # models.household_composition.components.cohabitation_overlay.cohabitation_flag
    "cohabitation": frozenset({"person_id", "year", "cohabiting"}),
}

#: The manifest: run-phase function -> {frame name -> columns the phase reads
#: from that frame}.  Each entry is the set the function requires to EXIST on the
#: frame; a phase that reads a column absent from the frame's schema is exactly
#: the run-2 crash.
PHASE_FRAME_COLUMN_READS: dict[str, dict[str, frozenset[str]]] = {
    # m6_inputs.assemble_m6_inputs._require_columns (m6_inputs.py:393-407)
    "assemble_m6_inputs.require_columns": {
        "demographic_panel": frozenset(
            {"person_id", "period", "age", "weight", "interview"}
        ),
        "death_records": frozenset({"person_id", "sex", "death_year"}),
        "earnings_panel": frozenset(
            {"person_id", "period", "earnings", "age", "weight"}
        ),
    },
    # m6_cells.build_anchor_frame (m6_cells.py:122-140)
    "build_anchor_frame": {
        "demographic_panel": frozenset(
            {"person_id", "period", "weight", "interview"}
        ),
    },
    # m6_cells.presence_by_wave (m6_cells.py:143-150)
    "presence_by_wave": {
        "demographic_panel": frozenset({"person_id", "period", "weight"}),
    },
    # m6_cells.mortality_slices (m6_cells.py:159-218)
    "mortality_slices": {
        "demographic_panel": frozenset(
            {"person_id", "period", "age", "weight"}
        ),
        "death_records": frozenset({"person_id", "sex", "death_year"}),
    },
    # m6_cells.marital_tables_from_panel (m6_cells.py:234-282)
    "marital_tables_from_panel": {
        "marital.person_years": frozenset({"person_id", "year", "age", "sex"}),
        "marital.events": frozenset({"person_id", "year", "age", "sex"}),
    },
    # m6_cells.disability_pairs (m6_cells.py:294-334)
    "disability_pairs": {
        "disability_status": frozenset(
            {"person_id", "period", "age", "disabled"}
        ),
        "death_records": frozenset({"person_id", "sex"}),
    },
    # m6_cells.earnings_frame (m6_cells.py:337+)
    "earnings_frame": {
        "earnings_panel": frozenset(
            {"person_id", "period", "earnings", "age", "weight"}
        ),
    },
    # m6_population.build_realized_population (m6_population.py) -- the crash site.
    # `sex` is required from `death_records`, NOT `demographic_panel` (§2.8.3f).
    "build_realized_population": {
        "demographic_panel": frozenset(
            {"person_id", "period", "age", "interview"}
        ),
        "death_records": frozenset({"person_id", "sex"}),
        "marital.person_years": frozenset(
            {
                "person_id",
                "year",
                "marital_state",
                "marriage_duration",
                "years_since_dissolution",
            }
        ),
        "household.person_waves": frozenset(
            {
                "person_id",
                "year",
                "coresident_spouse",
                "coresident_parent",
                "coresident_child",
                "coresident_grandchild",
                "multigen",
                "hh_size",
            }
        ),
        "cohabitation": frozenset({"person_id", "year", "cohabiting"}),
        "disability_panel.person_years": frozenset(
            {"person_id", "period", "status_code", "disabled", "retired"}
        ),
        "earnings_panel": frozenset(
            {"person_id", "period", "earnings", "age", "weight"}
        ),
    },
}


def audited_frame_names() -> frozenset[str]:
    """Every frame name the manifest references."""
    names: set[str] = set()
    for reads in PHASE_FRAME_COLUMN_READS.values():
        names.update(reads)
    return frozenset(names)


def audit_columns(schemas: Mapping[str, Collection[str]]) -> list[str]:
    """Return the contract violations of the manifest against ``schemas``.

    ``schemas`` maps a frame name to its available columns.  Each violation is a
    human-readable line naming the phase, frame, and the columns the phase reads
    that the frame's schema does not provide (plus any frame the manifest reads
    that ``schemas`` omits).  An empty list means the manifest is satisfiable.
    """
    violations: list[str] = []
    for phase, frame_reads in sorted(PHASE_FRAME_COLUMN_READS.items()):
        for frame_name, columns in sorted(frame_reads.items()):
            if frame_name not in schemas:
                violations.append(
                    f"{phase}: frame {frame_name!r} has no schema entry"
                )
                continue
            missing = set(columns) - set(schemas[frame_name])
            if missing:
                violations.append(
                    f"{phase} reads {sorted(missing)} absent from "
                    f"{frame_name!r}"
                )
    return violations
