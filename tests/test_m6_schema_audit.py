"""Static column-contract audit against the committed loader schemas.

The manifest in :mod:`populace_dynamics.harness.m6_schema_audit` enumerates every
column each real-frame-reading M6 phase requires from each certified frame.  Here
it is checked against the committed schemas verified from the loader source; the
companion PSID suite checks it against the real loaders.  Neither computes a
scored statistic.  This is the unit-tier form of the check whose absence let the
second registered gate_m6 run crash (grading #42 comment 4972045579).
"""

from __future__ import annotations

from populace_dynamics.harness.m6_schema_audit import (
    COMMITTED_FRAME_SCHEMAS,
    PHASE_FRAME_COLUMN_READS,
    audit_columns,
    audited_frame_names,
)


def test_manifest_reads_are_satisfied_by_committed_schemas():
    assert audit_columns(COMMITTED_FRAME_SCHEMAS) == []


def test_every_manifest_frame_has_a_committed_schema():
    assert audited_frame_names() <= set(COMMITTED_FRAME_SCHEMAS)


def test_demographic_panel_schema_excludes_sex():
    # The correction itself (§2.8.3f): the demo frame carries no sex, and the
    # realized-population builder must not read it from there.
    assert "sex" not in COMMITTED_FRAME_SCHEMAS["demographic_panel"]
    build = PHASE_FRAME_COLUMN_READS["build_realized_population"]
    assert "sex" not in build["demographic_panel"]
    # Person sex is read from the death-records frame instead.
    assert "sex" in build["death_records"]


def test_audit_flags_a_phase_reading_a_column_absent_from_the_frame():
    # Discriminating: dropping a required column from a frame's schema must be
    # reported, so a green audit is meaningful rather than vacuous.
    broken = dict(COMMITTED_FRAME_SCHEMAS)
    broken["demographic_panel"] = COMMITTED_FRAME_SCHEMAS[
        "demographic_panel"
    ] - {"interview"}
    violations = audit_columns(broken)
    assert any(
        "interview" in line and "demographic_panel" in line
        for line in violations
    )


def test_audit_flags_a_frame_with_no_schema_entry():
    broken = {
        name: cols
        for name, cols in COMMITTED_FRAME_SCHEMAS.items()
        if name != "death_records"
    }
    assert any("death_records" in line for line in audit_columns(broken))
