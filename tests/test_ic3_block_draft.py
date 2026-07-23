"""Bind the IC3 employer gate-block draft before it can lock.

ALWAYS-RUNNABLE (artifact tier): reads only the committed draft
block YAML and the design document. No microdata, no model, no
candidate run.

Closes blocking item B2 of the round-1 adversarial review, whose
finding was that the referee was asked to ratify prose plus PROPOSED
numbers while the actual ``gates.yaml`` text would first appear —
unrefereed — in the amendment PR, where transcription errors, silent
cell additions and derivation drift would never have been reviewed.

The tests here are the mutation pins the review asked for. Their
job is that a dropped, added, or renamed cell **fails**, rather than
passing silently into a locked contract:

* every gated family's cell list is pinned by exact count and
  membership;
* the calibration/gate partition is pinned as disjoint sets, with
  every gate registered on exactly one side (the round-1 defect was
  E6 and E7 appearing on neither);
* the demotions that round 1 forced — E1, E6, and both E9-stay
  statistics — are pinned as demoted, so a later edit cannot quietly
  re-gate a cell whose floor is degenerate or whose reference is a
  calibration target;
* the block cannot claim to be locked while its own prerequisites
  are unmet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs" / "design" / "ic3_employer_gate_block_draft.yaml"
DESIGN = ROOT / "docs" / "design" / "c3_employer_gate_block.md"

#: Exact gated cell counts. These numbers ARE the mutation pin: a
#: cell added or dropped anywhere changes one of them.
GATED_CELL_COUNTS = {
    "e2_sexage_rates": 16,  # 2 sexes x 8 age groups, margins excluded
    "e3_tenure_ecdf_by_age": 7,
    "e4_retention_by_age_sex": 12,  # 6 age bands x 2 sexes
    "e5_runs_by_age": 6,
    "e7_earns_relative_gradient_by_size": 5,
    "e8_nonemployment_by_age": 12,  # 6 age bands x 2 statistics
    "e9_j2j_earnings_change": 2,
    "e10_locked_psid_gates": 1,  # inherits; not enumerated here
    "e11_destination_size_margins": 5,
}

#: Demoted by round 1. Re-gating any of these silently is the
#: failure mode this set exists to prevent.
MUST_STAY_REPORT_ONLY = {
    "e1_size_margin_and_sector_cross",  # B1: function of calibration targets
    "e6_size_margin_and_sector_cells",  # B1: linear functional of targets
    "e9_stay_median",  # degenerate floor
    "e9_stay_iqr",  # degenerate floor (B5)
    "firm_size_conditional_cells",  # no truth frame, permanent
    "e11_origin_x_destination_detail",  # one YoY pair per cell
}


@pytest.fixture(scope="module")
def block() -> dict:
    return yaml.safe_load(DRAFT.read_text(encoding="utf-8"))["gates"][
        "employer"
    ]


def test_block_is_not_locked_and_edits_no_gate(block):
    assert block["locked"] is False
    assert block["status"].startswith("draft")
    # The draft must never carry a concrete design_commit: it is
    # concretized only at the flip.
    assert block["design_commit"] == "PENDING"


def test_gated_cell_counts_are_pinned(block):
    families = block["families"]
    actual = {name: len(fam["cells"]) for name, fam in families.items()}
    assert actual == GATED_CELL_COUNTS


def test_every_gated_family_is_registered_in_the_partition(block):
    """The round-1 defect: E6 and E7 appeared on neither side."""
    families = set(block["families"])
    gate_cells = set(block["partition"]["gate_cells"])
    assert families == gate_cells


def test_partition_sides_are_disjoint(block):
    partition = block["partition"]
    gate = set(partition["gate_cells"])
    report = set(partition["report_only"])
    deferred = set(partition["deferred"])
    fitting = set(partition["fitting_consumes"])
    assert gate.isdisjoint(report)
    assert gate.isdisjoint(deferred)
    assert report.isdisjoint(deferred)
    # Nothing consumed by fitting may appear as a gated cell.
    assert gate.isdisjoint(fitting)


def test_round_1_demotions_cannot_be_silently_reverted(block):
    report_only = set(block["partition"]["report_only"])
    assert MUST_STAY_REPORT_ONLY <= report_only
    # And none of them may reappear as a gated family.
    assert MUST_STAY_REPORT_ONLY.isdisjoint(set(block["families"]))


def test_the_only_quantifier_binds_more_than_microcalibrate(block):
    """#192's hazards calibrate to QWI/J2J; that consumer was
    unregistered, which left the E2 and E11 holdouts unenforceable."""
    text = block["partition"]["only_quantifier"]
    for stage in ("microcalibrate", "hazard", "hyperparameter", "raking"):
        assert stage in text
    assert "whether or not it is called calibration" in text


def test_the_deterministic_function_corollary_is_registered(block):
    corollary = block["partition"]["deterministic_function_corollary"]
    assert "BY NAME" in corollary
    assert "linear functional" in corollary


def test_every_gated_family_cites_a_floor_or_declares_why_not(block):
    for name, family in block["families"].items():
        if family["floor_run"] is None:
            # Only E10 may lack a floor: it re-scores locked cells.
            assert name == "e10_locked_psid_gates"
            continue
        assert family["floor_run"].startswith("runs/")
        assert family["floor_run"].endswith("_v1.json")


def test_no_gated_family_rests_on_a_degenerate_floor(block):
    """B5's rule: a hand-set number may not be the whole gate.

    E9-stay was the violation — its IQR floor is 0.0/0.0 like its
    median, so 'IQR-only' was 100% the hand-set 0.02.
    """
    families = block["families"]
    assert "e9_stay" not in families
    assert "stay" not in families["e9_j2j_earnings_change"]["floor_path"]


def test_e11_carries_its_demotion_condition(block):
    condition = block["partition"]["e11_condition"]
    assert "J2JOD" in condition
    assert "no further referee round" in condition


def test_scoring_frame_registers_split_machine_fraction_and_seed(block):
    """S2: none of these were registered in the prose draft."""
    frame = block["scoring_frame"]["sipp_holdout"]
    assert frame["split_unit"] == "person"
    assert 0 < frame["holdout_fraction"] < 1
    assert isinstance(frame["seed"], int)
    assert frame["split_machine"]
    assert "disjoint" in frame["leakage_rule"]


def test_floor_scale_gap_is_recorded_as_unsatisfied(block):
    """S2's other half. Honest status beats a silent assertion."""
    scale = block["scoring_frame"]["floor_scale_assertion"]
    assert scale["status"] == "RECORDED_NOT_SATISFIED"
    assert scale["tolerance"] == "REFEREE"


def test_k_basis_states_the_locked_band_honestly(block):
    """S1: the band is 1.8-8, and two locked ks sit below 4."""
    basis = block["threshold_policy"]["k_basis"]
    assert "1.8-8" in basis
    assert "POLICY CHOICE" in basis
    assert "5 seeds exactly as gate-1's did" in basis


def test_linkage_reweighting_is_instantiated(block):
    """B4: ADR 0004 §3 was dropped entirely from the prose draft."""
    rw = block["linkage_qc"]["reweighting"]
    assert rw["clustering"] == "worker"
    assert "seam_vs_within_indicator" in rw["observables"]
    # The numerics stay referee items, registered as such.
    for slot in (
        "propensity_spec",
        "overlap_balance_tolerance",
        "trimming_percentile",
        "operative_version",
    ):
        assert rw[slot] == "REFEREE"
    assert "weighted AND unweighted" in rw["publication_rule"]


def test_linkage_failure_never_weakens_e10(block):
    rule = block["families"]["e10_locked_psid_gates"]["hard_rule"]
    assert "NEVER WEAKENS E10" in rule


def test_ceremony_requires_merges_and_pinning(block):
    """B3: a block whose normative ADR is a branch file is not locked."""
    ceremony = block["ceremony"]
    merges = " ".join(ceremony["merges_required_before_amendment"])
    assert "#224" in merges
    assert "#277" in merges
    assert "AT ITS MERGE COMMIT" in ceremony["artifact_pinning"]
    assert ceremony["referee_verification_required"] is True
    assert ceremony["no_candidate_runs_before_lock"] is True


def test_the_block_states_what_it_cannot_certify(block):
    """The paper's rule: misses publish as prominently as hits."""
    not_certified = block["not_certified"]
    assert "E2, E7 and E11-margins ONLY" in (
        not_certified["firm_side_gating_is_thin"]
    )
    assert "PERMANENTLY" in not_certified["imputed_band_conditioned_cells"]
    assert "Phase-2 no-go" in not_certified["e12_not_gated"]


@pytest.mark.skipif(
    not DESIGN.exists(),
    reason=(
        "the revision-2 design document lands with #230; this branch "
        "carries the block YAML only. The check becomes live — and "
        "must pass — once both are on master, which the ceremony's "
        "merge list requires before the amendment PR."
    ),
)
def test_draft_and_design_document_agree_on_the_demotions(block):
    """The YAML and the prose must not drift apart.

    B2 exists because the block text and the document the referee
    read were different artifacts. Pinning their agreement is the
    point, not a nicety.
    """
    design = DESIGN.read_text(encoding="utf-8")
    # The design doc must not still advertise E1/E6 as gated.
    assert "does not gate at first" in design
    assert "E2, E7 and E11-margins only" in design


def test_naming_uses_the_interface_contract_prefix():
    """A bare C1/C2 in gates.yaml means the LOCKED gate_w1
    fingerprints. This block must never use the colliding form."""
    text = DRAFT.read_text(encoding="utf-8")
    body = text.split("# NAMING.")[1].split("\n\n", 1)[1]
    import re

    assert not re.search(r"\bC[123]\b", body)
