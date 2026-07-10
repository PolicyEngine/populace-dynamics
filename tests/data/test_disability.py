"""Tests for the PSID disability reader, hazards, and code verification.

Two tiers, mirroring the other data-reader tests:

* Always-runnable: the pure code map and SAS value-label / format-
  assignment parsers, the value-code verification against a synthetic
  formats file (pass and the release-drift raise), the label-verified
  read on a synthetic ``ind2023er`` fixture, and the incidence / recovery
  / conversion / prevalence hazards on hand-built person-year frames --
  no real PSID needed.
* Skipped off-machine (needs the staged individual file): the real read
  covers the 20 employment-status waves, the value codes verify against
  the release's own formats, disability prevalence rises with age, and
  the grid-adjacent transition pairs sit on the 1-2 year grid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import disability
from tests.data.psid_fixtures import write_product

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID ind2023er not staged",
)


# --------------------------------------------------------------------------
# Constants -- pure
# --------------------------------------------------------------------------
def test_code_map_and_flags():
    assert disability.EMPLOYMENT_STATUS_CODES[disability.DISABLED_CODE] == (
        "disabled"
    )
    assert disability.EMPLOYMENT_STATUS_CODES[disability.RETIRED_CODE] == (
        "retired"
    )
    assert disability.EMPLOYMENT_STATUS_CODES[1] == "working"
    # 0 "Inap." and 9 "NA/DK" are not ascertained states.
    assert 0 not in disability.VALID_STATUS_CODES
    assert 9 not in disability.VALID_STATUS_CODES
    assert disability.VALID_STATUS_CODES == frozenset({1, 2, 3, 4, 5, 6, 7, 8})


# --------------------------------------------------------------------------
# SAS value-label / format-assignment parsers -- pure
# --------------------------------------------------------------------------
_FORMATS_SAS = """\
   VALUE ES99F
         1 = 'Working now'
         2 = 'Only temporarily laid off'
         4 = 'Retired'
         5 = 'Permanently disabled'
         9 = 'DK; NA; refused'
         0 = 'Inap.:  born or moved in after the interview; other'
             'continuation line that should be ignored'
   ;
   VALUE ES01F
         1 = 'Working now'
         5 = 'Permanently disabled'
   ;
   FORMAT
      ES99       ES99F.
      ES01       ES01F.
   ;
"""


def _write_formats(tmp_path: Path, body: str) -> Path:
    product = tmp_path / "ind2023er"
    product.mkdir(parents=True, exist_ok=True)
    (product / "IND2023ER_formats.sas").write_text(body)
    return product / "IND2023ER_formats.sas"


def test_parse_sas_value_labels(tmp_path):
    path = _write_formats(tmp_path, _FORMATS_SAS)
    labels = disability.parse_sas_value_labels(path)
    assert labels["ES99F"][5] == "Permanently disabled"
    assert labels["ES99F"][1] == "Working now"
    assert labels["ES99F"][4] == "Retired"
    assert labels["ES01F"] == {1: "Working now", 5: "Permanently disabled"}


def test_parse_sas_format_assignments(tmp_path):
    path = _write_formats(tmp_path, _FORMATS_SAS)
    assigns = disability.parse_sas_format_assignments(path)
    assert assigns["ES99"] == "ES99F"
    assert assigns["ES01"] == "ES01F"
    # Value-label lines never leak into the assignment map.
    assert "1" not in assigns and "5" not in assigns


# --------------------------------------------------------------------------
# Value-code verification against the formats file -- pure
# --------------------------------------------------------------------------
def _status_sps_fields() -> list[tuple[str, int, str, list[int]]]:
    """Minimal ind fixture fields resolving 2 employment-status waves."""
    return [
        ("ER30001", 2, "1968 INTERVIEW NUMBER", [1, 2]),
        ("ER30002", 3, "PERSON NUMBER   68", [1, 1]),
        ("ES99", 1, "EMPLOYMENT STATUS   99", [5, 1]),
        ("ES01", 1, "EMPLOYMENT STATUS   01", [1, 5]),
    ]


def _write_status_product(tmp_path: Path) -> Path:
    write_product(
        tmp_path / "ind2023er",
        "IND2023ER.sps",
        "IND2023ER.txt",
        _status_sps_fields(),
    )
    return tmp_path


def test_verify_codes_passes_on_good_formats(tmp_path):
    _write_status_product(tmp_path)
    _write_formats(tmp_path, _FORMATS_SAS)
    result = disability.verify_employment_status_codes(data_dir=tmp_path)
    assert set(result) == {1999, 2001}
    assert result[1999]["code5_label"] == "Permanently disabled"
    assert result[1999]["format"] == "ES99F"


def test_verify_codes_falls_back_to_naming_convention(tmp_path):
    """A missing FORMAT-assignment block falls back to the <var>F name."""
    _write_status_product(tmp_path)
    no_assign = _FORMATS_SAS.split("   FORMAT")[0]
    _write_formats(tmp_path, no_assign)
    result = disability.verify_employment_status_codes(data_dir=tmp_path)
    assert result[2001]["format"] == "ES01F"


def test_verify_codes_raises_when_disabled_code_renumbered(tmp_path):
    """Release drift (code 5 no longer 'disabled') fails loudly."""
    _write_status_product(tmp_path)
    bad = _FORMATS_SAS.replace("5 = 'Permanently disabled'", "5 = 'Retired'")
    _write_formats(tmp_path, bad)
    with pytest.raises(ValueError, match="expected a 'disabled' label"):
        disability.verify_employment_status_codes(data_dir=tmp_path)


# --------------------------------------------------------------------------
# read_disability_status on a synthetic ind fixture -- pure
# --------------------------------------------------------------------------
_READER_FIELDS = [
    ("ER30001", 2, "1968 INTERVIEW NUMBER", [10, 10, 20, 30, 50]),
    ("ER30002", 3, "PERSON NUMBER   68", [1, 1, 1, 1, 1]),
    # 1999 wave
    ("A99", 2, "AGE OF INDIVIDUAL   99", [45, 55, 30, 40, 64]),
    ("S99", 2, "SEQUENCE NUMBER   99", [1, 1, 1, 30, 1]),
    ("W99", 2, "INDIVIDUAL WEIGHT   99", [10, 12, 5, 8, 7]),
    ("E99", 1, "EMPLOYMENT STATUS   99", [1, 5, 0, 1, 5]),
    # 2001 wave
    ("A01", 2, "AGE OF INDIVIDUAL   01", [47, 57, 32, 42, 66]),
    ("S01", 2, "SEQUENCE NUMBER   01", [1, 1, 1, 1, 1]),
    ("W01", 2, "INDIVIDUAL WEIGHT   01", [11, 13, 5, 8, 7]),
    ("E01", 1, "EMPLOYMENT STATUS   01", [5, 1, 9, 1, 4]),
]


def test_read_status_filters_and_decodes(tmp_path):
    write_product(
        tmp_path / "ind2023er",
        "IND2023ER.sps",
        "IND2023ER.txt",
        _READER_FIELDS,
    )
    df = disability.read_disability_status(data_dir=tmp_path)
    # person 20001 is Inap. (0) in 99 and NA (9) in 01 -> both dropped.
    assert 20001 not in set(df.person_id)
    # person 30001 is out-of-family (sequence 30) in 99 -> that row dropped,
    # but present in 01.
    p3 = df[df.person_id == 30001]
    assert list(p3.period) == [2001]
    # Decodes and flags.
    row = df[(df.person_id == 10001) & (df.period == 2001)].iloc[0]
    assert row.status_code == 5
    assert row.status == "disabled"
    assert bool(row.disabled) is True
    assert bool(row.retired) is False


def test_read_status_missing_concept_raises(tmp_path):
    """Renaming every EMPLOYMENT STATUS label resolves no wave and raises."""
    renamed = {"E99": "LABOR FORCE STATE   99", "E01": "LABOR FORCE STATE  01"}
    bad = [
        (name, w, renamed.get(name, lab), v)
        for name, w, lab, v in _READER_FIELDS
    ]
    write_product(
        tmp_path / "ind2023er", "IND2023ER.sps", "IND2023ER.txt", bad
    )
    with pytest.raises(ValueError, match="matched no wave variables"):
        disability.read_disability_status(data_dir=tmp_path)


# --------------------------------------------------------------------------
# Panel + hazards on hand-built frames -- pure
# --------------------------------------------------------------------------
def _hand_status() -> pd.DataFrame:
    """person-year states: 1001 onset, 1002 recovery, 5001 conversion."""
    rows = [
        # 1001 male: working(45) -> disabled(47): incidence in 40-49
        (1001, 1999, 1, 45),
        (1001, 2001, 5, 47),
        # 1002 female: disabled(55) -> working(57): recovery in 50-59
        (1002, 1999, 5, 55),
        (1002, 2001, 1, 57),
        # 5001 male: disabled(64) -> retired(66): conversion at 60-67
        (5001, 1999, 5, 64),
        (5001, 2001, 4, 66),
        # 6001 male: disabled(52) -> disabled(54): stays disabled (no exit)
        (6001, 1999, 5, 52),
        (6001, 2001, 5, 54),
    ]
    df = pd.DataFrame(
        rows, columns=["person_id", "period", "status_code", "age"]
    )
    df["status"] = df["status_code"].map(disability.EMPLOYMENT_STATUS_CODES)
    df["disabled"] = df["status_code"] == disability.DISABLED_CODE
    df["retired"] = df["status_code"] == disability.RETIRED_CODE
    df["weight"] = 1.0
    df["sequence"] = 1
    return df


def _hand_deaths() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1001, 1002, 5001, 6001],
            "sex": ["male", "female", "male", "male"],
        }
    )


def test_build_panel_pairs_and_censoring():
    panel = disability.build_disability_panel(_hand_status(), _hand_deaths())
    assert len(panel.person_years) == 8
    # Every person here has two grid-adjacent (2-yr) waves -> one pair each.
    assert len(panel.pairs) == 4
    assert set(panel.pairs.interval) == {2}
    onset = panel.pairs[panel.pairs.person_id == 1001].iloc[0]
    assert bool(onset.from_disabled) is False
    assert bool(onset.to_disabled) is True


def test_incidence_and_recovery_cells():
    panel = disability.build_disability_panel(_hand_status(), _hand_deaths())
    inc = disability.incidence_cells(panel, weighted=False)
    # 1001 is the only not-disabled origin (40-49 male) and it onsets.
    assert inc["incidence.40-49|male"]["rate"] == pytest.approx(1.0)
    assert inc["incidence.40-49|male"]["n_events"] == 1
    rec = disability.recovery_cells(panel, weighted=False)
    # 1002 (50-59 female) recovers; 6001 (50-59 male) does not.
    assert rec["recovery.50-59|female"]["rate"] == pytest.approx(1.0)
    assert rec["recovery.50-59|male"]["rate"] == pytest.approx(0.0)
    assert rec["recovery.50-59|male"]["n_at_risk"] == 1


def test_conversion_and_prevalence_cells():
    panel = disability.build_disability_panel(_hand_status(), _hand_deaths())
    conv = disability.conversion_cells(panel, weighted=False)
    # 5001 (male) is the only retired-entry in 60-67 and comes from disabled.
    cell = conv["conversion.retired_from_disabled|male"]
    assert cell["rate"] == pytest.approx(1.0)
    assert cell["n_events"] == 1
    assert cell["n_at_risk"] == 1
    prev = disability.prevalence_cells(panel, weighted=False)
    # 6001 contributes two disabled person-years at 50-59 male; 1002's 55
    # disabled year is female.
    assert prev["prevalence.50-59|male"]["n_events"] == 2
    assert prev["prevalence.50-59|female"]["n_events"] == 1


def test_reference_moments_key_set_is_fixed():
    panel = disability.build_disability_panel(_hand_status(), _hand_deaths())
    cells = disability.reference_moments(panel)
    # 5 bands x 2 sexes x {incidence, recovery, prevalence} + 2 conversion.
    assert len(cells) == 5 * 2 * 3 + 2
    assert "conversion.retired_from_disabled|female" in cells


def test_person_subset_restricts_moments():
    panel = disability.build_disability_panel(_hand_status(), _hand_deaths())
    subset = disability.incidence_cells(
        panel, person_ids={1002}, weighted=False
    )
    # 1002 is never a not-disabled origin, so no onset exposure.
    assert subset["incidence.40-49|male"]["n_at_risk"] == 0


# --------------------------------------------------------------------------
# Real ind2023er -- skipped off-machine
# --------------------------------------------------------------------------
@needs_real_ind
def test_real_wave_coverage_and_state_domain():
    df = disability.read_disability_status()
    waves = sorted(int(w) for w in df.period.unique())
    assert waves == [
        1982,
        1983,
        1993,
        1994,
        1995,
        1996,
        1997,
        1999,
        2001,
        2003,
        2005,
        2007,
        2009,
        2011,
        2013,
        2015,
        2017,
        2019,
        2021,
        2023,
    ]
    # Only ascertained states survive the read.
    assert set(df.status_code.unique()) <= disability.VALID_STATUS_CODES


@needs_real_ind
def test_real_value_codes_verify_against_formats():
    result = disability.verify_employment_status_codes()
    assert len(result) == 20
    assert "disabl" in result[2023]["code5_label"].lower()
    assert "work" in result[2023]["code1_label"].lower()


@needs_real_ind
def test_real_prevalence_rises_with_age():
    from populace_dynamics.data import deaths

    status = disability.read_disability_status()
    panel = disability.build_disability_panel(
        status, deaths.read_death_records()
    )
    prev = disability.prevalence_cells(panel)
    for sex in ("male", "female"):
        young = prev[f"prevalence.20-29|{sex}"]["rate"]
        mid = prev[f"prevalence.50-59|{sex}"]["rate"]
        assert young < mid
    # Grid-adjacent pairs sit on the 1-2 year PSID grid.
    assert set(panel.pairs.interval.unique()) <= {1, 2}
    assert len(panel.pairs) > 100_000
