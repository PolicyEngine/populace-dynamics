"""Tests for person-period panel construction from PSID products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from populace_dynamics.data import panels

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_data = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID data not staged",
)


@pytest.fixture()
def mini_ind_dir(tmp_path: Path) -> Path:
    """A tiny synthetic ind2023er product: 2 waves, 3 persons.

    Positions and the SPS setup generate from one field table so the
    fixture cannot drift from its own layout. Fields cover every
    demographic concept for waves 1969 and 1970; person 1002 moves
    out in 1970 (sequence 71), exercising the presence filter, and
    1970's weight exists only in longitudinal form, exercising the
    concept fallback.
    """
    fields = [
        ("ER30001", 2, "1968 INTERVIEW NUMBER", [1, 1, 2]),
        ("ER30002", 3, "PERSON NUMBER   68", [1, 2, 1]),
        ("A1", 2, "AGE OF INDIVIDUAL   69", [30, 28, 40]),
        ("S1", 2, "SEQUENCE NUMBER   69", [1, 2, 1]),
        ("R1", 2, "RELATIONSHIP TO HEAD   69", [10, 30, 10]),
        ("W1", 2, "INDIVIDUAL WEIGHT   69", [10, 11, 12]),
        ("I1", 2, "1969 INTERVIEW NUMBER", [7, 7, 8]),
        ("A2", 2, "AGE OF INDIVIDUAL   70", [31, 29, 41]),
        ("S2", 2, "SEQUENCE NUMBER   70", [1, 71, 1]),
        ("R2", 2, "RELATION TO REFERENCE PERSON   70", [10, 30, 10]),
        ("W2", 2, "CORE INDIVIDUAL LONGITUDINAL WEIGHT 70", [12, 13, 14]),
        ("I2", 2, "1970 INTERVIEW NUMBER", [9, 9, 11]),
    ]
    product = tmp_path / "ind2023er"
    product.mkdir()
    specs = []
    position = 1
    for name, width, _, _ in fields:
        end = position + width - 1
        specs.append(f"      {name:<15} {position} - {end}")
        position += width
    label_lines = [f'   {name:<12} "{label}"' for name, _, label, _ in fields]
    sps = (
        "DATA LIST FILE = PSID FIXED /\n"
        + "\n".join(specs)
        + "\n.\n\nVARIABLE LABELS\n"
        + "\n".join(label_lines)
        + "\n.\n"
    )
    (product / "IND2023ER.sps").write_text(sps)
    lines = []
    for person in range(3):
        line = "".join(
            f"{values[person]:>{width}}" for _, width, _, values in fields
        )
        lines.append(line)
    (product / "IND2023ER.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_label_year_trailing_and_leading_forms():
    assert panels.label_year("AGE OF INDIVIDUAL   68") == 1968
    assert panels.label_year("AGE OF INDIVIDUAL   23") == 2023
    assert panels.label_year("1999 INTERVIEW NUMBER") == 1999
    assert panels.label_year("RELEASE NUMBER") is None


def test_wave_variables_raises_on_ambiguity():
    labels = {
        "X1": "RELATION TO HEAD                      95",
        "X2": "RELATIONSHIP TO RESPONDENT            95",
    }
    with pytest.raises(ValueError, match="ambiguous"):
        panels.wave_variables(labels, r"RELATION")
    tightened = panels.wave_variables(
        labels, r"^RELATION(SHIP)? TO (HEAD|REFERENCE PERSON)"
    )
    assert tightened == {1995: "X1"}


def test_ind_person_period_melts_and_packs_ids(mini_ind_dir: Path):
    long = panels.ind_person_period(
        {"age": (r"^AGE OF INDIVIDUAL",)},
        data_dir=mini_ind_dir,
    )
    assert set(long.columns) == {"person_id", "period", "age"}
    assert sorted(long.person_id.unique()) == [1001, 1002, 2001]
    assert sorted(long.period.unique()) == [1969, 1970]
    row = long[(long.person_id == 1001) & (long.period == 1970)]
    assert row.age.iloc[0] == 31


def test_weight_fallback_covers_both_waves(mini_ind_dir: Path):
    long = panels.ind_person_period(
        {
            "weight": (
                r"^INDIVIDUAL WEIGHT|CROSS-SECTION WT",
                r"LONGITUDINAL W",
            )
        },
        data_dir=mini_ind_dir,
    )
    assert sorted(long.period.unique()) == [1969, 1970]
    w = long[(long.person_id == 2001)].sort_values("period")
    assert list(w.weight) == [12, 14]


def test_max_period_field_caps_the_wide_source_read(mini_ind_dir: Path):
    long = panels.ind_person_period(
        {"age": (r"^AGE OF INDIVIDUAL",)},
        data_dir=mini_ind_dir,
        max_period=1969,
    )
    assert long["period"].unique().tolist() == [1969]


def test_demographic_panel_presence_filter(mini_ind_dir: Path):
    everyone = panels.demographic_panel(
        data_dir=mini_ind_dir, in_family_only=False
    )
    present = panels.demographic_panel(data_dir=mini_ind_dir)
    assert len(everyone) == 6
    # Person 1002 moved out in 1970 (sequence 71) and drops.
    assert len(present) == 5
    gone = present[(present.person_id == 1002) & (present.period == 1970)]
    assert gone.empty


def test_missing_concept_errors_clearly(mini_ind_dir: Path):
    with pytest.raises(ValueError, match="matched no wave variables"):
        panels.ind_person_period(
            {"earnings": (r"TOTAL LABOR INCOME",)},
            data_dir=mini_ind_dir,
        )


@needs_real_data
def test_real_demographic_panel_shape_and_sanity():
    panel = panels.demographic_panel()
    assert panel.period.min() == 1969
    assert panel.period.max() == 2023
    assert panel.period.nunique() == 42
    # Tens of thousands of persons; ~0.9M present person-periods
    # (verified 892,639 on the 2023 release).
    assert panel.person_id.nunique() > 30_000
    assert len(panel) > 800_000
    # Ages plausible for present persons; weights non-negative.
    assert panel.age.between(0, 125).mean() > 0.999
    assert (panel.weight >= 0).all()
    # person-period uniqueness.
    assert not panel.duplicated(["person_id", "period"]).any()


@needs_real_data
def test_real_panel_feeds_panel_view_projection():
    from populace_dynamics.harness import panel as hpanel

    demo = panels.demographic_panel()
    recent = demo[demo.period >= 1999]
    view = hpanel.PanelView(
        name="psid_age_pairs",
        id_column="person_id",
        period_column="period",
        value_columns=("age",),
        weight_column="weight",
        window=2,
        period_step=2,
    )
    points, _ = hpanel.project_panel(recent, view)
    assert points.shape[0] > 100_000
    # Ages advance about two years across a biennial window.
    diffs = points[:, 1] - points[:, 0]
    assert np.median(diffs) == 2


@pytest.fixture()
def mislabeled_earnings_dir(tmp_path: Path) -> Path:
    """An ind2023er product whose earnings variable label is wrong."""
    product = tmp_path / "ind2023er"
    product.mkdir()
    sps = (
        "DATA LIST FILE = PSID FIXED /\n"
        "      ER30001         1 - 2         ER30002         3 - 5\n"
        "      ER33537N        6 - 12\n"
        ".\n\nVARIABLE LABELS\n"
        '   ER30001      "1968 INTERVIEW NUMBER"\n'
        '   ER30002      "PERSON NUMBER   68"\n'
        '   ER33537N     "TOTAL ANNUAL EARNINGS IN 1998         99"\n'
        ".\n"
    )
    (product / "IND2023ER.sps").write_text(sps)
    (product / "IND2023ER.txt").write_text(" 1  1  50000\n")
    return tmp_path


def test_earnings_panel_rejects_mislabeled_release(
    mislabeled_earnings_dir: Path,
):
    with pytest.raises(ValueError, match="release layout may have changed"):
        panels.individual_earnings_panel(data_dir=mislabeled_earnings_dir)


@needs_real_data
def test_real_individual_earnings_panel():
    panel = panels.individual_earnings_panel()
    assert sorted(panel.period.unique()) == [1997, 1999, 2001]
    # Verified 61,066 rows over 24,429 persons on the 2023 release.
    assert len(panel) > 55_000
    assert panel.person_id.nunique() > 20_000
    # Missing sentinel filtered; weights positive.
    assert panel.earnings.max() < 9_999_999
    assert (panel.weight > 0).all()
    # Zeros retained: the nonemployment margin stays scoreable.
    assert (panel.earnings == 0).mean() > 0.1


@needs_real_data
def test_real_earnings_noise_floor_reproduces_committed_run():
    """The committed first-noise-floor artifact reproduces at seed 0."""
    import json

    from populace_dynamics.harness import panel as hpanel

    committed = json.loads(
        Path("runs/noise_floor_psid_earnings_9701.json").read_text()
    )
    full = panels.individual_earnings_panel()
    prime = full[(full.age >= 25) & (full.age <= 59)]
    view = hpanel.PanelView(
        name="psid_earnings_pairs_9701",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=2,
        period_step=2,
    )
    floor = hpanel.noise_floor(prime, view, seed=0)
    recorded = committed["noise_floor_seeds_0_4"]
    # Seed-0 values sit within the recorded 5-seed spread.
    for key in ("c2st_auc", "prdc_coverage"):
        stats = recorded[key]
        assert abs(floor[key] - stats["mean"]) < max(4 * stats["sd"], 0.03)
