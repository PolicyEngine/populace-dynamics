"""Tests for family-file labor income and the earnings-history merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from populace_dynamics.data import family

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _write_product(directory, sps_name, txt_name, fields):
    """Write a fixture PSID product from field tables (positions fixed).

    ``fields`` is a list of ``(name, width, label, values)`` tuples;
    the setup file's DATA LIST positions are derived from the widths so
    they cannot drift, and the fixed-width text is right-justified to
    match.
    """
    directory.mkdir(parents=True, exist_ok=True)
    specs = []
    position = 1
    for name, width, _, _ in fields:
        end = position + width - 1
        specs.append(f"      {name:<15} {position} - {end}")
        position += width
    labels = "\n".join(
        f'   {name:<12} "{label}"' for name, _, label, _ in fields
    )
    sps = (
        "DATA LIST FILE = PSID FIXED /\n"
        + "\n".join(specs)
        + "\n.\n\nVARIABLE LABELS\n"
        + labels
        + "\n.\n"
    )
    (directory / sps_name).write_text(sps)
    n = len(fields[0][3])
    lines = [
        "".join(f"{values[i]:>{width}}" for _, width, _, values in fields)
        for i in range(n)
    ]
    (directory / txt_name).write_text("\n".join(lines) + "\n")


@pytest.fixture()
def mini_family_dir(tmp_path: Path) -> Path:
    """ind2023er + one 1994 family wave, three persons in two families.

    Family 7 has head 1001 and wife 1002; family 8 has head 2001.
    Head labor 30000/40000; wife labor 12000. The 1994 wave also
    carries the head wage/misc accuracy pair and the wife direct-total
    accuracy flag, so the max-of-components logic is pinned: family 7's
    head has wage code 1 and misc code 3 (max 3), family 8's head has
    wage code 5 and misc code 0 (max 5); the wife carries direct code
    2. Both products generate from field tables so positions cannot
    drift, and a _formats.sps decoy checks the glob exclusion.
    """
    _write_product(
        tmp_path / "ind2023er",
        "IND2023ER.sps",
        "IND2023ER.txt",
        [
            ("ER30001", 2, "1968 INTERVIEW NUMBER", [1, 1, 2]),
            ("ER30002", 3, "PERSON NUMBER   68", [1, 2, 1]),
            ("A1", 2, "AGE OF INDIVIDUAL   94", [40, 38, 50]),
            ("S1", 2, "SEQUENCE NUMBER   94", [1, 2, 1]),
            ("R1", 2, "RELATIONSHIP TO HEAD   94", [10, 20, 10]),
            (
                "W1",
                2,
                "CORE INDIVIDUAL LONGITUDINAL WEIGHT 94",
                [15, 14, 16],
            ),
            ("I1", 2, "1994 INTERVIEW NUMBER", [7, 7, 8]),
        ],
    )
    _write_product(
        tmp_path / "family" / "1994",
        "FAM1994ER.sps",
        "FAM1994ER.txt",
        [
            ("ER2002", 2, "1994 INTERVIEW #", [7, 8]),
            ("ER4123", 1, "ACC WAGES AND SALARIES OF HEAD-1993", [1, 5]),
            ("ER4137", 1, "ACC MISC LABOR INCOME OF HEAD-1993", [3, 0]),
            ("ER4140", 6, "LABOR INCOME OF HEAD-1993", [30000, 40000]),
            ("ER4144", 6, "LABOR INCOME OF WIFE-1993", [12000, 0]),
            ("ER4145", 1, "ACC LABOR INCOME OF WIFE-1993", [2, 0]),
        ],
    )
    (tmp_path / "family" / "1994" / "FAM1994ER_formats.sps").write_text(
        "decoy\n"
    )
    return tmp_path


@pytest.fixture()
def flagless_head_dir(tmp_path: Path) -> Path:
    """ind2023er + a 1976 family wave, where the head total has no flag.

    1976 is the documented per-role gap: the wife carries a wage
    accuracy flag (``ACC WIFES ANN WAG``) but the head's total labor
    income carries none, so ``earnings_acc`` must default to 0 for the
    head and take the wife's code for the wife. Family 7 has head 1001
    and wife 1002. Uses the exact _PRE94 income variables and the
    adjudicated wife accuracy variable for 1976.
    """
    _write_product(
        tmp_path / "ind2023er",
        "IND2023ER.sps",
        "IND2023ER.txt",
        [
            ("ER30001", 2, "1968 INTERVIEW NUMBER", [1, 1]),
            ("ER30002", 3, "PERSON NUMBER   68", [1, 2]),
            ("A1", 2, "AGE OF INDIVIDUAL   76", [40, 38]),
            ("S1", 2, "SEQUENCE NUMBER   76", [1, 2]),
            ("R1", 2, "RELATIONSHIP TO HEAD   76", [1, 2]),
            ("W1", 2, "INDIVIDUAL WEIGHT   76", [15, 14]),
            ("I1", 2, "1976 INTERVIEW NUMBER", [7, 7]),
        ],
    )
    _write_product(
        tmp_path / "family" / "1976",
        "FAM1976.sps",
        "FAM1976.txt",
        [
            ("V4302", 2, "1976 ID NUMBER", [7]),
            ("V4379", 6, "WIFES ANNUAL WAGE H25", [9000]),
            ("V4380", 1, "ACC WIFES ANN WAG", [4]),
            ("V5031", 6, "HEAD TOTAL LABOR Y", [30000]),
        ],
    )
    return tmp_path


def test_family_merge_assigns_roles_and_reference_year(
    mini_family_dir: Path,
):
    panel = family.family_earnings_panel(
        waves=(1994,), data_dir=mini_family_dir
    )
    assert sorted(panel.period.unique()) == [1993]
    by_person = panel.set_index("person_id")
    assert by_person.loc[1001, "earnings"] == 30000
    assert by_person.loc[1001, "role"] == "head"
    assert by_person.loc[1002, "earnings"] == 12000
    assert by_person.loc[1002, "role"] == "spouse"
    assert by_person.loc[2001, "earnings"] == 40000


def test_earnings_acc_is_max_of_role_components(mini_family_dir: Path):
    """Head acc = max(wage, misc); spouse acc = the wife total flag."""
    panel = family.family_earnings_panel(
        waves=(1994,), data_dir=mini_family_dir
    )
    by_person = panel.set_index("person_id")
    # Head 1001: max(wage 1, misc 3) == 3; head 2001: max(5, 0) == 5.
    assert by_person.loc[1001, "earnings_acc"] == 3
    assert by_person.loc[2001, "earnings_acc"] == 5
    # Spouse 1002 takes the wife direct-total flag, code 2.
    assert by_person.loc[1002, "earnings_acc"] == 2
    assert panel.earnings_acc.dtype.kind == "i"


def test_read_family_labor_exposes_per_role_acc(mini_family_dir: Path):
    """read_family_labor carries head_acc/spouse_acc, max over components."""
    frame = family.read_family_labor(1994, data_dir=mini_family_dir)
    assert list(frame.columns) == [
        "interview",
        "head_labor",
        "spouse_labor",
        "head_acc",
        "spouse_acc",
    ]
    # Row order is family order (7, 8): head_acc max(1,3)=3 then max(5,0)=5.
    assert list(frame.head_acc) == [3, 5]
    assert list(frame.spouse_acc) == [2, 0]


def test_flagless_head_defaults_acc_to_zero(flagless_head_dir: Path):
    """1976 head total carries no accuracy flag -> earnings_acc == 0."""
    frame = family.read_family_labor(1976, data_dir=flagless_head_dir)
    assert list(frame.head_acc) == [0]  # head has no acc variable in 1976
    assert list(frame.spouse_acc) == [4]  # wife wage flag resolves
    panel = family.family_earnings_panel(
        waves=(1976,), data_dir=flagless_head_dir
    )
    by_person = panel.set_index("person_id")
    assert by_person.loc[1001, "role"] == "head"
    assert by_person.loc[1001, "earnings_acc"] == 0
    assert by_person.loc[1002, "role"] == "spouse"
    assert by_person.loc[1002, "earnings_acc"] == 4


def test_acc_label_mismatch_raises(mini_family_dir: Path):
    """A drifted accuracy label fails loudly, like the income vars."""
    sps = mini_family_dir / "family" / "1994" / "FAM1994ER.sps"
    sps.write_text(
        sps.read_text().replace(
            "ACC WAGES AND SALARIES OF HEAD-1993",
            "ACC WAGES AND SALARIES OF HEAD-1992",
        )
    )
    with pytest.raises(ValueError, match="does not"):
        family.read_family_labor(1994, data_dir=mini_family_dir)


def test_waves_without_acc_are_documented():
    """The flagless-wave constant names exactly the two earliest waves."""
    assert family.WAVES_WITHOUT_ACC == (1968, 1969)
    # Those waves are resolved (have income) but carry no acc entry.
    for wave in family.WAVES_WITHOUT_ACC:
        assert wave in family.FAMILY_WAVES
        assert wave not in family._ACC_COMPONENTS


def test_reference_year_mismatch_raises(mini_family_dir: Path):
    sps = mini_family_dir / "family" / "1994" / "FAM1994ER.sps"
    sps.write_text(sps.read_text().replace("HEAD-1993", "HEAD-1992"))
    with pytest.raises(ValueError, match="reference year"):
        family.read_family_labor(1994, data_dir=mini_family_dir)


def test_out_of_range_wave_rejected():
    # 1998 is not an interview year (annual ends 1997, biennial
    # resumes 1999).
    with pytest.raises(ValueError, match="outside the resolved range"):
        family.read_family_labor(1998)


@needs_real_family
def test_real_all_waves_resolve():
    for wave in family.FAMILY_WAVES:
        frame = family.read_family_labor(wave, nrows=5)
        assert list(frame.columns) == [
            "interview",
            "head_labor",
            "spouse_labor",
            "head_acc",
            "spouse_acc",
        ]


@needs_real_family
def test_real_family_earnings_panel():
    panel = family.family_earnings_panel()
    # Verified 405,348 rows over 34,404 persons on the 2023 release
    # (reference years 1968-2022; 1,981 persons carry 35+ observed
    # years).
    assert len(panel) > 380_000
    assert panel.person_id.nunique() > 30_000
    assert panel.period.min() == 1968
    assert panel.period.max() == 2022
    assert panel.period.nunique() == 42
    depth = panel.groupby("person_id").period.nunique()
    assert (depth >= 35).sum() > 1_500
    assert panel.earnings.max() < 9_999_998
    assert set(panel.role.unique()) == {"head", "spouse"}
    assert not panel.duplicated(["person_id", "period"]).any()
    # The accuracy column is a small non-negative integer code; the two
    # earliest reference years (1967-1968 waves, WAVES_WITHOUT_ACC) and
    # any unflagged observation are 0.
    assert "earnings_acc" in panel.columns
    assert panel.earnings_acc.dtype.kind == "i"
    assert panel.earnings_acc.min() == 0
    assert panel.earnings_acc.max() <= 9
    without_acc_periods = {w - 1 for w in family.WAVES_WITHOUT_ACC}
    flagless = panel[panel.period.isin(without_acc_periods)]
    assert (flagless.earnings_acc == 0).all()


@needs_real_family
def test_real_2021_head_acc_share_matches_measured():
    """Anchor: 2021 wave positive-head-labor share at code 1 is 6.5%.

    Measured 433/6651 = 6.51% via ER81627 (ACC WAGES AND SALARIES OF
    RP-2020) on the 2023 release; including the misc-labor component
    leaves the code-1 share unchanged (misc is all zero that wave).
    """
    panel = family.family_earnings_panel()
    heads_2020 = panel[
        (panel.period == 2020) & (panel.role == "head") & (panel.earnings > 0)
    ]
    share_code_1 = (heads_2020.earnings_acc == 1).mean()
    assert share_code_1 == pytest.approx(0.065, abs=0.01)


@needs_real_family
def test_real_family_noise_floor_reproduces_committed_run():
    import json

    from populace_dynamics.harness import panel as hpanel

    committed = json.loads(
        Path("runs/noise_floor_psid_family_9822.json").read_text()
    )
    panel = family.family_earnings_panel()
    prime = panel[
        (panel.age >= 25) & (panel.age <= 59) & (panel.period >= 1998)
    ]
    view = hpanel.PanelView(
        name="psid_family_earnings_pairs_9822",
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
    for key in ("c2st_auc", "prdc_coverage"):
        stats = recorded[key]
        assert abs(floor[key] - stats["mean"]) < max(4 * stats["sd"], 0.03)


def _prime_family_panel():
    panel = family.family_earnings_panel()
    return panel[
        (panel.age >= 25)
        & (panel.age <= 59)
        & (panel.period >= 1998)
        & (panel.period <= 2022)
        & (panel.weight > 0)
    ]


@needs_real_family
def test_real_family_runs_noise_floor_reproduces_committed_run():
    """Window-3 floor artifact (the runs view gate 1 locks)."""
    import json

    from populace_dynamics.harness import panel as hpanel

    committed = json.loads(
        Path("runs/noise_floor_psid_family_runs_9822.json").read_text()
    )
    prime = _prime_family_panel()
    view = hpanel.PanelView(
        name="psid_family_earnings_runs_9822",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=3,
        period_step=2,
    )
    windows, _ = hpanel.project_panel(prime, view)
    assert len(windows) == committed["n_windows"]
    floor = hpanel.noise_floor(prime, view, seed=0)
    recorded = committed["noise_floor_seeds_0_4"]
    for key in ("c2st_auc", "prdc_coverage"):
        stats = recorded[key]
        assert floor[key] == pytest.approx(stats["values"][0])
        assert abs(floor[key] - stats["mean"]) < max(4 * stats["sd"], 0.03)


@needs_real_family
def test_real_family_ctx20_floor_reproduces_committed_run():
    """Candidate-context floor: 20% vs 20% of persons, seed 0."""
    import json

    from populace_dynamics.harness import panel as hpanel

    committed = json.loads(
        Path("runs/noise_floor_psid_family_ctx20_9822.json").read_text()
    )
    prime = _prime_family_panel()
    view = hpanel.PanelView(
        name="psid_family_earnings_ctx20_9822",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=2,
        period_step=2,
    )
    forty, _ = hpanel.split_panel_by_person(
        prime, "person_id", fraction=0.4, seed=1000
    )
    a, b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=0
    )
    score = hpanel.panel_scorecard(a, b, view, seed=0)
    recorded = committed["noise_floor_seeds_0_4"]
    for key in ("c2st_auc", "prdc_coverage", "energy_distance"):
        stats = recorded[key]
        assert score[key] == pytest.approx(stats["values"][0])


@needs_real_family
def test_real_family_runs_ctx20_floor_reproduces_committed_run():
    """Deployment-scale window-3 floor (runs-view derivation basis)."""
    import json

    from populace_dynamics.harness import panel as hpanel

    committed = json.loads(
        Path("runs/noise_floor_psid_family_runs_ctx20_9822.json").read_text()
    )
    prime = _prime_family_panel()
    view = hpanel.PanelView(
        name="psid_family_earnings_runs_ctx20_9822",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=3,
        period_step=2,
    )
    forty, _ = hpanel.split_panel_by_person(
        prime, "person_id", fraction=0.4, seed=1000
    )
    a, b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=0
    )
    score = hpanel.panel_scorecard(a, b, view, seed=0)
    recorded = committed["noise_floor_seeds_0_4"]
    for key in ("c2st_auc", "prdc_coverage"):
        stats = recorded[key]
        assert score[key] == pytest.approx(stats["values"][0])
