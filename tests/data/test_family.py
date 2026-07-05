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


@pytest.fixture()
def mini_family_dir(tmp_path: Path) -> Path:
    """ind2023er + one 1994 family wave, three persons in two families.

    Family 7 has head 1001 and wife 1002; family 8 has head 2001.
    Head labor 30000/40000; wife labor 12000. Both products generate
    from field tables so positions cannot drift, and a _formats.sps
    decoy checks the glob exclusion.
    """

    def write_product(directory, sps_name, txt_name, fields):
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

    write_product(
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
    write_product(
        tmp_path / "family" / "1994",
        "FAM1994ER.sps",
        "FAM1994ER.txt",
        [
            ("ER2002", 2, "1994 INTERVIEW #", [7, 8]),
            ("ER4140", 6, "LABOR INCOME OF HEAD-1993", [30000, 40000]),
            ("ER4144", 6, "LABOR INCOME OF WIFE-1993", [12000, 0]),
        ],
    )
    (tmp_path / "family" / "1994" / "FAM1994ER_formats.sps").write_text(
        "decoy\n"
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


def test_reference_year_mismatch_raises(mini_family_dir: Path):
    sps = mini_family_dir / "family" / "1994" / "FAM1994ER.sps"
    sps.write_text(sps.read_text().replace("HEAD-1993", "HEAD-1992"))
    with pytest.raises(ValueError, match="reference year"):
        family.read_family_labor(1994, data_dir=mini_family_dir)


def test_out_of_range_wave_rejected():
    with pytest.raises(ValueError, match="outside the resolved range"):
        family.read_family_labor(1985)


@needs_real_family
def test_real_all_waves_resolve():
    for wave in family.FAMILY_WAVES:
        frame = family.read_family_labor(wave, nrows=5)
        assert list(frame.columns) == [
            "interview",
            "head_labor",
            "spouse_labor",
        ]


@needs_real_family
def test_real_family_earnings_panel():
    panel = family.family_earnings_panel()
    # Verified 207,806 rows over 29,142 persons on the 2023 release.
    assert len(panel) > 190_000
    assert panel.person_id.nunique() > 25_000
    assert panel.period.min() == 1993
    assert panel.period.max() == 2022
    assert panel.earnings.max() < 9_999_998
    assert set(panel.role.unique()) == {"head", "spouse"}
    assert not panel.duplicated(["person_id", "period"]).any()


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
