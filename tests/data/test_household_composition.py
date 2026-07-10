"""Tests for the gate-2b household-composition panel + moments.

Two tiers, matching the rest of the data suite. An always-runnable tier
exercises the pure construction logic on synthetic
:func:`populace_dynamics.data.relmap.relationship_map`-shaped frames (the
household roster, the coresidence-direction convention, the
multigenerational rule, the demographic join filters, the reference-moment
cell schema), and an integration tier reads the real staged MX23REL +
ind2023er products and pins the panel's shape. The integration tier skips
when the PSID files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import household_composition as hc

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_relmap = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID relationship matrix (MX23REL) not staged",
)
needs_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID individual file (ind2023er) not staged",
)


def _row(
    year: int,
    inum: int,
    ego: int,
    ego_rp: int,
    mx8: int,
    alter: int,
    alter_rp: int,
) -> dict:
    """One relationship-map row (relmap.relationship_map output shape)."""
    return {
        "interview_year": year,
        "interview_number": inum,
        "ego_person_id": ego,
        "ego_sequence": 1,
        "ego_rel_to_rp": ego_rp,
        "ego_rel_to_alter": mx8,
        "alter_person_id": alter,
        "alter_sequence": 1,
        "alter_rel_to_rp": alter_rp,
    }


# Codes: MX7 rel-to-RP 10=RP 20=spouse 30=child 60=grandchild.
# MX8 ego-to-alter 10=self 20=spouse 50=parent 30=child 66=grandparent
# 60=grandchild.
def _household_a(year: int = 2001, inum: int = 5) -> list[dict]:
    """RP(1001) + spouse(1002) + child(1003): 2 generations."""
    return [
        _row(year, inum, 1001, 10, 10, 1001, 10),  # RP self
        _row(year, inum, 1002, 20, 10, 1002, 20),  # spouse self
        _row(year, inum, 1003, 30, 10, 1003, 30),  # child self
        _row(year, inum, 1001, 10, 20, 1002, 20),  # RP -> spouse
        _row(year, inum, 1001, 10, 50, 1003, 30),  # RP -> child (is parent)
        _row(year, inum, 1002, 20, 20, 1001, 10),  # spouse -> RP
        _row(year, inum, 1002, 20, 50, 1003, 30),  # spouse -> child
        _row(year, inum, 1003, 30, 30, 1001, 10),  # child -> RP (is child)
        _row(year, inum, 1003, 30, 30, 1002, 20),  # child -> spouse
    ]


def _household_b(year: int = 2001, inum: int = 6) -> list[dict]:
    """RP(2001) + child(2002) + grandchild(2003): 3 generations."""
    return [
        _row(year, inum, 2001, 10, 10, 2001, 10),
        _row(year, inum, 2002, 30, 10, 2002, 30),
        _row(year, inum, 2003, 60, 10, 2003, 60),
        _row(year, inum, 2001, 10, 50, 2002, 30),  # RP -> child
        _row(year, inum, 2001, 10, 66, 2003, 60),  # RP -> grandchild
        _row(year, inum, 2002, 30, 30, 2001, 10),  # child -> RP
        _row(year, inum, 2002, 30, 50, 2003, 60),  # child -> grandchild
        _row(year, inum, 2003, 60, 60, 2001, 10),  # grandchild -> RP
        _row(year, inum, 2003, 60, 30, 2002, 30),  # grandchild -> child
    ]


def _household_c(year: int = 2001, inum: int = 7) -> list[dict]:
    """RP(3001) alone: 1-person household."""
    return [_row(year, inum, 3001, 10, 10, 3001, 10)]


def _demo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1001, 2001, 40, 100.0),
            (1002, 2001, 42, 100.0),
            (1003, 2001, 15, 100.0),
            (2001, 2001, 70, 50.0),
            (2002, 2001, 45, 50.0),
            (2003, 2001, 16, 50.0),
            (3001, 2001, 30, 200.0),
        ],
        columns=["person_id", "period", "age", "weight"],
    )


def _sex() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1001, "female"),
            (1002, "male"),
            (1003, "female"),
            (2001, "female"),
            (2002, "male"),
            (2003, "male"),
            (3001, "female"),
        ],
        columns=["person_id", "sex"],
    )


def _roster() -> pd.DataFrame:
    rows = _household_a() + _household_b() + _household_c()
    return hc.household_roster(pd.DataFrame(rows))


# --------------------------------------------------------------------------
# Roster + coresidence direction (the load-bearing convention)
# --------------------------------------------------------------------------
def test_roster_has_one_row_per_person_wave():
    roster = _roster()
    assert len(roster) == 7  # 3 + 3 + 1 members
    assert set(roster["person_id"]) == {
        1001,
        1002,
        1003,
        2001,
        2002,
        2003,
        3001,
    }


def test_household_size_counts_distinct_members():
    roster = _roster().set_index("person_id")
    assert roster.loc[1001, "hh_size"] == 3
    assert roster.loc[2001, "hh_size"] == 3
    assert roster.loc[3001, "hh_size"] == 1


def test_coresidence_direction_is_ego_relative():
    """ego coded child -> coresident PARENT; ego coded parent -> coresident
    CHILD. Reading MX8 as the alter's role would invert every flag."""
    roster = _roster().set_index("person_id")
    # The child (1003) lives WITH a parent, not with a child.
    assert bool(roster.loc[1003, "coresident_parent"])
    assert not bool(roster.loc[1003, "coresident_child"])
    assert not bool(roster.loc[1003, "coresident_spouse"])
    # The reference person (1001) lives with a spouse and a child.
    assert bool(roster.loc[1001, "coresident_spouse"])
    assert bool(roster.loc[1001, "coresident_child"])
    assert not bool(roster.loc[1001, "coresident_parent"])


def test_grandparent_grandchild_direction():
    roster = _roster().set_index("person_id")
    # RP (2001) has a coresident grandchild; grandchild (2003) has a
    # coresident parent (2002) but is not itself flagged grandchild-holder.
    assert bool(roster.loc[2001, "coresident_grandchild"])
    assert bool(roster.loc[2003, "coresident_parent"])
    assert not bool(roster.loc[2003, "coresident_grandchild"])


def test_multigen_is_three_lineal_generations():
    roster = _roster().set_index("person_id")
    # Household B spans RP/child/grandchild (gens 0/-1/-2) -> multigen.
    for pid in (2001, 2002, 2003):
        assert bool(roster.loc[pid, "multigen"]), pid
    # Household A is only 2 generations (RP+spouse / child).
    for pid in (1001, 1002, 1003):
        assert not bool(roster.loc[pid, "multigen"]), pid
    # A 1-person household is never multigen.
    assert not bool(roster.loc[3001, "multigen"])


def test_pre1983_era_generation_frame():
    """The abbreviated pre-1983 rel-to-RP frame still resolves the lineal
    generations it carries (child=3, grandchild=6)."""
    rows = [
        _row(1975, 9, 5001, 1, 10, 5001, 1),  # RP self
        _row(1975, 9, 5002, 3, 10, 5002, 3),  # child self
        _row(1975, 9, 5003, 6, 10, 5003, 6),  # grandchild self
        _row(1975, 9, 5001, 1, 66, 5003, 6),  # RP -> grandchild
    ]
    roster = hc.household_roster(pd.DataFrame(rows)).set_index("person_id")
    assert roster.loc[5001, "hh_size"] == 3
    assert bool(roster.loc[5001, "multigen"])  # gens 0 / -1 / -2
    assert bool(roster.loc[5001, "coresident_grandchild"])


# --------------------------------------------------------------------------
# Demographic join filters
# --------------------------------------------------------------------------
def test_join_keeps_usable_adult_person_waves():
    pw = hc.join_demographics(_roster(), _demo(), _sex())
    assert len(pw) == 7
    assert set(pw.columns) >= {
        "person_id",
        "year",
        "age",
        "band",
        "sex",
        "weight",
        "hh_size",
        "multigen",
    }
    assert (pw["age"] >= hc.START_AGE).all()
    assert pw.loc[pw.person_id == 1003, "band"].iloc[0] == "15-24"


def test_join_drops_child_zero_weight_and_uncoded_sex():
    # child under 15, zero weight, age sentinel 999, na sex.
    extra_rows = pd.DataFrame(
        [
            _row(2001, 8, 4001, 10, 10, 4001, 10),  # age 10 -> dropped
            _row(2001, 8, 4002, 10, 10, 4002, 10),  # weight 0 -> dropped
            _row(2001, 8, 4003, 10, 10, 4003, 10),  # age 999 -> dropped
            _row(2001, 8, 4004, 10, 10, 4004, 10),  # na sex -> dropped
        ]
    )
    roster = hc.household_roster(
        pd.DataFrame(_household_a() + _household_b() + _household_c()).pipe(
            lambda df: pd.concat([df, extra_rows], ignore_index=True)
        )
    )
    demo2 = pd.concat(
        [
            _demo(),
            pd.DataFrame(
                [
                    (4001, 2001, 10, 100.0),
                    (4002, 2001, 30, 0.0),
                    (4003, 2001, 999, 100.0),
                    (4004, 2001, 30, 100.0),
                ],
                columns=["person_id", "period", "age", "weight"],
            ),
        ]
    )
    sex2 = pd.concat(
        [
            _sex(),
            pd.DataFrame(
                [
                    (4001, "male"),
                    (4002, "male"),
                    (4003, "female"),
                    (4004, "na"),
                ],
                columns=["person_id", "sex"],
            ),
        ]
    )
    pw = hc.join_demographics(roster, demo2, sex2)
    assert set(pw["person_id"]) == {
        1001,
        1002,
        1003,
        2001,
        2002,
        2003,
        3001,
    }


# --------------------------------------------------------------------------
# Reference-moment cell schema + aggregation map
# --------------------------------------------------------------------------
def test_reference_moments_cell_schema_and_rate():
    pw = hc.join_demographics(_roster(), _demo(), _sex())
    panel = hc.HouseholdCompositionPanel(
        person_waves=pw, attrs=pw[["person_id"]].drop_duplicates()
    )
    cells = hc.reference_moments(panel, weighted=True)
    # 5 band families x 7 bands x 2 sexes + 5 hh_size + 6 aggregates.
    assert len(cells) == 5 * 7 * 2 + 5 + 6
    for cell in cells.values():
        assert set(cell) == {"rate", "num_wt", "den_wt", "n_events"}
        if cell["den_wt"] > 0:
            assert cell["rate"] == pytest.approx(
                cell["num_wt"] / cell["den_wt"]
            )
    # hh_size shares over the fixture sum to 1 (every person is in one).
    total = (
        sum(cells[f"hh_size.{k}"]["rate"] for k in (1, 2, 3, 4))
        + cells["hh_size.5+"]["rate"]
    )
    assert total == pytest.approx(1.0)


def test_aggregation_members_map_is_coherent():
    members = hc.aggregation_members()
    assert set(members) == {
        "coresident_parent.45+|female",
        "coresident_parent.45+|male",
        "coresident_grandchild.55+|female",
        "coresident_grandchild.55+|male",
        "multigen.55+|female",
        "multigen.55+|male",
    }
    # coresident_parent.45+ pools the four 45+ per-band cells.
    assert members["coresident_parent.45+|female"] == [
        "coresident_parent.45-54|female",
        "coresident_parent.55-64|female",
        "coresident_parent.65-74|female",
        "coresident_parent.75+|female",
    ]
    # 55+ aggregates pool exactly three bands.
    assert len(members["multigen.55+|male"]) == 3


def test_person_ids_subset_restricts_moments():
    pw = hc.join_demographics(_roster(), _demo(), _sex())
    panel = hc.HouseholdCompositionPanel(
        person_waves=pw, attrs=pw[["person_id"]].drop_duplicates()
    )
    only_c = hc.reference_moments(panel, {3001}, weighted=True)
    # 3001 is a 30-year-old woman living alone: hh_size.1 share is 1.0,
    # she is in the 25-34|female denominator but has no coresident spouse,
    # and she is absent from every other band x sex denominator.
    assert only_c["hh_size.1"]["rate"] == pytest.approx(1.0)
    assert only_c["coresident_spouse.25-34|female"]["den_wt"] == 200.0
    assert only_c["coresident_spouse.25-34|female"]["rate"] == 0.0
    assert only_c["coresident_spouse.65-74|male"]["den_wt"] == 0.0


# --------------------------------------------------------------------------
# Integration: the real staged panel (skips without PSID)
# --------------------------------------------------------------------------
@needs_relmap
@needs_ind
def test_real_panel_builds_and_moment_grid_is_complete():
    panel = hc.build_household_panel()
    assert len(panel.person_waves) > 100_000
    assert panel.attrs["person_id"].is_unique
    cells = hc.reference_moments(panel, weighted=True)
    assert len(cells) == 5 * 7 * 2 + 5 + 6
    # coresident_parent peaks young, coresident_child peaks middle-age:
    # the direction convention holds on real data.
    assert (
        cells["coresident_parent.15-24|female"]["rate"]
        > cells["coresident_parent.55-64|female"]["rate"]
    )
    assert (
        cells["coresident_child.35-44|female"]["rate"]
        > cells["coresident_child.15-24|female"]["rate"]
    )
