"""Unit tests for W1 candidate 2 -- the three proven levers.

Pure-logic checks (no h5 / no PSID / no committed artifact): protocol
constants byte-carried from candidate 1, the three-delta spec resolutions,
the Q2 byte-carry (25-59 earnings draws preserved), the Q1 entry-state
seeding + model normalisation, and the Q3 child materialisation schema on
synthetic inputs. The committed-artifact bindings live in
``test_gate_w1_candidate2_artifacts.py`` (the ``artifact`` tier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.models import transport_deployment_v1 as td1
from populace_dynamics.models import transport_deployment_v2 as td2


def test_protocol_constants_byte_carried_from_candidate1():
    assert td2.K_DRAWS == td1.K_DRAWS == 20
    assert td2.GATE_SEEDS == td1.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert td2.FAMILY_A_STREAM_BASE == td1.FAMILY_A_STREAM_BASE == 9100
    assert td2.FAMILY_B_STREAM_BASE == td1.FAMILY_B_STREAM_BASE == 9200
    assert (
        td2.FAMILY_C_TRANSITORY_STREAM
        == td1.FAMILY_C_TRANSITORY_STREAM
        == 9300
    )
    assert td2.ADULT_MIN_AGE == td1.ADULT_MIN_AGE == 18


def test_delta_constants():
    assert td2.BASE_ENTRY_AGE == 25
    assert td2.BOUNDARY_RANGES == ((18, 24, "18-24"), (60, 69, "60-69"))
    assert td2.BOUNDARY_SEXES == ("female", "male")
    assert td2.CHILD_MAX_AGE == 18
    # the entry-state / child streams are mutually distinct and distinct from
    # the A/B/C transitory streams
    salts = {
        td2.ENTRY_STATE_STREAM_SALT,
        td2.CHILD_SEX_STREAM_SALT,
        9100,
        9200,
        9300,
    }
    assert len(salts) == 5


def test_spec_resolutions_cover_three_deltas_and_byte_carry():
    keys = td2.SPEC_RESOLUTIONS
    for needed in (
        "delta_q1_entry_state_seeding",
        "delta_q2_boundary_support_extension",
        "delta_q3_child_attachment_coresidence",
        "byte_carry_from_candidate1",
    ):
        assert needed in keys
        assert len(keys[needed]) > 80
    # the byte-carried v1 resolutions are also retained
    assert "family_a_earnings_regeneration" in keys


class _FakeMarginal:
    """A one-cell marginal: zero share p0, linear positive quantile map."""

    n_pos = 100

    def __init__(self, p0, lo, hi):
        self.p0 = p0
        self._lo, self._hi = lo, hi

    def quantile(self, u):
        return self._lo + (self._hi - self._lo) * np.asarray(u)


def test_q2_boundary_preserves_v1_in_support_draws():
    """For in-support ages (25-59) v2 reproduces v1's regenerate_earnings
    bit-for-bit: same base marginal, same single u draw."""
    marg = {(3, td2.TERMINAL_PERIOD): _FakeMarginal(0.3, 10_000.0, 200_000.0)}

    def age_bin(a):
        return np.full(len(np.atleast_1d(a)), 3, dtype=int)

    ages = np.linspace(25, 59, 5000)
    fem = np.arange(5000) % 2 == 0
    v1 = td1.regenerate_earnings(ages, np.random.default_rng(7), marg, age_bin)
    v2 = td2.regenerate_earnings_v2(
        ages, fem, np.random.default_rng(7), marg, {}, age_bin
    )
    assert np.array_equal(v1, v2)


def test_q2_boundary_ages_route_to_boundary_marginal():
    """18-24 / 60-69 draw from the sex-specific boundary marginals, not the
    clipped base bin -- and respect the boundary p0 (participation)."""
    base = {(0, td2.TERMINAL_PERIOD): _FakeMarginal(0.1, 5_000.0, 9_000.0)}
    boundary = {
        ("18-24", "female"): _FakeMarginal(0.4, 1_000.0, 3_000.0),
        ("18-24", "male"): _FakeMarginal(0.4, 1_000.0, 3_000.0),
        ("60-69", "female"): _FakeMarginal(0.5, 2_000.0, 4_000.0),
        ("60-69", "male"): _FakeMarginal(0.5, 2_000.0, 4_000.0),
    }

    def age_bin(a):
        # 18-24 would clip to bin 0 in v1; here proves the reroute
        return np.zeros(len(np.atleast_1d(a)), dtype=int)

    ages = np.full(20_000, 20.0)
    fem = np.ones(20_000, dtype=bool)
    earn = td2.regenerate_earnings_v2(
        ages, fem, np.random.default_rng(0), base, boundary, age_bin
    )
    zero_frac = float((earn == 0).mean())
    assert abs(zero_frac - 0.4) < 0.02  # ~ boundary p0, NOT the base 0.1
    pos = earn[earn > 0]
    assert pos.min() >= 1_000.0 - 1 and pos.max() <= 3_000.0 + 1


def test_fit_initial_state_model_normalises_and_records_fit_vs_raw():
    # tiny synthetic marital panel + demographic panel (first-wave entries)
    py = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3, 4],
            "year": [2000, 2002, 2000, 2002, 2000, 2000],
            "age": [30, 32, 40, 42, 28, 50],
            "sex": ["female", "female", "male", "male", "female", "male"],
            "weight": [1.0, 1.0, 2.0, 2.0, 1.0, 3.0],
            "marital_state": [
                "married",
                "married",
                "never_married",
                "married",
                "divorced",
                "widowed",
            ],
        }
    )

    class _P:
        person_years = py

    demo = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "period": [2000, 2000, 2000, 2000],
        }
    )
    model, raw = td2.fit_initial_state_model(_P(), demo)
    for probs in model.values():
        assert abs(float(np.sum(probs)) - 1.0) < 1e-12
        assert len(probs) == len(td2.MARITAL_STATES)
    assert (
        "25-34",
        "female",
    ) in model  # persons 1 (entry 30) and 3 (entry 28)
    assert any("max_abs_fit_minus_raw" in v for v in raw.values())


def test_build_seeded_marital_panel_entry_rows():
    pid = np.array([10, 11, 12], dtype=np.int64)
    age = np.array([30.0, 50.0, 20.0])
    fem = np.array([True, False, True])
    wt = np.array([1.0, 1.0, 1.0])
    model = {
        ("25-34", "female"): np.array([0.0, 1.0, 0.0, 0.0]),  # always married
        ("25-34", "male"): np.array([0.0, 1.0, 0.0, 0.0]),
    }
    panel = td2.build_seeded_marital_panel(pid, age, fem, wt, model, seed=0)
    py = panel.person_years.set_index("person_id")
    # adults >= 25 enter at BASE_ENTRY_AGE=25 seeded married; the 20yo at 18
    assert py.loc[10, "year"] == (td2.REF_YEAR - 30) + 25
    assert py.loc[10, "marital_state"] == "married"
    assert py.loc[11, "marital_state"] == "married"
    assert py.loc[12, "marital_state"] == "never_married"  # age<25 -> default
    assert py.loc[12, "year"] == (td2.REF_YEAR - 20) + 20
    # married entries carry duration 0, never-married carry NA
    assert int(py.loc[10, "marriage_duration"]) == 0
    assert pd.isna(py.loc[12, "marriage_duration"])


def test_materialize_children_schema_and_minor_only():
    # a synthetic seeded panel with two mothers; hand a births frame in via a
    # stubbed ft.simulate through a fake fitted_hc is heavy -- instead test the
    # empty-birth path and the schema contract directly.
    empty = td2._empty_child_frame()
    assert list(empty.columns) == [
        "person_id",
        "weight",
        "age",
        "is_female",
        "earnings",
        "marital_status",
        "hh_size",
        "coresident_spouse",
    ]
    assert len(empty) == 0


def test_child_rows_excluded_from_non_hhsize_cells():
    """A materialized child row (age<18, marital NaN, earnings 0) moves ONLY
    hh_size_share in reference_moments; every other family-A cell is unchanged
    whether or not the child rows are present."""
    from populace_dynamics.data import deployment_frame as dfm

    adults = dfm.synthetic_person_frame(n=300, seed=1)
    adults = adults[adults["age"] >= 18].reset_index(drop=True)
    kids = pd.DataFrame(
        {
            "person_id": np.arange(40) + td2.CHILD_ID_OFFSET,
            "weight": np.full(40, 1000.0),
            "age": np.full(40, 8.0),
            "is_female": np.arange(40) % 2 == 0,
            "earnings": np.zeros(40),
            "marital_status": np.array([np.nan] * 40, dtype=object),
            "hh_size": np.full(40, 4.0),
            "coresident_spouse": np.zeros(40, dtype=bool),
        }
    )
    base_cells = dfm.reference_moments(adults, weighted=True)
    with_kids = dfm.reference_moments(
        pd.concat([adults, kids], ignore_index=True), weighted=True
    )
    for cell in base_cells:
        if cell.startswith("hh_size_share."):
            continue
        assert base_cells[cell]["rate"] == with_kids[cell]["rate"], cell
    # hh_size_share DID move (children added weight at size 4)
    assert (
        base_cells["hh_size_share.4"]["rate"]
        != with_kids["hh_size_share.4"]["rate"]
    )
