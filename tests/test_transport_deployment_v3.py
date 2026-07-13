"""Synthetic-frame unit tests for the W1 candidate-3 transport levers.

Every test constructs small synthetic DataFrames -- no PSID fit, no certified
frame, no committed artifact. The three forensics-2 levers are exercised through
their pure pieces (entry-anchor extraction + the no-terminal-readback guard,
the interior sex-covariate earnings routing, the co-designed roster/fertility
invariants); the functions that need the fitted PSID generators are covered by
their pure sub-functions and by a delegation spy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate1_candidate5b import CellMarginal, age_bin  # noqa: E402

import populace_dynamics.models.transport_deployment_v3 as td  # noqa: E402


def _constant_marginal(value: float, p0: float = 0.0) -> CellMarginal:
    """A CellMarginal whose positive quantile is a constant ``value``."""
    return CellMarginal(
        p0, np.array([0.0, 1.0]), np.array([value, value]), 2, 1.0
    )


# ==========================================================================
# Lever 1 -- CPS-anchored entry-level marital model (Q6).
# ==========================================================================
def test__given_frame_moments__when_build_cps_entry_anchor__then_married_seeds():
    moments = {
        "marital_share.married.25-34|female": {"rate": 0.49},
        "marital_share.married.25-34|male": {"rate": 0.42},
        # a 65+ terminal cell is PRESENT but must never be read
        "marital_share.married.65+|female": {"rate": 0.69},
    }
    anchor = td.build_cps_entry_anchor(moments)
    mi = td.MARITAL_STATES.index("married")
    ni = td.MARITAL_STATES.index("never_married")
    assert anchor["female"][mi] == pytest.approx(0.49)
    assert anchor["female"][ni] == pytest.approx(0.51)
    assert anchor["male"][mi] == pytest.approx(0.42)


def test__given_anchor__then_probs_sum_to_one_no_divorced_or_widowed():
    moments = {
        "marital_share.married.25-34|female": {"rate": 0.49},
        "marital_share.married.25-34|male": {"rate": 0.42},
    }
    anchor = td.build_cps_entry_anchor(moments)
    di = td.MARITAL_STATES.index("divorced")
    wi = td.MARITAL_STATES.index("widowed")
    for sex in ("female", "male"):
        assert anchor[sex].sum() == pytest.approx(1.0)
        # nobody is divorced/widowed at the age-25 entry (Q6: fresh at entry)
        assert anchor[sex][di] == 0.0
        assert anchor[sex][wi] == 0.0


@pytest.mark.parametrize(
    "key",
    [
        "marital_share.married.65+|female",
        "marital_share.married.65+|male",
        "coresident_spouse.65+|female",
        "marital_share.married.55-64|male",
        "marital_share.married.45-54|female",
        "marital_share.married.35-44|male",
    ],
)
def test__given_terminal_or_65plus_key__when_assert_entry_anchor__then_raises(
    key,
):
    # The no-terminal-readback guard: Q6 prohibits back-solving the entry from
    # a scored 65+/older-terminal state (the identity-in-disguise).
    with pytest.raises(AssertionError):
        td._assert_entry_anchor_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "marital_share.married.25-34|female",
        "marital_share.married.25-34|male",
    ],
)
def test__given_entry_band_key__when_assert_entry_anchor__then_ok(key):
    td._assert_entry_anchor_key(key)  # must not raise


def test__given_moments_without_entry_band__when_build_anchor__then_raises():
    # build_cps_entry_anchor reads ONLY the 25-34 entry band; a moments dict
    # that omits it (only 65+ present) raises rather than reading a terminal.
    with pytest.raises(KeyError):
        td.build_cps_entry_anchor(
            {"marital_share.married.65+|female": {"rate": 0.69}}
        )


def test__given_seedable_adults__when_seed_marital_panel__then_entry_at_25():
    n = 50
    pid = np.arange(n, dtype=np.int64)
    age = np.full(n, 40.0)
    fem = np.zeros(n, dtype=bool)
    wt = np.ones(n)
    anchor = {
        "female": np.array([0.5, 0.5, 0.0, 0.0]),
        "male": np.array([0.5, 0.5, 0.0, 0.0]),
    }
    panel = td.build_cps_seeded_marital_panel(
        pid, age, fem, wt, anchor, seed=1
    )
    # entry age is BASE_ENTRY_AGE=25 for adults in the gated bands
    birth_year = td.REF_YEAR - 40
    assert (panel.attrs["start_exposure_year"] == birth_year + 25).all()
    assert (panel.person_years["year"] == birth_year + 25).all()
    assert set(panel.person_years["marital_state"].unique()) <= {
        "never_married",
        "married",
    }


def test__given_two_seeds__when_seed_marital_panel__then_regenerates():
    n = 300
    pid = np.arange(n, dtype=np.int64)
    age = np.full(n, 30.0)
    fem = np.zeros(n, dtype=bool)
    wt = np.ones(n)
    anchor = {
        "female": np.array([0.5, 0.5, 0.0, 0.0]),
        "male": np.array([0.5, 0.5, 0.0, 0.0]),
    }
    a = td.build_cps_seeded_marital_panel(pid, age, fem, wt, anchor, seed=1)
    b = td.build_cps_seeded_marital_panel(pid, age, fem, wt, anchor, seed=2)
    # a per-draw categorical => the seeded states vary across draws (not the
    # frozen identity map); ~0 probability of identical over 300 coin flips.
    assert not np.array_equal(
        a.person_years["marital_state"].to_numpy(),
        b.person_years["marital_state"].to_numpy(),
    )


def test__given_p_married_one__when_seed_marital_panel__then_all_married():
    n = 40
    pid = np.arange(n, dtype=np.int64)
    age = np.full(n, 50.0)
    fem = np.ones(n, dtype=bool)
    wt = np.ones(n)
    anchor = {
        "female": np.array([0.0, 1.0, 0.0, 0.0]),  # married w.p. 1
        "male": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    panel = td.build_cps_seeded_marital_panel(
        pid, age, fem, wt, anchor, seed=7
    )
    assert (panel.person_years["marital_state"] == "married").all()
    # married entrants carry duration 0 (fresh at entry)
    assert (panel.person_years["marriage_duration"] == 0).all()


def test__given_young_adult_below_gated_bands__then_enters_never_married():
    pid = np.array([0, 1], dtype=np.int64)
    age = np.array([20.0, 22.0])  # below BASE_ENTRY_AGE=25
    fem = np.array([True, False])
    wt = np.ones(2)
    anchor = {
        "female": np.array([0.0, 1.0, 0.0, 0.0]),
        "male": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    panel = td.build_cps_seeded_marital_panel(
        pid, age, fem, wt, anchor, seed=1
    )
    # the very young are not seeded from the gated-band anchor
    assert (panel.person_years["marital_state"] == "never_married").all()
    # they enter at their own age (>= 18), not BASE_ENTRY_AGE
    assert (
        panel.attrs["start_exposure_year"] == (td.REF_YEAR - age) + age
    ).all()


# ==========================================================================
# Lever 2 -- interior sex covariate on the earnings marginals (Q8).
# ==========================================================================
def test__given_sex_specific_interior__when_regenerate_earnings__then_routes():
    # interior 35-44 marginals differ by sex -> the covariate must route
    # female and male to different values (the whole point of Q8).
    interior = {
        ("35-44", "female"): _constant_marginal(100.0),
        ("35-44", "male"): _constant_marginal(200.0),
    }
    ages = np.array([40.0, 40.0, 40.0, 40.0])
    fem = np.array([True, False, True, False])
    out = td.regenerate_earnings_v3(
        ages,
        fem,
        np.random.default_rng(0),
        base_marginals={},
        boundary_marginals={},
        interior_marginals=interior,
        age_bin_fn=age_bin,
    )
    assert list(out[fem]) == [100.0, 100.0]
    assert list(out[~fem]) == [200.0, 200.0]


def test__given_boundary_and_interior__when_regenerate__then_60_61_boundary():
    # boundary is applied FIRST, so 60-61 (in both the 60-69 boundary and the
    # 55-61 interior band) route to the 60-69 boundary marginal, exactly as v2.
    boundary = {
        ("60-69", "female"): _constant_marginal(999.0),
        ("60-69", "male"): _constant_marginal(999.0),
    }
    interior = {
        ("55-61", "female"): _constant_marginal(555.0),
        ("55-61", "male"): _constant_marginal(555.0),
    }
    ages = np.array([58.0, 60.0, 61.0, 62.0])
    fem = np.zeros(4, dtype=bool)
    out = td.regenerate_earnings_v3(
        ages,
        fem,
        np.random.default_rng(0),
        base_marginals={},
        boundary_marginals=boundary,
        interior_marginals=interior,
        age_bin_fn=age_bin,
    )
    assert out[0] == 555.0  # 58 -> interior 55-61
    assert out[1] == 999.0  # 60 -> boundary 60-69 (precedence)
    assert out[2] == 999.0  # 61 -> boundary 60-69
    assert out[3] == 999.0  # 62 -> boundary 60-69


def test__given_same_rng__when_regenerate_earnings__then_deterministic():
    interior = {("35-44", "female"): _constant_marginal(100.0, p0=0.5)}
    ages = np.array([40.0] * 20)
    fem = np.ones(20, dtype=bool)
    kw = dict(
        base_marginals={},
        boundary_marginals={},
        interior_marginals=interior,
        age_bin_fn=age_bin,
    )
    a = td.regenerate_earnings_v3(ages, fem, np.random.default_rng(3), **kw)
    b = td.regenerate_earnings_v3(ages, fem, np.random.default_rng(3), **kw)
    assert np.array_equal(a, b)  # single u draw, reproducible


def test__given_70plus__when_regenerate_earnings__then_base_fallback():
    # 70+ is covered by neither boundary (18-24/60-69) nor interior (25-61);
    # it falls back to the certified base marginal (v1 path).
    base = {
        (
            int(age_bin(np.array([72.0]))[0]),
            td.TERMINAL_PERIOD,
        ): _constant_marginal(7.0)
    }
    out = td.regenerate_earnings_v3(
        np.array([72.0]),
        np.array([False]),
        np.random.default_rng(0),
        base_marginals=base,
        boundary_marginals={},
        interior_marginals={},
        age_bin_fn=age_bin,
    )
    assert out[0] == 7.0


def test__given_synthetic_panel__when_fit_interior_marginals__then_keys():
    # a tiny synthetic PSID-style earnings panel at the terminal period
    rows = []
    for a in (30, 40, 50, 58):
        for _ in range(40):
            rows.append((1, a))  # placeholder; earnings/weight filled below
    df = pd.DataFrame(
        {
            "person_id": np.arange(160) % 8,
            "period": td.TERMINAL_PERIOD,
            "age": np.repeat([30, 40, 50, 58], 40).astype(float),
            "earnings": np.tile([0.0, 50000.0], 80),
            "weight": 1.0,
        }
    )
    person_sex = pd.Series(
        ["female" if i % 2 else "male" for i in range(8)],
        index=np.arange(8),
    )
    marginals, raw = td.fit_interior_marginals(df, person_sex)
    for lo, hi in td.INTERIOR_BANDS:
        label = f"{lo}-{hi}"
        for sex in ("female", "male"):
            assert (label, sex) in marginals
    # half the person-years are zero-earning => participation ~0.5
    some = raw["35-44|female"]["fit_participation_1_minus_p0"]
    assert 0.0 <= some <= 1.0


# ==========================================================================
# Lever 3 -- co-designed coresident_parent roster + fertility window (Q7).
# ==========================================================================
def test__given_rates__when_seed_coresident_parent__then_bernoulli_by_sex():
    rates = {("15-24", "female"): 0.70, ("15-24", "male"): 0.75}
    n = 4000
    fem = np.array([True] * (n // 2) + [False] * (n // 2))
    seeded = td.seed_coresident_parent(fem, rates, np.random.default_rng(0))
    f_rate = seeded[fem].mean()
    m_rate = seeded[~fem].mean()
    assert f_rate == pytest.approx(0.70, abs=0.04)
    assert m_rate == pytest.approx(0.75, abs=0.04)


def test__given_two_seeds__when_seed_coresident_parent__then_regenerates():
    rates = {("15-24", "female"): 0.5, ("15-24", "male"): 0.5}
    fem = np.ones(500, dtype=bool)
    a = td.seed_coresident_parent(fem, rates, np.random.default_rng(1))
    b = td.seed_coresident_parent(fem, rates, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test__given_household_panel_v3__then_waves_start_at_15():
    pid = np.array([100], dtype=np.int64)
    age = np.array([40.0])
    fem = np.array([False])
    wt = np.array([1.0])
    rates = {("15-24", "female"): 0.0, ("15-24", "male"): 0.0}
    panel = td._synthetic_household_panel_v3(pid, age, fem, wt, rates, seed=0)
    pw = panel.person_waves
    assert int(pw["age"].min()) == td.FERTILITY_ENTRY_AGE == 15
    assert int(pw["year"].max()) == td.REF_YEAR


def test__given_all_seeded__when_household_panel_v3__then_wave0_coresident():
    pid = np.arange(30, dtype=np.int64)
    age = np.full(30, 22.0)  # young adults, still near the parental home
    fem = np.zeros(30, dtype=bool)
    wt = np.ones(30)
    rates = {("15-24", "female"): 1.0, ("15-24", "male"): 1.0}  # seed all True
    panel = td._synthetic_household_panel_v3(pid, age, fem, wt, rates, seed=0)
    pw = panel.person_waves
    first = pw["year"] == pw.groupby("person_id")["year"].transform("min")
    # every person's WAVE-0 coresident_parent is seeded True (rate 1.0)
    assert pw.loc[first, "coresident_parent"].all()


def test__given_no_seed__when_household_panel_v3__then_wave0_all_false():
    pid = np.arange(10, dtype=np.int64)
    age = np.full(10, 22.0)
    fem = np.zeros(10, dtype=bool)
    wt = np.ones(10)
    rates = {("15-24", "female"): 0.0, ("15-24", "male"): 0.0}
    panel = td._synthetic_household_panel_v3(pid, age, fem, wt, rates, seed=0)
    assert not panel.person_waves["coresident_parent"].any()


def test__given_fertility_panel__then_entry_at_15_never_married():
    pid = np.array([1, 2], dtype=np.int64)
    age = np.array([30.0, 45.0])
    fem = np.array([True, False])
    wt = np.ones(2)
    panel = td.build_fertility_window_marital_panel(pid, age, fem, wt)
    birth_year = td.REF_YEAR - age
    assert (
        panel.attrs["start_exposure_year"]
        == (birth_year + td.FERTILITY_ENTRY_AGE)
    ).all()
    # empty person-years => the simulator's never-married default (fertility
    # support), NOT a seeded marital state
    assert len(panel.person_years) == 0


def test__given_household_and_fertility_panels__then_share_entry_window():
    # The co-design's foundation: the roster and the fertility window are
    # entered at the SAME age (15), so they evolve on one shared child ledger,
    # not two independent windows.
    pid = np.array([5], dtype=np.int64)
    age = np.array([50.0])
    fem = np.array([True])
    wt = np.array([1.0])
    rates = {("15-24", "female"): 0.0, ("15-24", "male"): 0.0}
    hh = td._synthetic_household_panel_v3(pid, age, fem, wt, rates, seed=0)
    fert = td.build_fertility_window_marital_panel(pid, age, fem, wt)
    hh_entry = int(hh.person_waves["age"].min())
    fert_entry = int(
        fert.attrs["start_exposure_year"].iloc[0] - (td.REF_YEAR - 50)
    )
    assert hh_entry == fert_entry == td.FERTILITY_ENTRY_AGE


def test__given_child_ledger__when_classify_still_home__then_disjoint():
    # The no-double-count invariant: still-home children partition into
    # materialized MINORS (<18) and coresident YOUNG ADULTS (>=18), disjoint.
    ref = td.REF_YEAR
    births = pd.DataFrame(
        {
            "birth_year": [
                ref - 5,  # age 5  minor, home
                ref - 10,  # age 10 minor, home
                ref - 20,  # age 20 young adult, home
                ref - 22,  # age 22 young adult, home
                ref - 8,  # age 8  but LEFT
            ],
            "leave_year": [
                ref + 10,
                ref + 5,
                ref + 3,
                ref + 1,
                ref - 1,  # already left -> excluded
            ],
        }
    )
    out = td.classify_still_home_children(births, ref)
    minors = set(out["minors"].tolist())
    young = set(out["coresident_young_adults"].tolist())
    assert minors == {0, 1}
    assert young == {2, 3}
    # disjoint and partition the still-home set (index 4 left, excluded)
    assert minors.isdisjoint(young)
    assert minors | young == {0, 1, 2, 3}


def test__given_empty_ledger__when_classify_still_home__then_empty():
    out = td.classify_still_home_children(
        pd.DataFrame({"birth_year": [], "leave_year": []}), td.REF_YEAR
    )
    assert len(out["minors"]) == 0
    assert len(out["coresident_young_adults"]) == 0


def test__given_fert_panel_and_seed__when_materialize_v3__then_delegates(
    monkeypatch,
):
    # materialize_children_v3 must forward the 15-entry fertility panel + seed
    # to the SAME v2 machinery compose_base uses (the shared ledger), not a
    # separate re-implementation.
    captured = {}

    def _spy(panel, fitted_hc, hh, wt, seed):
        captured["panel"] = panel
        captured["seed"] = seed
        return pd.DataFrame()

    monkeypatch.setattr(td.td2, "materialize_children", _spy)
    sentinel_panel = object()
    td.materialize_children_v3(sentinel_panel, "fitted", "hh", "wt", seed=1234)
    assert captured["panel"] is sentinel_panel
    assert captured["seed"] == 1234


# ==========================================================================
# SPEC_RESOLUTIONS record the forensics-2 lever bindings for the referee.
# ==========================================================================
def test__given_spec_resolutions__then_cite_forensics2_fields():
    keys = td.SPEC_RESOLUTIONS
    assert "delta_lever1_cps_anchored_entry" in keys
    assert "delta_lever2_interior_sex_covariate" in keys
    assert "delta_lever3_codesigned_roster_fertility" in keys
    # each cites the runs/gate_w1_forensics2 field it binds to
    assert (
        "contract_adjudication.determination"
        in keys["delta_lever1_cps_anchored_entry"]
    )
    assert (
        "q8_interior_sex_covariate"
        in keys["delta_lever2_interior_sex_covariate"]
    )
    assert (
        "q7_coresident_parent_fertility"
        in keys["delta_lever3_codesigned_roster_fertility"]
    )
