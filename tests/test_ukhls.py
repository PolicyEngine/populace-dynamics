"""Tests for the UKHLS reader and transition estimator.

The committed tables in ``data/external/`` are aggregated,
cell-suppressed derivatives of UKHLS Waves 1-15 (salvaged from
archived policyengine-uk-data#346); tests against them always run.
Estimator tests use hermetic synthetic panels. Raw-microdata tests
skip when no UKHLS staging directory is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import ukhls

STATES = {"IN_WORK", "UNEMPLOYED", "RETIRED", "INACTIVE"}


# ---------------------------------------------------------------
# Committed aggregate tables (always runnable)
# ---------------------------------------------------------------


def test_employment_table_schema_and_suppression():
    df = pd.read_csv(ukhls.EMPLOYMENT_TABLE)
    assert list(df.columns) == [
        "age_band",
        "sex",
        "state_from",
        "state_to",
        "count",
        "probability",
    ]
    assert (df["count"] >= ukhls.MIN_CELL_COUNT).all()
    assert set(df["sex"]) == {"MALE", "FEMALE"}
    assert set(df["state_from"]) <= STATES
    assert set(df["state_to"]) <= STATES


def test_employment_table_normalised():
    df = pd.read_csv(ukhls.EMPLOYMENT_TABLE)
    sums = df.groupby(["age_band", "sex", "state_from"])["probability"].sum()
    assert np.allclose(sums, 1.0)


def test_decile_table_schema_and_normalised():
    df = pd.read_csv(ukhls.DECILE_TABLE)
    assert (df["count"] >= ukhls.MIN_CELL_COUNT).all()
    assert set(df["decile_from"]) <= set(range(1, 11))
    assert set(df["decile_to"]) <= set(range(1, 11))
    sums = df.groupby(["age_band", "sex", "decile_from"])["probability"].sum()
    assert np.allclose(sums, 1.0)


def test_published_spot_checks():
    """Rates match the external published statistics cited in the
    provenance note (ONS LFS flow rates; IFS decile persistence)."""
    emp = ukhls.load_employment_transitions()
    stay = emp[("25-29", "MALE", "IN_WORK")]["IN_WORK"]
    assert 0.93 <= stay <= 0.98
    exit_u = emp[("25-29", "MALE", "UNEMPLOYED")]["IN_WORK"]
    assert 0.30 <= exit_u <= 0.45

    dec = pd.read_csv(ukhls.DECILE_TABLE)
    diag = dec[dec["decile_from"] == dec["decile_to"]]
    avg_stay = (diag["count"].sum()) / dec["count"].sum()
    assert 0.30 <= avg_stay <= 0.50


def test_loaders_return_nested_dicts():
    emp = ukhls.load_employment_transitions()
    assert all(len(k) == 3 for k in emp)
    dec = ukhls.load_income_decile_transitions()
    key = next(iter(dec))
    assert isinstance(key[2], int)
    assert all(isinstance(d, int) for d in dec[key])


# ---------------------------------------------------------------
# Estimator logic (hermetic synthetic panel)
# ---------------------------------------------------------------


def _synthetic_panel(
    n: int = 4000, p_stay: float = 0.8, seed: int = 0
) -> pd.DataFrame:
    """Two-wave panel of 30-year-old men alternating between two
    jbstat codes with a planted persistence probability."""
    rng = np.random.default_rng(seed)
    state_t = rng.integers(0, 2, size=n)  # 0=employed, 1=unemployed
    stay = rng.random(n) < p_stay
    state_t1 = np.where(stay, state_t, 1 - state_t)
    jb = {0: 2, 1: 3}  # EMPLOYED / UNEMPLOYED codes
    rows = []
    for i in range(n):
        for wave, st in ((1, state_t[i]), (2, state_t1[i])):
            rows.append(
                {
                    "pidp": i,
                    "wave": wave,
                    "age_dv": 30,
                    "sex": 1,
                    "jbstat": jb[int(st)],
                    "fimngrs_dv": 1000.0 + 100 * int(st),
                }
            )
    return pd.DataFrame(rows)


def test_estimator_recovers_planted_rates():
    df = _synthetic_panel(p_stay=0.8)
    out = ukhls.estimate_employment_transitions(df)
    row = out[
        (out["age_band"] == "30-34")
        & (out["state_from"] == "IN_WORK")
        & (out["state_to"] == "IN_WORK")
    ]
    assert len(row) == 1
    assert abs(row["probability"].iloc[0] - 0.8) < 0.03


def test_estimator_suppresses_small_cells():
    # 5 people < MIN_CELL_COUNT: everything suppressed, empty output.
    df = _synthetic_panel(n=5, p_stay=1.0)
    out = ukhls.estimate_employment_transitions(df)
    assert (out["count"] >= ukhls.MIN_CELL_COUNT).all()
    assert out.empty


def test_estimator_drops_missing_and_renormalises():
    df = _synthetic_panel(n=500, p_stay=0.7)
    df.loc[df.index[:20], "jbstat"] = -9  # missing codes
    out = ukhls.estimate_employment_transitions(df)
    sums = out.groupby(["age_band", "sex", "state_from"])["probability"].sum()
    assert np.allclose(sums, 1.0)


def test_four_state_label():
    assert ukhls.four_state_label(2) == "IN_WORK"
    assert ukhls.four_state_label(1) == "IN_WORK"
    assert ukhls.four_state_label(3) == "UNEMPLOYED"
    assert ukhls.four_state_label(4) == "RETIRED"
    assert ukhls.four_state_label(7) == "INACTIVE"
    assert ukhls.four_state_label(-8) == "MISSING"
    assert ukhls.four_state_label(None) == "MISSING"


def test_age_band_edges():
    assert ukhls._age_band(16) == "16-19"
    assert ukhls._age_band(74) == "70-74"
    assert ukhls._age_band(120) == "75-120"
    assert ukhls._age_band(15) is None
    assert ukhls._age_band(121) is None
    assert ukhls._age_band("x") is None


def test_decile_estimator_hermetic():
    df = _synthetic_panel(n=2000, p_stay=0.9)
    out = ukhls.estimate_income_decile_transitions(df)
    sums = out.groupby(["age_band", "sex", "decile_from"])["probability"].sum()
    assert np.allclose(sums, 1.0)


# ---------------------------------------------------------------
# Raw microdata (skips off-machine)
# ---------------------------------------------------------------


@pytest.mark.skipif(
    not ukhls.ukhls_dir().exists(),
    reason="UKHLS microdata not staged on this machine",
)
def test_load_wave_from_staged_data():
    df = ukhls.load_wave("a")
    assert "pidp" in df.columns
    assert "age_dv" in df.columns
    assert (df["wave"] == 1).all()
