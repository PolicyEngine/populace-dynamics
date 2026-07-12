"""Unit math for the gate-2c candidate-2 first-marriage earnings modifier.

No external data, artifact, or oracle: the modifier's fit (residual real /
certified), its marginal-preservation normalization (``sum_t m * phi_cert =
1``), its shrinkage, and its application (only first-marriage event weights
scale; the pooled band marginal is preserved) are exercised on hand-built
frames and a constant mock certified hazard.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.models import couple_formation_sim_v2 as v2
from populace_dynamics.models.couple_formation_sim_v2 import (
    FirstMarriageEarningsModifier,
    apply_first_marriage_modifier,
    fit_first_marriage_modifier,
)


# --------------------------------------------------------------------------
# FirstMarriageEarningsModifier: constraint, lookup, record
# --------------------------------------------------------------------------
def _neutral_modifier() -> FirstMarriageEarningsModifier:
    ones = np.ones((2, 4, 3))
    phi = np.full((2, 4, 3), 1.0 / 3.0)
    return FirstMarriageEarningsModifier(
        m_norm=ones.copy(),
        m_raw=ones.copy(),
        m_shrunk=ones.copy(),
        phi_cert=phi,
        z_norm=np.ones((2, 4)),
        n_events=np.full((2, 4, 3), 100, dtype=int),
        alpha=8.0,
        meta={},
    )


def test_constraint_neutral_modifier_holds():
    mod = _neutral_modifier()
    con = mod.constraint_per_band()
    assert np.allclose(con, 1.0)
    assert mod.constraint_max_abs_dev() <= 1e-12


def test_constraint_detects_violation():
    mod = _neutral_modifier()
    bad = mod.m_norm.copy()
    bad[0, 0, 0] = 2.0  # sum_t m*phi now 4/3 for female 18-24
    mod2 = FirstMarriageEarningsModifier(
        m_norm=bad,
        m_raw=mod.m_raw,
        m_shrunk=mod.m_shrunk,
        phi_cert=mod.phi_cert,
        z_norm=mod.z_norm,
        n_events=mod.n_events,
        alpha=mod.alpha,
        meta={},
    )
    assert mod2.constraint_max_abs_dev() > 0.3


def test_lookup_vectorized_and_out_of_band_is_neutral():
    mod = _neutral_modifier()
    m = mod.m_norm.copy()
    m[0, 0, 0] = 1.5  # female, 18-24, t1
    m[1, 1, 2] = 0.7  # male, 25-34, t3
    mod = FirstMarriageEarningsModifier(
        m_norm=m,
        m_raw=mod.m_raw,
        m_shrunk=mod.m_shrunk,
        phi_cert=mod.phi_cert,
        z_norm=mod.z_norm,
        n_events=mod.n_events,
        alpha=mod.alpha,
        meta={},
    )
    got = mod.lookup(
        np.array(["female", "male", "female"]),
        np.array(["18-24", "25-34", "10-17"]),  # last band is out of range
        np.array([1, 3, 2]),
    )
    assert got[0] == pytest.approx(1.5)
    assert got[1] == pytest.approx(0.7)
    assert got[2] == pytest.approx(1.0)  # out-of-band -> neutral


def test_fit_vs_raw_record_structure():
    rec = _neutral_modifier().fit_vs_raw_record()
    assert rec["alpha"] == 8.0
    assert len(rec["cells"]) == 24
    assert "t1.18-24|female" in rec["cells"]
    assert rec["constraint_max_abs_dev_from_one"] <= 1e-12


# --------------------------------------------------------------------------
# apply_first_marriage_modifier: only first-marriage weights scale
# --------------------------------------------------------------------------
def _events_and_exposure():
    events = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [20, 22, 30, 40],
            "sex": ["female", "female", "male", "male"],
            "weight": [10.0, 20.0, 30.0, 40.0],
            "transition": [
                "first_marriage",
                "first_marriage",
                "remarriage",
                "first_marriage",
            ],
            "tercile": [1, 3, 2, 2],
            "fm_band": ["18-24", "18-24", "25-34", "35-44"],
        }
    )
    exposure = pd.DataFrame(
        {
            "person_id": [1, 2, 5, 6],
            "age": [20, 22, 30, 40],
            "sex": ["female", "female", "male", "male"],
            "weight": [100.0, 100.0, 100.0, 100.0],
            "marital_state": [
                "never_married",
                "never_married",
                "divorced",
                "never_married",
            ],
            "tercile": [1, 3, 2, 2],
            "fm_band": ["18-24", "18-24", "25-34", "35-44"],
        }
    )
    return events, exposure


def test_apply_scales_only_first_marriage_weights():
    events, exposure = _events_and_exposure()
    mod = _neutral_modifier()
    m = mod.m_norm.copy()
    m[0, 0, 0] = 1.5  # female 18-24 t1  -> person 1
    m[0, 0, 2] = 0.5  # female 18-24 t3  -> person 2
    # keep the female 18-24 band marginal preserved: phi = event shares.
    ev_w = np.array([10.0, 20.0])  # t1, t3 (t2 has no event -> 0)
    phi = mod.phi_cert.copy()
    phi[0, 0, :] = [ev_w[0] / 30.0, 0.0, ev_w[1] / 30.0]
    z = (m[0, 0] * phi[0, 0]).sum()
    m[0, 0] = m[0, 0] / z  # renormalize so sum_t m*phi = 1
    mod = FirstMarriageEarningsModifier(
        m_norm=m,
        m_raw=mod.m_raw,
        m_shrunk=mod.m_shrunk,
        phi_cert=phi,
        z_norm=mod.z_norm,
        n_events=mod.n_events,
        alpha=mod.alpha,
        meta={},
    )
    new_events, check = apply_first_marriage_modifier(events, mod, exposure)
    # remarriage row (person 3) weight untouched.
    r = new_events.set_index("person_id")
    assert r.loc[3, "weight"] == pytest.approx(30.0)
    # first-marriage weights scaled by m_norm(cell).
    assert r.loc[1, "weight"] == pytest.approx(10.0 * m[0, 0, 0])
    assert r.loc[2, "weight"] == pytest.approx(20.0 * m[0, 0, 2])
    # other columns byte-identical (only weight changed).
    for col in ("age", "sex", "transition", "tercile", "fm_band"):
        assert list(new_events[col]) == list(events[col])
    assert check["n_first_marriage_events_reweighted"] == 3


def test_apply_preserves_female_18_24_pooled_marginal():
    events, exposure = _events_and_exposure()
    mod = _neutral_modifier()
    m = mod.m_norm.copy()
    phi = mod.phi_cert.copy()
    # event-share normalization on female 18-24 -> exact pooled preservation.
    phi[0, 0, :] = [10.0 / 30.0, 0.0, 20.0 / 30.0]
    raw = np.array([1.5, 1.0, 0.5])
    m[0, 0] = raw / (raw * phi[0, 0]).sum()
    mod = FirstMarriageEarningsModifier(
        m_norm=m,
        m_raw=mod.m_raw,
        m_shrunk=mod.m_shrunk,
        phi_cert=phi,
        z_norm=mod.z_norm,
        n_events=mod.n_events,
        alpha=mod.alpha,
        meta={},
    )
    _, check = apply_first_marriage_modifier(events, mod, exposure)
    # female|18-24 pooled hazard unchanged when phi = the draw's event share.
    assert check["realized_pooled_band_hazard_abs_ln"][
        "18-24|female"
    ] == pytest.approx(0.0, abs=1e-12)
    assert check["constraint_holds"] is True


def test_apply_raises_on_constraint_violation():
    events, exposure = _events_and_exposure()
    mod = _neutral_modifier()
    bad = mod.m_norm.copy()
    bad[0, 0, 0] = 5.0  # sum_t m*phi far from 1 on female 18-24
    mod = FirstMarriageEarningsModifier(
        m_norm=bad,
        m_raw=mod.m_raw,
        m_shrunk=mod.m_shrunk,
        phi_cert=mod.phi_cert,
        z_norm=mod.z_norm,
        n_events=mod.n_events,
        alpha=mod.alpha,
        meta={},
    )
    with pytest.raises(RuntimeError, match="marginal-preservation"):
        apply_first_marriage_modifier(events, mod, exposure)


# --------------------------------------------------------------------------
# fit_first_marriage_modifier: residual math with a constant mock hazard
# --------------------------------------------------------------------------
def _mock_panel_constant_gradient():
    """One band (18-24), female: t1/t2/t3 with a known hazard gradient.

    Each tercile has 100 never-married person-years (weight 1); events give
    hazards 0.15 / 0.10 / 0.05 so, against a constant certified hazard 0.10,
    the raw modifier is 1.5 / 1.0 / 0.5.
    """
    rows_py = []
    rows_ev = []
    pid = 1
    counts = {1: 15, 2: 10, 3: 5}
    for terc in (1, 2, 3):
        for _ in range(100):
            rows_py.append(
                {
                    "person_id": pid,
                    "year": 2010,
                    "age": 20,
                    "sex": "female",
                    "weight": 1.0,
                    "marital_state": "never_married",
                }
            )
            pid += 1
        # events (a subset of the same-tercile persons)
        for j in range(counts[terc]):
            rows_ev.append(
                {
                    "person_id": 10000 + terc * 100 + j,
                    "age": 20,
                    "sex": "female",
                    "weight": 1.0,
                    "transition": "first_marriage",
                }
            )
    py = pd.DataFrame(rows_py)
    ev = pd.DataFrame(rows_ev)
    mpanel = SimpleNamespace(person_years=py, events=ev)
    # tercile by person_id: first 100 -> t1, next 100 -> t2, last 100 -> t3;
    # event ids encode the tercile too.
    earn = {}
    p = 1
    for terc in (1, 2, 3):
        for _ in range(100):
            earn[p] = {1: 0.0, 2: 1.0, 3: 2.0}[terc]
            p += 1
    for terc in (1, 2, 3):
        for j in range(counts[terc]):
            earn[10000 + terc * 100 + j] = {1: 0.0, 2: 1.0, 3: 2.0}[terc]
    # cuts so value 0 -> t1, 1 -> t2, 2 -> t3.
    axis = SimpleNamespace(earn=earn, cuts=(0.5, 1.5))
    components = SimpleNamespace(
        first_marriage=SimpleNamespace(
            predict=lambda age, is_male, decade: np.full(len(age), 0.10)
        )
    )
    return components, mpanel, axis, set(earn)


def test_fit_raw_modifier_is_real_over_certified():
    components, mpanel, axis, train_ids = _mock_panel_constant_gradient()
    mod = fit_first_marriage_modifier(
        components, mpanel, axis, train_ids, alpha=0.0
    )
    fi = 0  # female
    bi = 0  # 18-24
    # certified hazard 0.10 constant; real 0.15/0.10/0.05 -> raw 1.5/1.0/0.5.
    assert mod.m_raw[fi, bi, 0] == pytest.approx(1.5, rel=1e-6)
    assert mod.m_raw[fi, bi, 1] == pytest.approx(1.0, rel=1e-6)
    assert mod.m_raw[fi, bi, 2] == pytest.approx(0.5, rel=1e-6)
    # equal exposure -> phi_cert = 1/3 each; Z = (1.5+1+0.5)/3 = 1 -> m_norm
    # equals m_raw and the constraint holds exactly.
    assert np.allclose(mod.phi_cert[fi, bi], 1.0 / 3.0)
    assert mod.constraint_max_abs_dev() <= 1e-12
    assert np.allclose(mod.m_norm[fi, bi], [1.5, 1.0, 0.5], atol=1e-6)


def test_fit_shrinkage_pulls_thin_cells_toward_neutral():
    components, mpanel, axis, train_ids = _mock_panel_constant_gradient()
    m0 = fit_first_marriage_modifier(
        components, mpanel, axis, train_ids, alpha=0.0
    )
    m50 = fit_first_marriage_modifier(
        components, mpanel, axis, train_ids, alpha=50.0
    )
    fi, bi = 0, 0
    # heavy shrinkage moves each tercile modifier toward 1 (pre-norm).
    for ti in (0, 2):
        assert abs(m50.m_shrunk[fi, bi, ti] - 1.0) < abs(
            m0.m_shrunk[fi, bi, ti] - 1.0
        )
    # the normalization constraint still holds exactly after shrinkage.
    assert m50.constraint_max_abs_dev() <= 1e-12


def test_fit_matched_cell_gets_neutral_modifier():
    components, mpanel, axis, train_ids = _mock_panel_constant_gradient()
    mod = fit_first_marriage_modifier(
        components, mpanel, axis, train_ids, alpha=0.0
    )
    # tercile 2's real hazard equals the certified 0.10 -> m_raw == 1 (no
    # double-count of a gradient the core already carries).
    assert mod.m_raw[0, 0, 1] == pytest.approx(1.0, rel=1e-6)


def test_module_pins():
    assert v2.MODIFIER_SHRINKAGE_ALPHA == 8.0
    assert v2.GATED_MARGINAL_BANDS == ("18-24", "25-34")
    assert v2.N_DECILES == 10
    assert len(v2.CERTIFIED_SPEC.sha256) == 64
