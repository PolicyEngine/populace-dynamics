"""Unit tests for W1 candidate 1 -- the first full transport deployment.

Pure-logic checks (no h5 / no PSID / no committed artifact): protocol
constants, the spec-resolution schema, and the earnings / DI / fingerprint
helpers on synthetic inputs. The committed-artifact bindings live in
``test_gate_w1_candidate1_artifacts.py`` (the ``artifact`` tier).
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from populace_dynamics.models import transport_deployment_v1 as td


def test_protocol_constants_match_registration():
    assert td.K_DRAWS == 20
    assert td.GATE_SEEDS == (0, 1, 2, 3, 4)
    assert td.FAMILY_A_STREAM_BASE == 9100
    assert td.FAMILY_B_STREAM_BASE == 9200
    # A / B / C transitory streams are mutually distinct
    assert len({9100, 9200, td.FAMILY_C_TRANSITORY_STREAM}) == 3


def test_spec_resolutions_cover_every_family():
    keys = td.SPEC_RESOLUTIONS
    for needed in (
        "frozen_generator_fit",
        "family_a_earnings_regeneration",
        "family_a_marital_regeneration",
        "family_a_household_regeneration",
        "family_b_di_simulation",
        "family_b_di_prevalence",
        "family_b_conversion",
        "family_c_history_transport",
    ):
        assert needed in keys
        assert len(keys[needed]) > 40


def test_di_anchor_bands_partition_ages():
    labels = [b[0] for b in td.DI_ANCHOR_BANDS]
    assert labels[0] == "under30" and labels[-1] == "60-fra"
    # contiguous, non-overlapping over 0..66
    prev_hi = -1
    for _label, lo, hi in td.DI_ANCHOR_BANDS:
        assert lo == prev_hi + 1
        prev_hi = hi


class _FakeMarginal:
    """A one-cell marginal: zero share p0, linear positive quantile map."""

    n_pos = 100

    def __init__(self, p0, lo, hi):
        self.p0 = p0
        self._lo, self._hi = lo, hi

    def quantile(self, u):
        return self._lo + (self._hi - self._lo) * np.asarray(u)

    def rank(self, y):
        return float(
            np.clip((y - self._lo) / (self._hi - self._lo), 0.001, 0.999)
        )


def test_regenerate_earnings_respects_zero_share_and_support():
    marg = {(3, td.TERMINAL_PERIOD): _FakeMarginal(0.3, 10_000.0, 200_000.0)}

    def age_bin(a):
        return np.full(len(np.atleast_1d(a)), 3, dtype=int)

    ages = np.full(20_000, 45.0)
    rng = np.random.default_rng(0)
    earn = td.regenerate_earnings(ages, rng, marg, age_bin)
    zero_frac = float((earn == 0).mean())
    assert abs(zero_frac - 0.3) < 0.02  # ~p0 zeros
    pos = earn[earn > 0]
    assert pos.min() >= 10_000.0 - 1 and pos.max() <= 200_000.0 + 1


def test_kendall_tau_extremes():
    order = ["a", "b", "c", "d"]
    assert td._kendall_tau(order, order) == pytest.approx(1.0)
    assert td._kendall_tau(order, order[::-1]) == pytest.approx(-1.0)
    # one adjacent swap of 4 items -> tau 2/3
    assert td._kendall_tau(["a", "c", "b", "d"], order) == pytest.approx(2 / 3)


def test_swap_realised():
    # required swap_pair [x, y] means y should end up ABOVE x
    assert td._swap_realised(["y", "z", "x", "w"], ["x", "y"]) is True
    assert td._swap_realised(["x", "y", "z", "w"], ["x", "y"]) is False
    assert td._swap_realised(["a"], ["x", "y"]) is None


def test_m4_band_of_excludes_children_and_over_cap():
    bands = ((20, 29), (30, 39), (40, 49), (50, 59), (60, 66))
    assert td._m4_band_of(10, bands) is None  # child -> no DI
    assert td._m4_band_of(19, bands) is None
    assert td._m4_band_of(25, bands) == (20, 29)
    assert td._m4_band_of(66, bands) == (60, 66)
    assert td._m4_band_of(70, bands) is None  # past FRA -> converted


def test_simulate_di_conditions_only_on_age_sex():
    import pandas as pd

    bands = ((20, 29), (30, 39), (40, 49), (50, 59), (60, 66))
    prev = {}
    for lo, hi in bands:
        for sex in ("female", "male"):
            prev[(f"{lo}-{hi}", lo, hi, sex)] = 0.5
    gens = types.SimpleNamespace(m4_prevalence=prev, m4_bands=bands)
    persons = pd.DataFrame(
        {"age": [10, 25, 40, 55, 70], "is_female": [True] * 5}
    )
    rng = np.random.default_rng(0)
    on_di = td.simulate_di_status(persons, gens, rng)
    # children (10) and past-cap (70) never on DI
    assert not on_di[0] and not on_di[4]


@pytest.mark.skipif(
    "POPULACE_DYNAMICS_FRAME_PICKLE" not in os.environ,
    reason="needs the certified frame export (policyengine.py .venv)",
)
def test_frame_pin_matches_certified():
    assert td.dfm.CERTIFIED_PIN["artifact_sha256"].startswith("c2065b64")
    assert td.dfm.CERTIFIED_PIN["reference_period"] == "2024"
