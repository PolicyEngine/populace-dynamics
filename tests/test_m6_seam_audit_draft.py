"""DRAFT synthetic characterizations for the omnibus M6 seam audit.

These tests intentionally pin current defect behavior.  They are evidence for
``AUDIT.md``, not proposed fixes, and are not registered in a tier manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import disability
from populace_dynamics.engine.assembly import _merge_period_columns
from populace_dynamics.engine.earnings_domain import EarningsDomainAdapter
from populace_dynamics.engine.loop import (
    PeriodContext,
    SyntheticPersonIdAllocator,
)
from populace_dynamics.engine.steps import (
    AgeSexMortalityModel,
    apply_earnings,
    materialize_maternal_births,
)
from populace_dynamics.engine.support import (
    PresenceBasis,
    StartWaveWeightSnapshot,
)
from populace_dynamics.harness.m6_projection import (
    prepare_gated_realized_support,
    prepare_projected_disability,
)
from populace_dynamics.harness.panel import split_panel_by_person


class _GlobalDomainGenerator:
    """Minimal fitted surface whose only real-domain ID is absent from a side."""

    realized_earn_2014_by_person = {2: 100.0}
    u_w_by_person = {2: 0.1}

    def materialize_initial_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["u_w"] = 0.1
        out["realized_earn_2014"] = 100.0
        out["realized_earn_2012"] = 90.0
        out["earnings"] = 100.0
        out["gen_earn_w2"] = 100.0
        out["gen_earn_w4"] = 90.0
        return out

    def generate(
        self, frame: pd.DataFrame, year: int, rng: np.random.Generator
    ) -> np.ndarray:
        del year, rng
        return np.full(len(frame), 100.0)


def test_draft_off_band_disability_pair_trips_exact_support_guard():
    person_years = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "period": [2015, 2017, 2015, 2017],
            "age": [18, 20, 30, 32],
            "sex": ["female", "female", "male", "male"],
            "weight": [9.0, 9.0, 9.0, 9.0],
            "disabled": [False, False, False, True],
        }
    )
    anchor = pd.DataFrame({"person_id": [1, 2], "weight": [1.0, 2.0]})
    projected = prepare_projected_disability(
        disability.DisabilityPanel(person_years, pd.DataFrame()), anchor
    )
    truth = projected[projected["age"].between(20, 66)].copy()

    with pytest.raises(ValueError, match="identical projection and truth"):
        prepare_gated_realized_support(
            projected,
            truth,
            realized_presence=projected[["person_id", "start_wave"]],
            start_weights=StartWaveWeightSnapshot.from_frame(
                anchor, boundary_period=2014
            ),
            presence_basis=PresenceBasis.START_OF_INTERVAL,
            period_column="start_wave",
        )


def test_draft_side_local_child_id_aliases_global_earnings_domain():
    adapter = EarningsDomainAdapter(_GlobalDomainGenerator())
    side = pd.DataFrame(
        {
            "person_id": [1],
            "year": [2014],
            "age": [30],
            "sex": ["female"],
            "weight": [1.0],
        }
    )
    side = adapter.materialize_initial_frame(side)
    context = PeriodContext(
        period_index=1,
        year=2015,
        draw_index=0,
        metadata={
            "synthetic_id_allocator": SyntheticPersonIdAllocator(next_id=2)
        },
    )
    roster = materialize_maternal_births(
        side.assign(year=2015),
        pd.DataFrame({"parent_person_id": [1], "birth_year": [2015]}),
        context,
        np.random.default_rng(0),
    )
    assert roster.loc[roster["age"] == 0, "person_id"].tolist() == [2]

    with pytest.raises(ValueError, match="marker disagrees"):
        apply_earnings(
            roster,
            context,
            np.random.default_rng(1),
            model=adapter,
        )


def test_draft_partial_period_update_erases_unmatched_live_state():
    roster = pd.DataFrame(
        {
            "person_id": [1, 2],
            "marital_state": ["married", "divorced"],
        }
    )
    updates = pd.DataFrame(
        {"person_id": [1], "year": [2016], "marital_state": ["widowed"]}
    )

    merged = _merge_period_columns(
        roster, updates, year=2016, columns=("marital_state",)
    )

    assert merged.loc[merged["person_id"] == 1, "marital_state"].item() == (
        "widowed"
    )
    assert pd.isna(
        merged.loc[merged["person_id"] == 2, "marital_state"].item()
    )


def test_draft_domain_first_floor_split_changes_full_anchor_membership():
    full = pd.DataFrame({"person_id": [1, 2, 3], "household_id": [1, 2, 3]})
    domain = full[full["person_id"].isin({1, 3})]

    full_left, _ = split_panel_by_person(full, "person_id", seed=1)
    domain_left, _ = split_panel_by_person(domain, "person_id", seed=1)

    assert set(full_left["person_id"]) & {1, 3} == {3}
    assert set(domain_left["person_id"]) == set()


def test_draft_mortality_open_top_assigns_zero_probability_above_120():
    model = AgeSexMortalityModel(
        bands=((0, 84), (85, 120)),
        probability={
            ("0-84", "female"): 0.0,
            ("0-84", "male"): 0.0,
            ("85+", "female"): 1.0,
            ("85+", "male"): 1.0,
        },
    )
    frame = pd.DataFrame({"age": [120, 121], "sex": ["female", "female"]})

    assert model.probabilities(frame).tolist() == [1.0, 0.0]
