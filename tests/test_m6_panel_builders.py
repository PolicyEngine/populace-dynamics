"""Synthetic coverage for the registered M6 native-panel builders."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace_dynamics.data import household_composition as hc
from populace_dynamics.data import transitions
from populace_dynamics.engine.loop import PeriodContext
from populace_dynamics.engine.marital import (
    _simulate_candidate16_with_generators,
)
from populace_dynamics.engine.panel_builders import (
    PANEL_BUILDER_INPUTS_KEY,
    PanelBuilderInputs,
    household_panel_builder,
    marital_panel_builder,
)
from populace_dynamics.models.family_transitions.components.initial_states import (
    EntryWidowedCells,
    ObservedInitialStates,
    SupportComposition,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)


class _ConstantFirstMarriage:
    def __init__(self, probability: float = 0.0):
        self.probability = probability

    def predict(self, age, is_male, birth_decade):
        del is_male, birth_decade
        return np.full(len(age), self.probability)


def _events_schema() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": pd.Series(dtype="int64"),
            "year": pd.Series(dtype="int64"),
            "age": pd.Series(dtype="int64"),
            "sex": pd.Series(dtype="object"),
            "weight": pd.Series(dtype="float64"),
            "transition": pd.Series(dtype="object"),
            "marriage_duration": pd.Series(dtype="Int64"),
            "years_since_dissolution": pd.Series(dtype="Int64"),
            "origin": pd.Series(dtype="object"),
        }
    )


def _marital_source() -> transitions.MaritalPanel:
    attrs = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "sex": ["female", "male", "female"],
            "birth_year": [1980.0, 1981.0, 1982.0],
            "most_recent_report_year": [2023.0, 2018.0, 2023.0],
            "n_marriages": [7.0, 0.0, 2.0],
            "death_year": pd.array([pd.NA, 2018, pd.NA], dtype="Int64"),
            "censor_year": [2023.0, 2018.0, 2023.0],
            "weight": [11.0, 12.0, 13.0],
            "start_exposure_year": [1995.0, 1996.0, 1997.0],
        }
    )
    person_years = transitions._person_years_frame(attrs)
    person_years["marital_state"] = "never_married"
    person_years["marriage_duration"] = pd.array(
        [pd.NA] * len(person_years), dtype="Int64"
    )
    person_years["years_since_dissolution"] = pd.array(
        [pd.NA] * len(person_years), dtype="Int64"
    )
    anchor_row = (person_years["person_id"] == 1) & (
        person_years["year"] == 2015
    )
    person_years.loc[anchor_row, "marital_state"] = "married"
    person_years.loc[anchor_row, "marriage_duration"] = 10
    return transitions.MaritalPanel(
        person_years=person_years,
        events=_events_schema(),
        attrs=attrs,
    )


def _household_source() -> hc.HouseholdCompositionPanel:
    rows = []
    years_by_person = {
        1: (2013, 2015, 2017, 2019, 2021, 2023),
        2: (2015, 2017, 2019, 2021, 2023),
        3: (2015, 2017),
        4: (2015,),
    }
    for person_id, years in years_by_person.items():
        for year in years:
            anchor_state = year in (2015, 2017)
            age = year - (1980 + person_id)
            rows.append(
                {
                    "person_id": person_id,
                    "year": year,
                    "age": age,
                    "band": hc._band_of(age),
                    "sex": "female" if person_id % 2 else "male",
                    "weight": float(person_id * 100 + year),
                    "hh_size": 2 if anchor_state else 99,
                    "coresident_spouse": anchor_state,
                    "coresident_parent": person_id == 1 and anchor_state,
                    "coresident_child": person_id == 2 and anchor_state,
                    "coresident_grandchild": False if anchor_state else True,
                    "multigen": person_id == 2 and anchor_state,
                }
            )
    person_waves = hc._add_transitions(pd.DataFrame(rows))
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    return hc.HouseholdCompositionPanel(person_waves, attrs)


def _inputs() -> PanelBuilderInputs:
    anchor = pd.DataFrame(
        {
            "person_id": [1, 2, 4, 5],
            "household_id": [101, 202, 404, 505],
            "weight": [11.0, 12.0, 14.0, 15.0],
            "anchor_wave": [2015, 2019, 2017, 2015],
        }
    )
    cohabitation = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 4],
            "year": [2015, 2019, 2021, 2015],
            "cohabiting": [True, False, True, True],
        }
    )
    return PanelBuilderInputs(
        anchor=anchor,
        marital=_marital_source(),
        household=_household_source(),
        cohabitation=cohabitation,
    )


def _context(inputs: PanelBuilderInputs, *, year: int = 2015) -> PeriodContext:
    return PeriodContext(
        period_index=year - 2014,
        year=year,
        draw_index=0,
        metadata={PANEL_BUILDER_INPUTS_KEY: inputs},
    )


def _components(
    person_ids: set[int],
    *,
    divorce_probability: float = 0.0,
    remarriage_probability: float = 0.0,
) -> FittedFamilyTransitions:
    empty_entry = EntryWidowedCells(
        key_sorted=np.array([], dtype=np.int64),
        years_since_dissolution_sorted=np.array([], dtype=np.int64),
        person_id_sorted=np.array([], dtype=np.int64),
        weight_sorted=np.array([], dtype=np.float64),
    )
    initial = ObservedInitialStates(
        marriage_residual_by_person={},
        entry_widowed=empty_entry,
        support=SupportComposition(
            pd.Series(
                0,
                index=pd.Index(sorted(person_ids), name="person_id"),
            )
        ),
    )
    fertility = {
        (age, parity, decade): 0.0
        for age in range(15, 50)
        for parity in range(4)
        for decade in (1970, 1980, 1990)
    }
    remarriage = {
        (age_band, ysd_band, origin, sex): remarriage_probability
        for age_band in range(5)
        for ysd_band in range(3)
        for origin in ("divorced", "widowed")
        for sex in ("female", "male")
    }
    gaps = {
        sex: {band: np.array([0], dtype=np.int64) for band in range(4)}
        for sex in ("female", "male")
    }
    return FittedFamilyTransitions(
        first_marriage=_ConstantFirstMarriage(),
        divorce=np.full((4, 2), divorce_probability),
        widowhood=SimpleNamespace(lookup=np.zeros((7, 2, 2))),
        remarriage=remarriage,
        fertility=fertility,
        initial_states=initial,
        spousal_age_gaps=gaps,
        implementation_ids={},
    )


def test_marital_builder_intersects_attrs_and_uses_per_person_anchor():
    inputs = _inputs()
    current = pd.DataFrame({"person_id": [999], "year": [2020]})

    panel, holdout_ids = marital_panel_builder(current, _context(inputs))

    # Person 2 is in the anchor and MH universes but their realized censor
    # precedes their 2019 anchor.  Person 3 is MH-covered but not anchored.
    assert holdout_ids == {1}
    assert panel.attrs["person_id"].tolist() == [1]
    assert panel.attrs.loc[0, "start_exposure_year"] == 2015.0
    assert panel.attrs.loc[0, "censor_year"] == 2022.0
    assert panel.attrs.loc[0, "n_marriages"] == 7.0
    assert panel.attrs.loc[0, "weight"] == 11.0
    assert panel.person_years[["person_id", "year"]].values.tolist() == [
        [1, 2015]
    ]
    assert panel.person_years.loc[0, "marital_state"] == "married"
    assert panel.person_years.loc[0, "marriage_duration"] == 10
    assert panel.events.empty

    source = inputs.marital
    assert list(panel.attrs.columns) == list(source.attrs.columns)
    assert list(panel.person_years.columns) == list(
        source.person_years.columns
    )
    assert list(panel.events.columns) == list(source.events.columns)
    assert panel.attrs.dtypes.equals(source.attrs.dtypes)
    assert panel.person_years.dtypes.equals(source.person_years.dtypes)
    assert panel.events.dtypes.equals(source.events.dtypes)


def test_marital_builder_is_frame_and_invocation_year_independent():
    inputs = _inputs()
    first, first_ids = marital_panel_builder(
        pd.DataFrame({"person_id": [1], "year": [2015]}),
        _context(inputs, year=2015),
    )
    later, later_ids = marital_panel_builder(
        pd.DataFrame({"person_id": [], "year": []}),
        _context(inputs, year=2021),
    )

    assert first_ids == later_ids
    pd.testing.assert_frame_equal(first.attrs, later.attrs)
    pd.testing.assert_frame_equal(first.person_years, later.person_years)
    pd.testing.assert_frame_equal(first.events, later.events)


def test_marital_seed_extends_open_episode_with_realized_duration():
    inputs = _inputs()
    panel, holdout_ids = marital_panel_builder(
        pd.DataFrame(), _context(inputs)
    )

    simulated, _ = _simulate_candidate16_with_generators(
        panel,
        holdout_ids,
        _components(holdout_ids),
        np.random.default_rng(5200),
        np.random.default_rng(6200),
    )

    person = simulated.person_years.sort_values("year")
    assert person["year"].tolist() == list(range(2015, 2023))
    assert person["marital_state"].tolist() == ["married"] * 8
    assert person["marriage_duration"].tolist() == list(range(10, 18))
    assert person["years_since_dissolution"].isna().all()
    # Lifetime MH18 is the byte-identical input seed and is carried-inert;
    # certified assembly replaces it with the simulated episode count.
    assert panel.attrs.loc[0, "n_marriages"] == 7.0
    assert simulated.attrs.loc[0, "n_marriages"] == 1.0


def test_marital_core_appends_certified_change_points_and_durations():
    inputs = _inputs()
    panel, holdout_ids = marital_panel_builder(
        pd.DataFrame(), _context(inputs)
    )

    simulated, _ = _simulate_candidate16_with_generators(
        panel,
        holdout_ids,
        _components(
            holdout_ids,
            divorce_probability=1.0,
            remarriage_probability=1.0,
        ),
        np.random.default_rng(5200),
        np.random.default_rng(6200),
    )

    person = simulated.person_years.set_index("year")
    # Exact-year changes apply only after the event year.  The seed marriage
    # dissolves in 2015, remarriage occurs in 2016, and the new marriage first
    # appears in entering-year state in 2017 with duration one.
    assert person.loc[2015, "marital_state"] == "married"
    assert person.loc[2015, "marriage_duration"] == 10
    assert person.loc[2016, "marital_state"] == "divorced"
    assert person.loc[2016, "years_since_dissolution"] == 1
    assert person.loc[2017, "marital_state"] == "married"
    assert person.loc[2017, "marriage_duration"] == 1
    events = simulated.events.set_index("year")
    assert events.loc[2015, "transition"] == "divorce"
    assert events.loc[2016, "transition"] == "remarriage"
    assert events.loc[2016, "years_since_dissolution"] == 1


def test_household_builder_uses_realized_support_and_anchor_state_only():
    inputs = _inputs()
    panel, holdout_ids = household_panel_builder(
        pd.DataFrame({"person_id": [999], "year": [2020]}),
        _context(inputs),
    )

    # Person 1 has an exact 2015 household seed.  Person 2's anchor is 2019,
    # and person 4 is excluded because their household row is only in 2015,
    # before their 2017 anchor.
    assert holdout_ids == {1, 2}
    assert panel.attrs["person_id"].tolist() == [1, 2]
    one = panel.person_waves[panel.person_waves["person_id"] == 1]
    two = panel.person_waves[panel.person_waves["person_id"] == 2]
    assert one["year"].tolist() == [2015, 2017, 2019, 2021]
    assert two["year"].tolist() == [2019, 2021]
    assert one["weight"].tolist() == [11.0] * 4
    assert two["weight"].tolist() == [12.0] * 2

    # Post-anchor source values were deliberately set to 99/True.  Every
    # projected support row instead carries the person's own anchor state.
    assert one["hh_size"].tolist() == [2] * 4
    assert one["coresident_parent"].tolist() == [True] * 4
    assert one["coresident_grandchild"].tolist() == [False] * 4
    assert one["cohabiting"].tolist() == [True] * 4
    assert two["hh_size"].tolist() == [99] * 2
    assert two["multigen"].tolist() == [False] * 2
    # The sparse flag has a later positive row for person 2, but no exact hit
    # at their 2019 anchor, so the seed is false rather than future-informed.
    assert two["cohabiting"].tolist() == [False] * 2

    assert one["has_next"].tolist() == [True, True, True, False]
    assert one["next_coresident_parent"].iloc[:3].tolist() == [True] * 3
    assert pd.isna(one["next_coresident_parent"].iloc[-1])


def test_household_builder_preserves_reader_schema_bytes_and_rebuilds_links():
    inputs = _inputs()
    panel, _ = household_panel_builder(pd.DataFrame(), _context(inputs))
    source = inputs.household.person_waves

    assert list(panel.person_waves.columns) == [*source.columns, "cohabiting"]
    for column in source.columns:
        assert panel.person_waves[column].dtype == source[column].dtype
    assert panel.person_waves["cohabiting"].dtype == bool

    # No stale source transition survives the per-person anchor/support slice.
    two = panel.person_waves[panel.person_waves["person_id"] == 2]
    assert two["has_next"].tolist() == [True, False]
    assert two["next_multigen"].iloc[0] == two["multigen"].iloc[1]
    assert pd.isna(two["next_multigen"].iloc[1])


def test_from_realized_histories_uses_anchor_weight_reader(monkeypatch):
    anchor = _inputs().anchor
    expected = _marital_source()
    captured = {}

    def fake_build(records, deaths, weights):
        captured["records"] = records
        captured["deaths"] = deaths
        captured["weights"] = weights.copy()
        return expected

    monkeypatch.setattr(transitions, "build_marital_panel", fake_build)
    records = pd.DataFrame({"person_id": [1]})
    deaths = pd.DataFrame({"person_id": [1], "death_year": [pd.NA]})
    household = _household_source()
    cohabitation = pd.DataFrame(columns=["person_id", "year", "cohabiting"])

    result = PanelBuilderInputs.from_realized_histories(
        anchor=anchor,
        marriage_records=records,
        death_records=deaths,
        household=household,
        cohabitation=cohabitation,
    )

    assert result.marital is expected
    assert captured["records"] is records
    assert captured["deaths"] is deaths
    pd.testing.assert_series_equal(
        captured["weights"],
        anchor.set_index("person_id")["weight"],
    )
