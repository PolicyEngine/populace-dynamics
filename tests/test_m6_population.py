"""Real-schema coverage of the M6 realized-anchor population law.

The demographic fixture carries the exact seven columns
:func:`populace_dynamics.data.panels.demographic_panel` emits
(``person_id, period, age, sequence, relationship, weight, interview`` -- and
**no** ``sex``), and person sex is supplied by a separate real-schema death-
records sibling, mirroring :func:`populace_dynamics.data.deaths.read_death_records`.
This is deliberate: the earlier fixture baked ``sex`` into the demo frame, so it
passed while the real-frame path could not (grading #42 comment 4972045579).
"""

from __future__ import annotations

import pandas as pd
import pytest

from populace_dynamics.data import (
    disability,
    household_composition,
    transitions,
)
from populace_dynamics.engine.loop import (
    SCHEDULED_ENTRIES_KEY,
    MaritalStepResult,
    PeriodModules,
    ProjectionEngine,
)
from populace_dynamics.engine.panel_builders import PanelBuilderInputs
from populace_dynamics.harness.m6_population import build_realized_population


def _inputs():
    anchor = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": pd.Series([101, 202, 303], dtype="int64"),
            "weight": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
            "anchor_wave": pd.Series([2015, 2017, 2019], dtype="int64"),
        }
    )
    # Exactly the real demographic_panel schema: seven columns, no sex.
    demo = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "period": [2015, 2017, 2019],
            "age": [31, 42, 53],
            "sequence": [1, 2, 1],
            "relationship": [10, 20, 10],
            "weight": [1.0, 2.0, 3.0],
            "interview": [101, 202, 303],
        }
    )
    # The sibling that actually carries person sex, mirroring the real
    # read_death_records schema (ER32000 -> sex; person-constant).
    death_records = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "sex_code": pd.Series([2, 1, 2], dtype="int64"),
            "sex": pd.array(["female", "male", "female"], dtype="string"),
            "death_code": pd.Series([0, 0, 0], dtype="int64"),
            "death_status": pd.array(
                ["not_deceased", "not_deceased", "not_deceased"],
                dtype="string",
            ),
            "death_year": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "death_year_lo": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "death_year_hi": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
        }
    )
    marital_py = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "year": [2015, 2017, 2019],
            "marital_state": ["married", "divorced", "never_married"],
            "marriage_duration": pd.array([4, pd.NA, pd.NA], dtype="Int64"),
            "years_since_dissolution": pd.array(
                [pd.NA, 2, pd.NA], dtype="Int64"
            ),
        }
    )
    marital = transitions.MaritalPanel(
        person_years=marital_py,
        events=pd.DataFrame(),
        attrs=pd.DataFrame({"person_id": [1, 2, 3]}),
    )
    household_pw = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "year": [2015, 2017, 2019],
            "coresident_spouse": [True, False, False],
            "coresident_parent": [False, False, True],
            "coresident_child": [True, False, False],
            "coresident_grandchild": [False, False, False],
            "multigen": [False, False, False],
            "hh_size": [3, 1, 2],
        }
    )
    household = household_composition.HouseholdCompositionPanel(
        person_waves=household_pw,
        attrs=pd.DataFrame({"person_id": [1, 2, 3]}),
    )
    dis_py = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "period": [2015, 2017, 2019],
            "status_code": [1, 5, 4],
            "disabled": [False, True, False],
            "retired": [False, False, True],
        }
    )
    dis = disability.DisabilityPanel(dis_py, pd.DataFrame())
    builders = PanelBuilderInputs(
        anchor=anchor,
        marital=marital,
        household=household,
        cohabitation=pd.DataFrame(
            {
                "person_id": [1, 2],
                "year": [2015, 2017],
                "cohabiting": [True, False],
            }
        ),
    )
    earnings = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "period": [2014, 2016, 2018],
            "age": [30, 41, 52],
            "weight": [99.0, 99.0, 99.0],
            "earnings": [10.0, 20.0, 30.0],
        }
    )
    return demo, death_records, earnings, dis, builders


def test_later_openers_enter_only_at_their_realized_anchor():
    demo, death_records, earnings, dis, builders = _inputs()
    population = build_realized_population(
        demographic_panel=demo,
        death_records=death_records,
        earnings_panel=earnings,
        disability_panel=dis,
        panel_builder_inputs=builders,
        earnings_domain_ids={1},
    )

    assert population.initial_slice["person_id"].tolist() == [1]
    assert population.initial_slice["year"].tolist() == [2014]
    assert population.initial_slice["age"].tolist() == [31]
    assert population.initial_slice["cohabiting"].tolist() == [True]
    assert sorted(population.scheduled_entries_by_year) == [2017, 2019]
    assert population.scheduled_entries_by_year[2017][
        "person_id"
    ].tolist() == [2]
    assert population.scheduled_entries_by_year[2017]["year"].tolist() == [
        2016
    ]
    assert population.scheduled_entries_by_year[2019]["age"].tolist() == [53]
    assert population.earnings_domain_ids == frozenset({1})
    assert not population.scheduled_entries_by_year[2017][
        "earnings_domain"
    ].item()
    assert not population.scheduled_entries_by_year[2017]["cohabiting"].item()


def test_earnings_support_is_exact_row_existence_not_flow_presence():
    demo, death_records, earnings, dis, builders = _inputs()
    population = build_realized_population(
        demographic_panel=demo,
        death_records=death_records,
        earnings_panel=earnings,
        disability_panel=dis,
        panel_builder_inputs=builders,
        earnings_domain_ids={1},
    )

    assert set(population.earnings_support["period"]) == {2014, 2016, 2018}
    assert population.presence[2015] == {1}
    assert population.presence[2017] == {2}


def test_person_sex_is_sourced_from_the_death_records_sibling():
    demo, death_records, earnings, dis, builders = _inputs()
    population = build_realized_population(
        demographic_panel=demo,
        death_records=death_records,
        earnings_panel=earnings,
        disability_panel=dis,
        panel_builder_inputs=builders,
        earnings_domain_ids={1},
    )

    # Person 1 seeds at 2015 (the initial slice); sex comes from the sibling,
    # never from the demo frame, which does not carry it.
    assert "sex" not in demo.columns
    assert population.initial_slice["sex"].tolist() == ["female"]
    # Later openers keep their coded sex at their realized anchor wave.
    assert population.scheduled_entries_by_year[2017]["sex"].tolist() == [
        "male"
    ]


def test_anchor_person_without_coded_sex_raises():
    demo, death_records, earnings, dis, builders = _inputs()
    # Drop person 2 from the sex sibling: an anchor person with no coded sex.
    dropped = death_records[death_records["person_id"] != 2].reset_index(
        drop=True
    )
    with pytest.raises(ValueError, match="no coded sex"):
        build_realized_population(
            demographic_panel=demo,
            death_records=dropped,
            earnings_panel=earnings,
            disability_panel=dis,
            panel_builder_inputs=builders,
            earnings_domain_ids={1},
        )


def test_projection_loop_activates_openers_at_anchor_with_global_ordinals():
    observed = []

    def mortality(frame, context, rng):
        del rng
        observed.append(
            (
                context.year,
                tuple(frame["person_id"]),
                dict(context.person_ordinals),
            )
        )
        return frame.copy()

    def aging(frame, context, rng):
        del rng
        out = frame.copy()
        out["year"] = context.year
        return out

    def passthrough(frame, context, rng):
        del context, rng
        return frame.copy()

    def marital(frame, context, rng):
        del frame, context, rng
        return MaritalStepResult(pd.DataFrame(), pd.DataFrame())

    def reader(frame, context, marital_result, rng):
        del context, marital_result, rng
        return frame.copy()

    engine = ProjectionEngine(
        PeriodModules(
            mortality=mortality,
            aging=aging,
            marital_core=marital,
            fertility=reader,
            disability=passthrough,
            earnings=passthrough,
            claiming=passthrough,
            household_composition=reader,
        )
    )
    initial = pd.DataFrame({"person_id": [1], "year": [2014]})
    result = engine.project(
        initial,
        end_year=2019,
        draw_index=0,
        metadata={
            SCHEDULED_ENTRIES_KEY: {
                2017: pd.DataFrame({"person_id": [2], "year": [2016]}),
                2019: pd.DataFrame({"person_id": [3], "year": [2018]}),
            }
        },
    )

    assert [tuple(frame["person_id"]) for frame in result.slices] == [
        (1,),
        (1,),
        (1,),
        (1, 2),
        (1, 2),
        (1, 2, 3),
    ]
    assert observed[0][2] == {1: 0, 2: 1, 3: 2}
    assert observed[2][1] == (1, 2)
    assert observed[4][1] == (1, 2, 3)


def test_person_sex_map_raises_on_conflicting_coded_values():
    from populace_dynamics.harness.m6_population import _person_sex_map

    # Two conflicting coded sexes for one person must raise, not first-win.
    conflicting = pd.DataFrame(
        {
            "person_id": [1, 1],
            "sex": pd.array(["female", "male"], dtype="string"),
        }
    )
    with pytest.raises(ValueError, match="conflicting"):
        _person_sex_map(conflicting)


def test_demo_fixture_matches_the_committed_demographic_schema():
    from populace_dynamics.harness.m6_schema_audit import (
        COMMITTED_FRAME_SCHEMAS,
    )

    # The fixture must not silently re-flatter: its columns are exactly the
    # committed real demographic_panel schema (no sex).
    demo, *_ = _inputs()
    assert set(demo.columns) == COMMITTED_FRAME_SCHEMAS["demographic_panel"]
