"""Source-data assembly for the candidate-9 household registry.

The load order, positive-weight selection, panel construction, and reusable
seed-independent frames are copied from
``scripts/run_gate2b_candidate9.py:275-336``.  Unlike the frozen runner, this
module resolves those frames through the named registry components and never
imports a ``household_composition_sim*`` module.
"""

from __future__ import annotations

from typing import Any

from populace_dynamics.data import (
    births,
    deaths,
    marriage,
    panels,
    relmap,
    transitions,
)
from populace_dynamics.data import household_composition as hc


def marriage_order_map(marriage_records):
    """Return datable per-person marriage order.

    Copied from ``scripts/run_gate2b_candidate9.py:275-283``.
    """
    episodes = marriage.marriage_episodes(marriage_records)
    episodes = episodes[episodes["start_year"].notna()].copy()
    episodes["start_year"] = episodes["start_year"].astype("int64")
    episodes = episodes.sort_values(["person_id", "start_year"])
    episodes["order"] = episodes.groupby("person_id").cumcount() + 1
    return episodes[["person_id", "start_year", "order"]].drop_duplicates(
        ["person_id", "start_year"]
    )


def load_sources() -> dict[str, Any]:
    """Load every source and seed-independent frame used by candidate 9.

    Copied operation-for-operation from
    ``scripts/run_gate2b_candidate9.py:286-336``.  Imports are local because
    the component modules are the owners of their copied data adapters.
    """
    from populace_dynamics.models.household_composition.components.child_attribution import (
        build_child_record_exposure,
        father_link_births_with_child,
        father_marital_by_year,
        parent_child_coresidence_pairs,
    )
    from populace_dynamics.models.household_composition.components.household_size import (
        parent_link_counts,
    )
    from populace_dynamics.models.household_composition.components.legal_spouse_residual import (
        legal_spouse_flag,
    )
    from populace_dynamics.models.household_composition.components.nonfamily_bridge import (
        family_unit_sizes,
    )

    relationship_map = relmap.relationship_map()
    demographic_panel = panels.demographic_panel()
    death_records = deaths.read_death_records()
    roster = hc.household_roster(relationship_map)
    person_waves = hc.join_demographics(
        roster, demographic_panel, death_records
    )
    attrs = (
        person_waves[["person_id"]].drop_duplicates().reset_index(drop=True)
    )
    household_panel = hc.HouseholdCompositionPanel(
        person_waves=person_waves,
        attrs=attrs,
    )

    marriage_records = marriage.marriage_history()
    birth_records = births.birth_history()
    positive_weight = demographic_panel[demographic_panel.weight > 0]
    person_weight = (
        positive_weight.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    marital_panel = transitions.build_marital_panel(
        marriage_records,
        death_records,
        person_weight,
    )
    order_map = marriage_order_map(marriage_records)

    father_links_child = father_link_births_with_child(birth_records)
    parent_pairs = parent_child_coresidence_pairs(relationship_map)
    family_sizes = family_unit_sizes(relationship_map)
    legal_flag = legal_spouse_flag(relationship_map)
    parent_counts = parent_link_counts(relationship_map)
    marital_by_year = father_marital_by_year(marital_panel)
    child_record_exposure = build_child_record_exposure(
        father_links_child,
        parent_pairs,
        marital_by_year,
        demographic_panel,
        relationship_map,
    )
    return {
        "hh": household_panel,
        "mpanel": marital_panel,
        "demo": demographic_panel,
        "mh": marriage_records,
        "bh": birth_records,
        "rel_map": relationship_map,
        "order_map": order_map,
        "father_links_child": father_links_child,
        "parent_pairs": parent_pairs,
        "marital_by_year": marital_by_year,
        "fu_sizes": family_sizes,
        "legal_flag": legal_flag,
        "parent_counts": parent_counts,
        "child_record_expo": child_record_exposure,
    }
