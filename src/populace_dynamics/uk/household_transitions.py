"""Household transition mechanics for the UK panel pipeline.

Ported from archived
`policyengine-uk-data#346 <https://github.com/PolicyEngine/policyengine-uk-data/pull/346>`_
per populace#148 / ADR 0002, onto
:class:`~populace_dynamics.uk.dataset.UKPanelDataset`.

Complements :mod:`populace_dynamics.uk.demographic_ageing` (mortality,
fertility, age increment) with the life-cycle events that change
household and benefit-unit composition:

- ``apply_marriages``: pair up single adults and merge their benunits.
- ``apply_separations``: split married benunits into two singles;
  children attach to the mother by default.
- ``apply_children_leaving_home``: move adult dependents out of their
  parents' benunit + household into a new one of their own.
- ``apply_migration``: add immigrant rows and drop emigrant rows.
- ``apply_employment_transitions`` / ``apply_income_decile_transitions``:
  within-person labour-market and income-mobility change, rule-based or
  UKHLS-matrix-driven.

**Review fixes applied** (blockers on #346's second pass):

1. Immigration donors are drawn proportionally to the donor's
   household weight (the source sampled donors unweighted, biasing
   the cloned immigrant population toward low-weight records).
2. Every person-removal path (emigration) routes through
   :func:`~populace_dynamics.uk.dataset.prune_orphaned_entities`, the
   same helper mortality uses, so no removal can leave benunits or
   households behind with no surviving members.

Design notes
------------
Benefit-unit semantics: marriage merges two single benunits into one;
separation splits a two-adult benunit — only ``person_benunit_id``
(and, for leaving-home and separation, ``person_household_id``) is
rewritten. Weights on surviving units are preserved by summing the
weights of merging units.

Randomness: all functions take an explicit ``numpy.random.Generator``
so the pipeline is reproducible given a seed, and all are pure with
respect to the input dataset.

Out of scope (as in the source): same-sex pairing, multi-benunit
households not assembled from a parent couple, second / subsequent
marriages, and family-unit migration.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from populace_dynamics.uk.dataset import (
    UKPanelDataset,
    prune_orphaned_entities,
)
from populace_dynamics.uk.demographic_ageing import (
    AGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
    SEX_COLUMN,
)

__all__ = [
    "DEFAULT_MARRIAGE_RATES",
    "DEFAULT_SEPARATION_RATES",
    "DEFAULT_LEAVING_HOME_RATES",
    "DEFAULT_NET_MIGRATION_RATES",
    "apply_marriages",
    "apply_separations",
    "apply_children_leaving_home",
    "apply_migration",
    "apply_employment_transitions",
    "apply_income_decile_transitions",
]


# -- Age-indexed marriage rates, per unmarried adult per year. ---------
# ONS Marriage Statistics for England and Wales, historic post-2000
# averages; callers can override via ``marriage_rates``.
DEFAULT_MARRIAGE_RATES: dict[str, dict[int, float]] = {
    MALE_VALUE: {
        **{age: 0.0 for age in range(0, 18)},
        **{age: 0.002 for age in range(18, 20)},
        **{age: 0.010 for age in range(20, 25)},
        **{age: 0.030 for age in range(25, 30)},
        **{age: 0.035 for age in range(30, 35)},
        **{age: 0.025 for age in range(35, 40)},
        **{age: 0.015 for age in range(40, 50)},
        **{age: 0.008 for age in range(50, 65)},
        **{age: 0.003 for age in range(65, 121)},
    },
    FEMALE_VALUE: {
        **{age: 0.0 for age in range(0, 18)},
        **{age: 0.004 for age in range(18, 20)},
        **{age: 0.018 for age in range(20, 25)},
        **{age: 0.035 for age in range(25, 30)},
        **{age: 0.032 for age in range(30, 35)},
        **{age: 0.020 for age in range(35, 40)},
        **{age: 0.010 for age in range(40, 50)},
        **{age: 0.004 for age in range(50, 65)},
        **{age: 0.001 for age in range(65, 121)},
    },
}


def apply_marriages(
    dataset: UKPanelDataset,
    marriage_rates: Mapping[str, Mapping[int, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> UKPanelDataset:
    """Pair up single adults and merge their benunits.

    For each single adult, draw Bernoulli(rate[age, sex]) — winners
    enter a "want to marry" pool and are matched to an opposite-sex
    partner in the same region, preferring similar age. The partners'
    benunits merge; the first partner's household is kept and the
    second (plus their benunit's dependents) moves in.

    ``marriage_rates=None`` uses :data:`DEFAULT_MARRIAGE_RATES`; pass
    ``{}`` to disable.
    """
    if rng is None:
        rng = np.random.default_rng()

    if marriage_rates is None:
        marriage_rates = DEFAULT_MARRIAGE_RATES
    if not marriage_rates:
        return dataset

    ds = dataset.copy()

    singles = _identify_single_adults(ds)
    if singles.empty:
        return ds

    singles = _draw_want_to_marry(singles, marriage_rates, rng)
    if singles["wants_to_marry"].sum() == 0:
        return ds

    pairs = _match_pairs(singles, rng)
    if not pairs:
        return ds

    return _merge_pairs(ds, pairs)


def _identify_single_adults(ds: UKPanelDataset) -> pd.DataFrame:
    """Single adults keyed by ``person_id`` with their benunit,
    household, age, gender and region.

    A benunit is single if it has exactly one adult; adults in
    multi-adult benunits (existing couples) are excluded.
    """
    person = ds.person
    benunit_adult_counts = (
        person.assign(
            _is_adult=(person[AGE_COLUMN].astype(int) >= 18).astype(int)
        )
        .groupby("person_benunit_id")["_is_adult"]
        .sum()
        .rename("benunit_adults")
    )
    single_benunits = benunit_adult_counts[benunit_adult_counts == 1].index

    adults = person[
        (person[AGE_COLUMN].astype(int) >= 18)
        & person["person_benunit_id"].isin(single_benunits)
    ].copy()

    if "region" in ds.household.columns:
        hh_region = ds.household.set_index("household_id")["region"]
        adults["_region"] = (
            adults["person_household_id"]
            .map(hh_region)
            .fillna("_UNKNOWN")
            .values
        )
    else:
        adults["_region"] = "_UNKNOWN"
    return adults


def _draw_want_to_marry(
    singles: pd.DataFrame,
    rates: Mapping[str, Mapping[int, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw Bernoulli(rate[age, sex]) for each single; mark winners."""
    probs = np.array(
        [
            float(rates.get(str(sex), {}).get(int(age), 0.0))
            for age, sex in zip(
                singles[AGE_COLUMN].astype(int), singles[SEX_COLUMN]
            )
        ],
        dtype=float,
    )
    draws = rng.random(size=len(singles))
    singles = singles.copy()
    singles["wants_to_marry"] = draws < probs
    return singles


def _match_pairs(
    singles: pd.DataFrame, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """(male_person_id, female_person_id) pairs for matched singles.

    Greedy match within region: for each male who wants to marry (in
    randomised order), pick the closest-age unmatched female in the
    same region. One-sided excess intent is discarded.
    """
    wants = singles[singles["wants_to_marry"]]
    if wants.empty:
        return []

    pairs: list[tuple[int, int]] = []
    for _region, group in wants.groupby("_region"):
        males = group[group[SEX_COLUMN] == MALE_VALUE].copy()
        females = group[group[SEX_COLUMN] == FEMALE_VALUE].copy()
        if males.empty or females.empty:
            continue

        males = males.sample(
            frac=1.0, random_state=int(rng.integers(2**31 - 1))
        )

        female_ages = females[AGE_COLUMN].astype(int).to_numpy()
        female_ids = females["person_id"].to_numpy()
        female_taken = np.zeros(len(females), dtype=bool)

        for _, male_row in males.iterrows():
            male_age = int(male_row[AGE_COLUMN])
            candidate_scores = np.where(
                female_taken,
                np.inf,
                np.abs(female_ages - male_age).astype(float),
            )
            best = int(np.argmin(candidate_scores))
            if candidate_scores[best] == np.inf:
                break
            pairs.append((int(male_row["person_id"]), int(female_ids[best])))
            female_taken[best] = True

    return pairs


def _merge_pairs(
    ds: UKPanelDataset, pairs: list[tuple[int, int]]
) -> UKPanelDataset:
    """For each (male_id, female_id) pair, merge their benunits.

    The male's benunit absorbs the female's; the male's household
    absorbs the female's. Emptied unit rows are dropped with their
    weight folded into the surviving unit. (A simplification — real
    newlyweds often form a new household; keeping the male-side IDs
    preserves panel continuity without minting new IDs.)
    """
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    person_by_id = person.set_index("person_id")
    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    bu_reassign: dict[int, int] = {}
    hh_reassign: dict[int, int] = {}

    for male_id, female_id in pairs:
        m = person_by_id.loc[male_id]
        f = person_by_id.loc[female_id]
        m_bu, f_bu = int(m["person_benunit_id"]), int(f["person_benunit_id"])
        m_hh, f_hh = int(m["person_household_id"]), int(
            f["person_household_id"]
        )

        # Chase transitive reassignments so units touched by two pairs
        # land on the final destination.
        dest_bu = _resolve(bu_reassign, m_bu)
        src_bu = _resolve(bu_reassign, f_bu)
        if dest_bu != src_bu:
            bu_reassign[src_bu] = dest_bu

        dest_hh = _resolve(hh_reassign, m_hh)
        src_hh = _resolve(hh_reassign, f_hh)
        if dest_hh != src_hh:
            hh_reassign[src_hh] = dest_hh

    bu_final = {k: _resolve(bu_reassign, k) for k in bu_reassign}
    hh_final = {k: _resolve(hh_reassign, k) for k in hh_reassign}

    if bu_final:
        person["person_benunit_id"] = person["person_benunit_id"].replace(
            bu_final
        )
    if hh_final:
        person["person_household_id"] = person["person_household_id"].replace(
            hh_final
        )

    if bu_final and bu_weight_col is not None:
        benunit = _fold_weights(benunit, "benunit_id", bu_final, bu_weight_col)
    if hh_final and hh_weight_col is not None:
        household = _fold_weights(
            household, "household_id", hh_final, hh_weight_col
        )

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


def _resolve(reassign: dict[int, int], key: int) -> int:
    while key in reassign:
        key = reassign[key]
    return key


def _find_weight_col(df: pd.DataFrame) -> str | None:
    for candidate in ("benunit_weight", "household_weight"):
        if candidate in df.columns:
            return candidate
    return None


# -- Age-indexed annual divorce (split) rates, per married person. -----
# ONS divorce statistics for England and Wales, smoothed 1990-2020
# approximations, indexed by the couple's mean age.
DEFAULT_SEPARATION_RATES: dict[int, float] = {
    **{age: 0.000 for age in range(0, 20)},
    **{age: 0.012 for age in range(20, 25)},
    **{age: 0.020 for age in range(25, 35)},
    **{age: 0.017 for age in range(35, 45)},
    **{age: 0.010 for age in range(45, 55)},
    **{age: 0.005 for age in range(55, 65)},
    **{age: 0.002 for age in range(65, 121)},
}


def apply_separations(
    dataset: UKPanelDataset,
    separation_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    *,
    children_stay_with_female: bool = True,
) -> UKPanelDataset:
    """Split married benunits with probability ``separation_rates[age]``.

    For each two-adult benunit, draw Bernoulli at the rate indexed by
    the mean of the two adults' ages. On success one adult stays on
    the existing IDs, the other moves to a fresh benunit and
    household. Children remain by default with the female adult (the
    HBAI "resident parent" convention).

    ``separation_rates=None`` uses :data:`DEFAULT_SEPARATION_RATES`;
    ``{}`` disables.
    """
    if rng is None:
        rng = np.random.default_rng()
    if separation_rates is None:
        separation_rates = DEFAULT_SEPARATION_RATES
    if not separation_rates:
        return dataset

    ds = dataset.copy()

    person = ds.person
    adult_mask = person[AGE_COLUMN].astype(int) >= 18
    adults = person[adult_mask]

    adult_counts = adults.groupby("person_benunit_id").size()
    couple_benunits = adult_counts[adult_counts == 2].index.tolist()
    if not couple_benunits:
        return ds

    mean_ages = (
        adults[adults["person_benunit_id"].isin(couple_benunits)]
        .groupby("person_benunit_id")[AGE_COLUMN]
        .mean()
        .astype(int)
    )
    probs = mean_ages.map(lambda a: float(separation_rates.get(int(a), 0.0)))
    draws = rng.random(size=len(probs))
    to_split = mean_ages.index[draws < probs.to_numpy()].tolist()
    if not to_split:
        return ds

    return _execute_separations(
        ds,
        couple_benunits_to_split=to_split,
        children_stay_with_female=children_stay_with_female,
        rng=rng,
    )


def _execute_separations(
    ds: UKPanelDataset,
    *,
    couple_benunits_to_split: list[int],
    children_stay_with_female: bool,
    rng: np.random.Generator,
) -> UKPanelDataset:
    """Apply the actual row moves for the drawn separations."""
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    new_bu_rows: list[dict] = []
    new_hh_rows: list[dict] = []

    for bu_id in couple_benunits_to_split:
        bu_rows = person[person["person_benunit_id"] == bu_id]
        adults_in_bu = bu_rows[bu_rows[AGE_COLUMN].astype(int) >= 18]
        if len(adults_in_bu) != 2:
            continue

        sexes = adults_in_bu[SEX_COLUMN].to_numpy()
        ids = adults_in_bu["person_id"].to_numpy()

        # Whichever adult does NOT keep the children moves out; if the
        # rule cannot apply (same-sex pair), the younger adult moves.
        if (
            children_stay_with_female
            and FEMALE_VALUE in sexes
            and MALE_VALUE in sexes
        ):
            mover_id = int(ids[np.where(sexes != FEMALE_VALUE)[0][0]])
        elif (
            (not children_stay_with_female)
            and FEMALE_VALUE in sexes
            and MALE_VALUE in sexes
        ):
            mover_id = int(ids[np.where(sexes == FEMALE_VALUE)[0][0]])
        else:
            younger = adults_in_bu.sort_values(AGE_COLUMN).iloc[0]
            mover_id = int(younger["person_id"])

        original_hh_id = int(adults_in_bu.iloc[0]["person_household_id"])

        person.loc[person["person_id"] == mover_id, "person_benunit_id"] = (
            next_bu_id
        )
        person.loc[person["person_id"] == mover_id, "person_household_id"] = (
            next_hh_id
        )

        if bu_weight_col is not None:
            original_bu_weight = float(
                benunit.loc[
                    benunit["benunit_id"] == bu_id, bu_weight_col
                ].iloc[0]
            )
            new_bu_rows.append(
                {
                    "benunit_id": next_bu_id,
                    bu_weight_col: original_bu_weight,
                }
            )

        if hh_weight_col is not None:
            original_hh_weight = float(
                household.loc[
                    household["household_id"] == original_hh_id,
                    hh_weight_col,
                ].iloc[0]
            )
            new_hh_row = {
                "household_id": next_hh_id,
                hh_weight_col: original_hh_weight,
            }
            if "region" in household.columns:
                new_hh_row["region"] = str(
                    household.loc[
                        household["household_id"] == original_hh_id,
                        "region",
                    ].iloc[0]
                )
            new_hh_rows.append(new_hh_row)

        next_bu_id += 1
        next_hh_id += 1

    if new_bu_rows:
        benunit = pd.concat(
            [benunit, pd.DataFrame(new_bu_rows, columns=benunit.columns)],
            ignore_index=True,
        )
    if new_hh_rows:
        household = pd.concat(
            [
                household,
                pd.DataFrame(new_hh_rows, columns=household.columns),
            ],
            ignore_index=True,
        )

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


# -- Age-indexed rates of adults leaving the parental home. ------------
# ONS LFS "Young adults living with parents" series.
DEFAULT_LEAVING_HOME_RATES: dict[int, float] = {
    **{age: 0.0 for age in range(0, 16)},
    16: 0.05,
    17: 0.08,
    **{age: 0.12 for age in range(18, 22)},
    **{age: 0.10 for age in range(22, 25)},
    **{age: 0.08 for age in range(25, 30)},
    **{age: 0.05 for age in range(30, 35)},
    **{age: 0.02 for age in range(35, 40)},
    **{age: 0.005 for age in range(40, 121)},
}


def apply_children_leaving_home(
    dataset: UKPanelDataset,
    leaving_home_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    *,
    min_age: int = 16,
) -> UKPanelDataset:
    """Move adult children out of their parents' benunit and household.

    For each person aged ``min_age``+ attached to a benunit containing
    another adult, draw Bernoulli at the age-indexed rate; leavers get
    a fresh benunit and household seeded from the original household's
    weight.

    ``leaving_home_rates=None`` uses
    :data:`DEFAULT_LEAVING_HOME_RATES`; ``{}`` disables.
    """
    if rng is None:
        rng = np.random.default_rng()
    if leaving_home_rates is None:
        leaving_home_rates = DEFAULT_LEAVING_HOME_RATES
    if not leaving_home_rates:
        return dataset

    ds = dataset.copy()

    person = ds.person
    eligible_ids = _identify_adult_dependents(person, min_age=min_age)
    if eligible_ids.empty:
        return ds

    probs = eligible_ids["age"].map(
        lambda a: float(leaving_home_rates.get(int(a), 0.0))
    )
    draws = rng.random(size=len(eligible_ids))
    leaving = eligible_ids[draws < probs.to_numpy()]
    if leaving.empty:
        return ds

    return _execute_leaving_home(ds, leaving["person_id"].tolist())


def _identify_adult_dependents(
    person: pd.DataFrame, *, min_age: int
) -> pd.DataFrame:
    """Adults who appear to be living in a parent's home.

    Two FRS/HBAI shapes:

    1. **Young dependent adult on parents' benunit** — a benunit with
       3+ adults; the two oldest are assumed the parental couple, the
       rest are candidates.
    2. **Adult child with own benunit inside the parental household**
       — a single-adult benunit whose household contains another
       benunit that also has adults.

    Two-adult benunits are never flagged (the resident parental pair).
    """
    adult_mask = person[AGE_COLUMN].astype(int) >= min_age
    adults = person[adult_mask].copy()
    if adults.empty:
        return pd.DataFrame(columns=["person_id", "age"])

    bu_adult_counts = adults.groupby("person_benunit_id").size()

    case1_ids: list[int] = []
    multi_adult_bus = bu_adult_counts[bu_adult_counts >= 3].index
    for bu in multi_adult_bus:
        bu_adults = adults[adults["person_benunit_id"] == bu].sort_values(
            by=AGE_COLUMN, ascending=False
        )
        case1_ids.extend(bu_adults["person_id"].iloc[2:].tolist())

    benunit_is_single_adult = bu_adult_counts == 1
    single_adult_bu_ids = set(
        benunit_is_single_adult[benunit_is_single_adult].index
    )

    case2_ids: list[int] = []
    for _hh_id, group in adults.groupby("person_household_id"):
        benunits_in_hh = set(group["person_benunit_id"].unique())
        if len(benunits_in_hh) < 2:
            continue
        for bu_id in benunits_in_hh:
            if bu_id not in single_adult_bu_ids:
                continue
            others_with_adults = [
                other
                for other in benunits_in_hh
                if other != bu_id and bu_adult_counts.get(other, 0) >= 1
            ]
            if not others_with_adults:
                continue
            case2_ids.extend(
                group[group["person_benunit_id"] == bu_id][
                    "person_id"
                ].tolist()
            )

    eligible_ids = set(case1_ids) | set(case2_ids)
    if not eligible_ids:
        return pd.DataFrame(columns=["person_id", "age"])

    eligible = adults[adults["person_id"].isin(eligible_ids)].copy()
    eligible = eligible.rename(columns={AGE_COLUMN: "age"})
    return eligible[["person_id", "age"]].reset_index(drop=True)


def _execute_leaving_home(
    ds: UKPanelDataset, leaving_person_ids: list[int]
) -> UKPanelDataset:
    """Move each leaver to a fresh benunit + household."""
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    new_bu_rows: list[dict] = []
    new_hh_rows: list[dict] = []

    for person_id in leaving_person_ids:
        row = person[person["person_id"] == person_id].iloc[0]
        original_hh_id = int(row["person_household_id"])
        original_bu_id = int(row["person_benunit_id"])

        person.loc[person["person_id"] == person_id, "person_benunit_id"] = (
            next_bu_id
        )
        person.loc[person["person_id"] == person_id, "person_household_id"] = (
            next_hh_id
        )

        if bu_weight_col is not None:
            bu_weight_source = benunit.loc[
                benunit["benunit_id"] == original_bu_id, bu_weight_col
            ]
            weight = (
                float(bu_weight_source.iloc[0])
                if len(bu_weight_source)
                else 1.0
            )
            new_bu_rows.append(
                {"benunit_id": next_bu_id, bu_weight_col: weight}
            )

        if hh_weight_col is not None:
            hh_weight_source = household.loc[
                household["household_id"] == original_hh_id,
                hh_weight_col,
            ]
            weight = (
                float(hh_weight_source.iloc[0])
                if len(hh_weight_source)
                else 1.0
            )
            new_hh_row = {
                "household_id": next_hh_id,
                hh_weight_col: weight,
            }
            if "region" in household.columns:
                region_source = household.loc[
                    household["household_id"] == original_hh_id, "region"
                ]
                new_hh_row["region"] = (
                    str(region_source.iloc[0]) if len(region_source) else ""
                )
            new_hh_rows.append(new_hh_row)

        next_bu_id += 1
        next_hh_id += 1

    if new_bu_rows:
        benunit = pd.concat(
            [benunit, pd.DataFrame(new_bu_rows, columns=benunit.columns)],
            ignore_index=True,
        )
    if new_hh_rows:
        household = pd.concat(
            [
                household,
                pd.DataFrame(new_hh_rows, columns=household.columns),
            ],
            ignore_index=True,
        )

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


def _fold_weights(
    df: pd.DataFrame,
    id_col: str,
    reassign: dict[int, int],
    weight_col: str,
) -> pd.DataFrame:
    """Sum absorbed rows' weights into their destination, then drop."""
    df = df.copy()
    df[id_col] = df[id_col].astype(int)
    mapping = df.set_index(id_col)[weight_col]
    for src, dest in reassign.items():
        if src in mapping.index and dest in mapping.index:
            df.loc[df[id_col] == dest, weight_col] = float(
                mapping.loc[dest]
            ) + float(mapping.loc[src])
    df = df[~df[id_col].isin(reassign.keys())]
    return df


# -- Net migration rates (immigration minus emigration) by age. --------
# ONS Long-Term International Migration, smoothed 2015-2023 averages.
DEFAULT_NET_MIGRATION_RATES: dict[int, float] = {
    **{age: 0.001 for age in range(0, 15)},
    **{age: 0.003 for age in range(15, 18)},
    **{age: 0.012 for age in range(18, 25)},
    **{age: 0.010 for age in range(25, 35)},
    **{age: 0.003 for age in range(35, 50)},
    **{age: 0.001 for age in range(50, 65)},
    **{age: 0.000 for age in range(65, 121)},
}


def _person_sampling_weights(ds: UKPanelDataset) -> pd.Series | None:
    """Per-person sampling weight from the household weight column.

    Review fix: donor draws must be weight-proportional so cloned
    immigrants reflect the weighted population, not the raw records.
    Returns ``None`` when the household table carries no weight
    column (uniform fallback).
    """
    hh_weight_col = _find_weight_col(ds.household)
    if hh_weight_col is None:
        return None
    hh_weights = ds.household.set_index("household_id")[hh_weight_col]
    weights = ds.person["person_household_id"].map(hh_weights)
    weights = weights.fillna(0.0).astype(float)
    if weights.sum() <= 0:
        return None
    return weights


def apply_migration(
    dataset: UKPanelDataset,
    net_migration_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
) -> UKPanelDataset:
    """Add immigrants (positive net) or remove emigrants (negative net).

    Net migration is a per-capita age-indexed rate. Per age cohort the
    expected delta is drawn as Poisson; positive deltas clone donor
    rows from the same-age cohort (weight-proportionally — review
    fix), negative deltas remove randomly-drawn rows, with orphaned
    entities pruned (review fix).

    ``net_migration_rates=None`` uses
    :data:`DEFAULT_NET_MIGRATION_RATES`; ``{}`` disables.
    """
    if rng is None:
        rng = np.random.default_rng()
    if net_migration_rates is None:
        net_migration_rates = DEFAULT_NET_MIGRATION_RATES
    if not net_migration_rates:
        return dataset

    ds = dataset.copy()

    ages = ds.person[AGE_COLUMN].astype(int).to_numpy()
    cohort_sizes = pd.Series(ages).value_counts().to_dict()
    sampling_weights = _person_sampling_weights(ds)

    emigrate_ids: list[int] = []
    immigrate_donor_ids: list[int] = []

    for age, n_in_cohort in cohort_sizes.items():
        rate = float(net_migration_rates.get(int(age), 0.0))
        if rate == 0.0 or n_in_cohort == 0:
            continue
        expected = rate * n_in_cohort
        # Poisson so small rates average correctly over many runs.
        delta = int(rng.poisson(lam=abs(expected)))
        if delta == 0:
            continue
        cohort_mask = ds.person[AGE_COLUMN].astype(int) == int(age)
        if rate > 0:
            donors = ds.person[cohort_mask]
            donor_weights = (
                sampling_weights[cohort_mask]
                if sampling_weights is not None
                and float(sampling_weights[cohort_mask].sum()) > 0
                else None
            )
            chosen = donors.sample(
                n=min(delta, len(donors)),
                random_state=int(rng.integers(2**31 - 1)),
                replace=True,
                weights=donor_weights,
            )
            immigrate_donor_ids.extend(chosen["person_id"].tolist())
        else:
            leavers = ds.person[cohort_mask]
            chosen = leavers.sample(
                n=min(delta, len(leavers)),
                random_state=int(rng.integers(2**31 - 1)),
                replace=False,
            )
            emigrate_ids.extend(chosen["person_id"].tolist())

    if emigrate_ids:
        ds.person = ds.person[
            ~ds.person["person_id"].isin(emigrate_ids)
        ].reset_index(drop=True)
        ds = prune_orphaned_entities(ds)
    if immigrate_donor_ids:
        ds = _append_immigrants(ds, immigrate_donor_ids)

    return ds


def _append_immigrants(
    ds: UKPanelDataset, donor_person_ids: list[int]
) -> UKPanelDataset:
    """Append immigrant rows cloned from donors with fresh IDs.

    Each donor becomes one new person (single-migrant modelling) in a
    fresh single-adult benunit and single-person household, inheriting
    the donor's age, sex and income attributes.
    """
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_person_id = int(person["person_id"].max()) + 1
    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    new_people: list[dict] = []
    new_benunits: list[dict] = []
    new_households: list[dict] = []

    donor_rows = person.set_index("person_id")

    for donor_id in donor_person_ids:
        donor = donor_rows.loc[donor_id]
        row = donor.to_dict()
        row["person_id"] = next_person_id
        row["person_benunit_id"] = next_bu_id
        row["person_household_id"] = next_hh_id
        new_people.append(row)

        if bu_weight_col is not None:
            new_benunits.append({"benunit_id": next_bu_id, bu_weight_col: 1.0})

        if hh_weight_col is not None:
            hh_row: dict = {
                "household_id": next_hh_id,
                hh_weight_col: 1.0,
            }
            if "region" in household.columns:
                # Seed region from the donor's original household.
                donor_hh_id = int(donor["person_household_id"])
                src = household.loc[
                    household["household_id"] == donor_hh_id, "region"
                ]
                hh_row["region"] = str(src.iloc[0]) if len(src) else ""
            new_households.append(hh_row)

        next_person_id += 1
        next_bu_id += 1
        next_hh_id += 1

    if new_people:
        person = pd.concat(
            [person, pd.DataFrame(new_people, columns=person.columns)],
            ignore_index=True,
        )
    if new_benunits:
        benunit = pd.concat(
            [
                benunit,
                pd.DataFrame(new_benunits, columns=benunit.columns),
            ],
            ignore_index=True,
        )
    if new_households:
        household = pd.concat(
            [
                household,
                pd.DataFrame(new_households, columns=household.columns),
            ],
            ignore_index=True,
        )

    ds.person = person
    ds.benunit = benunit
    ds.household = household
    return ds


# -- Rule-based employment and income transitions ----------------------
# The rule-based path implements the three first-order dynamics:
# retirement at state-pension age, wage drift, and random job loss /
# gain. When UKHLS matrices are supplied (via
# populace_dynamics.data.ukhls.load_employment_transitions) the loss /
# gain rules are replaced by empirical state draws.

DEFAULT_STATE_PENSION_AGE = 66

DEFAULT_JOB_LOSS_RATE = 0.03
DEFAULT_JOB_GAIN_RATE = 0.05
DEFAULT_WAGE_DRIFT = 0.04  # nominal growth per year (CPI + small real)


def apply_employment_transitions(
    dataset: UKPanelDataset,
    *,
    state_pension_age: int = DEFAULT_STATE_PENSION_AGE,
    job_loss_rate: float = DEFAULT_JOB_LOSS_RATE,
    job_gain_rate: float = DEFAULT_JOB_GAIN_RATE,
    wage_drift: float = DEFAULT_WAGE_DRIFT,
    ukhls_rates: (
        Mapping[tuple[str, str, str], Mapping[str, float]] | None
    ) = None,
    rng: np.random.Generator | None = None,
) -> UKPanelDataset:
    """Apply one year of labour-market transitions.

    Two modes:

    - **Rule-based** (default): retirement at SPA -> wage drift ->
      rule-based job loss / gain.
    - **UKHLS-driven** (``ukhls_rates`` supplied): each working-age
      person's four-state label is redrawn from the empirical
      age-band x sex matrix (output of
      :func:`populace_dynamics.data.ukhls.load_employment_transitions`).
      Retirement at SPA and wage drift still run.
    """
    if rng is None:
        rng = np.random.default_rng()

    ds = dataset.copy()

    person = ds.person
    ages = person[AGE_COLUMN].astype(int).to_numpy()

    emp_col = "employment_income"
    self_col = (
        "self_employment_income"
        if "self_employment_income" in person.columns
        else None
    )
    status_col = (
        "employment_status" if "employment_status" in person.columns else None
    )

    has_labour_income = (
        person[emp_col] > 0
        if emp_col in person.columns
        else pd.Series([False] * len(person))
    )
    if self_col is not None:
        has_labour_income = has_labour_income | (person[self_col] > 0)

    # --- 1. Retirement at SPA ----------------------------------------
    reaching_spa = ages >= state_pension_age
    if emp_col in person.columns:
        person.loc[reaching_spa, emp_col] = 0.0
    if self_col is not None:
        person.loc[reaching_spa, self_col] = 0.0
    if status_col is not None:
        person.loc[reaching_spa, status_col] = "RETIRED"

    # --- 2. Wage drift for remaining workers -------------------------
    active_workers = (ages < state_pension_age) & (
        has_labour_income.to_numpy()
    )
    if emp_col in person.columns:
        person.loc[active_workers, emp_col] = person.loc[
            active_workers, emp_col
        ] * (1.0 + wage_drift)
    if self_col is not None:
        person.loc[active_workers, self_col] = person.loc[
            active_workers, self_col
        ] * (1.0 + wage_drift)

    # --- 3. Labour-market transitions --------------------------------
    working_age = (ages >= 18) & (ages < state_pension_age)
    employed_mask = working_age & has_labour_income.to_numpy()
    unemployed_mask = working_age & ~has_labour_income.to_numpy()

    if ukhls_rates is not None:
        _apply_ukhls_employment_transitions(
            person=person,
            ages=ages,
            state_pension_age=state_pension_age,
            ukhls_rates=ukhls_rates,
            emp_col=emp_col,
            self_col=self_col,
            status_col=status_col,
            rng=rng,
        )
    else:
        if job_loss_rate > 0 and employed_mask.any():
            draws = rng.random(size=len(person))
            loses = employed_mask & (draws < job_loss_rate)
            if emp_col in person.columns:
                person.loc[loses, emp_col] = 0.0
            if self_col is not None:
                person.loc[loses, self_col] = 0.0
            if status_col is not None:
                person.loc[loses, status_col] = "UNEMPLOYED"

        if (
            job_gain_rate > 0
            and unemployed_mask.any()
            and emp_col in person.columns
        ):
            draws = rng.random(size=len(person))
            gains = unemployed_mask & (draws < job_gain_rate)
            if gains.any():
                currently_employed = person[
                    (person[emp_col] > 0) & (ages < state_pension_age)
                ]
                if not currently_employed.empty:
                    age_values = (
                        currently_employed[AGE_COLUMN].astype(int).to_numpy()
                    )
                    income_values = currently_employed[emp_col].to_numpy()
                    gainer_ages = (
                        person.loc[gains, AGE_COLUMN].astype(int).to_numpy()
                    )
                    new_incomes = []
                    for ga in gainer_ages:
                        diffs = np.abs(age_values - ga)
                        best = np.where(diffs == diffs.min())[0]
                        chosen = int(rng.choice(best))
                        new_incomes.append(float(income_values[chosen]))
                    person.loc[gains, emp_col] = new_incomes
                    if status_col is not None:
                        person.loc[gains, status_col] = "FT_EMPLOYED"

    ds.person = person.reset_index(drop=True)
    return ds


# -- UKHLS transition application --------------------------------------

# 5-year bands matching populace_dynamics.data.ukhls._age_band.
_UKHLS_AGE_EDGES = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 121]


def _ukhls_age_band_label(age: int) -> str | None:
    if age < _UKHLS_AGE_EDGES[0] or age >= _UKHLS_AGE_EDGES[-1]:
        return None
    for lo, hi in zip(_UKHLS_AGE_EDGES, _UKHLS_AGE_EDGES[1:]):
        if lo <= age < hi:
            return f"{lo}-{hi - 1}"
    return None


def _current_four_state(
    employment_income: float,
    has_self_employment: bool,
    age: int,
    state_pension_age: int,
) -> str:
    """Classify current labour-market state into the four-state enum."""
    if age >= state_pension_age:
        return "RETIRED"
    if employment_income > 0 or has_self_employment:
        return "IN_WORK"
    if 18 <= age < state_pension_age:
        return "UNEMPLOYED"
    return "INACTIVE"


def _apply_ukhls_employment_transitions(
    *,
    person: pd.DataFrame,
    ages: np.ndarray,
    state_pension_age: int,
    ukhls_rates: Mapping[tuple[str, str, str], Mapping[str, float]],
    emp_col: str,
    self_col: str | None,
    status_col: str | None,
    rng: np.random.Generator,
) -> None:
    """In-place: redraw working-age states from UKHLS matrices."""
    if emp_col not in person.columns:
        return
    emp_income = person[emp_col].to_numpy(dtype=float)
    has_self = (
        (person[self_col].to_numpy(dtype=float) > 0)
        if self_col is not None
        else np.zeros(len(person), dtype=bool)
    )
    sexes = (
        person[SEX_COLUMN].astype(str).to_numpy()
        if SEX_COLUMN in person.columns
        else np.array([MALE_VALUE] * len(person), dtype=object)
    )

    # Donor pool of currently employed people for fresh IN_WORK
    # entrants' incomes.
    employed_now_mask = (emp_income > 0) & (ages < state_pension_age)
    donor_ages = ages[employed_now_mask]
    donor_incomes = emp_income[employed_now_mask]

    new_emp_income = emp_income.copy()
    new_self_income = (
        person[self_col].to_numpy(dtype=float).copy()
        if self_col is not None
        else None
    )
    new_status: list[str | None] = (
        list(person[status_col].astype(str))
        if status_col is not None
        else [None] * len(person)
    )

    for idx in range(len(person)):
        age = int(ages[idx])
        if age < 18 or age >= state_pension_age:
            continue
        band = _ukhls_age_band_label(age)
        if band is None:
            continue
        sex = str(sexes[idx])
        cur_state = _current_four_state(
            emp_income[idx], bool(has_self[idx]), age, state_pension_age
        )
        probs = ukhls_rates.get((band, sex, cur_state))
        if not probs:
            continue
        states = list(probs.keys())
        weights = np.array([probs[s] for s in states], dtype=float)
        weights = weights / weights.sum() if weights.sum() else weights
        if weights.sum() == 0:
            continue
        next_state = str(rng.choice(states, p=weights))
        if next_state == cur_state:
            continue

        if next_state == "IN_WORK":
            if donor_ages.size:
                diffs = np.abs(donor_ages - age)
                best = np.where(diffs == diffs.min())[0]
                chosen = int(rng.choice(best))
                new_emp_income[idx] = float(donor_incomes[chosen])
            if status_col is not None:
                new_status[idx] = "FT_EMPLOYED"
        elif next_state in ("UNEMPLOYED", "INACTIVE"):
            new_emp_income[idx] = 0.0
            if new_self_income is not None:
                new_self_income[idx] = 0.0
            if status_col is not None:
                new_status[idx] = (
                    "UNEMPLOYED"
                    if next_state == "UNEMPLOYED"
                    else "OTHER_INACTIVE"
                )
        elif next_state == "RETIRED":
            new_emp_income[idx] = 0.0
            if new_self_income is not None:
                new_self_income[idx] = 0.0
            if status_col is not None:
                new_status[idx] = "RETIRED"

    person[emp_col] = new_emp_income
    if new_self_income is not None and self_col is not None:
        person[self_col] = new_self_income
    if status_col is not None:
        person[status_col] = new_status


def apply_income_decile_transitions(
    dataset: UKPanelDataset,
    decile_rates: Mapping[tuple[str, str, int], Mapping[int, float]],
    rng: np.random.Generator | None = None,
    *,
    income_col: str = "employment_income",
    min_age: int = 18,
) -> UKPanelDataset:
    """Move people between income deciles via UKHLS transitions.

    For each working-age person with positive ``income_col``, assign a
    within-dataset decile (by age-band x sex), draw a destination
    decile from ``decile_rates[(age_band, sex, decile_from)]`` (output
    of :func:`populace_dynamics.data.ukhls.load_income_decile_transitions`),
    and rescale income by the ratio of destination to origin decile
    medians. Suppressed (missing) cells pass through unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not decile_rates:
        return dataset

    ds = dataset.copy()
    person = ds.person
    if income_col not in person.columns:
        return ds

    ages = person[AGE_COLUMN].astype(int).to_numpy()
    sexes = (
        person[SEX_COLUMN].astype(str).to_numpy()
        if SEX_COLUMN in person.columns
        else np.array([MALE_VALUE] * len(person), dtype=object)
    )
    income = person[income_col].to_numpy(dtype=float)

    eligible = (ages >= min_age) & (income > 0)
    if not eligible.any():
        return ds

    age_bands = np.array(
        [
            _ukhls_age_band_label(int(a)) if eligible[i] else None
            for i, a in enumerate(ages)
        ],
        dtype=object,
    )

    # Rank into deciles within each (age_band, sex) cell among
    # eligibles so current decile is defined on a comparable
    # population.
    cell_median: dict[tuple[str, str, int], float] = {}
    current_decile = np.full(len(person), fill_value=-1, dtype=int)
    df = pd.DataFrame(
        {
            "age_band": age_bands,
            "sex": sexes,
            "income": income,
            "eligible": eligible,
        }
    )
    for (ab, sx), group in df[df["eligible"]].groupby(
        ["age_band", "sex"], observed=True
    ):
        if ab is None:
            continue
        ranks = group["income"].rank(method="first")
        # duplicates="drop" can leave NaN labels in degenerate cells
        # (fewer distinct ranks than bins); those people pass through
        # unchanged via the d_from < 1 guard below.
        deciles = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
        deciles = (deciles + 1).fillna(-1).astype(int)
        current_decile[group.index] = deciles.to_numpy()
        for d in range(1, 11):
            cell_sub = group.loc[deciles == d, "income"]
            if len(cell_sub):
                cell_median[(ab, sx, d)] = float(cell_sub.median())

    for idx in range(len(person)):
        if not eligible[idx]:
            continue
        ab = age_bands[idx]
        if ab is None:
            continue
        sx = str(sexes[idx])
        d_from = int(current_decile[idx])
        if d_from < 1:
            continue
        probs = decile_rates.get((ab, sx, d_from))
        if not probs:
            continue
        dests = [int(d) for d in probs.keys()]
        weights = np.array([probs[d] for d in dests], dtype=float)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        d_to = int(rng.choice(dests, p=weights))
        if d_to == d_from:
            continue

        target = cell_median.get((ab, sx, d_to))
        if target is None or target <= 0:
            continue
        origin = cell_median.get((ab, sx, d_from))
        if origin is None or origin <= 0:
            continue
        person.loc[person.index[idx], income_col] = float(income[idx]) * (
            target / origin
        )

    return ds
