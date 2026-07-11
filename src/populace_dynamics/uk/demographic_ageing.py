"""Demographic ageing for the UK panel pipeline.

Ported from archived
`policyengine-uk-data#346 <https://github.com/PolicyEngine/policyengine-uk-data/pull/346>`_
per populace#148 / ADR 0002, onto
:class:`~populace_dynamics.uk.dataset.UKPanelDataset`.

Given a base dataset and a number of years to advance, produces a new
dataset in which:

- every surviving person's ``age`` is incremented by the step size,
- a fraction of persons die each year according to an age-indexed
  mortality table (their rows are removed),
- a fraction of women of reproductive age give birth each year
  according to an age-indexed fertility table (new person rows are
  appended, attached to the mother's benefit unit and household, with
  fresh non-colliding ``person_id`` values).

**Review fix applied** (blocker on #346's second pass): mortality now
prunes benunit and household rows that no surviving person references
(via :func:`~populace_dynamics.uk.dataset.prune_orphaned_entities`);
the source only filtered the person table, leaving fully-deceased
entities behind with live weights.

Mortality rates accept two shapes — age-only ``Mapping[int, float]``
or sex-specific ``Mapping[str, Mapping[int, float]]`` keyed by
``"MALE"`` / ``"FEMALE"``. Real ONS rates are available via
:mod:`populace_dynamics.uk.ons_rates`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from populace_dynamics.uk.dataset import (
    UKPanelDataset,
    prune_orphaned_entities,
)

__all__ = [
    "AgeOnlyRates",
    "SexSpecificRates",
    "MortalityRates",
    "DEFAULT_MORTALITY_RATES_PLACEHOLDER",
    "DEFAULT_FERTILITY_RATES_PLACEHOLDER",
    "age_dataset",
]

# Type aliases — either age-only or sex-specific age-indexed rates.
AgeOnlyRates = Mapping[int, float]
SexSpecificRates = Mapping[str, Mapping[int, float]]
MortalityRates = AgeOnlyRates | SexSpecificRates


AGE_COLUMN = "age"
SEX_COLUMN = "gender"
FEMALE_VALUE = "FEMALE"
MALE_VALUE = "MALE"

MIN_FERTILE_AGE = 15
MAX_FERTILE_AGE = 49

# Placeholder rates, NOT from ONS: shipped only so the module is
# usable end-to-end in tests and smoke-runs. Pass ons_rates output
# for real tables.
DEFAULT_MORTALITY_RATES_PLACEHOLDER: Mapping[int, float] = {
    **{age: 0.0005 for age in range(0, 15)},
    **{age: 0.001 for age in range(15, 40)},
    **{age: 0.001 * (1.08 ** (age - 40)) for age in range(40, 60)},
    **{age: min(0.5, 0.01 * (1.1 ** (age - 60))) for age in range(60, 100)},
    **{age: 0.5 for age in range(100, 121)},
}

DEFAULT_FERTILITY_RATES_PLACEHOLDER: Mapping[int, float] = {
    **{15: 0.005, 16: 0.01, 17: 0.02, 18: 0.03, 19: 0.04},
    **{age: 0.05 for age in range(20, 25)},
    **{age: 0.09 for age in range(25, 30)},
    **{age: 0.10 for age in range(30, 35)},
    **{age: 0.06 for age in range(35, 40)},
    **{age: 0.02 for age in range(40, 45)},
    **{age: 0.005 for age in range(45, 50)},
}


def age_dataset(
    base: UKPanelDataset,
    years: int,
    *,
    seed: int = 0,
    mortality_rates: MortalityRates | None = None,
    fertility_rates: Mapping[int, float] | None = None,
) -> UKPanelDataset:
    """Return a demographically-aged copy of ``base``, ``years`` forward.

    The base dataset is not mutated. One year is applied at a time so
    mortality and fertility are stochastic-independent per year.

    Args:
        base: the starting dataset.
        years: non-negative number of whole years to advance. ``0``
            returns a straight copy.
        seed: random seed. Identical seeds produce identical outputs.
        mortality_rates: per-year death probabilities, age-only or
            sex-specific (see module docstring). ``{}`` disables
            mortality; ``None`` uses the placeholder defaults. Ages
            not in the table default to 0.
        fertility_rates: age-of-mother -> per-year birth probability.
            ``{}`` disables fertility; ``None`` uses the placeholder
            defaults. Applied only to ``gender == "FEMALE"``.

    Raises:
        ValueError: if ``years`` is negative.
    """
    if years < 0:
        raise ValueError(f"years must be non-negative, got {years}.")

    if mortality_rates is None:
        mortality_rates = DEFAULT_MORTALITY_RATES_PLACEHOLDER
    if fertility_rates is None:
        fertility_rates = DEFAULT_FERTILITY_RATES_PLACEHOLDER

    rng = np.random.default_rng(seed)
    aged = base.copy()

    for _ in range(int(years)):
        aged = _apply_mortality(aged, mortality_rates, rng)
        aged = _apply_fertility(aged, fertility_rates, rng)
        aged.person[AGE_COLUMN] = aged.person[AGE_COLUMN].astype(int) + 1

    return aged


def _is_sex_specific(rates: MortalityRates) -> bool:
    """Detect whether ``rates`` is keyed by sex label or by age.

    Empty mappings are treated as age-only (disabled).
    """
    if not rates:
        return False
    sample_key = next(iter(rates))
    return isinstance(sample_key, str)


def _apply_mortality(
    dataset: UKPanelDataset,
    mortality_rates: MortalityRates,
    rng: np.random.Generator,
) -> UKPanelDataset:
    if not mortality_rates:
        return dataset
    person = dataset.person
    ages = person[AGE_COLUMN].astype(int).to_numpy()

    if _is_sex_specific(mortality_rates):
        # Unknown sex labels fall through to 0 (no mortality) so extra
        # categories do not silently crash; missing ages inside a
        # known sex block also default to 0, matching the age-only
        # path.
        if SEX_COLUMN in person.columns:
            sexes = person[SEX_COLUMN].to_numpy()
        else:
            sexes = np.array([MALE_VALUE] * len(person), dtype=object)
        rates = np.array(
            [
                float(mortality_rates.get(str(s), {}).get(int(a), 0.0))
                for a, s in zip(ages, sexes, strict=True)
            ],
            dtype=float,
        )
    else:
        rates = np.array(
            [float(mortality_rates.get(int(a), 0.0)) for a in ages],
            dtype=float,
        )

    draws = rng.random(size=ages.shape[0])
    survives = draws >= rates
    dataset.person = person.loc[survives].reset_index(drop=True)
    # Review fix: deaths must not leave benunits/households behind
    # with no surviving members.
    return prune_orphaned_entities(dataset)


def _apply_fertility(
    dataset: UKPanelDataset,
    fertility_rates: Mapping[int, float],
    rng: np.random.Generator,
) -> UKPanelDataset:
    if not fertility_rates:
        return dataset
    person = dataset.person
    if SEX_COLUMN not in person.columns:
        return dataset

    ages = person[AGE_COLUMN].astype(int).to_numpy()
    sexes = person[SEX_COLUMN].to_numpy()
    rates = np.array(
        [
            (
                fertility_rates.get(int(a), 0.0)
                if (
                    s == FEMALE_VALUE
                    and MIN_FERTILE_AGE <= a <= MAX_FERTILE_AGE
                )
                else 0.0
            )
            for a, s in zip(ages, sexes, strict=True)
        ],
        dtype=float,
    )
    draws = rng.random(size=ages.shape[0])
    gives_birth = draws < rates
    n_births = int(gives_birth.sum())
    if n_births == 0:
        return dataset

    mother_rows = person.loc[gives_birth]
    max_existing_id = int(person["person_id"].max())
    new_ids = np.arange(
        max_existing_id + 1, max_existing_id + 1 + n_births, dtype=int
    )
    # Sex ratio at birth in the UK is ~1.05 boys per girl; a 50/50
    # split is within sampling noise for this v1 placeholder. Plain
    # object strings here — pandas-extension dtypes are not valid
    # numpy dtypes; ``pd.concat`` coerces back at the end.
    new_sex = np.array(
        rng.choice([MALE_VALUE, FEMALE_VALUE], size=n_births), dtype=object
    )

    newborns = _build_newborn_rows(
        template=person,
        mother_rows=mother_rows,
        new_ids=new_ids,
        new_sex=new_sex,
    )
    dataset.person = pd.concat(
        [person, newborns], ignore_index=True
    ).reset_index(drop=True)
    return dataset


def _build_newborn_rows(
    template: pd.DataFrame,
    mother_rows: pd.DataFrame,
    new_ids: np.ndarray,
    new_sex: np.ndarray,
) -> pd.DataFrame:
    """Construct newborn person rows matching ``template``'s columns.

    Numeric columns default to 0 and object columns to empty string,
    except IDs, age, gender, and the benunit/household links inherited
    from the mother.
    """
    n = len(new_ids)
    data: dict[str, np.ndarray] = {}
    for col in template.columns:
        series = template[col]
        if pd.api.types.is_numeric_dtype(series):
            data[col] = np.zeros(n, dtype=series.dtype)
        else:
            data[col] = np.array([""] * n, dtype=object)

    data["person_id"] = new_ids.astype(template["person_id"].dtype)
    data[AGE_COLUMN] = np.zeros(n, dtype=template[AGE_COLUMN].dtype)
    data[SEX_COLUMN] = new_sex
    if "person_benunit_id" in template.columns:
        data["person_benunit_id"] = (
            mother_rows["person_benunit_id"]
            .to_numpy()
            .astype(template["person_benunit_id"].dtype)
        )
    if "person_household_id" in template.columns:
        data["person_household_id"] = (
            mother_rows["person_household_id"]
            .to_numpy()
            .astype(template["person_household_id"].dtype)
        )

    return pd.DataFrame(data, columns=template.columns)
