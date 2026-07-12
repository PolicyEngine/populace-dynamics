"""CPS-observable cross-sectional moments on the certified populace frame.

The **deployment frame** for the W1 transport gate (#113 M5, #100 W1): the
certified populace US file onto which the PSID-estimated generators are
deployed. This module computes the **family-A** targets -- the
cross-sectional joints the certified file already pins, which a faithful
transport must reproduce as its deployed/terminal cross-section:

* ``earnings_participation`` / ``earnings_profile`` / ``earnings_p90p50`` /
  ``earnings_p50p10`` -- the earnings x age x sex distribution
  (participation share, the age-earnings hump, within-cell dispersion);
* ``marital_share`` -- marital composition x age x sex;
* ``hh_size_share`` / ``coresident_spouse`` -- household composition.

Every cell is a rate expressed as ``num_wt / den_wt`` (a weighted share or a
weighted-quantile ratio), so the symmetric scale-free ``|ln(rate_a/rate_b)|``
half-split statistic -- identical in shape to the locked gate-2a/2b/2c
statistic -- applies uniformly. :func:`reference_moments` is a pure function
of a person frame and is unit-testable on synthetic rows;
:func:`load_certified_persons` is the only function that touches the h5 (it
imports pandas/pytables lazily so importing this module stays light).

The frame is the L0-sparse-refit certified default (:data:`CERTIFIED_PIN`,
obtained from ``policyengine.py``'s ``pe.us.model.release_bundle``); moments
are built only on columns verified populated (the sparse file zeroes
untargeted inputs -- :func:`assert_columns_populated` fails loudly if a
future file zeroes one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# The certified deployment frame -- pinned from pe.us.model.release_bundle
# (policyengine 4.18.8; captured 2026-07-12). The build resolves this exact
# artifact by revision + sha256; the pin is the estimand's identity.
# --------------------------------------------------------------------------
CERTIFIED_PIN: dict[str, str] = {
    "bundle_id": "us-4.18.8",
    "country_id": "us",
    "policyengine_version": "4.18.8",
    "model_package": "policyengine-us",
    "model_version": "1.752.2",
    "data_package": "populace-data",
    "data_version": "0.1.0",
    "hf_repo_id": "policyengine/populace-us",
    "hf_filename": "populace_us_2024.h5",
    "hf_repo_type": "dataset",
    "dataset": "populace_us_2024",
    "revision": (
        "populace-us-2024-sparse-l0-refit-57k-71a0887-national-only-20260701"
    ),
    "artifact_sha256": (
        "c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5"
    ),
    "reference_period": "2024",
}

# CPS A_MARITL recode -> collapsed marital status (Census ASEC codes).
MARITAL_MAP: dict[int, str] = {
    1: "married",  # married, civilian spouse present
    2: "married",  # married, Armed Forces spouse present
    3: "married",  # married, spouse absent (exc. separated)
    4: "widowed",
    5: "divorced",
    6: "separated",
    7: "never_married",
}
MARITAL_STATUSES = (
    "married",
    "widowed",
    "divorced",
    "separated",
    "never_married",
)
#: Spouse-present codes (living-with-a-spouse indicator).
SPOUSE_PRESENT_CODES = (1, 2)

#: The source columns family A reads, with a plausible-support floor. The
#: sparse-57k default zeroes untargeted inputs; a future file that zeroes one
#: of these must fail loudly, not silently gate a degenerate moment.
REQUIRED_SOURCE_COLUMNS: dict[str, float] = {
    "age": 0.90,
    "is_female": 0.40,
    "employment_income_before_lsr": 0.30,
    "A_MARITL": 0.99,
    "person_household_id": 0.99,
}

# Age bands (lower, upper inclusive). Named so the derivations tier pins them.
EARN_BANDS: tuple[tuple[int, int], ...] = (
    (18, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 61),
    (62, 69),
)
DISPERSION_BANDS: tuple[tuple[int, int], ...] = (
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 61),
)
ADULT_BANDS: tuple[tuple[int, int], ...] = (
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 200),
)
#: The reference cell for the age-earnings profile ratio (both sexes).
PROFILE_REF_BAND: tuple[int, int] = (35, 44)
HH_SIZE_CATEGORIES = ("1", "2", "3", "4", "5plus")


def _band_label(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 200 else f"{lo}-{hi}"


def _sex_label(is_female: bool) -> str:
    return "female" if is_female else "male"


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Interpolated weighted quantile (type-7 style on the weight CDF)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


@dataclass(frozen=True)
class _Cell:
    num_wt: float
    den_wt: float
    n_events: int
    rate_unweighted: float


def _share_cell(
    frame: pd.DataFrame, mask: np.ndarray, weight: np.ndarray
) -> _Cell:
    """A weighted share: num = sum(w[mask]), den = sum(w), events = |mask|."""
    den = float(weight.sum())
    num = float(weight[mask].sum())
    n = int(mask.sum())
    unw = float(mask.mean()) if mask.size else 0.0
    return _Cell(num_wt=num, den_wt=den, n_events=n, rate_unweighted=unw)


def _ratio_cell(
    num_value: float, den_value: float, n_events: int, unweighted: float
) -> _Cell:
    """A weighted-quantile ratio expressed as num_wt / den_wt."""
    return _Cell(
        num_wt=float(num_value),
        den_wt=float(den_value),
        n_events=int(n_events),
        rate_unweighted=float(unweighted),
    )


def _emit(cells: dict[str, dict[str, Any]], key: str, cell: _Cell) -> None:
    rate = cell.num_wt / cell.den_wt if cell.den_wt else 0.0
    cells[key] = {
        "rate": float(rate),
        "rate_unweighted": float(cell.rate_unweighted),
        "num_wt": float(cell.num_wt),
        "den_wt": float(cell.den_wt),
        "n_events": int(cell.n_events),
    }


def reference_moments(
    persons: pd.DataFrame, weighted: bool = True
) -> dict[str, dict[str, Any]]:
    """Family-A CPS-observable cross-sectional moments on a person frame.

    ``persons`` carries one row per person with columns ``weight``, ``age``,
    ``is_female`` (bool), ``earnings`` (labor earnings, before LSR),
    ``marital_status`` (one of :data:`MARITAL_STATUSES` or NaN for
    non-adults), ``hh_size`` (int), ``coresident_spouse`` (bool). Every
    returned cell is ``rate == num_wt / den_wt``.
    """
    age = persons["age"].to_numpy(dtype=np.float64)
    female = persons["is_female"].to_numpy(dtype=bool)
    earn = persons["earnings"].to_numpy(dtype=np.float64)
    w = (
        persons["weight"].to_numpy(dtype=np.float64)
        if weighted
        else np.ones(len(persons), dtype=np.float64)
    )
    cells: dict[str, dict[str, Any]] = {}

    # --- earnings participation + profile + dispersion, by age band x sex ---
    ref_lo, ref_hi = PROFILE_REF_BAND
    ref_mask = (age >= ref_lo) & (age <= ref_hi) & (earn > 0)
    ref_median = (
        weighted_quantile(earn[ref_mask], w[ref_mask], 0.5)
        if ref_mask.any()
        else float("nan")
    )

    for lo, hi in EARN_BANDS:
        band = _band_label(lo, hi)
        in_band = (age >= lo) & (age <= hi)
        for is_f in (True, False):
            sex = _sex_label(is_f)
            cell_mask = in_band & (female == is_f)
            if not cell_mask.any():
                continue
            wc = w[cell_mask]
            earner = earn[cell_mask] > 0
            # participation
            _emit(
                cells,
                f"earnings_participation.{band}|{sex}",
                _share_cell(persons, earner, wc),
            )
            # profile + dispersion among earners
            e = earn[cell_mask][earner]
            we = wc[earner]
            n_earn = int(earner.sum())
            if n_earn > 0 and ref_median == ref_median and ref_median > 0:
                med = weighted_quantile(e, we, 0.5)
                _emit(
                    cells,
                    f"earnings_profile.{band}|{sex}",
                    _ratio_cell(med, ref_median, n_earn, med / ref_median),
                )
            if (lo, hi) in DISPERSION_BANDS and n_earn > 0:
                p90 = weighted_quantile(e, we, 0.9)
                p50 = weighted_quantile(e, we, 0.5)
                p10 = weighted_quantile(e, we, 0.1)
                if p50 > 0:
                    _emit(
                        cells,
                        f"earnings_p90p50.{band}|{sex}",
                        _ratio_cell(p90, p50, n_earn, p90 / p50),
                    )
                if p10 > 0:
                    _emit(
                        cells,
                        f"earnings_p50p10.{band}|{sex}",
                        _ratio_cell(p50, p10, n_earn, p50 / p10),
                    )

    # --- marital composition x age x sex ---
    status = persons["marital_status"].to_numpy(dtype=object)
    for lo, hi in ADULT_BANDS:
        band = _band_label(lo, hi)
        in_band = (age >= lo) & (age <= hi)
        for is_f in (True, False):
            sex = _sex_label(is_f)
            base = (
                in_band
                & (female == is_f)
                & pd.notna(persons["marital_status"]).to_numpy()
            )
            if not base.any():
                continue
            wc = w[base]
            st = status[base]
            for s in MARITAL_STATUSES:
                _emit(
                    cells,
                    f"marital_share.{s}.{band}|{sex}",
                    _share_cell(persons, st == s, wc),
                )

    # --- household composition ---
    hh = persons["hh_size"].to_numpy()
    for cat in HH_SIZE_CATEGORIES:
        if cat == "5plus":
            mask = hh >= 5
        else:
            mask = hh == int(cat)
        _emit(cells, f"hh_size_share.{cat}", _share_cell(persons, mask, w))

    cores = persons["coresident_spouse"].to_numpy(dtype=bool)
    for lo, hi in ADULT_BANDS:
        band = _band_label(lo, hi)
        in_band = (age >= lo) & (age <= hi)
        for is_f in (True, False):
            sex = _sex_label(is_f)
            base = in_band & (female == is_f)
            if not base.any():
                continue
            _emit(
                cells,
                f"coresident_spouse.{band}|{sex}",
                _share_cell(persons, cores[base], w[base]),
            )

    return cells


def cell_families() -> dict[str, str]:
    """The prefix -> human description map (documented in the artifact)."""
    return {
        "earnings_participation": (
            "weighted share with positive labor earnings, by age band x sex"
        ),
        "earnings_profile": (
            "weighted median earnings / prime-age (35-44, both sexes) median "
            "-- the age-earnings profile, by age band x sex"
        ),
        "earnings_p90p50": (
            "within-cell weighted p90/p50 earnings ratio (upper dispersion)"
        ),
        "earnings_p50p10": (
            "within-cell weighted p50/p10 earnings ratio (lower dispersion)"
        ),
        "marital_share": (
            "weighted share in each marital status, by age band x sex"
        ),
        "hh_size_share": (
            "person-level weighted share by household size (1/2/3/4/5+)"
        ),
        "coresident_spouse": (
            "weighted share living with a spouse (A_MARITL spouse-present), "
            "by age band x sex"
        ),
    }


def assert_columns_populated(person_table: pd.DataFrame) -> dict[str, float]:
    """Fail loudly if a required source column is (near-)all-zero.

    The certified default is the L0-sparse-refit file, which zeroes
    untargeted inputs; family A must never build a moment on a zeroed column
    (that would silently gate a degenerate distribution). Returns the observed
    non-zero fractions for the artifact's populated-columns disclosure.
    """
    fractions: dict[str, float] = {}
    for col, floor in REQUIRED_SOURCE_COLUMNS.items():
        if col not in person_table.columns:
            raise ValueError(f"certified frame missing column: {col}")
        v = person_table[col].fillna(0).to_numpy()
        frac = float((v != 0).mean())
        fractions[col] = frac
        if frac < floor:
            raise ValueError(
                f"column {col} non-zero fraction {frac:.3f} below floor "
                f"{floor:.2f} -- the sparse frame appears to zero it; refusing "
                "to build a degenerate moment"
            )
    return fractions


def load_certified_persons(
    h5_path: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Read the certified populace h5 into the family-A person frame.

    Returns ``(persons, populated_fractions)``. Reads the ``/person`` and
    ``/household`` entity tables with pandas (pytables); the person weight is
    the person's household weight (the PolicyEngine calibrated weight). Imports
    are local so importing this module needs neither pytables nor the h5.
    """
    with pd.HDFStore(h5_path, mode="r") as store:
        person = store["/person"]
        household = store["/household"]

    fractions = assert_columns_populated(person)

    hw = household.set_index("household_id")["household_weight"]
    weight = person["person_household_id"].map(hw).to_numpy(dtype=np.float64)
    hh_counts = person["person_household_id"].value_counts()
    hh_size = person["person_household_id"].map(hh_counts).to_numpy()

    maritl = person["A_MARITL"].astype(int)
    marital_status = maritl.map(MARITAL_MAP)
    coresident_spouse = maritl.isin(SPOUSE_PRESENT_CODES).to_numpy()

    earnings = (
        person["employment_income_before_lsr"].fillna(0.0)
        + person.get(
            "self_employment_income_before_lsr",
            pd.Series(0.0, index=person.index),
        ).fillna(0.0)
    ).to_numpy(dtype=np.float64)

    persons = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "household_id": person["person_household_id"].to_numpy(),
            "weight": weight,
            "age": person["age"].to_numpy(dtype=np.float64),
            "is_female": person["is_female"].astype(bool).to_numpy(),
            "earnings": earnings,
            "marital_status": marital_status.to_numpy(dtype=object),
            "hh_size": hh_size,
            "coresident_spouse": coresident_spouse,
        }
    )
    return persons, fractions


def synthetic_person_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A deterministic synthetic person frame for unit tests (no h5)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=n)
    is_female = rng.random(n) < 0.5
    # earnings decline with a hump around 45; ~60% participate under 62.
    participate = rng.random(n) < np.where(age < 62, 0.7, 0.2)
    peak = 60000 * np.exp(-((age - 45) ** 2) / (2 * 15**2))
    earnings = np.where(
        participate, np.maximum(peak * rng.lognormal(0, 0.4, n), 1.0), 0.0
    )
    codes = rng.choice(
        [1, 4, 5, 6, 7], size=n, p=[0.55, 0.05, 0.15, 0.05, 0.20]
    )
    codes = np.where(age < 25, 7, codes)
    household_id = rng.integers(0, max(2, n // 2), size=n)
    hh_counts = pd.Series(household_id).value_counts()
    return pd.DataFrame(
        {
            "person_id": np.arange(n),
            "household_id": household_id,
            "weight": rng.uniform(500, 5000, size=n),
            "age": age.astype(float),
            "is_female": is_female,
            "earnings": earnings,
            "marital_status": pd.Series(codes)
            .map(MARITAL_MAP)
            .to_numpy(object),
            "hh_size": pd.Series(household_id).map(hh_counts).to_numpy(),
            "coresident_spouse": np.isin(codes, SPOUSE_PRESENT_CODES),
        }
    )
