"""Build the mortality foundation floors: PSID hazards vs NCHS + noise.

REPORTED ANCHOR, NOT A GATE RUN. Like the other ``runs/`` floors, this
reads no gate and, on its own, decides nothing; it is committed evidence
pinned by a reproduction test. It is the data + floors + sd basis that a
future differential-mortality gate (issue #74 Phase B) would derive its
pre-registered thresholds from -- the ceremony (floors -> thresholds
with machine-bound derivations -> adversarial referee round ->
verification -> ratification) comes AFTER, exactly as issue #74's
pre-registered bar (comment 4907496891) requires. The proposed
thresholds are recorded here (``proposed_thresholds_note``) explicitly
marked NOT RATIFIED.

Two products, both on PSID person-interval exposure:

(a) **External anchor -- PSID vs NCHS.** Weighted PSID central death
    rates by age band x sex, compared to the NCHS 2023 US period life
    tables aggregated to the same bands. The PSID/NCHS ratio is reported
    per band x sex and per exposure window; PSID mortality undercount is
    a known literature fact, so the ratios are REPORTED honestly, never
    calibrated away.

(b) **Internal noise floor.** A person-disjoint 50/50 half-split
    (``split_panel_by_person``, ``fraction=0.5``, seeds 0-4 -- the
    harness ``noise_floor`` machinery) of the SAME band x sex hazard
    statistics, giving the across-seed mean/sd (the ``|log hazard
    ratio|`` between two independent real halves) a future gate would
    turn into thresholds as ``round(mean + k * sd)`` -- the
    ``tests/test_gates_derivations.py`` convention.

Exposure construction (person-interval, single-year slices)
-----------------------------------------------------------
Exposure is built from :func:`populace_dynamics.data.panels.demographic_panel`
-- the population layer that :func:`family.family_earnings_panel` itself
draws on -- rather than the prime-age 25-59 earnings filter, because
mortality differentials concentrate at ages outside the earnings window
(the 80-85 retiree rows anchor 1 targets). The earnings panel is a
prime-age subset of this same population machinery.

The PSID wave calendar is annual 1969-1997 then biennial 1999-2023, so
the grid gap after an observed wave is 1 year (through 1996), or 2 years
(1997 and every biennial wave). For each person observed present in a
responding family unit (sequence 1-20, valid age, positive weight) at
wave ``w`` with the next grid wave ``w'``:

* the person is credited ``w' - w`` single-year exposure slices at ages
  ``age_w, age_w + 1, ...`` (aged forward through the interval);
* a death is counted iff the person's EXACT year of death ``d`` (from
  :mod:`populace_dynamics.data.deaths`) satisfies ``w <= d < w'`` -- i.e.
  the person was observed alive at the start of the interval containing
  their death. The slice at year ``d`` carries exposure 0.5 (mid-year
  death convention) and one death event; later slices in the interval
  carry nothing (the person is dead).

Biennial-panel caveats, documented not hidden:

1. **Deaths only in the immediately-following interval.** A death is
   counted only when it falls in the one grid interval after an observed
   wave. A person who attrites and then dies two or more intervals later
   is right-censored at their last observed wave and their death is NOT
   counted -- a real ascertainment gap that is part of the honest PSID
   undercount, not corrected here.
2. **Start-wave weight and sex.** Each slice carries the person's weight
   at the interval's start wave and their constant sex; the biennial
   interval's two slices share that start-wave weight.
3. **Age at start.** Slices age forward within an interval, but the
   band is assigned per slice-age, so a 2-year interval can straddle a
   band boundary; attribution is by the slice's own age.
4. **Period pooling.** The ``all`` window pools 1969-2022 interval
   deaths against the NCHS 2023 period table (a named
   population/period-concept delta): historical mortality was higher, so
   pooling OLDER decades biases the PSID rate UPWARD relative to 2023 --
   working AGAINST the undercount. A ``recent`` window (intervals
   starting 2011+) is reported alongside to bracket that period effect.

Run from the repository root with the PSID individual file staged::

    .venv/bin/python scripts/build_mortality_floors.py

It needs no populace-fit (real-vs-real / real-vs-external only).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import deaths, panels
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "mortality_floors_v1.json"
NCHS_PATH = ROOT / "data" / "external" / "nchs_life_tables_2023.json"
ARTIFACT_SCHEMA_VERSION = "mortality_floors.v1"

#: Age bands (closed single-year bounds; the top band is open at
#: MAX_AGE). Ten-year adult bands spanning the working-age-through-elderly
#: range the Social Security reforms concern (including the 75-84 and 85+
#: retiree bands anchor 1's 80-85 rows need).
BANDS: tuple[tuple[int, int], ...] = (
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)
MAX_AGE = 120
SEXES = ("male", "female")
SEEDS = (0, 1, 2, 3, 4)

#: Intervals whose START wave is >= this year form the "recent" window
#: that brackets the period effect against the NCHS 2023 period table.
RECENT_START_YEAR = 2011

#: Minimum death count (on the weaker half, worst seed) for a band x sex
#: cell to be proposed as gate-eligible. 20 is NCHS's own reliability
#: floor: NCHS flags death rates based on fewer than 20 deaths as
#: unreliable (Technical Notes to the mortality reports). Cells below it
#: are report-only, the mortality analog of the PIA floor's d1/d2.
MIN_DEATHS_FOR_GATE = 20


def band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < MAX_AGE else f"{lo}+"


BAND_LABELS: tuple[str, ...] = tuple(band_label(lo, hi) for lo, hi in BANDS)


def _band_of(age: int) -> str | None:
    for lo, hi in BANDS:
        if lo <= age <= hi:
            return band_label(lo, hi)
    return None


def _key(band: str, sex: str) -> str:
    """Flat band x sex key used across the artifact (``"75-84|male"``)."""
    return f"{band}|{sex}"


# --------------------------------------------------------------------------
# Person-interval exposure slices
# --------------------------------------------------------------------------
def build_exposure_slices(
    demo: pd.DataFrame, death_records: pd.DataFrame
) -> pd.DataFrame:
    """Single-year exposure/death slices per observed person-interval.

    See the module docstring for the construction. Returns a frame with
    one row per single-year slice: ``person_id``, ``sex``, ``weight``,
    ``age``, ``band``, ``exposure`` (1.0, or 0.5 for the death year),
    ``death`` (1.0 in the death-year slice, else 0.0), and ``start_wave``
    (the observed wave anchoring the interval, for windowing).
    """
    grid = sorted(int(w) for w in demo.period.unique())
    next_wave = {w: grid[i + 1] for i, w in enumerate(grid[:-1])}

    obs = demo[(demo.age <= MAX_AGE) & (demo.weight > 0)].copy()
    sex_by_person = death_records.set_index("person_id")["sex"]
    dyear_by_person = death_records.set_index("person_id")["death_year"]
    obs["sex"] = obs.person_id.map(sex_by_person)
    obs["dyear"] = obs.person_id.map(dyear_by_person)  # Int64, NA if not exact
    obs = obs[obs.sex.isin(SEXES)].copy()
    obs["next_wave"] = obs.period.map(next_wave)
    obs = obs[obs.next_wave.notna()].copy()  # drop the terminal wave
    obs["next_wave"] = obs.next_wave.astype(int)
    obs["length"] = obs.next_wave - obs.period

    w = obs.period.to_numpy(dtype=np.int64)
    nxt = obs.next_wave.to_numpy(dtype=np.int64)
    d = obs.dyear.astype("float64").to_numpy()  # NaN where not exact
    has_death = obs.dyear.notna().to_numpy()
    death_in_interval = has_death & (d >= w) & (d < nxt)

    frames: list[pd.DataFrame] = []
    # Slice 0 (age at wave, calendar year w).
    s0_death = death_in_interval & (d == w)
    frames.append(
        pd.DataFrame(
            {
                "person_id": obs.person_id.to_numpy(),
                "sex": obs.sex.to_numpy(),
                "weight": obs.weight.to_numpy(dtype=np.float64),
                "age": obs.age.to_numpy(dtype=np.int64),
                "start_wave": w,
                "exposure": np.where(s0_death, 0.5, 1.0),
                "death": s0_death.astype(np.float64),
            }
        )
    )
    # Slice 1 (only for 2-year intervals where the person did not die in
    # slice 0): age + 1, calendar year w + 1.
    two = (obs.length.to_numpy() == 2) & ~s0_death
    if two.any():
        d1 = d[two]
        w1 = w[two]
        s1_death = death_in_interval[two] & (d1 == (w1 + 1))
        frames.append(
            pd.DataFrame(
                {
                    "person_id": obs.person_id.to_numpy()[two],
                    "sex": obs.sex.to_numpy()[two],
                    "weight": obs.weight.to_numpy(dtype=np.float64)[two],
                    "age": obs.age.to_numpy(dtype=np.int64)[two] + 1,
                    "start_wave": w1,
                    "exposure": np.where(s1_death, 0.5, 1.0),
                    "death": s1_death.astype(np.float64),
                }
            )
        )

    slices = pd.concat(frames, ignore_index=True)
    slices["band"] = slices.age.map(_band_of)
    slices = slices[slices.band.notna()].reset_index(drop=True)
    return slices


def weighted_hazards(slices: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Weighted central death rate ``m = sum(w*death)/sum(w*exposure)``.

    Returns ``{band|sex: {"psid_m", "psid_exposure_py", "psid_deaths_wt",
    "psid_deaths_unwt"}}`` for every band x sex with any exposure.
    """
    df = slices.copy()
    df["we"] = df.weight * df.exposure
    df["wd"] = df.weight * df.death
    grouped = df.groupby(["band", "sex"], observed=True).agg(
        we=("we", "sum"),
        wd=("wd", "sum"),
        n_death_unwt=("death", "sum"),
    )
    out: dict[str, dict[str, float]] = {}
    for (band, sex), row in grouped.iterrows():
        out[_key(band, sex)] = {
            "psid_m": float(row.we and row.wd / row.we),
            "psid_exposure_py": float(row.we),
            "psid_deaths_wt": float(row.wd),
            "psid_deaths_unwt": int(round(row.n_death_unwt)),
        }
    return out


# --------------------------------------------------------------------------
# NCHS band central rates (from the committed reference)
# --------------------------------------------------------------------------
def nchs_band_rates(nchs: dict[str, Any]) -> dict[str, float]:
    """NCHS life-table central death rate per band x sex.

    For band ``[a, b]`` the life-table central rate is
    ``(l_a - l_{b+1}) / (T_a - T_{b+1})`` -- deaths in the band over
    person-years lived in the band. For the open top band ``[85, .]``,
    ``b + 1`` is beyond the terminal age, so ``l_{b+1} = T_{b+1} = 0``
    and the rate is ``l_85 / T_85`` (deaths and person-years of the whole
    open tail). This is exactly comparable to the PSID central rate.
    """
    rates: dict[str, float] = {}
    for sex in SEXES:
        rows = {r["age"]: r for r in nchs["tables"][sex]}
        lx = {a: rows[a]["lx"] for a in rows}
        tx = {a: rows[a]["Tx"] for a in rows}
        for lo, hi in BANDS:
            deaths_b = lx[lo] - lx.get(hi + 1, 0.0)
            py_b = tx[lo] - tx.get(hi + 1, 0.0)
            rates[_key(band_label(lo, hi), sex)] = deaths_b / py_b
    return rates


# --------------------------------------------------------------------------
# External anchor (PSID vs NCHS) per exposure window
# --------------------------------------------------------------------------
def external_anchor(
    slices: pd.DataFrame,
    nchs_rates: dict[str, float],
    *,
    start_year_min: int | None,
) -> dict[str, Any]:
    """PSID-vs-NCHS ratio table for one exposure window.

    ``start_year_min`` restricts to intervals whose start wave is >= it
    (``None`` = all intervals). The ratio ``psid_m / nchs_M`` is reported
    honestly per band x sex; < 1 is the mortality undercount.
    """
    window = slices
    if start_year_min is not None:
        window = slices[slices.start_wave >= start_year_min]
    haz = weighted_hazards(window)

    by_band_sex: dict[str, Any] = {}
    ratios_by_band_all_sex: list[float] = []
    for band in BAND_LABELS:
        for sex in SEXES:
            key = _key(band, sex)
            psid = haz.get(key)
            nchs_m = nchs_rates[key]
            if psid is None or psid["psid_m"] <= 0:
                by_band_sex[key] = {
                    "psid_m": None if psid is None else psid["psid_m"],
                    "psid_exposure_py": (
                        None if psid is None else psid["psid_exposure_py"]
                    ),
                    "psid_deaths_unwt": (
                        0 if psid is None else psid["psid_deaths_unwt"]
                    ),
                    "nchs_M": nchs_m,
                    "ratio": None,
                }
                continue
            ratio = psid["psid_m"] / nchs_m
            ratios_by_band_all_sex.append(ratio)
            by_band_sex[key] = {
                "psid_m": psid["psid_m"],
                "psid_exposure_py": psid["psid_exposure_py"],
                "psid_deaths_wt": psid["psid_deaths_wt"],
                "psid_deaths_unwt": psid["psid_deaths_unwt"],
                "nchs_M": nchs_m,
                "ratio": ratio,
            }

    total_exposure = float((window.weight * window.exposure).sum())
    total_deaths = int(round(float(window.death.sum())))
    return {
        "start_year_min": start_year_min,
        "n_slices": int(len(window)),
        "total_exposure_py_weighted": total_exposure,
        "total_death_events_unwt": total_deaths,
        "by_band_sex": by_band_sex,
        "ratio_summary": {
            "n_estimable_cells": len(ratios_by_band_all_sex),
            "median_ratio": (
                float(np.median(ratios_by_band_all_sex))
                if ratios_by_band_all_sex
                else None
            ),
            "min_ratio": (
                float(np.min(ratios_by_band_all_sex))
                if ratios_by_band_all_sex
                else None
            ),
            "max_ratio": (
                float(np.max(ratios_by_band_all_sex))
                if ratios_by_band_all_sex
                else None
            ),
        },
    }


# --------------------------------------------------------------------------
# Internal noise floor (person-disjoint 50/50 half-split, seeds 0-4)
# --------------------------------------------------------------------------
def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/sd/min/max and raw values (float64, ddof=1 sd), the floor
    convention shared with the other ``runs/`` artifacts."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "values": [float(v) for v in arr],
    }


def measure_seed_halfsplit(seed: int, slices: pd.DataFrame) -> dict[str, Any]:
    """One seed: split persons 50/50, band x sex hazards on each half.

    Returns the two halves' hazards and, per band x sex, the absolute
    log hazard ratio ``|ln(m_A / m_B)|`` and the absolute percent gap --
    the sampling-noise floor statistics. A band x sex with zero deaths on
    either half has an undefined ratio (``None``): denominator-fragile,
    the mortality analog of the PIA floor's d1/d2.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        slices, "person_id", fraction=0.5, seed=seed
    )
    haz_a = weighted_hazards(side_a)
    haz_b = weighted_hazards(side_b)

    cells: dict[str, Any] = {}
    for band in BAND_LABELS:
        for sex in SEXES:
            key = _key(band, sex)
            a = haz_a.get(key)
            b = haz_b.get(key)
            m_a = a["psid_m"] if a else 0.0
            m_b = b["psid_m"] if b else 0.0
            defined = m_a > 0 and m_b > 0
            cells[key] = {
                "m_a": float(m_a),
                "m_b": float(m_b),
                "n_death_a": int(a["psid_deaths_unwt"]) if a else 0,
                "n_death_b": int(b["psid_deaths_unwt"]) if b else 0,
                "log_ratio_abs": (
                    float(abs(np.log(m_a / m_b))) if defined else None
                ),
                "pct_diff_abs": (
                    float(abs(m_a - m_b) / m_b * 100.0) if defined else None
                ),
            }
    return {
        "seed": seed,
        "n_persons_side_a": int(side_a.person_id.nunique()),
        "n_persons_side_b": int(side_b.person_id.nunique()),
        "cells": cells,
    }


def pool_internal_floor(
    per_seed: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Across-seed floor per band x sex, plus the stability partition.

    Returns ``(noise_floor_seeds_0_4, band_sex_stability)``:

    * ``noise_floor_seeds_0_4[key]`` follows the committed-floor
      convention (``mean``/``sd``/``values`` of ``|log hazard ratio|``),
      so a future gate derives ``round(mean + k * sd)`` from it exactly
      as ``test_gates_derivations`` binds. Only cells DEFINED on every
      seed carry a floor block (an undefined seed has no sd basis).
    * ``band_sex_stability[key]`` records how many seeds the cell was
      defined on and the minimum half-death count, the machine-visible
      evidence for which cells a future gate can bind (defined on all
      seeds) versus report-only (denominator-fragile).
    """
    noise_floor: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for band in BAND_LABELS:
        for sex in SEXES:
            key = _key(band, sex)
            log_ratios = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
            pct_diffs = [s["cells"][key]["pct_diff_abs"] for s in per_seed]
            min_deaths = min(
                min(s["cells"][key]["n_death_a"], s["cells"][key]["n_death_b"])
                for s in per_seed
            )
            defined_seeds = sum(v is not None for v in log_ratios)
            stability[key] = {
                "defined_seeds": defined_seeds,
                "n_seeds": len(per_seed),
                "min_deaths_either_half": int(min_deaths),
                "gate_eligible": (
                    defined_seeds == len(per_seed)
                    and min_deaths >= MIN_DEATHS_FOR_GATE
                ),
            }
            if defined_seeds == len(per_seed):
                floor_block = _summary([float(v) for v in log_ratios])
                floor_block["pct_diff_abs"] = _summary(
                    [float(v) for v in pct_diffs]
                )
                noise_floor[key] = floor_block
    return noise_floor, stability


# --------------------------------------------------------------------------
# Proposed thresholds note (NOT RATIFIED)
# --------------------------------------------------------------------------
def proposed_thresholds_note(
    stability: dict[str, Any], anchor_all: dict[str, Any]
) -> str:
    gate_eligible = sorted(
        k for k, v in stability.items() if v["gate_eligible"]
    )
    report_only = sorted(
        k for k, v in stability.items() if not v["gate_eligible"]
    )
    median_ratio = anchor_all["ratio_summary"]["median_ratio"]
    return (
        "PROPOSED VALIDATION STANDARD FOR THE DIFFERENTIAL-MORTALITY "
        "COMPONENT -- NOT RATIFIED.\n\n"
        "This is a proposal for the future gate ceremony (issue #74 "
        "Phase B), recorded so the standard exists BEFORE any scored use, "
        "per issue #74 comment 4907496891. It changes no locked value and "
        "no model reads it. Ratification requires the full lock ceremony: "
        "floors -> thresholds with machine-bound derivations -> an "
        "adversarial referee round -> verification -> maintainer "
        "ratification by merge. The k values below are PLACEHOLDERS for "
        "that ceremony to set, not settled numbers.\n\n"
        "STATISTIC. Per age band x sex, the weighted PSID central death "
        "rate m(band, sex) from the person-interval exposure in this "
        "artifact, and a synthetic candidate's m(band, sex) built the "
        "same way. The candidate-vs-PSID discrepancy is scored as the "
        "absolute log hazard ratio |ln(m_candidate / m_PSID)| per cell "
        "(symmetric, scale-free for rates).\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split "
        "|log hazard ratio| floor in internal_noise_floor.noise_floor_"
        "seeds_0_4 (mean/sd across seeds 0-4). A proposed per-cell "
        "threshold is round(mean + k * sd) at the shared derivation "
        "convention (tests/test_gates_derivations.py), with a PROPOSED "
        "k in [2, 3] for the ceremony to fix.\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY CELLS. Gate only cells whose "
        "real-vs-real floor is defined on every seed AND whose weaker "
        "half carries at least 20 deaths on the worst seed -- NCHS's own "
        "reliability floor (it flags death rates based on fewer than 20 "
        f"deaths as unreliable): {gate_eligible}. Report-only "
        f"(denominator-fragile, the mortality analog of the PIA floor's "
        f"d1/d2): {report_only}. This partition is derived from the "
        "committed floor's band_sex_stability, not hand-picked.\n\n"
        "EXTERNAL ANCHOR. The NCHS 2023 US period life tables "
        "(data/external/nchs_life_tables_2023.json), aggregated to these "
        "bands. Because PSID undercounts mortality (all reported ratios "
        f"< 1; median PSID/NCHS ratio ~{median_ratio:.2f} on the all-"
        "window), the anchor must NOT gate a level match to NCHS -- that "
        "would reject reality. The proposed external standard is a SHAPE "
        "check: (i) m(band, sex) monotone non-decreasing in age within "
        "each sex; (ii) male m >= female m in every adult band; (iii) "
        "the PSID/NCHS level ratio reported per cell and its departure "
        "from 1 documented as the undercount, with a named "
        "population/period-concept delta (PSID pools 1969-2022 interval "
        "deaths; NCHS is the 2023 period). A candidate that reproduces "
        "PSID's own hazards inherits this documented offset; the anchor "
        "certifies the gradient, not the level.\n\n"
        "BASELINE CONVENTION. Mortality feeds benefit levels through "
        "survival to and beyond claiming; any scored reform must state "
        "whether it runs against the scheduled or payable baseline (issue "
        "#74 protocol note 1). This note fixes none of that; it fixes the "
        "component's estimation/validation standard only."
    )


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _sha_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(verbose: bool = True) -> dict[str, Any]:
    started = time.time()

    nchs = json.loads(NCHS_PATH.read_text())
    nchs_rates = nchs_band_rates(nchs)

    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    slices = build_exposure_slices(demo, death_records)
    n_persons = int(slices.person_id.nunique())
    if verbose:
        print(
            f"exposure: {len(slices)} slices, {n_persons} persons, "
            f"{int(slices.death.sum())} death events"
        )

    anchor_all = external_anchor(slices, nchs_rates, start_year_min=None)
    anchor_recent = external_anchor(
        slices, nchs_rates, start_year_min=RECENT_START_YEAR
    )

    per_seed = [measure_seed_halfsplit(s, slices) for s in SEEDS]
    noise_floor, stability = pool_internal_floor(per_seed)
    if verbose:
        for band in BAND_LABELS:
            for sex in SEXES:
                key = _key(band, sex)
                cell = anchor_all["by_band_sex"][key]
                fl = noise_floor.get(key)
                ratio_str = (
                    "NA" if cell["ratio"] is None else f"{cell['ratio']:.2f}"
                )
                sd_str = "NA" if fl is None else f"{fl['sd']:.3f}"
                print(
                    f"  {key:>12}: PSID_m={(cell['psid_m'] or 0):.5f} "
                    f"NCHS_M={cell['nchs_M']:.5f} ratio={ratio_str} "
                    f"floor_sd={sd_str}"
                )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "mortality_floors_v1",
        "reported_anchor_not_gated": True,
        "component": "differential mortality (issue #74 Phase B; task B1)",
        "purpose": (
            "The mortality foundation: weighted PSID death hazards by age "
            "band x sex against the NCHS 2023 US life tables (external "
            "anchor, undercount reported honestly), plus the person-"
            "disjoint half-split noise floor (the sd basis a future "
            "differential-mortality gate would derive thresholds from). "
            "This reads no gate and changes no gate on its own; the "
            "pre-registered gate ceremony (issue #74 comment 4907496891) "
            "comes after. See proposed_thresholds_note (NOT RATIFIED)."
        ),
        "data": {
            "psid_population": (
                "populace_dynamics.data.panels.demographic_panel "
                "(sequence 1-20 present persons, all ages, waves 1969-"
                "2023) -- the population layer family_earnings_panel draws "
                "on; joined to per-person sex (ER32000) and exact year of "
                "death (ER32050) from populace_dynamics.data.deaths"
            ),
            "n_persons_with_exposure": n_persons,
            "psid_wave_calendar": "annual 1969-1997, biennial 1999-2023",
            "death_record_counts": {
                "exact_year": int(
                    (death_records.death_status == "exact").sum()
                ),
                "range_coded": int(
                    (death_records.death_status == "range").sum()
                ),
                "na_year": int((death_records.death_status == "na_dk").sum()),
                "not_deceased": int(
                    (death_records.death_status == "not_deceased").sum()
                ),
                "note": (
                    "Only exact-year deaths enter the hazards; range-coded "
                    "and NA-year deaths (a small minority) are excluded and "
                    "counted here for transparency."
                ),
            },
        },
        "exposure_construction": {
            "unit": "single-year person-interval slice",
            "rule": (
                "For each person observed present (sequence 1-20, valid "
                "age, positive weight) at wave w with next grid wave w', "
                "credit w'-w single-year slices aged forward from age_w; "
                "count a death iff the exact death year d satisfies "
                "w <= d < w' (observed alive at interval start). The "
                "death-year slice carries exposure 0.5 and one death; "
                "later slices in the interval carry nothing."
            ),
            "weight": "person weight at the interval's start wave",
            "biennial_caveats": [
                "deaths counted only in the one grid interval after an "
                "observed wave; later deaths of attriters are right-"
                "censored and NOT counted (part of the honest undercount)",
                "the 'all' window pools 1969-2022 interval deaths against "
                "the NCHS 2023 period table (named period-concept delta); "
                "older decades bias PSID UPWARD, against the undercount",
                "start-wave weight shared by a biennial interval's two "
                "slices; band assigned per slice-age",
            ],
        },
        "age_bands": list(BAND_LABELS),
        "sexes": list(SEXES),
        "external_anchor": {
            "nchs_reference_file": str(NCHS_PATH.relative_to(ROOT)),
            "nchs_vintage_year": nchs["vintage_year"],
            "nchs_citation": nchs["report"]["nvsr_citation"],
            "nchs_reference_sha256": _sha_of_file(NCHS_PATH),
            "nchs_source_file_sha256": {
                pop: meta["sha256"]
                for pop, meta in nchs["fetch"]["source_files"].items()
            },
            "band_central_rate_formula": (
                "(l_a - l_{b+1}) / (T_a - T_{b+1}); open top band "
                "[85,.] = l_85 / T_85"
            ),
            "undercount_note": (
                "PSID mortality undercount is a known literature fact. "
                "Every ratio below is REPORTED as measured, not "
                "calibrated: ratio < 1 means PSID observes fewer deaths "
                "per person-year than the NCHS period population."
            ),
            "windows": {"all": anchor_all, "recent": anchor_recent},
        },
        "internal_noise_floor": {
            "method": (
                "person-disjoint 50/50 half-split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction=0.5, seeds 0-4) of the same band x sex hazard "
                "statistics on the all-window slices; the floor statistic "
                "is |ln(m_A / m_B)| between the two independent real "
                "halves -- the sampling-noise sd basis a future gate would "
                "turn into round(mean + k*sd) thresholds"
            ),
            "window": "all (every interval; start_year_min=None)",
            "seeds": list(SEEDS),
            "noise_floor_seeds_0_4": noise_floor,
            "band_sex_stability": stability,
            "per_seed": per_seed,
        },
        "proposed_thresholds_note": proposed_thresholds_note(
            stability, anchor_all
        ),
        "revision_pins": {
            "populace_dynamics_sha": _git_sha(ROOT),
            "nchs_reference_sha256": _sha_of_file(NCHS_PATH),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
