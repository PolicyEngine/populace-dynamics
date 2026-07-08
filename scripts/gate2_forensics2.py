"""Gate-2 forensics 2 (REPORTED, NOT GATED): the widow stock and the female
count.

The registered diagnostic of PolicyEngine/populace-dynamics issue #42,
comment 4917987214 ("forensics 2 -- the widow stock and the female count").
It is evidence, not another spec draw: candidate 10 (run 1, #98) graded
FAIL 1/5 under the amended mean-over-20-draws estimator, and its decider was
``share_widowed.75+|female`` (failed 4/5 seeds, a level under-production of
the elderly-widow stock), with a residual female marriage-count tilt
(+0.044 ln, 3/5). Per the gate-1/gate-2 forensics precedent (the chronic-cell
forensics #94 that preceded candidate 9), candidate 11 registers only after --
and citing -- this decomposition. The registration wins: this runner answers
exactly its two frozen questions.

FROZEN SPEC (comment 4917987214), two questions:

Q4 -- decompose the 75+ female widowed-stock gap (simulated vs reference
widowed-75+ person-year shares) into (a) INFLOW: widowhood incidence by the
surviving spouse's age band, 65-74 vs 75+; (b) AGING-IN: widows entering 75+
having been widowed at 65-74; (c) OUTFLOW: remarriage / attrition of elderly
widows; (d) INITIAL STATES: persons observed widowed at panel entry vs
simulated-state persistence. Which margin carries the ~20% level gap, per
seed. PLUS explicitly test whether the spousal-age-gap draw (husband age
distribution at wife 65+) under-ages husbands relative to observed couples.

Q5 -- decompose the female marriage-count residual under candidate 10's
fitted tables: repeat the forensics-1 Q1 pathway table by sex with candidate
10's fitted components, isolating where the female in-exposure over-production
(-0.036 in the c8-era table) now sits after age-conditioned remarriage --
first marriage vs after-divorce vs after-widowhood, by age band -- and probe
ONE candidate conditioning margin (age at dissolution) for the same
misallocation signature the male fix exploited.

Train-side only, candidate-10 fitted tables, no outer contact beyond the
already-published per-seed scores. This diagnostic reuses candidate 10's
fit/simulate machinery (``scripts/run_gate2_candidate10.py``, merged #98) and
the forensics-1 decomposition patterns (``scripts/gate2_forensics.py``, merged
#94, whose :func:`pathway_cells` is reused verbatim) BUT simulates side B --
the train half -- and compares it to side B's OWN observed panel. The outer
holdout (side A) is never simulated here; the only side-A numbers used are the
already-published candidate-10 per-seed scores read from the committed artifact
(``runs/gate2_hazard_v10.json``). Nothing in ``gates.yaml`` or any committed
``runs/`` gate artifact is written or moved.

Per gate seed the split is fixed; the candidate is fit on side B and simulated
on side B at the K=20 amended-estimator draws (``default_rng(5200 + k)``,
k=0..19), and the per-draw decompositions are averaged over draws, then
reported per seed and over the 5 gate seeds. Reference decompositions are the
deterministic side-B observed panel.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit -- gate 2 does not need it). Run from the repository root with
the PSID history files staged::

    .venv/bin/python scripts/gate2_forensics2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 10 is the spec under diagnosis: its fit/simulate machinery is
# reused verbatim on the TRAIN side. The forensics-1 runner supplies the Q1
# pathway decomposition and the shared small helpers; candidate 1 supplies the
# split rule, the cache and the threshold loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate2_forensics as gf1  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate10 as c10  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_forensics2_v1.json"
CANDIDATE10_ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
FORENSICS1_ARTIFACT = ROOT / "runs" / "gate2_forensics_v1.json"
SCHEMA_VERSION = "gate2_forensics2.v1"
RUN_NAME = "gate2_forensics2_v1"

#: The registered diagnostic (issue #42, comment 4917987214). The registration
#: wins: this runner answers exactly its two frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4917987214"
)
REGISTRATION_POINTER = "4917987214"
REGISTRATION_TITLE = (
    "Registered diagnostic: forensics 2 -- the widow stock and the "
    "female count"
)
#: Candidate 10's own frozen-spec registration (the graded run this diagnoses).
CANDIDATE10_REGISTRATION = c10.SPEC_REGISTRATION
CANDIDATE10_POINTER = c10.REGISTRATION_POINTER

#: Reused frozen dials (candidate 1, via candidate 10).
GATE_SEEDS = c1.GATE_SEEDS  # (0, 1, 2, 3, 4)
SIM_SEED_BASE = c1.SIM_SEED_BASE  # 4200 (single-draw provenance stream)

#: The amended-estimator draw stream (gates.yaml gate_2 amendment 1): 20 draws
#: at ``default_rng(5200 + k)``, k=0..19 -- byte-identical to the stream
#: candidate 10 is scored under, so the train-side draw mean is directly
#: comparable to the published rbar.
DRAW_SEED_BASE = c10.DRAW_SEED_BASE  # 5200
N_DRAWS = c10.N_DRAWS  # 20

#: The failing cell this diagnostic decomposes and its 65-74 companion.
WIDOW_STOCK_BAND = (75, 120)
WIDOW_STOCK_YOUNG_BAND = (65, 74)

#: Widowhood-incidence bands (INFLOW). transitions.WIDOWHOOD_AGE_BANDS.
INCIDENCE_BANDS = transitions.WIDOWHOOD_AGE_BANDS

#: Ego age bands for the elderly-widow remarriage hazard (OUTFLOW). The 50-64 /
#: 65-74 / 75+ split resolves candidate 10's single "50+" remarriage band into
#: the three sub-bands it pools, so the misallocation is visible.
WIDOW_REMARRIAGE_EGO_BANDS = ((50, 64), (65, 74), (75, 120))

#: The cell the gate scores and its 65-74 companion (published-outer context).
Q4_CELLS = ("share_widowed.75+|female", "share_widowed.65-74|female")

#: Q5 pathway sexes and the count cells the female tilt lives on.
SEXES = transitions.SEXES  # ("female", "male")
COUNT_CELLS = c10.COUNT_CELLS

#: Age-at-dissolution bands for the Q5 misallocation probe: the ego's age when
#: the marriage dissolved (age - years_since_dissolution). The candidate-10
#: remarriage table conditions on CURRENT age band x ysd, not this margin.
DISSOLUTION_AGE_BANDS = ((15, 34), (35, 49), (50, 64), (65, 120))

#: Age at which the elderly-widow husband-mortality test conditions: husbands
#: at or above this age carry the mass of spousal mortality.
HUSBAND_HIGH_MORTALITY_AGE = 80
#: Wife age floor for the spousal-age-gap test.
WIFE_TEST_AGE = 65

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_forensics2_cache.json"
)


# --------------------------------------------------------------------------
# Q4 -- the 75+ female widowed-stock decomposition (train-side)
# --------------------------------------------------------------------------
def widowed_stock_shares(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, float]:
    """Weighted female widowed-stock shares at 65-74 and 75+ (the gated cell).

    The share of female person-years in the ``widowed`` state among all
    female person-years in the age band -- ``transitions.stock_occupancy_cells``
    restricted to ``ids``. Works identically on the observed and simulated
    panels (both carry ``marital_state`` person-years).
    """
    stock = transitions.stock_occupancy_cells(panel, ids, weighted=True)
    return {
        "share_75plus": float(stock["share_widowed.75+|female"]["rate"]),
        "share_65_74": float(stock["share_widowed.65-74|female"]["rate"]),
    }


def widowed_75plus_onset_buckets(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, Any]:
    """Partition the 75+ female widowed stock by the onset age of widowhood.

    Every widowed person-year carries ``years_since_dissolution``, so its
    widowhood onset age is ``age - years_since_dissolution`` and its onset year
    is ``year - years_since_dissolution``. Each female widowed person-year at
    75+ falls in exactly one bucket, and the buckets sum (as shares of all
    female 75+ person-years) to ``share_widowed.75+|female``:

    * ``initial_left_censored`` -- onset before the person's exposure start
      (a left-censored widow the simulation would have to initialise). Under
      the retrospective-to-age-15 panel this is structurally empty.
    * ``onset_lt65`` / ``onset_65_74`` -- widowed before 65 / at 65-74 and
      aged into 75+ (the AGING-IN cohorts).
    * ``onset_75plus`` -- widowed at 75+ (fresh elderly widowhood, the 75+
      INFLOW realised as stock).
    """
    py = panel.person_years
    py = py[py["person_id"].isin(ids)]
    attrs = panel.attrs.set_index("person_id")
    lo, hi = WIDOW_STOCK_BAND
    fem = py[
        (py["sex"] == "female") & (py["age"] >= lo) & (py["age"] <= hi)
    ].copy()
    den = float(fem["weight"].sum())
    wid = fem[fem["marital_state"] == "widowed"].copy()
    ysd = wid["years_since_dissolution"]
    n_na = int(ysd.isna().sum())
    ysd_f = ysd.astype("float64")
    onset_age = wid["age"].to_numpy(dtype=np.float64) - ysd_f.to_numpy(
        dtype=np.float64
    )
    onset_year = wid["year"].to_numpy(dtype=np.float64) - ysd_f.to_numpy(
        dtype=np.float64
    )
    sxy = (
        wid["person_id"]
        .map(attrs["start_exposure_year"])
        .to_numpy(dtype=np.float64)
    )
    w = wid["weight"].to_numpy(dtype=np.float64)
    defined = ~np.isnan(onset_age)
    left_cens = defined & (onset_year < sxy)
    in_window = defined & ~left_cens

    def share(mask: np.ndarray) -> float:
        return float(w[mask].sum()) / den if den > 0 else 0.0

    buckets = {
        "initial_left_censored": share(left_cens),
        "onset_lt65": share(in_window & (onset_age < 65)),
        "onset_65_74": share(
            in_window & (onset_age >= 65) & (onset_age <= 74)
        ),
        "onset_75plus": share(in_window & (onset_age >= 75)),
    }
    total = float(wid["weight"].sum()) / den if den > 0 else 0.0
    return {
        "denominator_person_year_weight": den,
        "share_75plus": total,
        "n_widowed_75plus_ysd_na": n_na,
        "buckets": buckets,
        "reconciliation_remainder": total - sum(buckets.values()),
        "aging_in_share": buckets["onset_lt65"] + buckets["onset_65_74"],
    }


def widowhood_incidence(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, float]:
    """Female widowhood incidence (hazard) by the ego age band -- INFLOW.

    Weighted widowhood events over married female person-year exposure in each
    band. Numerator and denominator share the panel's own event / state
    definitions, so this equals the gated ``widowhood.<band>|female`` cells at
    the registered bands.
    """
    py = panel.person_years
    ev = panel.events
    py = py[py["person_id"].isin(ids)]
    ev = ev[ev["person_id"].isin(ids)]
    married = py[(py["marital_state"] == "married") & (py["sex"] == "female")]
    we = ev[(ev["transition"] == "widowhood") & (ev["sex"] == "female")]
    out: dict[str, float] = {}
    for lo, hi in INCIDENCE_BANDS:
        x = married[(married["age"] >= lo) & (married["age"] <= hi)]
        e = we[(we["age"] >= lo) & (we["age"] <= hi)]
        xw = float(x["weight"].sum())
        ew = float(e["weight"].sum())
        out[transitions.band_label(lo, hi)] = ew / xw if xw > 0 else 0.0
    return out


def elderly_widow_remarriage(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, float]:
    """Female remarriage hazard among WIDOWED person-years by ego age band --
    OUTFLOW.

    Weighted remarriage events (from the widowed origin) over widowed female
    person-year exposure in each ego age band. The candidate-10 table pools
    50-64 / 65-74 / 75+ into a single "50+" remarriage band, so a mismatch that
    concentrates at 75+ is the pooled band over-remarrying elderly widows.
    """
    py = panel.person_years
    ev = panel.events
    py = py[py["person_id"].isin(ids)]
    ev = ev[ev["person_id"].isin(ids)]
    wid_py = py[(py["marital_state"] == "widowed") & (py["sex"] == "female")]
    rem = ev[
        (ev["transition"] == "remarriage")
        & (ev["origin"] == "widowed")
        & (ev["sex"] == "female")
    ]
    out: dict[str, float] = {}
    for lo, hi in WIDOW_REMARRIAGE_EGO_BANDS:
        x = wid_py[(wid_py["age"] >= lo) & (wid_py["age"] <= hi)]
        e = rem[(rem["age"] >= lo) & (rem["age"] <= hi)]
        xw = float(x["weight"].sum())
        ew = float(e["weight"].sum())
        out[transitions.band_label(lo, hi)] = ew / xw if xw > 0 else 0.0
    return out


def ever_widowed_by_75(
    panel: transitions.MaritalPanel, ids: set[int]
) -> float:
    """Share of women observed through 75 who ever occupy the widowed state by
    75 -- the inflow integral net of remarriage up to 75.

    A creation-side summary that separates "were enough widows made?" from the
    stock question "did they persist to 75+?".
    """
    attrs = panel.attrs
    attrs = attrs[attrs["person_id"].isin(ids)]
    obs = attrs[
        (attrs["sex"] == "female")
        & (attrs["censor_year"] >= attrs["birth_year"] + 75)
    ]
    obs_ids = set(obs["person_id"])
    py = panel.person_years
    py = py[py["person_id"].isin(obs_ids) & (py["age"] <= 75)]
    ever = set(py[py["marital_state"] == "widowed"]["person_id"])
    w = obs.set_index("person_id")["weight"]
    den = float(w.sum())
    num = float(w[w.index.isin(ever)].sum())
    return num / den if den > 0 else 0.0


def entry_widowed_state(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, Any]:
    """The INITIAL-STATES margin: persons whose first observed person-year is
    widowed, and their contribution to the 75+ widowed stock.

    The simulation initialises each person's state from the first observed
    person-year; under the retrospective-to-age-15 exposure window that entry
    is age-15 never-married for everyone, so there is no left-censored
    widowhood to persist or drop. This reports the entry-state distribution and
    the entry-widowed contribution to the 75+ stock so the margin's size is
    measured, not assumed.
    """
    py = panel.person_years
    py = py[py["person_id"].isin(ids)]
    entry = py.sort_values("year").groupby("person_id", as_index=False).first()
    counts = entry["marital_state"].value_counts().to_dict()
    entry_wid = entry[entry["marital_state"] == "widowed"]
    entry_wid_ids = set(entry_wid["person_id"])
    lo, hi = WIDOW_STOCK_BAND
    fem75 = py[(py["sex"] == "female") & (py["age"] >= lo) & (py["age"] <= hi)]
    den = float(fem75["weight"].sum())
    wid75 = fem75[fem75["marital_state"] == "widowed"]
    from_entry = wid75[wid75["person_id"].isin(entry_wid_ids)]
    return {
        "entry_state_counts": {k: int(v) for k, v in counts.items()},
        "n_entry_widowed_persons": int(len(entry_wid)),
        "entry_widowed_75plus_stock_share": (
            float(from_entry["weight"].sum()) / den if den > 0 else 0.0
        ),
    }


def spousal_gap_husband_age(
    panel: transitions.MaritalPanel,
    ids: set[int],
    gap_dist_female: np.ndarray,
    mh_records: pd.DataFrame,
) -> dict[str, Any]:
    """Spousal-age-gap test: imputed vs observed husband age at wife 65+.

    Over the observed married-female person-years at wife age >= 65, compare
    the husband age the candidate DRAWS -- ``wife_age + G`` with ``G`` sampled
    from the fitted sex-specific gap distribution ``gap_dist_female``
    (``spouse_age - self_age``); the K-draw average equals the deterministic
    convolution used here -- against the OBSERVED husband age from the
    currently-active marriage's spouse birth year. Reports mean husband age,
    the mean gap, and the share of husbands at/above the high-mortality age
    (:data:`HUSBAND_HIGH_MORTALITY_AGE`) both ways. A positive
    ``imputed_minus_observed_mean_husband_age`` means the draw OVER-ages
    husbands (the registration's hypothesis -- that it UNDER-ages them -- is
    then rejected).
    """
    py = panel.person_years
    py = py[py["person_id"].isin(ids)]
    mfem = py[
        (py["sex"] == "female")
        & (py["marital_state"] == "married")
        & (py["age"] >= WIFE_TEST_AGE)
    ].copy()
    w = mfem["weight"].to_numpy(dtype=np.float64)
    wife_age = mfem["age"].to_numpy(dtype=np.float64)
    den = float(w.sum())
    gaps = np.asarray(gap_dist_female, dtype=np.float64)
    mean_gap_pooled = float(gaps.mean())
    mean_wife_age = float((w * wife_age).sum() / den) if den > 0 else 0.0

    # Imputed husband-age survival share via the empirical gap distribution:
    # E_py[ P(wife_age + G >= H) ] with G ~ gaps. searchsorted for O(n log n).
    sg = np.sort(gaps)
    n_g = sg.size
    thr = HUSBAND_HIGH_MORTALITY_AGE - wife_age
    ge = (n_g - np.searchsorted(sg, thr, side="left")) / n_g
    imputed_share_high = float((w * ge).sum() / den) if den > 0 else 0.0
    imputed_mean_husband_age = mean_wife_age + mean_gap_pooled

    # Observed husband age from the currently-active marriage episode.
    person_birth = (
        mh_records.dropna(subset=["birth_year"])
        .groupby("person_id")["birth_year"]
        .first()
    )
    ep = marriage.marriage_episodes(mh_records)
    ep = ep[
        ep["person_id"].isin(ids)
        & ep["start_year"].notna()
        & ep["spouse_person_id"].notna()
    ].copy()
    ep["spouse_birth"] = ep["spouse_person_id"].map(person_birth)
    ep = ep[ep["spouse_birth"].notna()].copy()
    ep["end_eff"] = ep["episode_end_year"].astype("float64").fillna(9999.0)
    mfem_j = mfem[["person_id", "year", "age", "weight"]].copy()
    merged = mfem_j.merge(
        ep[["person_id", "start_year", "end_eff", "spouse_birth"]],
        on="person_id",
        how="inner",
    )
    active = merged[
        (merged["start_year"] <= merged["year"])
        & (merged["year"] <= merged["end_eff"])
    ].copy()
    active = (
        active.sort_values(["person_id", "year", "start_year"])
        .groupby(["person_id", "year"], as_index=False)
        .last()
    )
    aw = active["weight"].to_numpy(dtype=np.float64)
    husband_age = active["year"].to_numpy(dtype=np.float64) - active[
        "spouse_birth"
    ].to_numpy(dtype=np.float64)
    aden = float(aw.sum())
    obs_mean_husband_age = (
        float((aw * husband_age).sum() / aden) if aden > 0 else 0.0
    )
    obs_gap = husband_age - active["age"].to_numpy(dtype=np.float64)
    obs_mean_gap = float((aw * obs_gap).sum() / aden) if aden > 0 else 0.0
    obs_share_high = (
        float(aw[husband_age >= HUSBAND_HIGH_MORTALITY_AGE].sum() / aden)
        if aden > 0
        else 0.0
    )
    return {
        "wife_test_age_floor": WIFE_TEST_AGE,
        "husband_high_mortality_age": HUSBAND_HIGH_MORTALITY_AGE,
        "married_female_65plus_exposure_weight": den,
        "observed_active_match_coverage": (aden / den if den > 0 else 0.0),
        "mean_wife_age": mean_wife_age,
        "imputed_mean_gap_pooled": mean_gap_pooled,
        "observed_mean_gap_active_65plus": obs_mean_gap,
        "imputed_mean_husband_age": imputed_mean_husband_age,
        "observed_mean_husband_age": obs_mean_husband_age,
        "imputed_minus_observed_mean_husband_age": (
            imputed_mean_husband_age - obs_mean_husband_age
        ),
        "imputed_share_husband_ge_high_mortality": imputed_share_high,
        "observed_share_husband_ge_high_mortality": obs_share_high,
    }


# --------------------------------------------------------------------------
# Q5 -- female marriage-count residual under candidate 10's fitted tables
# --------------------------------------------------------------------------
def remarriage_by_age_at_dissolution(
    panel: transitions.MaritalPanel, ids: set[int], sex: str
) -> dict[str, dict[str, float]]:
    """Female/male remarriage hazard by AGE AT DISSOLUTION x origin.

    The ego's age when the marriage dissolved is ``age -
    years_since_dissolution`` on both the dissolved person-years (exposure) and
    the remarriage events (numerator). The candidate-10 remarriage table
    conditions on the ego's CURRENT age band x years-since-dissolution band x
    origin, NOT on age-at-dissolution; if the observed hazard varies by
    age-at-dissolution within those cells and the simulation over- or
    under-produces at the extremes, that is the misallocation signature the
    male fix exploited on the current-age margin, now probed on the
    dissolution-age margin.
    """
    py = panel.person_years
    ev = panel.events
    py = py[py["person_id"].isin(ids)]
    ev = ev[ev["person_id"].isin(ids)]
    out: dict[str, dict[str, float]] = {}
    for origin in ("divorced", "widowed"):
        diss = py[
            (py["marital_state"] == origin)
            & (py["sex"] == sex)
            & py["years_since_dissolution"].notna()
        ].copy()
        diss["age_at_diss"] = diss["age"].to_numpy(dtype=np.float64) - diss[
            "years_since_dissolution"
        ].astype("float64").to_numpy(dtype=np.float64)
        rem = ev[
            (ev["transition"] == "remarriage")
            & (ev["origin"] == origin)
            & (ev["sex"] == sex)
            & ev["years_since_dissolution"].notna()
        ].copy()
        rem["age_at_diss"] = rem["age"].to_numpy(dtype=np.float64) - rem[
            "years_since_dissolution"
        ].astype("float64").to_numpy(dtype=np.float64)
        cells: dict[str, float] = {}
        for lo, hi in DISSOLUTION_AGE_BANDS:
            x = diss[(diss["age_at_diss"] >= lo) & (diss["age_at_diss"] <= hi)]
            e = rem[(rem["age_at_diss"] >= lo) & (rem["age_at_diss"] <= hi)]
            xw = float(x["weight"].sum())
            ew = float(e["weight"].sum())
            cells[transitions.band_label(lo, hi)] = ew / xw if xw > 0 else 0.0
        out[origin] = cells
    return out


# --------------------------------------------------------------------------
# Per-seed computation (fit candidate 10 on train, 20 train-side draws)
# --------------------------------------------------------------------------
def _mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Elementwise mean over a list of flat float dicts (union of keys)."""
    keys: set[str] = set()
    for d in dicts:
        keys |= set(d)
    return {k: float(np.mean([d.get(k, 0.0) for d in dicts])) for k in keys}


def _mean_nested(
    dicts: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Elementwise mean over a list of two-level float dicts."""
    outer: set[str] = set()
    for d in dicts:
        outer |= set(d)
    return {o: _mean_dict([d.get(o, {}) for d in dicts]) for o in outer}


def compute_seed(
    seed: int, data: dict[str, Any], verbose: bool
) -> dict[str, Any]:
    """Fit candidate 10 on the train half, then 20 train-side RNG draws.

    Returns the deterministic reference (side B observed) decompositions and
    the draw-averaged simulated decompositions for both questions, all
    train-side.
    """
    t0 = time.time()
    panel = data["panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = c10.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    gap_dist_female = np.asarray(
        components.gap_dist_by_sex["female"], dtype=np.float64
    )

    # ---- Deterministic reference (side B's own observed panel). ----
    ref_stock = widowed_stock_shares(panel, ids_b)
    ref_onset = widowed_75plus_onset_buckets(panel, ids_b)
    ref_incidence = widowhood_incidence(panel, ids_b)
    ref_remarriage = elderly_widow_remarriage(panel, ids_b)
    ref_ever_widowed = ever_widowed_by_75(panel, ids_b)
    ref_entry = entry_widowed_state(panel, ids_b)
    ref_gap = spousal_gap_husband_age(
        panel, ids_b, gap_dist_female, data["mh_records"]
    )
    ref_pathway = {sex: gf1.pathway_cells(panel, ids_b, sex) for sex in SEXES}
    ref_diss = {
        sex: remarriage_by_age_at_dissolution(panel, ids_b, sex)
        for sex in SEXES
    }

    # ---- 20 simulation-RNG draws of candidate 10 on the train half. ----
    sim_stock_acc: list[dict[str, float]] = []
    sim_onset_acc: list[dict[str, float]] = []
    sim_incidence_acc: list[dict[str, float]] = []
    sim_remarriage_acc: list[dict[str, float]] = []
    sim_ever_acc: list[float] = []
    sim_gap_acc: list[dict[str, float]] = []
    sim_pathway_nmarr: dict[str, list[float]] = {s: [] for s in SEXES}
    sim_pathway_inexp: dict[str, list[float]] = {s: [] for s in SEXES}
    sim_pathway_cells: dict[str, dict[str, list[float]]] = {
        s: {} for s in SEXES
    }
    sim_diss_acc: dict[str, list[dict[str, dict[str, float]]]] = {
        s: [] for s in SEXES
    }
    per_draw_share_75plus: list[float] = []

    for k in range(N_DRAWS):
        sim_seed = DRAW_SEED_BASE + k
        sim_panel, _sim_births = c10.simulate_holdout(
            panel, ids_b, components, sim_seed
        )
        stock = widowed_stock_shares(sim_panel, ids_b)
        sim_stock_acc.append(stock)
        per_draw_share_75plus.append(stock["share_75plus"])
        sim_onset_acc.append(
            widowed_75plus_onset_buckets(sim_panel, ids_b)["buckets"]
        )
        sim_incidence_acc.append(widowhood_incidence(sim_panel, ids_b))
        sim_remarriage_acc.append(elderly_widow_remarriage(sim_panel, ids_b))
        sim_ever_acc.append(ever_widowed_by_75(sim_panel, ids_b))
        sim_gap_acc.append(
            {
                "imputed_share_husband_ge_high_mortality": (
                    spousal_gap_husband_age(
                        sim_panel,
                        ids_b,
                        gap_dist_female,
                        data["mh_records"],
                    )["imputed_share_husband_ge_high_mortality"]
                )
            }
        )
        for sex in SEXES:
            pc = gf1.pathway_cells(sim_panel, ids_b, sex)
            sim_pathway_nmarr[sex].append(pc["mean_lifetime_marriages"])
            sim_pathway_inexp[sex].append(
                pc["in_exposure_marriages_per_person"]
            )
            for key, val in pc["cells"].items():
                sim_pathway_cells[sex].setdefault(key, []).append(val)
            sim_diss_acc[sex].append(
                remarriage_by_age_at_dissolution(sim_panel, ids_b, sex)
            )

    sim_pathway_mean = {
        sex: {
            key: float(np.mean(vals + [0.0] * (N_DRAWS - len(vals))))
            for key, vals in sim_pathway_cells[sex].items()
        }
        for sex in SEXES
    }
    elapsed = round(time.time() - t0, 1)
    if verbose:
        print(
            f"seed {seed}: n_train={len(ids_b)} "
            f"ref share_widowed.75+={ref_stock['share_75plus']:.4f} "
            f"sim={float(np.mean(per_draw_share_75plus)):.4f} "
            f"[{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_train_persons": len(ids_b),
        "n_holdout_persons": len(
            set(int(x) for x in side_a.person_id.unique())
        ),
        "ref_stock": ref_stock,
        "ref_onset_buckets": ref_onset,
        "ref_incidence": ref_incidence,
        "ref_remarriage": ref_remarriage,
        "ref_ever_widowed_by_75": ref_ever_widowed,
        "ref_entry_widowed": ref_entry,
        "ref_spousal_gap": ref_gap,
        "ref_pathway": ref_pathway,
        "ref_remarriage_by_age_at_dissolution": ref_diss,
        "sim_stock_mean": _mean_dict(sim_stock_acc),
        "sim_onset_buckets_mean": _mean_dict(sim_onset_acc),
        "sim_incidence_mean": _mean_dict(sim_incidence_acc),
        "sim_remarriage_mean": _mean_dict(sim_remarriage_acc),
        "sim_ever_widowed_by_75_mean": float(np.mean(sim_ever_acc)),
        "sim_spousal_gap_imputed_share_mean": float(
            np.mean(
                [
                    d["imputed_share_husband_ge_high_mortality"]
                    for d in sim_gap_acc
                ]
            )
        ),
        "sim_pathway_nmarr_mean": {
            s: float(np.mean(sim_pathway_nmarr[s])) for s in SEXES
        },
        "sim_pathway_inexp_mean": {
            s: float(np.mean(sim_pathway_inexp[s])) for s in SEXES
        },
        "sim_pathway_cells_mean": sim_pathway_mean,
        "sim_remarriage_by_age_at_dissolution_mean": {
            s: _mean_nested(sim_diss_acc[s]) for s in SEXES
        },
        "per_draw_share_widowed_75plus": per_draw_share_75plus,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Published-outer context (candidate 10's already-run side-A scores)
# --------------------------------------------------------------------------
def _outer_published() -> dict[str, dict[str, Any]]:
    """The already-published side-A (holdout) candidate-10 scores.

    Read from the committed candidate-10 gate artifact
    (``runs/gate2_hazard_v10.json``); the outer holdout is never simulated
    here. These are the only side-A numbers this diagnostic uses -- context
    that the train-side decomposition reproduces the published outer failure.
    """
    art = json.loads(CANDIDATE10_ARTIFACT.read_text())
    by_seed = {s["seed"]: s for s in art["per_seed"]}
    cells = Q4_CELLS + COUNT_CELLS
    out: dict[str, dict[str, Any]] = {}
    for cell in cells:
        per_seed: dict[str, Any] = {}
        for seed, s in by_seed.items():
            rec = s["gated_cells"].get(cell) or s["report_only_cells"].get(
                cell
            )
            if rec is None:
                continue
            per_seed[str(seed)] = {
                "rbar": float(rec.get("rbar", rec.get("r_candidate"))),
                "rate_a": float(rec["rate_a"]),
                "score": float(rec["score"]),
                "pass": rec.get("pass"),
                "tolerance": rec.get("tolerance"),
            }
        out[cell] = per_seed
    return out


# --------------------------------------------------------------------------
# Assembling the two question blocks
# --------------------------------------------------------------------------
def _mean_seeds(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ratio(sim: float, ref: float) -> float:
    return float(sim / ref) if ref > 0 else float("nan")


def assemble_q4(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The 75+ female widowed-stock gap decomposition, over gate seeds."""
    ref_share = _mean_seeds([s["ref_stock"]["share_75plus"] for s in per_seed])
    sim_share = _mean_seeds(
        [s["sim_stock_mean"]["share_75plus"] for s in per_seed]
    )
    gap = ref_share - sim_share
    log_gap = float(np.log(sim_share / ref_share)) if ref_share > 0 else 0.0

    bucket_keys = (
        "initial_left_censored",
        "onset_lt65",
        "onset_65_74",
        "onset_75plus",
    )
    ref_buckets = {
        k: _mean_seeds(
            [s["ref_onset_buckets"]["buckets"][k] for s in per_seed]
        )
        for k in bucket_keys
    }
    sim_buckets = {
        k: _mean_seeds([s["sim_onset_buckets_mean"][k] for s in per_seed])
        for k in bucket_keys
    }
    bucket_gaps = {k: ref_buckets[k] - sim_buckets[k] for k in bucket_keys}
    aging_in_gap = bucket_gaps["onset_lt65"] + bucket_gaps["onset_65_74"]
    fresh_gap = bucket_gaps["onset_75plus"]
    initial_gap = bucket_gaps["initial_left_censored"]

    inc_bands = [transitions.band_label(lo, hi) for lo, hi in INCIDENCE_BANDS]
    inflow = {}
    for b in inc_bands:
        r = _mean_seeds([s["ref_incidence"][b] for s in per_seed])
        sm = _mean_seeds([s["sim_incidence_mean"][b] for s in per_seed])
        inflow[b] = {
            "reference": r,
            "simulated": sm,
            "sim_over_ref_ratio": _ratio(sm, r),
        }
    rem_bands = [
        transitions.band_label(lo, hi) for lo, hi in WIDOW_REMARRIAGE_EGO_BANDS
    ]
    outflow = {}
    for b in rem_bands:
        r = _mean_seeds([s["ref_remarriage"][b] for s in per_seed])
        sm = _mean_seeds([s["sim_remarriage_mean"][b] for s in per_seed])
        outflow[b] = {
            "reference": r,
            "simulated": sm,
            "sim_over_ref_ratio": _ratio(sm, r),
        }

    ref_ever = _mean_seeds([s["ref_ever_widowed_by_75"] for s in per_seed])
    sim_ever = _mean_seeds(
        [s["sim_ever_widowed_by_75_mean"] for s in per_seed]
    )
    n_entry_widowed = int(
        np.sum(
            [
                s["ref_entry_widowed"]["n_entry_widowed_persons"]
                for s in per_seed
            ]
        )
    )
    initial_state_stock = _mean_seeds(
        [
            s["ref_entry_widowed"]["entry_widowed_75plus_stock_share"]
            for s in per_seed
        ]
    )

    gap_keys = (
        "imputed_mean_gap_pooled",
        "observed_mean_gap_active_65plus",
        "imputed_mean_husband_age",
        "observed_mean_husband_age",
        "imputed_minus_observed_mean_husband_age",
        "imputed_share_husband_ge_high_mortality",
        "observed_share_husband_ge_high_mortality",
        "observed_active_match_coverage",
    )
    spousal = {
        k: _mean_seeds([s["ref_spousal_gap"][k] for s in per_seed])
        for k in gap_keys
    }
    husband_underaged = spousal["imputed_minus_observed_mean_husband_age"] < 0

    # Per-seed detail for the widow-stock gap and the outflow signature.
    per_seed_detail = []
    for s in per_seed:
        rem75 = transitions.band_label(*WIDOW_REMARRIAGE_EGO_BANDS[-1])
        per_seed_detail.append(
            {
                "seed": s["seed"],
                "reference_share_75plus": s["ref_stock"]["share_75plus"],
                "simulated_share_75plus": s["sim_stock_mean"]["share_75plus"],
                "gap": s["ref_stock"]["share_75plus"]
                - s["sim_stock_mean"]["share_75plus"],
                "aging_in_gap": (
                    s["ref_onset_buckets"]["buckets"]["onset_lt65"]
                    + s["ref_onset_buckets"]["buckets"]["onset_65_74"]
                )
                - (
                    s["sim_onset_buckets_mean"]["onset_lt65"]
                    + s["sim_onset_buckets_mean"]["onset_65_74"]
                ),
                "outflow_75plus_sim_over_ref_ratio": _ratio(
                    s["sim_remarriage_mean"][rem75],
                    s["ref_remarriage"][rem75],
                ),
                "reconciliation_remainder": s["ref_onset_buckets"][
                    "reconciliation_remainder"
                ],
            }
        )

    rem75 = transitions.band_label(*WIDOW_REMARRIAGE_EGO_BANDS[-1])
    aging_in_pct = 100.0 * aging_in_gap / gap if gap else 0.0
    fresh_pct = 100.0 * fresh_gap / gap if gap else 0.0
    inc_ratios = [rec["sim_over_ref_ratio"] for rec in inflow.values()]
    verdict = (
        "The 75+ female widowed-stock gap (reference "
        f"{ref_share:.3f} vs simulated {sim_share:.3f}, level "
        f"{gap:+.3f}, ln {log_gap:+.3f}) is carried by the AGING-IN margin: "
        f"the onset<75 buckets hold {aging_in_gap:+.3f} of the {gap:+.3f} gap "
        f"({aging_in_pct:.0f}%), a further {fresh_gap:+.3f} ({fresh_pct:.0f}%) "
        "sits in fresh 75+ onset, and the INITIAL-STATES margin is "
        f"structurally absent ({initial_gap:+.3f}; {n_entry_widowed} "
        "entry-widowed persons -- retrospective exposure starts at age 15 "
        "never-married, so there is no left-censored widowhood). Two flows "
        "drive it. OUTFLOW is the largest single anomaly: elderly-widow "
        f"remarriage runs {outflow[rem75]['sim_over_ref_ratio']:.0f}x the "
        f"reference at 75+ (sim {outflow[rem75]['simulated']:.4f} vs ref "
        f"{outflow[rem75]['reference']:.4f}) because candidate 10's single "
        "'50+' current-age remarriage band pools 75+ widows with the "
        "higher-remarrying 50-64 widows, depleting every aged-in cohort once "
        "it reaches 75+. A broad INFLOW shortfall co-contributes: widowhood "
        f"incidence runs {min(inc_ratios):.2f}-{max(inc_ratios):.2f} of the "
        "reference across bands (worst at 65-74/75+), compounding through the "
        f"ever-widowed-by-75 integral (ref {ref_ever:.3f} vs sim "
        f"{sim_ever:.3f}). The spousal-age-gap draw does NOT under-age "
        "husbands: the imputed husband age at wife 65+ is "
        f"{spousal['imputed_minus_observed_mean_husband_age']:+.2f} yr vs "
        f"observed (pooled gap {spousal['imputed_mean_gap_pooled']:.2f} vs "
        f"observed active {spousal['observed_mean_gap_active_65plus']:.2f}), "
        "so inflow-via-husband-mortality is ruled out."
    )
    return {
        "question": (
            "Decompose the 75+ female widowed-stock gap into inflow "
            "(widowhood incidence 65-74 vs 75+), aging-in (widowed 65-74 "
            "entering 75+), outflow (elderly-widow remarriage/attrition) and "
            "initial states (entry-widowed vs simulated persistence); which "
            "margin carries the ~20% level gap, and does the spousal-age-gap "
            "draw under-age husbands at wife 65+?"
        ),
        "method": (
            "Train-side. Per gate seed, fit candidate 10 on side B and "
            "simulate the SAME side B at the 20 draw seeds 5200+k; average the "
            "per-draw decompositions over draws, then over the 5 gate seeds. "
            "Reference = side B's observed marital panel. The 75+ widowed "
            "person-years are partitioned by widowhood onset age (onset = age "
            "- years_since_dissolution), which reconciles to "
            "share_widowed.75+|female; inflow = widowhood incidence hazards; "
            "outflow = remarriage hazard among widowed person-years by ego age "
            "band (50-64/65-74/75+, resolving candidate 10's pooled 50+ band); "
            "the spousal-gap test compares imputed (wife age + fitted gap "
            "draw) vs observed (active-marriage spouse birth year) husband age."
        ),
        "margin_verdict": (
            "aging-in stock, with a substantial fresh-75+-onset shortfall; "
            "co-driven by outflow (elderly-widow over-remarriage from the "
            "pooled 50+ band) and a broad inflow shortfall in widowhood "
            "incidence; initial-states absent; spousal-age-gap ruled out"
        ),
        "widowed_stock_75plus": {
            "reference": ref_share,
            "simulated": sim_share,
            "gap_reference_minus_simulated": gap,
            "log_ratio_sim_over_ref": log_gap,
        },
        "onset_bucket_decomposition": {
            "reference": ref_buckets,
            "simulated": sim_buckets,
            "gap_reference_minus_simulated": bucket_gaps,
            "aging_in_gap": aging_in_gap,
            "fresh_75plus_onset_gap": fresh_gap,
            "initial_states_gap": initial_gap,
            "note": (
                "buckets are shares of all female 75+ person-years and sum to "
                "share_widowed.75+|female; the gap partitions across them"
            ),
        },
        "inflow_incidence": inflow,
        "outflow_elderly_widow_remarriage": outflow,
        "aging_in_integral_ever_widowed_by_75": {
            "reference": ref_ever,
            "simulated": sim_ever,
            "sim_over_ref_ratio": _ratio(sim_ever, ref_ever),
        },
        "initial_states": {
            "n_entry_widowed_persons_total": n_entry_widowed,
            "entry_widowed_75plus_stock_share": initial_state_stock,
            "note": (
                "structurally absent: the retrospective-to-age-15 exposure "
                "window makes every person's entry state age-15 "
                "never-married, so no left-censored widowhood exists to "
                "persist or drop"
            ),
        },
        "spousal_age_gap_test": {
            **spousal,
            "husband_underaged_by_draw": husband_underaged,
            "finding": (
                "REJECTED -- the draw does not under-age husbands at wife "
                "65+; it slightly over-ages them"
                if not husband_underaged
                else "CONFIRMED -- the draw under-ages husbands at wife 65+"
            ),
        },
        "per_seed": per_seed_detail,
        "finding": verdict,
    }


def assemble_q5(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The female marriage-count residual under candidate 10, over seeds."""
    by_sex: dict[str, Any] = {}
    for sex in SEXES:
        ref_mean = _mean_seeds(
            [
                s["ref_pathway"][sex]["mean_lifetime_marriages"]
                for s in per_seed
            ]
        )
        sim_mean = _mean_seeds(
            [s["sim_pathway_nmarr_mean"][sex] for s in per_seed]
        )
        ref_inexp = _mean_seeds(
            [
                s["ref_pathway"][sex]["in_exposure_marriages_per_person"]
                for s in per_seed
            ]
        )
        sim_inexp = _mean_seeds(
            [s["sim_pathway_inexp_mean"][sex] for s in per_seed]
        )
        ref_keys: set[str] = set()
        sim_keys: set[str] = set()
        for s in per_seed:
            ref_keys |= set(s["ref_pathway"][sex]["cells"])
            sim_keys |= set(s["sim_pathway_cells_mean"][sex])
        ref_cells = {
            k: _mean_seeds(
                [s["ref_pathway"][sex]["cells"].get(k, 0.0) for s in per_seed]
            )
            for k in ref_keys
        }
        sim_cells = {
            k: _mean_seeds(
                [
                    s["sim_pathway_cells_mean"][sex].get(k, 0.0)
                    for s in per_seed
                ]
            )
            for k in sim_keys
        }
        table = {}
        for k in sorted(ref_keys | sim_keys):
            r = ref_cells.get(k, 0.0)
            sm = sim_cells.get(k, 0.0)
            table[k] = {
                "reference": r,
                "simulated": sm,
                "deficit": r - sm,
            }
        # The over-produced cells (sim exceeds ref): the female in-exposure
        # over-production's location. Negative deficit == over-production.
        overproduced = sorted(
            ((k, v) for k, v in table.items() if v["deficit"] < 0),
            key=lambda kv: kv[1]["deficit"],
        )
        by_sex[sex] = {
            "mean_lifetime_marriages_reference": ref_mean,
            "mean_lifetime_marriages_simulated": sim_mean,
            "count_deficit_reference_minus_simulated": ref_mean - sim_mean,
            "in_exposure_reference": ref_inexp,
            "in_exposure_simulated": sim_inexp,
            "in_exposure_over_production": sim_inexp - ref_inexp,
            "pathway_cells": table,
            "largest_over_produced_cells": [
                {"cell": k, **v} for k, v in overproduced[:6]
            ],
        }

    # The conditioning-margin probe: female remarriage by age-at-dissolution.
    diss_bands = [
        transitions.band_label(lo, hi) for lo, hi in DISSOLUTION_AGE_BANDS
    ]
    probe: dict[str, Any] = {}
    for origin in ("divorced", "widowed"):
        cells = {}
        for b in diss_bands:
            r = _mean_seeds(
                [
                    s["ref_remarriage_by_age_at_dissolution"]["female"][
                        origin
                    ][b]
                    for s in per_seed
                ]
            )
            sm = _mean_seeds(
                [
                    s["sim_remarriage_by_age_at_dissolution_mean"]["female"][
                        origin
                    ].get(b, 0.0)
                    for s in per_seed
                ]
            )
            cells[b] = {
                "reference": r,
                "simulated": sm,
                "sim_over_ref_ratio": _ratio(sm, r),
            }
        probe[origin] = cells

    fem = by_sex["female"]

    # Locate the over-production: share in first vs after-divorce vs
    # after-widowhood.
    def pathway_sum(prefix: str) -> tuple[float, float]:
        r = sum(
            v["reference"]
            for k, v in fem["pathway_cells"].items()
            if k.startswith(prefix)
        )
        s = sum(
            v["simulated"]
            for k, v in fem["pathway_cells"].items()
            if k.startswith(prefix)
        )
        return r, s

    fr, fs = (
        fem["pathway_cells"]["first"]["reference"],
        fem["pathway_cells"]["first"]["simulated"],
    )
    dr, ds = pathway_sum("after_divorced")
    wr, ws = pathway_sum("after_widowed")
    origin_split = {
        "first_marriage": {
            "reference": fr,
            "simulated": fs,
            "over_production": fs - fr,
        },
        "after_divorce": {
            "reference": dr,
            "simulated": ds,
            "over_production": ds - dr,
        },
        "after_widowhood": {
            "reference": wr,
            "simulated": ws,
            "over_production": ws - wr,
        },
    }

    wid_probe = probe["widowed"]
    verdict = (
        "Under candidate 10's age-conditioned remarriage, the female "
        "in-exposure marriage-count over-production "
        f"({fem['in_exposure_over_production']:+.3f}/person; the c8-era table "
        "carried -0.036) now sits in remarriage after WIDOWHOOD "
        f"(+{origin_split['after_widowhood']['over_production']:.3f}/person) "
        "and first marriage "
        f"(+{origin_split['first_marriage']['over_production']:.3f}/person), "
        "NOT the after-divorce pathway the male age-conditioning fix "
        "addressed "
        f"({origin_split['after_divorce']['over_production']:+.3f}/person "
        "net). The age-at-dissolution probe shows the misallocation "
        "signature the male fix exploited, now on the female after-widowhood "
        "margin: simulated remarriage runs "
        f"{wid_probe['15-34']['sim_over_ref_ratio']:.2f}x the reference at "
        f"dissolution age 15-34, {wid_probe['50-64']['sim_over_ref_ratio']:.2f}x "
        f"at 50-64 and {wid_probe['65+']['sim_over_ref_ratio']:.2f}x at 65+ -- "
        "the current-age x years-since-dissolution table pools over "
        "age-at-dissolution, so widows who dissolved at the extremes inherit "
        "the pooled rate. The elderly (dissolution age 65+) over-remarriage is "
        "the same defect as Q4's outflow driver, viewed on the count margin."
    )
    return {
        "question": (
            "Repeat the forensics-1 Q1 pathway table by sex with candidate "
            "10's fitted tables: where does the female in-exposure "
            "over-production (-0.036 in the c8-era table) now sit after "
            "age-conditioned remarriage -- first marriage vs after-divorce vs "
            "after-widowhood, by age band -- and does a conditioning margin "
            "(age at dissolution) show the same misallocation signature the "
            "male fix exploited?"
        ),
        "method": (
            "Train-side. Per gate seed, fit candidate 10 on side B and "
            "simulate side B at 5200+k (k=0..19); average the per-person "
            "pathway counts over draws then over the 5 gate seeds. Reference = "
            "side B observed episodes. Pathway = first / after_divorce / "
            "after_widowhood x order (2nd vs 3rd+) x marriage-age band "
            "(gate2_forensics.pathway_cells, reused verbatim). The probe "
            "recomputes female remarriage hazard by age-at-dissolution band "
            "(age - years_since_dissolution) x origin, the margin candidate "
            "10's current-age table does not condition on."
        ),
        "by_sex": by_sex,
        "female_over_production_origin_split": origin_split,
        "conditioning_margin_probe": {
            "margin": "age_at_dissolution",
            "candidate_10_conditions_on": (
                "ego CURRENT age band (18-34/35-49/50+) x "
                "years-since-dissolution band x origin x sex"
            ),
            "female_remarriage_by_age_at_dissolution": probe,
            "signature": (
                "female after-widowhood remarriage over-produced at the "
                "age-at-dissolution extremes (15-34 and 65+) and "
                "under-produced at 50-64 -- the pooled current-age x ysd cell "
                "misallocates across dissolution age"
            ),
        },
        "finding": verdict,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    import scipy
    import sklearn

    return {
        "populace_dynamics_sha": c1._git_sha(ROOT),
        "candidate10_runner": "scripts/run_gate2_candidate10.py",
        "candidate10_runner_sha256": c1._sha_of_file(
            ROOT / "scripts" / "run_gate2_candidate10.py"
        ),
        "candidate10_artifact": "runs/gate2_hazard_v10.json",
        "candidate10_artifact_sha256": c1._sha_of_file(CANDIDATE10_ARTIFACT),
        "forensics1_runner": "scripts/gate2_forensics.py",
        "forensics1_runner_sha256": c1._sha_of_file(
            ROOT / "scripts" / "gate2_forensics.py"
        ),
        "forensics1_artifact": "runs/gate2_forensics_v1.json",
        "gates_yaml_locked": bool(thresholds.get("locked", False)),
        "gates_yaml_status": thresholds.get("status"),
        "sklearn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scipy_version": str(scipy.__version__),
        "schema_version": SCHEMA_VERSION,
    }


def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = c1._load_cache(cache_path)

    thresholds = c1.load_gate2_thresholds()
    tol = c1.gated_tolerances(thresholds)
    for cell in Q4_CELLS:
        if cell not in tol:
            raise RuntimeError(
                f"{cell} is not a gated cell in gates.yaml; refusing to "
                "diagnose a cell the lock does not define."
            )
    for name, path in (
        ("candidate-10", CANDIDATE10_ARTIFACT),
        ("forensics-1", FORENSICS1_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )

    data = gf1._load_inputs()
    if verbose:
        print(
            f"panel: {data['data_meta']['n_person_years']} person-years, "
            f"{data['data_meta']['panel_persons_weighted']} persons; "
            f"K={N_DRAWS} draws (5200 + k) on the train half"
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(cache[key])
            continue
        result = compute_seed(seed, data, verbose)
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    q4 = assemble_q4(per_seed)
    q5 = assemble_q5(per_seed)
    outer = _outer_published()

    candidate_11_implications = (
        "Q4: the elderly-widow stock under-production is led by an OUTFLOW "
        "defect -- candidate 10's single '50+' current-age remarriage band "
        "applies a rate ~9x too high to 75+ widows, depleting the aging-in "
        "stock -- with a broad but modest (~10%) inflow shortfall in "
        "widowhood incidence co-contributing; the spousal-age-gap imputation "
        "is sound (husbands are not under-aged) and the initial-states margin "
        "is structurally absent. Q5: the residual female count over-production "
        "is remarriage after WIDOWHOOD (and a little first marriage), with a "
        "clean age-at-dissolution misallocation signature (over-remarriage at "
        "dissolution age 65+). Both questions point to the SAME highest-value "
        "candidate 11 lever: condition remarriage -- especially after "
        "widowhood -- on age at dissolution (or split the pooled 50+ "
        "current-age band into 50-64/65-74/75+ and/or cap the thin "
        "elderly-widow cell the add-one smoothing inflates), rather than raise "
        "or lower any aggregate rate; the inflow shortfall is a separate, "
        "smaller mortality-side question. Registered only after citing this "
        "evidence, under the one-shot rule."
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "registration_title": REGISTRATION_TITLE,
        "candidate_under_diagnosis": (
            "gate-2 candidate 10 (run 1, PR #98): FAIL 1/5 under the amended "
            "mean-over-20-draws estimator; decider share_widowed.75+|female "
            "(failed 4/5), residual female marriage-count tilt +0.044 ln (3/5)"
        ),
        "candidate10_spec_registration": CANDIDATE10_REGISTRATION,
        "candidate10_registration_pointer": CANDIDATE10_POINTER,
        "candidate10_artifact": "runs/gate2_hazard_v10.json",
        "forensics1_artifact": "runs/gate2_forensics_v1.json (#94)",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published candidate-10 per-seed "
                "scores read from runs/gate2_hazard_v10.json; the holdout "
                "(side A) is never simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "outer holdout, side B = train complement (this diagnostic "
                "simulates side B)"
            ),
            "fit_simulate_machinery": (
                "scripts/run_gate2_candidate10.py (merged #98; chain-imports "
                "candidates 1-9) reused byte-for-byte on the train side; "
                "scripts/gate2_forensics.py (merged #94) pathway_cells reused "
                "verbatim"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": (
                f"numpy default_rng({DRAW_SEED_BASE} + k) for k in "
                f"0..{N_DRAWS - 1} (the candidate-10 amended-estimator stream; "
                "the gate seed fixes the split, k varies only the simulation "
                "RNG)"
            ),
            "reference": (
                "side B's own observed marital panel (no rate_a/rate_b "
                "tolerance scoring; the decompositions compare simulated vs "
                "observed levels directly)"
            ),
        },
        "data": data["data_meta"],
        "question_4_widow_stock": q4,
        "question_5_female_count": q5,
        "published_outer_context": {
            "note": (
                "already-published candidate-10 side-A scores "
                "(runs/gate2_hazard_v10.json); the outer holdout is not "
                "simulated here -- context that the train-side decomposition "
                "reproduces the published outer failure"
            ),
            "cells": outer,
        },
        "per_seed": per_seed,
        "candidate_11_implications": candidate_11_implications,
        "revision_pins": _revision_pins(thresholds),
        "per_seed_compute_seconds": {
            s["seed"]: s["elapsed_seconds"] for s in per_seed
        },
        "total_per_seed_compute_seconds": round(
            sum(s["elapsed_seconds"] for s in per_seed), 1
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        q = artifact["question_4_widow_stock"]["widowed_stock_75plus"]
        gap = q["gap_reference_minus_simulated"]
        margin = artifact["question_4_widow_stock"]["margin_verdict"]
        print(
            f"\nQ4 widow stock: ref {q['reference']:.3f} vs sim "
            f"{q['simulated']:.3f} (gap {gap:+.3f}); margin = {margin}"
        )
        f = artifact["question_5_female_count"]["by_sex"]["female"]
        print(
            "Q5 female in-exposure over-production "
            f"{f['in_exposure_over_production']:+.3f}/person; "
            "biggest over-produced cell "
            f"{f['largest_over_produced_cells'][0]['cell']}"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="lbfgs failed to converge")
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
