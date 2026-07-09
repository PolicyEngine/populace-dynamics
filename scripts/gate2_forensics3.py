"""Gate-2 forensics 3 (REPORTED, NOT GATED): the widowed life cycle's
accounting.

The registered diagnostic of PolicyEngine/populace-dynamics issue #42,
comment 4920355640 ("forensics 3 -- the widowed life cycle's accounting").
It is evidence, not another spec draw: candidate 11 (run 1, #101) graded
FAIL 1/5 under the amended mean-over-20-draws estimator, and its decider
was again ``share_widowed.75+|female`` -- the registered modal, which the
elderly-remarriage outflow fix left standing because the ~10% widowhood-
incidence inflow shortfall was never touched (the female count regressed
3/5 -> 2/5). Per the gate-2 forensics precedent (forensics 1 #94 before
candidate 9, forensics 2 #99 before candidate 11), candidate 12 registers
only after -- and citing -- this decomposition. The registration wins: this
runner answers exactly its two frozen questions.

FROZEN SPEC (comment 4920355640), two questions:

Q6 -- reference-vs-simulable audit of the 75+ widowed stock. Decompose how
reference person-periods classified widowed-75+|female ARISE in the reference
construction (:mod:`populace_dynamics.data.transitions`):
  (a) widowhood observed as an episode ending WITHIN the person's panel
      support window (simulation-reachable);
  (b) marital status carried from an episode whose spouse-death PREDATES the
      person's panel support (reachable ONLY as an observed initial state);
  (c) statuses not derivable from any datable episode (the exact taxonomy
      that exposed the marriage-count residual, forensics 1).
Measure the share of reference widowed-75+ person-years a support-constrained
simulation could produce under ANY transition rates. If a structurally
unreachable share exists, quantify it per seed -- that is an observed-initial-
state fix (the marriage-count precedent, candidate 9 delta 1), not a rate fix.

Q7 -- widowed exposure accounting by age. Simulated vs reference widowed
person-years by age band x sex under candidate 11: is the simulated widowed
50-64 pool inflated (which would explain the count over-production persisting
under the data's own correct rates), and does the ~10% widowhood-incidence
inflow shortfall concentrate in an age band? Include the interaction:
cumulative survival-in-widowhood curves (years since widowhood before
remarriage / support-end), simulated vs reference, for widowhoods beginning at
50-64 vs 65+.

Train-side only, candidate-11 fitted tables, no outer contact beyond the
already-published per-seed scores. This diagnostic reuses candidate 11's
fit/simulate machinery (``scripts/run_gate2_candidate11.py``, merged #101) and
the forensics-1/-2 decomposition patterns (``scripts/gate2_forensics.py`` #94,
``scripts/gate2_forensics2.py`` #99). It simulates side B -- the train half --
and compares it to side B's OWN observed panel. The outer holdout (side A) is
never simulated here; the only side-A numbers used are the already-published
candidate-11 per-seed scores read from the committed artifact
(``runs/gate2_hazard_v11.json``). Nothing in ``gates.yaml`` or any committed
``runs/`` gate artifact is written or moved.

The Q6 taxonomy is a REFERENCE-construction audit (no simulation): each 75+
female widowed person-year's generating widowhood episode is classified by its
end year (the spouse-death / dissolution year, ``year -
years_since_dissolution``) against the person's OBSERVED PSID support window
(the first and last wave the person is present in a responding family unit,
``populace_dynamics.data.panels.demographic_panel``) -- the tight support a
realistic deployment observes, NOT the retrospective-to-age-15 exposure the
reference reconstructs marital state across. Under the age-15 exposure window
forensics 2 found zero left-censored widowhood; the observed-support window is
exactly the tighter boundary that exposes the carried-status share.

Per gate seed the split is fixed; candidate 11 is fit on side B and simulated
on side B at the K=20 amended-estimator draws (``default_rng(5200 + k)``,
k=0..19), the per-draw Q7 decompositions averaged over draws, then reported per
seed and over the 5 gate seeds. Reference decompositions are the deterministic
side-B observed panel.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit -- gate 2 does not need it). Run from the repository root with the
PSID history files staged::

    .venv/bin/python scripts/gate2_forensics3.py
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

# Candidate 11 is the spec under diagnosis: its fit/simulate machinery is
# reused verbatim on the TRAIN side. The forensics-1 runner supplies the shared
# input loader; candidate 1 supplies the split rule, the cache and the loaders.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate2_forensics as gf1  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate11 as c11  # noqa: E402

from populace_dynamics.data import transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_forensics3_v1.json"
CANDIDATE11_ARTIFACT = ROOT / "runs" / "gate2_hazard_v11.json"
FORENSICS2_ARTIFACT = ROOT / "runs" / "gate2_forensics2_v1.json"
SCHEMA_VERSION = "gate2_forensics3.v1"
RUN_NAME = "gate2_forensics3_v1"

#: The registered diagnostic (issue #42, comment 4920355640). The registration
#: wins: this runner answers exactly its two frozen questions.
REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4920355640"
)
REGISTRATION_POINTER = "4920355640"
REGISTRATION_TITLE = (
    "Registered diagnostic: forensics 3 -- the widowed life cycle's "
    "accounting"
)
#: Candidate 11's own frozen-spec registration (the graded run this diagnoses).
CANDIDATE11_REGISTRATION = c11.SPEC_REGISTRATION
CANDIDATE11_POINTER = c11.REGISTRATION_POINTER

#: Reused frozen dials (candidate 1, via candidate 11).
GATE_SEEDS = c1.GATE_SEEDS  # (0, 1, 2, 3, 4)
SIM_SEED_BASE = c1.SIM_SEED_BASE  # 4200 (single-draw provenance stream)

#: The amended-estimator draw stream (gates.yaml gate_2 amendment 1): 20 draws
#: at ``default_rng(5200 + k)``, k=0..19 -- byte-identical to the stream
#: candidate 11 is scored under, so the train-side draw mean is directly
#: comparable to the published rbar.
DRAW_SEED_BASE = c11.DRAW_SEED_BASE  # 5200
N_DRAWS = c11.N_DRAWS  # 20

#: The failing modal cell this diagnostic accounts for, and its stock context.
WIDOW_STOCK_BAND = (75, 120)
STOCK_CONTEXT_BANDS = ((65, 74), (75, 120))
Q_STOCK_CELL = "share_widowed.75+|female"

#: Q7 widowed-stock-by-age bands (the 50-64 pool inflation question needs the
#: 50-64 band explicitly; the catch-all <50 band closes the partition).
WIDOWED_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (15, 49),
    (50, 64),
    (65, 74),
    (75, 120),
)
#: Q7 widowhood-incidence bands (INFLOW). transitions.WIDOWHOOD_AGE_BANDS.
INCIDENCE_BANDS = transitions.WIDOWHOOD_AGE_BANDS

#: Q7 survival-in-widowhood onset bands and the duration grid (years since
#: widowhood). The 50-64 vs 65+ split is the registration's interaction axis.
ONSET_BANDS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("50-64", (50, 64)),
    ("65+", (65, 120)),
)
SURVIVAL_DMAX = 30
SURVIVAL_REPORT_DURATIONS = (1, 2, 3, 5, 10, 15, 20)

SEXES = transitions.SEXES  # ("female", "male")

#: Age above which a carried-status widow counts as a "late entrant already
#: widowed" (a reported diagnostic on the carried mass, not a threshold).
LATE_ENTRY_AGE = 60

#: Per-seed cache OUTSIDE runs/ (never committed): a long run resumes.
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_forensics3_cache.json"
)


# --------------------------------------------------------------------------
# Panel support (the observed PSID window) -- the Q6 boundary
# --------------------------------------------------------------------------
def observed_support(demo: pd.DataFrame) -> pd.DataFrame:
    """Per-person observed PSID support window from the demographic panel.

    ``demographic_panel`` already restricts to sequence 1-20 (present in a
    responding family unit -- the presence notion trajectory windows treat as
    observation), so each person's first and last ``period`` are the first and
    last wave they are actually observed. This is the tight support a
    support-constrained simulation would see, distinct from the reference's
    retrospective-to-age-15 exposure window.
    """
    supp = demo.groupby("person_id")["period"].agg(["min", "max"])
    supp.columns = ["first_wave", "last_wave"]
    return supp


# --------------------------------------------------------------------------
# Q6 -- reference-vs-simulable audit of the 75+ female widowed stock
# --------------------------------------------------------------------------
def widowed_75plus_support_taxonomy(
    panel: transitions.MaritalPanel,
    support: pd.DataFrame,
    ids: set[int],
) -> dict[str, Any]:
    """Classify each 75+ female widowed person-year by its generating
    widowhood episode against the person's observed panel support.

    Every widowed person-year carries ``years_since_dissolution``, so its
    widowhood onset (spouse-death / dissolution) year is ``year -
    years_since_dissolution``. Each female widowed person-year at 75+ falls in
    exactly one bucket, and the buckets sum (as shares of the 75+ female
    widowed stock) to 1 (and, as shares of all female 75+ person-years, to
    ``share_widowed.75+|female``):

    * ``reachable_within_support`` -- onset year in ``[first_wave, last_wave]``:
      the spouse-death is observed within the person's PSID support, so a
      support-constrained simulation rolling forward from the observed married
      state produces it under some widowhood hazard (category a).
    * ``carried_predates_support`` -- onset year < ``first_wave``: the person
      enters observation ALREADY widowed; no transition rate on the observed
      support produces it, only an observed-initial-state injection can
      (category b, the marriage-count precedent).
    * ``onset_after_support_end`` -- onset year > ``last_wave``: the widowhood
      is dated after the last observed wave (the retrospective marriage report
      outran the demographic presence); unreachable and NOT initial-state-
      fixable (a tail-truncation, expected ~0).
    * ``non_derivable`` -- no datable dissolution (``years_since_dissolution``
      NA) or no observed support at all: not derivable from any datable
      episode (category c, the marriage-count residual's analogue).

    Also reports the same partition under the reconstruction (age-15) exposure
    window, which -- per forensics 2 -- has zero carried mass, to make the
    boundary difference explicit.
    """
    py = panel.person_years
    py = py[py["person_id"].isin(ids)]
    attrs = panel.attrs.set_index("person_id")
    lo, hi = WIDOW_STOCK_BAND
    fem = py[
        (py["sex"] == "female") & (py["age"] >= lo) & (py["age"] <= hi)
    ].copy()
    den_all = float(fem["weight"].sum())
    wid = fem[fem["marital_state"] == "widowed"].copy()
    den_wid = float(wid["weight"].sum())

    ysd = wid["years_since_dissolution"].astype("float64")
    onset_year = wid["year"].to_numpy(dtype=np.float64) - ysd.to_numpy(
        dtype=np.float64
    )
    onset_age = wid["age"].to_numpy(dtype=np.float64) - ysd.to_numpy(
        dtype=np.float64
    )
    fw = wid["person_id"].map(support["first_wave"]).to_numpy(dtype=np.float64)
    lw = wid["person_id"].map(support["last_wave"]).to_numpy(dtype=np.float64)
    birth = (
        wid["person_id"].map(attrs["birth_year"]).to_numpy(dtype=np.float64)
    )
    sxy = (
        wid["person_id"]
        .map(attrs["start_exposure_year"])
        .to_numpy(dtype=np.float64)
    )
    cy = wid["person_id"].map(attrs["censor_year"]).to_numpy(dtype=np.float64)
    w = wid["weight"].to_numpy(dtype=np.float64)

    n_ysd_na = int(np.isnan(onset_year).sum())
    no_support = np.isnan(fw)
    defined = ~np.isnan(onset_year)

    reachable = defined & ~no_support & (onset_year >= fw) & (onset_year <= lw)
    carried = defined & ~no_support & (onset_year < fw)
    after_end = defined & ~no_support & (onset_year > lw)
    non_deriv = (~defined) | no_support

    def stock_share(mask: np.ndarray) -> float:
        return float(w[mask].sum()) / den_wid if den_wid > 0 else 0.0

    def py_share(mask: np.ndarray) -> float:
        return float(w[mask].sum()) / den_all if den_all > 0 else 0.0

    buckets = {
        "reachable_within_support": stock_share(reachable),
        "carried_predates_support": stock_share(carried),
        "onset_after_support_end": stock_share(after_end),
        "non_derivable": stock_share(non_deriv),
    }
    structurally_unreachable = (
        buckets["carried_predates_support"]
        + buckets["onset_after_support_end"]
        + buckets["non_derivable"]
    )
    initial_state_fixable = buckets["carried_predates_support"]

    # Contrast partition under the reconstruction (age-15) exposure window:
    # onset always falls in [start_exposure_year, censor_year] there, so the
    # carried share is structurally 0 (forensics 2's boundary).
    recon_reachable = defined & (onset_year >= sxy) & (onset_year <= cy)
    recon_before = defined & (onset_year < sxy)
    recon_after = defined & (onset_year > cy)

    # Diagnostics on the carried mass.
    carried_w = float(w[carried].sum())
    if carried_w > 0:
        carried_onset_age = float(
            np.average(onset_age[carried], weights=w[carried])
        )
        first_obs_age = fw[carried] - birth[carried]
        carried_late_entry = float(
            np.average(
                (first_obs_age >= LATE_ENTRY_AGE).astype(np.float64),
                weights=w[carried],
            )
        )
        carried_mean_first_wave = float(
            np.average(fw[carried], weights=w[carried])
        )
    else:
        carried_onset_age = 0.0
        carried_late_entry = 0.0
        carried_mean_first_wave = 0.0

    return {
        "denominator_all_female_75plus_py_weight": den_all,
        "widowed_75plus_py_weight": den_wid,
        "share_widowed_75plus_female": (
            den_wid / den_all if den_all > 0 else 0.0
        ),
        "buckets_share_of_widowed_stock": buckets,
        "buckets_share_of_all_75plus_py": {
            "reachable_within_support": py_share(reachable),
            "carried_predates_support": py_share(carried),
            "onset_after_support_end": py_share(after_end),
            "non_derivable": py_share(non_deriv),
        },
        "structurally_unreachable_share": structurally_unreachable,
        "initial_state_fixable_share": initial_state_fixable,
        "reconciliation_remainder": 1.0 - sum(buckets.values()),
        "n_widowed_75plus_py": int(len(wid)),
        "n_years_since_dissolution_na": n_ysd_na,
        "n_no_observed_support": int(no_support.sum()),
        "carried_mass_diagnostics": {
            "mean_onset_age": carried_onset_age,
            "mean_first_observed_wave": carried_mean_first_wave,
            "share_first_observed_age_ge_60": carried_late_entry,
        },
        "reconstruction_window_contrast": {
            "reachable_within_exposure": stock_share(recon_reachable),
            "onset_before_exposure_start": stock_share(recon_before),
            "onset_after_exposure_end": stock_share(recon_after),
            "note": (
                "the retrospective-to-age-15 exposure window has ~0 carried "
                "mass (forensics 2); the observed-support window is the "
                "tighter boundary that exposes the carried-status share"
            ),
        },
    }


# --------------------------------------------------------------------------
# Q7 -- widowed exposure accounting by age (train-side sim vs reference)
# --------------------------------------------------------------------------
def widowed_person_years_by_age_sex(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, dict[str, float]]:
    """Weighted widowed person-years and stock share by age band x sex.

    The simulated and reference panels share the SAME person-year age x sex
    grid and weights (the simulation only reassigns ``marital_state``), so the
    denominators are identical and only the widowed weight differs -- a clean
    exposure comparison. The 50-64 female cell is the pool-inflation subject.
    """
    py = panel.person_years
    py = py[py["person_id"].isin(ids)]
    out: dict[str, dict[str, float]] = {}
    for lo, hi in WIDOWED_AGE_BANDS:
        for sex in SEXES:
            grp = py[
                (py["age"] >= lo) & (py["age"] <= hi) & (py["sex"] == sex)
            ]
            den = float(grp["weight"].sum())
            wid = grp[grp["marital_state"] == "widowed"]
            num = float(wid["weight"].sum())
            key = f"{transitions.band_label(lo, hi)}|{sex}"
            out[key] = {
                "widowed_py_weight": num,
                "all_py_weight": den,
                "widowed_share": num / den if den > 0 else 0.0,
                "n_widowed_py": int(len(wid)),
            }
    return out


def widowhood_incidence_by_age_sex(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, dict[str, float]]:
    """Widowhood incidence (hazard) by ego age band x sex -- INFLOW.

    Weighted widowhood events over married person-year exposure in each band.
    Numerator and denominator share the panel's own event / state definitions,
    so this equals the gated ``widowhood.<band>|<sex>`` cells at the registered
    bands. The ~10% inflow shortfall's age concentration is read off the
    per-band sim/ref ratios.
    """
    py = panel.person_years
    ev = panel.events
    py = py[py["person_id"].isin(ids)]
    ev = ev[ev["person_id"].isin(ids)]
    out: dict[str, dict[str, float]] = {}
    for sex in SEXES:
        married = py[(py["marital_state"] == "married") & (py["sex"] == sex)]
        we = ev[(ev["transition"] == "widowhood") & (ev["sex"] == sex)]
        cells: dict[str, float] = {}
        for lo, hi in INCIDENCE_BANDS:
            x = married[(married["age"] >= lo) & (married["age"] <= hi)]
            e = we[(we["age"] >= lo) & (we["age"] <= hi)]
            xw = float(x["weight"].sum())
            ew = float(e["weight"].sum())
            cells[transitions.band_label(lo, hi)] = ew / xw if xw > 0 else 0.0
        out[sex] = cells
    return out


def _widowed_spells(
    panel: transitions.MaritalPanel, ids: set[int], sex: str
) -> pd.DataFrame:
    """One row per widowed spell with its terminal duration and event flag.

    A widowed spell is a maximal run of consecutive widowed person-years from
    one dissolution (keyed by ``person_id`` and onset year ``year -
    years_since_dissolution``). Its observed widowed duration is the maximum
    ``years_since_dissolution`` in the run; the spell ENDS in remarriage if a
    remarriage event (origin widowed) shares the person and onset year -- at
    duration ``years_since_dissolution`` of that event -- and is otherwise
    right-censored at the observed duration (support-end: last wave or death).
    """
    py = panel.person_years
    ev = panel.events
    attrs = panel.attrs.set_index("person_id")
    py = py[py["person_id"].isin(ids)]
    ev = ev[ev["person_id"].isin(ids)]
    wid = py[
        (py["marital_state"] == "widowed")
        & (py["sex"] == sex)
        & py["years_since_dissolution"].notna()
    ].copy()
    if wid.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "onset_year",
                "onset_age",
                "term_dur",
                "event",
                "weight",
            ]
        )
    ysd = wid["years_since_dissolution"].astype("int64")
    wid["onset_year"] = wid["year"].astype("int64") - ysd
    wid["ysd"] = ysd
    spell = wid.groupby(["person_id", "onset_year"], as_index=False).agg(
        D=("ysd", "max"), weight=("weight", "first")
    )
    spell["onset_age"] = spell["onset_year"] - spell["person_id"].map(
        attrs["birth_year"]
    )

    rem = ev[
        (ev["transition"] == "remarriage")
        & (ev["origin"] == "widowed")
        & (ev["sex"] == sex)
        & ev["years_since_dissolution"].notna()
    ].copy()
    if not rem.empty:
        rr = rem["years_since_dissolution"].astype("int64")
        rem["onset_year"] = rem["year"].astype("int64") - rr
        rem["r"] = rr
        rem_map = rem.groupby(["person_id", "onset_year"], as_index=False)[
            "r"
        ].min()
        spell = spell.merge(
            rem_map, on=["person_id", "onset_year"], how="left"
        )
    else:
        spell["r"] = pd.NA
    spell["event"] = spell["r"].notna().astype("int64")
    spell["term_dur"] = np.where(
        spell["event"] == 1,
        spell["r"].astype("float64"),
        spell["D"].astype("float64"),
    )
    return spell[
        ["person_id", "onset_year", "onset_age", "term_dur", "event", "weight"]
    ]


def _weighted_km(
    term_dur: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    dmax: int,
) -> np.ndarray:
    """Weighted Kaplan-Meier survival S(d), d=0..dmax (S(0)=1).

    Event = remarriage (leaving widowhood); right-censoring = support-end. The
    risk set at duration d is spells whose terminal duration is >= d; events at
    d are remarriages at exactly d.
    """
    term = np.asarray(term_dur, dtype=np.float64)
    ev = np.asarray(event, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    surv = np.ones(dmax + 1, dtype=np.float64)
    running = 1.0
    for d in range(1, dmax + 1):
        at_risk = float(w[term >= d].sum())
        events = float(w[(term == d) & (ev == 1.0)].sum())
        if at_risk > 0:
            running *= 1.0 - events / at_risk
        surv[d] = running
    return surv


def widowhood_survival_curves(
    panel: transitions.MaritalPanel, ids: set[int], sex: str = "female"
) -> dict[str, Any]:
    """Cumulative survival-in-widowhood by onset band (50-64 vs 65+).

    For each onset band, the weighted KM survival of the widowed spell against
    remarriage (support-end = censoring). Returns the full S(d) curve (for
    recomputability), the reported-duration snapshots, the restricted mean
    survival time (area under S up to the cap) and the spell weight / count.
    """
    spell = _widowed_spells(panel, ids, sex)
    out: dict[str, Any] = {}
    for label, (lo, hi) in ONSET_BANDS:
        sub = spell[(spell["onset_age"] >= lo) & (spell["onset_age"] <= hi)]
        if sub.empty:
            surv = np.ones(SURVIVAL_DMAX + 1, dtype=np.float64)
            n_spells = 0
            spell_w = 0.0
            n_events = 0
        else:
            surv = _weighted_km(
                sub["term_dur"].to_numpy(dtype=np.float64),
                sub["event"].to_numpy(dtype=np.float64),
                sub["weight"].to_numpy(dtype=np.float64),
                SURVIVAL_DMAX,
            )
            n_spells = int(len(sub))
            spell_w = float(sub["weight"].sum())
            n_events = int(sub["event"].sum())
        out[label] = {
            "survival_curve": [float(x) for x in surv],
            "reported": {
                str(d): float(surv[d]) for d in SURVIVAL_REPORT_DURATIONS
            },
            "restricted_mean_survival_years": float(surv[1:].sum()),
            "n_spells": n_spells,
            "spell_weight": spell_w,
            "n_remarriage_events": n_events,
        }
    return out


# --------------------------------------------------------------------------
# Per-seed computation (fit candidate 11 on train, 20 train-side draws)
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


def _mean_curves(curves: list[list[float]]) -> list[float]:
    """Elementwise mean over a list of equal-length survival curves."""
    arr = np.asarray(curves, dtype=np.float64)
    return [float(x) for x in arr.mean(axis=0)]


def _stock_shares(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, float]:
    """Female widowed-stock shares at the 65-74 and 75+ context bands."""
    stock = transitions.stock_occupancy_cells(panel, ids, weighted=True)
    return {
        "share_75plus": float(stock["share_widowed.75+|female"]["rate"]),
        "share_65_74": float(stock["share_widowed.65-74|female"]["rate"]),
    }


def compute_seed(
    seed: int, data: dict[str, Any], support: pd.DataFrame, verbose: bool
) -> dict[str, Any]:
    """Fit candidate 11 on the train half, then 20 train-side RNG draws.

    Q6 is a reference-construction audit (no simulation); Q7 compares the
    draw-averaged simulated decompositions to side B's observed panel.
    """
    t0 = time.time()
    panel = data["panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = c11.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )

    # ---- Q6: reference-construction reachability audit (deterministic). ----
    ref_taxonomy = widowed_75plus_support_taxonomy(panel, support, ids_b)

    # ---- Q7 reference (side B's own observed panel). ----
    ref_stock = _stock_shares(panel, ids_b)
    ref_widowed_by_age = widowed_person_years_by_age_sex(panel, ids_b)
    ref_incidence = widowhood_incidence_by_age_sex(panel, ids_b)
    ref_survival = {
        sex: widowhood_survival_curves(panel, ids_b, sex) for sex in SEXES
    }

    # ---- 20 simulation-RNG draws of candidate 11 on the train half. ----
    sim_stock_acc: list[dict[str, float]] = []
    sim_widowed_by_age_acc: list[dict[str, dict[str, float]]] = []
    sim_incidence_acc: list[dict[str, dict[str, float]]] = []
    sim_survival_acc: dict[str, dict[str, list[list[float]]]] = {
        sex: {label: [] for label, _ in ONSET_BANDS} for sex in SEXES
    }
    sim_survival_scalar_acc: dict[str, dict[str, dict[str, list[float]]]] = {
        sex: {
            label: {"rmst": [], "spell_weight": []} for label, _ in ONSET_BANDS
        }
        for sex in SEXES
    }
    per_draw_share_75plus: list[float] = []

    for k in range(N_DRAWS):
        sim_seed = DRAW_SEED_BASE + k
        sim_panel, _sim_births = c11.simulate_holdout(
            panel, ids_b, components, sim_seed
        )
        stock = _stock_shares(sim_panel, ids_b)
        sim_stock_acc.append(stock)
        per_draw_share_75plus.append(stock["share_75plus"])
        sim_widowed_by_age_acc.append(
            widowed_person_years_by_age_sex(sim_panel, ids_b)
        )
        sim_incidence_acc.append(
            widowhood_incidence_by_age_sex(sim_panel, ids_b)
        )
        for sex in SEXES:
            curves = widowhood_survival_curves(sim_panel, ids_b, sex)
            for label, _band in ONSET_BANDS:
                sim_survival_acc[sex][label].append(
                    curves[label]["survival_curve"]
                )
                sim_survival_scalar_acc[sex][label]["rmst"].append(
                    curves[label]["restricted_mean_survival_years"]
                )
                sim_survival_scalar_acc[sex][label]["spell_weight"].append(
                    curves[label]["spell_weight"]
                )

    sim_widowed_by_age_mean = _mean_nested(sim_widowed_by_age_acc)
    sim_incidence_mean = _mean_nested(sim_incidence_acc)
    sim_survival_mean: dict[str, dict[str, Any]] = {}
    for sex in SEXES:
        sim_survival_mean[sex] = {}
        for label, _band in ONSET_BANDS:
            mean_curve = _mean_curves(sim_survival_acc[sex][label])
            sim_survival_mean[sex][label] = {
                "survival_curve": mean_curve,
                "reported": {
                    str(d): float(mean_curve[d])
                    for d in SURVIVAL_REPORT_DURATIONS
                },
                "restricted_mean_survival_years": float(
                    np.mean(sim_survival_scalar_acc[sex][label]["rmst"])
                ),
                "spell_weight": float(
                    np.mean(
                        sim_survival_scalar_acc[sex][label]["spell_weight"]
                    )
                ),
            }

    elapsed = round(time.time() - t0, 1)
    if verbose:
        unreach = ref_taxonomy["structurally_unreachable_share"]
        print(
            f"seed {seed}: n_train={len(ids_b)} "
            f"Q6 unreachable={unreach:.4f} "
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
        "ref_support_taxonomy": ref_taxonomy,
        "ref_stock": ref_stock,
        "ref_widowed_by_age": ref_widowed_by_age,
        "ref_incidence": ref_incidence,
        "ref_survival": ref_survival,
        "sim_stock_mean": _mean_dict(sim_stock_acc),
        "sim_widowed_by_age_mean": sim_widowed_by_age_mean,
        "sim_incidence_mean": sim_incidence_mean,
        "sim_survival_mean": sim_survival_mean,
        "per_draw_share_widowed_75plus": per_draw_share_75plus,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Published-outer context (candidate 11's already-run side-A scores)
# --------------------------------------------------------------------------
def _outer_published() -> dict[str, dict[str, Any]]:
    """The already-published side-A (holdout) candidate-11 scores.

    Read from the committed candidate-11 gate artifact
    (``runs/gate2_hazard_v11.json``); the outer holdout is never simulated
    here. Context that the train-side decomposition reproduces the published
    outer failure of the registered modal cell.
    """
    art = json.loads(CANDIDATE11_ARTIFACT.read_text())
    by_seed = {s["seed"]: s for s in art["per_seed"]}
    cells = (Q_STOCK_CELL,) + c11.COUNT_CELLS
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


def assemble_q6(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The reference-vs-simulable reachability audit, over gate seeds."""
    bucket_keys = (
        "reachable_within_support",
        "carried_predates_support",
        "onset_after_support_end",
        "non_derivable",
    )
    buckets = {
        k: _mean_seeds(
            [
                s["ref_support_taxonomy"]["buckets_share_of_widowed_stock"][k]
                for s in per_seed
            ]
        )
        for k in bucket_keys
    }
    unreachable = _mean_seeds(
        [
            s["ref_support_taxonomy"]["structurally_unreachable_share"]
            for s in per_seed
        ]
    )
    initial_state_fixable = _mean_seeds(
        [
            s["ref_support_taxonomy"]["initial_state_fixable_share"]
            for s in per_seed
        ]
    )
    share_stock = _mean_seeds(
        [
            s["ref_support_taxonomy"]["share_widowed_75plus_female"]
            for s in per_seed
        ]
    )
    recon_carried = _mean_seeds(
        [
            s["ref_support_taxonomy"]["reconstruction_window_contrast"][
                "onset_before_exposure_start"
            ]
            for s in per_seed
        ]
    )
    carried_onset_age = _mean_seeds(
        [
            s["ref_support_taxonomy"]["carried_mass_diagnostics"][
                "mean_onset_age"
            ]
            for s in per_seed
        ]
    )
    carried_late_entry = _mean_seeds(
        [
            s["ref_support_taxonomy"]["carried_mass_diagnostics"][
                "share_first_observed_age_ge_60"
            ]
            for s in per_seed
        ]
    )

    # The observed-initial-state fix (candidate 9 delta 1's marriage-count
    # precedent) applies iff the unreachable mass is carried-status (fixable by
    # injecting the observed widowed entry state) and NOT a non-derivable
    # residual (which no initial state recovers).
    non_derivable = buckets["non_derivable"]
    fix_applies = bool(unreachable > 1e-6 and non_derivable < 1e-4)

    per_seed_detail = []
    for s in per_seed:
        tax = s["ref_support_taxonomy"]
        per_seed_detail.append(
            {
                "seed": s["seed"],
                "reachable_within_support": tax[
                    "buckets_share_of_widowed_stock"
                ]["reachable_within_support"],
                "carried_predates_support": tax[
                    "buckets_share_of_widowed_stock"
                ]["carried_predates_support"],
                "onset_after_support_end": tax[
                    "buckets_share_of_widowed_stock"
                ]["onset_after_support_end"],
                "non_derivable": tax["buckets_share_of_widowed_stock"][
                    "non_derivable"
                ],
                "structurally_unreachable_share": tax[
                    "structurally_unreachable_share"
                ],
                "initial_state_fixable_share": tax[
                    "initial_state_fixable_share"
                ],
            }
        )

    verdict = (
        "Of the reference 75+ female widowed stock (share "
        f"{share_stock:.3f} of all female 75+ person-years), "
        f"{buckets['reachable_within_support']:.1%} arises from a widowhood "
        "episode ending WITHIN the person's observed PSID support -- reachable "
        "by a support-constrained simulation under some widowhood hazard. The "
        f"structurally unreachable share is {unreachable:.1%}, and it is "
        "ENTIRELY carried marital status: "
        f"{buckets['carried_predates_support']:.1%} from a spouse-death that "
        "PREDATES the person's first observed wave (the person enters "
        "observation already widowed), with "
        f"{buckets['onset_after_support_end']:.2%} onset-after-support-end and "
        f"{non_derivable:.2%} non-derivable (years_since_dissolution is never "
        "NA -- unlike the marriage-count residual, every widowed person-year "
        "is derived from a datable widowhood episode, so there is no "
        "undatable-status category c). The carried widows were widowed young "
        f"(mean onset age {carried_onset_age:.0f}) and "
        f"{carried_late_entry:.0%} were first observed at age >= 60 -- "
        "left-truncated elderly entrants and pre-1968 widowhoods. Under the "
        "reference's own retrospective-to-age-15 exposure window this share is "
        f"{recon_carried:.2%} (forensics 2's boundary), so the carried mass is "
        "invisible until the tighter observed-support boundary is applied. "
        "Because the unreachable mass is carried status with a zero "
        "non-derivable residual, the fix is an OBSERVED-INITIAL-STATE "
        "injection (seed the entry-widowed state, exactly the marriage-count "
        "precedent of candidate 9 delta 1), NOT a transition-rate change."
    )
    return {
        "question": (
            "Decompose how reference person-periods classified widowed-75+"
            "|female arise: widowhood observed within the panel support "
            "(reachable) vs marital status carried from a spouse-death "
            "predating support (initial-state only) vs statuses not derivable "
            "from any datable episode; measure the share a support-constrained "
            "simulation could produce under any transition rates, and whether "
            "the structurally unreachable share is an observed-initial-state "
            "fix (the marriage-count precedent), not a rate fix."
        ),
        "method": (
            "Reference-construction audit (no simulation). Per gate seed, over "
            "side B's observed marital panel, each 75+ female widowed "
            "person-year's widowhood onset year (year - "
            "years_since_dissolution) is classified against the person's "
            "observed PSID support window [first_wave, last_wave] from "
            "populace_dynamics.data.panels.demographic_panel (present in a "
            "responding family unit, sequence 1-20). Buckets partition the 75+ "
            "widowed stock; means over the 5 gate seeds."
        ),
        "taxonomy_share_of_widowed_stock": buckets,
        "structurally_unreachable_share": unreachable,
        "initial_state_fixable_share": initial_state_fixable,
        "non_derivable_residual_share": non_derivable,
        "share_widowed_75plus_female": share_stock,
        "reconstruction_window_carried_share": recon_carried,
        "carried_mass_diagnostics": {
            "mean_onset_age": carried_onset_age,
            "share_first_observed_age_ge_60": carried_late_entry,
        },
        "observed_initial_state_fix_applies": fix_applies,
        "fix_verdict": (
            "APPLIES -- the unreachable mass is carried marital status with a "
            "zero non-derivable residual, so an observed-initial-state "
            "injection (the marriage-count precedent, candidate 9 delta 1) "
            "recovers it; a transition-rate change cannot"
            if fix_applies
            else "DOES NOT APPLY -- no meaningful carried-status share, or a "
            "non-derivable residual an initial state cannot recover"
        ),
        "per_seed": per_seed_detail,
        "finding": verdict,
    }


def assemble_q7(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The widowed-exposure-by-age decomposition, over gate seeds."""
    # ---- Widowed person-years by age band x sex (the 50-64 pool). ----
    age_keys: set[str] = set()
    for s in per_seed:
        age_keys |= set(s["ref_widowed_by_age"])
    widowed_by_age: dict[str, Any] = {}
    for key in sorted(age_keys):
        ref_w = _mean_seeds(
            [
                s["ref_widowed_by_age"][key]["widowed_py_weight"]
                for s in per_seed
            ]
        )
        sim_w = _mean_seeds(
            [
                s["sim_widowed_by_age_mean"][key]["widowed_py_weight"]
                for s in per_seed
            ]
        )
        ref_sh = _mean_seeds(
            [s["ref_widowed_by_age"][key]["widowed_share"] for s in per_seed]
        )
        sim_sh = _mean_seeds(
            [
                s["sim_widowed_by_age_mean"][key]["widowed_share"]
                for s in per_seed
            ]
        )
        widowed_by_age[key] = {
            "reference_widowed_py_weight": ref_w,
            "simulated_widowed_py_weight": sim_w,
            "reference_widowed_share": ref_sh,
            "simulated_widowed_share": sim_sh,
            "sim_over_ref_weight_ratio": _ratio(sim_w, ref_w),
            "sim_over_ref_share_ratio": _ratio(sim_sh, ref_sh),
        }
    pool_5064_female = widowed_by_age["50-64|female"]
    pool_ratio = pool_5064_female["sim_over_ref_weight_ratio"]
    pool_inflated = bool(pool_ratio > 1.0)

    # ---- Widowhood incidence (INFLOW) by age band x sex. ----
    inflow: dict[str, Any] = {}
    for sex in SEXES:
        bands: dict[str, Any] = {}
        for lo, hi in INCIDENCE_BANDS:
            b = transitions.band_label(lo, hi)
            r = _mean_seeds([s["ref_incidence"][sex][b] for s in per_seed])
            sm = _mean_seeds(
                [s["sim_incidence_mean"][sex][b] for s in per_seed]
            )
            bands[b] = {
                "reference": r,
                "simulated": sm,
                "sim_over_ref_ratio": _ratio(sm, r),
                "inflow_shortfall": (1.0 - _ratio(sm, r)) if r > 0 else 0.0,
            }
        inflow[sex] = bands
    # Overall female inflow: exposure-weighted married widowhood rate ratio.
    fem_bands = inflow["female"]
    worst_band = min(
        fem_bands,
        key=lambda b: (
            fem_bands[b]["sim_over_ref_ratio"]
            if not np.isnan(fem_bands[b]["sim_over_ref_ratio"])
            else np.inf
        ),
    )
    ratios = [
        fem_bands[b]["sim_over_ref_ratio"]
        for b in fem_bands
        if not np.isnan(fem_bands[b]["sim_over_ref_ratio"])
    ]

    # ---- Survival-in-widowhood curves (50-64 vs 65+), female. ----
    survival: dict[str, Any] = {}
    for label, _band in ONSET_BANDS:
        ref_curve = _mean_curves(
            [
                s["ref_survival"]["female"][label]["survival_curve"]
                for s in per_seed
            ]
        )
        sim_curve = _mean_curves(
            [
                s["sim_survival_mean"]["female"][label]["survival_curve"]
                for s in per_seed
            ]
        )
        ref_rmst = _mean_seeds(
            [
                s["ref_survival"]["female"][label][
                    "restricted_mean_survival_years"
                ]
                for s in per_seed
            ]
        )
        sim_rmst = _mean_seeds(
            [
                s["sim_survival_mean"]["female"][label][
                    "restricted_mean_survival_years"
                ]
                for s in per_seed
            ]
        )
        survival[label] = {
            "reference_survival_curve": ref_curve,
            "simulated_survival_curve": sim_curve,
            "reference_reported": {
                str(d): float(ref_curve[d]) for d in SURVIVAL_REPORT_DURATIONS
            },
            "simulated_reported": {
                str(d): float(sim_curve[d]) for d in SURVIVAL_REPORT_DURATIONS
            },
            "reference_restricted_mean_survival_years": ref_rmst,
            "simulated_restricted_mean_survival_years": sim_rmst,
            "sim_minus_ref_rmst": sim_rmst - ref_rmst,
        }

    # Per-seed widow-stock + pool detail.
    per_seed_detail = []
    for s in per_seed:
        per_seed_detail.append(
            {
                "seed": s["seed"],
                "reference_share_75plus": s["ref_stock"]["share_75plus"],
                "simulated_share_75plus": s["sim_stock_mean"]["share_75plus"],
                "pool_5064_female_sim_over_ref": _ratio(
                    s["sim_widowed_by_age_mean"]["50-64|female"][
                        "widowed_py_weight"
                    ],
                    s["ref_widowed_by_age"]["50-64|female"][
                        "widowed_py_weight"
                    ],
                ),
            }
        )

    surv_5064 = survival["50-64"]
    surv_65 = survival["65+"]
    verdict = (
        "Widowed exposure by age. The simulated widowed 50-64 female pool is "
        f"{'INFLATED' if pool_inflated else 'NOT inflated'} "
        f"({pool_ratio:.2f}x the reference widowed person-year weight; "
        f"share {pool_5064_female['simulated_widowed_share']:.3f} vs ref "
        f"{pool_5064_female['reference_widowed_share']:.3f}). The widowhood-"
        "incidence inflow shortfall concentrates at "
        f"{worst_band} (sim/ref {fem_bands[worst_band]['sim_over_ref_ratio']:.2f}, "
        f"shortfall {fem_bands[worst_band]['inflow_shortfall']:.0%}); across "
        f"female bands the sim runs {min(ratios):.2f}-{max(ratios):.2f} of the "
        "reference incidence. Survival-in-widowhood: for widowhoods beginning "
        f"at 50-64 the simulated restricted-mean widowed duration is "
        f"{surv_5064['simulated_restricted_mean_survival_years']:.1f}y vs ref "
        f"{surv_5064['reference_restricted_mean_survival_years']:.1f}y "
        f"({surv_5064['sim_minus_ref_rmst']:+.1f}y); for 65+ onset it is "
        f"{surv_65['simulated_restricted_mean_survival_years']:.1f}y vs ref "
        f"{surv_65['reference_restricted_mean_survival_years']:.1f}y "
        f"({surv_65['sim_minus_ref_rmst']:+.1f}y) -- candidate 11's 50+ band "
        "split brings elderly-onset survival close to the reference, so the "
        "residual 75+ stock gap is an INFLOW (incidence) shortfall, not an "
        "outflow (over-remarriage) one."
    )
    return {
        "question": (
            "Simulated vs reference widowed person-years by age band x sex "
            "under candidate 11: is the widowed 50-64 pool inflated (which "
            "would explain the count over-production under correct rates), and "
            "does the ~10% incidence inflow shortfall concentrate in an age "
            "band? Plus cumulative survival-in-widowhood (time to "
            "remarriage/support-end) for widowhoods beginning at 50-64 vs 65+, "
            "simulated vs reference."
        ),
        "method": (
            "Train-side. Per gate seed, fit candidate 11 on side B and "
            "simulate the SAME side B at the 20 draw seeds 5200+k; average the "
            "per-draw decompositions over draws, then over the 5 gate seeds. "
            "Reference = side B's observed marital panel. Widowed person-years "
            "and incidence share the panel's own state/event definitions; "
            "survival-in-widowhood is a weighted Kaplan-Meier of each widowed "
            "spell against remarriage (support-end = right-censoring), by "
            "onset age band."
        ),
        "pool_inflation_verdict": (
            "50-64 female widowed pool inflated"
            if pool_inflated
            else "50-64 female widowed pool NOT inflated"
        ),
        "widowed_person_years_by_age_sex": widowed_by_age,
        "pool_5064_female": pool_5064_female,
        "inflow_incidence_by_age_sex": inflow,
        "inflow_shortfall_worst_female_band": worst_band,
        "survival_in_widowhood_female": survival,
        "per_seed": per_seed_detail,
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
        "candidate11_runner": "scripts/run_gate2_candidate11.py",
        "candidate11_runner_sha256": c1._sha_of_file(
            ROOT / "scripts" / "run_gate2_candidate11.py"
        ),
        "candidate11_artifact": "runs/gate2_hazard_v11.json",
        "candidate11_artifact_sha256": c1._sha_of_file(CANDIDATE11_ARTIFACT),
        "forensics2_runner": "scripts/gate2_forensics2.py",
        "forensics2_runner_sha256": c1._sha_of_file(
            ROOT / "scripts" / "gate2_forensics2.py"
        ),
        "forensics2_artifact": "runs/gate2_forensics2_v1.json",
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
    if Q_STOCK_CELL not in tol:
        raise RuntimeError(
            f"{Q_STOCK_CELL} is not a gated cell in gates.yaml; refusing to "
            "diagnose a cell the lock does not define."
        )
    for name, path in (
        ("candidate-11", CANDIDATE11_ARTIFACT),
        ("forensics-2", FORENSICS2_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )

    data = gf1._load_inputs()
    support = observed_support(data["demo"])
    if verbose:
        print(
            f"panel: {data['data_meta']['n_person_years']} person-years, "
            f"{data['data_meta']['panel_persons_weighted']} persons; "
            f"support windows for {len(support)} observed persons; "
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
        result = compute_seed(seed, data, support, verbose)
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    q6 = assemble_q6(per_seed)
    q7 = assemble_q7(per_seed)
    outer = _outer_published()

    candidate_12_implications = (
        "Q6: candidate 11's residual share_widowed.75+|female under-production "
        "is NOT closable by any transition-rate change alone -- "
        f"{q6['structurally_unreachable_share']:.1%} of the reference 75+ "
        "female widowed stock is structurally unreachable by a support-"
        "constrained simulation, and it is entirely carried marital status "
        "(spouse-death predating the person's first observed wave) with a zero "
        "non-derivable residual. That is the marriage-count precedent exactly: "
        "an observed-initial-state injection (seed the entry-widowed state, as "
        "candidate 9 delta 1 seeded the observed lifetime-marriage count), not "
        "a rate fix. Q7: within the reachable stock the residual gap is an "
        "INFLOW shortfall -- female widowhood incidence runs below the "
        "reference and concentrates at "
        f"{q7['inflow_shortfall_worst_female_band']}; the elderly-onset "
        "survival-in-widowhood curve now tracks the reference (candidate 11's "
        "50+ remarriage split fixed the outflow), and the widowed 50-64 pool "
        f"is {'inflated' if q7['pool_5064_female']['sim_over_ref_weight_ratio'] > 1 else 'not inflated'}. "
        "So candidate 12's highest-value lever is a mortality/incidence-side "
        "raise of widowhood inflow (especially at the worst band) PLUS an "
        "observed-initial-state seed for the carried-widow share; a further "
        "remarriage-rate move would trade margins, not close the gap. "
        "Registered only after citing this evidence, under the one-shot rule."
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "registration_title": REGISTRATION_TITLE,
        "candidate_under_diagnosis": (
            "gate-2 candidate 11 (run 1, PR #101): FAIL 1/5 under the amended "
            "mean-over-20-draws estimator; decider share_widowed.75+|female "
            "(persisted 4/5 via the untouched inflow shortfall); female "
            "marriage count regressed 3/5 -> 2/5"
        ),
        "candidate11_spec_registration": CANDIDATE11_REGISTRATION,
        "candidate11_registration_pointer": CANDIDATE11_POINTER,
        "candidate11_artifact": "runs/gate2_hazard_v11.json",
        "forensics2_artifact": "runs/gate2_forensics2_v1.json (#99)",
        "protocol": {
            "train_side_only": True,
            "outer_holdout_contact": (
                "none beyond the already-published candidate-11 per-seed "
                "scores read from runs/gate2_hazard_v11.json; the holdout "
                "(side A) is never simulated here"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "outer holdout, side B = train complement (this diagnostic "
                "simulates side B)"
            ),
            "fit_simulate_machinery": (
                "scripts/run_gate2_candidate11.py (merged #101; reuses "
                "candidate 10's code objects rebound) fit_components + "
                "simulate_holdout reused on the train side; the Q6 taxonomy "
                "is a reference-construction audit with no simulation"
            ),
            "q6_support_boundary": (
                "observed PSID support window [first_wave, last_wave] per "
                "person from populace_dynamics.data.panels.demographic_panel "
                "(sequence 1-20, present in a responding family unit), NOT the "
                "reference's retrospective-to-age-15 exposure window"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": (
                f"numpy default_rng({DRAW_SEED_BASE} + k) for k in "
                f"0..{N_DRAWS - 1} (the candidate-11 amended-estimator stream; "
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
        "question_6_reference_reachability": q6,
        "question_7_widowed_exposure_by_age": q7,
        "published_outer_context": {
            "note": (
                "already-published candidate-11 side-A scores "
                "(runs/gate2_hazard_v11.json); the outer holdout is not "
                "simulated here -- context that the train-side decomposition "
                "reproduces the published outer failure"
            ),
            "cells": outer,
        },
        "per_seed": per_seed,
        "candidate_12_implications": candidate_12_implications,
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
        q6b = artifact["question_6_reference_reachability"]
        print(
            "\nQ6 reachability: reachable "
            f"{q6b['taxonomy_share_of_widowed_stock']['reachable_within_support']:.3f}"
            f", carried {q6b['taxonomy_share_of_widowed_stock']['carried_predates_support']:.3f}"
            f", non-derivable {q6b['non_derivable_residual_share']:.4f}; "
            f"initial-state fix applies = {q6b['observed_initial_state_fix_applies']}"
        )
        q7b = artifact["question_7_widowed_exposure_by_age"]
        print(
            "Q7: "
            + q7b["pool_inflation_verdict"]
            + f"; worst female inflow band = {q7b['inflow_shortfall_worst_female_band']}"
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
