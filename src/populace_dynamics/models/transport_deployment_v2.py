"""W1 transport deployment v2 -- the three proven levers.

Second candidate of the audited transport ladder, registered at issue #42
comment 4952253568 (FROZEN spec; the registration wins). Scored against the
AMENDED ``gate_w1`` contract (``gates.yaml`` post-#165: 55 gated cells = 53
family-A joints + 2 family-C fingerprint reversals; family B is fully
report-only). One-shot; publishes regardless.

Three deltas vs candidate 1 (:mod:`transport_deployment_v1`), each a
forensics-1 measured lever (``runs/gate_w1_forensics1_v1.json``); EVERYTHING
else is byte-carried from v1 (streams, K=20, the regenerated-surface rule, the
family-C procedures verbatim, the family-B M4 simulation):

* **Q1 -- entry-state seeding** (marital + coresident). The synthetic-panel
  marital chain seeds each frame adult's ENTRY marital state from a
  train-fitted initial-state model -- ``P(marital state | entry-age band,
  sex)`` estimated on PSID ENTRY states (each PSID person's marital state at
  their first observed demographic wave) -- at the youngest gated marital
  band's lower edge (age :data:`BASE_ENTRY_AGE`), replacing v1's
  never-married-at-18 default; the certified CANDIDATE_16 hazards then evolve
  it to the current age. This is the contract-permitted entry-state channel
  the forensics adjudicated (terminal ``A_MARITL`` seeding remains the
  prohibited identity; this seeds the ENTRY, never the scored terminal, from a
  band x sex MODEL that regenerates per draw).
* **Q2 -- boundary support extension** (earnings). The gate-1 earnings
  marginal gains a train-fitted extension over the boundary ages 18-24 and
  60-69 PLUS a sex covariate at those bands (v1 clipped them to the nearest
  25-59 bin, overshooting participation). The 25-59 draws are byte-identical
  to v1 (the single ``u`` draw is preserved); only the boundary ages reroute.
  The 18-24 participation cell's residual PSID-heads-vs-CPS coverage delta is
  named, not chased.
* **Q3 -- child attachment + coresidence composition** (household size). The
  deployed households attach children via the certified fertility/household
  machinery: CANDIDATE_9's ``hh_size`` already counts each ego's own children
  (``compose_base`` runs ``ft.simulate`` internally), so the only all-person
  gap is that CHILDREN are never emitted as scored person-rows -- v2
  materializes minor child-rows from the maternal ``ft.simulate`` births
  (mothers only, minors only, each carrying its mother's terminal ``hh_size``
  and weight) and unions them into the family-A frame so the LOCKED all-person
  ``hh_size_share`` is scoreable. The lone-adult over-production is repaired
  through the SAME entry-state seeding (Q1's coupling): the seeded marital
  panel drives the household generator's spouse/child dynamics.

The child rows carry ``marital_status = NaN`` and ``age < 18`` so they are
excluded from every family-A cell EXCEPT ``hh_size_share`` -- the delta is
surgical. Non-boundary earnings cells and the family-C fingerprints are
byte-carried from candidate 1 (a regression is recorded in the artifact).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition as hc
from populace_dynamics.models import transport_deployment_v1 as td
from populace_dynamics.models.household_composition.components.parental_home_exit import (  # noqa: E501
    DELTA_STREAM_TAG_V6,
    child_leave_years_refit,
)

# --------------------------------------------------------------------------
# Byte-carried protocol constants (identical to candidate 1).
# --------------------------------------------------------------------------
REF_YEAR = td.REF_YEAR
CPS_ID_OFFSET = td.CPS_ID_OFFSET
TERMINAL_PERIOD = td.TERMINAL_PERIOD
GATE_SEEDS = td.GATE_SEEDS
K_DRAWS = td.K_DRAWS
FAMILY_A_STREAM_BASE = td.FAMILY_A_STREAM_BASE
FAMILY_B_STREAM_BASE = td.FAMILY_B_STREAM_BASE
FAMILY_C_TRANSITORY_STREAM = td.FAMILY_C_TRANSITORY_STREAM
ADULT_MIN_AGE = td.ADULT_MIN_AGE
DI_ANCHOR_BANDS = td.DI_ANCHOR_BANDS

# --------------------------------------------------------------------------
# Q1 -- entry-state seeding constants.
# --------------------------------------------------------------------------
#: The gated marital / coresident age bands (dfm.ADULT_BANDS), used to bucket
#: the PSID entry-state initial-state model.
INITIAL_STATE_BANDS: tuple[tuple[int, int], ...] = dfm.ADULT_BANDS
MARITAL_STATES: tuple[str, ...] = (
    "never_married",
    "married",
    "divorced",
    "widowed",
)
#: Entry the synthetic marital chain at the youngest gated marital band's
#: lower edge -- the ``inferred earlier-age married stock`` the forensics
#: adjudication names -- so the certified hazards have a real window to evolve
#: (the 25-34 band lands via the short window; older bands evolve from the
#: seeded young-adult married stock). Adults under this age enter never-married
#: at 18 as in v1 (their marital cells are report-only, below ADULT_BANDS).
BASE_ENTRY_AGE = 25
#: Dedicated stream for the entry-state categorical draw (distinct from the
#: A / B / C transitory streams and the ft/hc simulation seeds).
ENTRY_STATE_STREAM_SALT = 0x5EED

# --------------------------------------------------------------------------
# Q2 -- boundary support extension constants.
# --------------------------------------------------------------------------
#: The out-of-25-59-support boundary age ranges the gate-1 fit is extended to
#: (train-fitted per sex at the terminal period). 60-69 covers the scored
#: 62-69 band (and the 60-61 tail of the 55-61 band); 18-24 the young cell.
BOUNDARY_RANGES: tuple[tuple[int, int, str], ...] = (
    (18, 24, "18-24"),
    (60, 69, "60-69"),
)
BOUNDARY_SEXES: tuple[str, ...] = ("female", "male")

# --------------------------------------------------------------------------
# Q3 -- child attachment constants.
# --------------------------------------------------------------------------
#: Disjoint id range for materialized minor child person-rows (above the
#: CPS_ID_OFFSET family-C range so no id ever collides).
CHILD_ID_OFFSET = 2_000_000_000
#: Dedicated stream for the child sex coin-flip.
CHILD_SEX_STREAM_SALT = 0xC17D
#: Coresident-minor age ceiling (below dfm.EARN_BANDS 18+ and ADULT_BANDS 25+
#: so a materialized child never enters any non-hh_size family-A cell).
CHILD_MAX_AGE = ADULT_MIN_AGE  # 18

SPEC_RESOLUTIONS: dict[str, str] = dict(td.SPEC_RESOLUTIONS)
SPEC_RESOLUTIONS.update(
    {
        "delta_q1_entry_state_seeding": (
            "Q1 (marital + coresident): the synthetic marital panel's ENTRY "
            "state is seeded from a train-fitted initial-state model "
            "P(marital_state | entry-age band, sex) estimated on PSID ENTRY "
            "states (each PSID person's reconstructed marital_state at their "
            "FIRST observed demographic wave, bucketed by entry-age band x "
            "sex, weighted), replacing v1's never-married-at-18 default. Each "
            "frame adult enters at BASE_ENTRY_AGE=25 (the youngest gated "
            "marital band's lower edge) with a state drawn from the 25-34 "
            "entry-band model; the certified CANDIDATE_16 hazards evolve it to "
            "the current age. Entry states (first-wave) are used, NOT the "
            "settled cross-section, because the cross-section over-states "
            "young-adult marriage (0.60 vs the frame's 0.48 at 25-34F); the "
            "entry-state model lands 0.467 there. CONTRACT-PERMITTED: the "
            "simulator (simulator.py:147-196) reads the panel's ENTRY state as "
            "the initial condition; this seeds the entry from a band x sex "
            "MODEL that regenerates per draw (across-draw sd > 0, never reads "
            "the person's own A_MARITL), so it is NOT the prohibited identity. "
            "Duration/years-since are seeded at 0 (fresh at entry)."
        ),
        "delta_q2_boundary_support_extension": (
            "Q2 (earnings): the gate-1 cell marginals gain a train-fitted "
            "boundary extension over ages 18-24 and 60-69, each fit PER SEX on "
            "the PSID family-earnings panel at the terminal period 2022 "
            "(fit_cell_marginals machinery, keyed (band, sex)). v1 clipped "
            "these ages to the nearest 25-59 bin (participation ~0.86), "
            "overshooting the boundary bands; the extension routes 18-24 and "
            "60-69 to their own marginals while 25-59 stays on the certified "
            "marginals. The single u = rng.random(n) draw is preserved, so "
            "every 25-59 earnings draw is byte-identical to v1; only the "
            "boundary ages reroute. Fit at period 2022 (not pooled) keeps the "
            "nominal scale aligned with the base marginals so the profile "
            "ratio is scale-consistent. The 18-24 participation cell carries "
            "an additional PSID-heads-vs-CPS-all-person coverage delta on top "
            "of the support gap -- named, not chased (it stays failing)."
        ),
        "delta_q3_child_attachment_coresidence": (
            "Q3 (hh_size): CANDIDATE_9's hh_size already counts each ego's own "
            "children (compose_base runs ft.simulate on the marital panel and "
            "sizes each household as 1 + spouse + child_counts + parents + "
            "nonfamily). The all-person gap is that CHILDREN are never emitted "
            "as scored person-rows (v1's regenerate_person_frame restricts to "
            "age >= 18). v2 (a) passes the SAME Q1-seeded marital panel to "
            "hc.simulate so the seeded married stock lifts spouse presence "
            "(the coresidence repair of the lone-adult over-production) and "
            "drives the certified fertility, and (b) materializes minor "
            "child-rows from the MATERNAL ft.simulate births (mothers only, so "
            "the two independent synthetic parents never double a child; "
            "minors only via child_leave_years_refit with leave_year > 2024, "
            "so coresident young adults -- already adult rows -- are not "
            "double-counted), each carrying its mother's terminal hh_size and "
            "weight, unioned into the family-A frame. Children carry "
            "marital_status = NaN and age < 18, so ONLY hh_size_share moves; "
            "all other family-A cells stay as the adult surface. The residual "
            "size-1 over-production (young adults simulated as lone rather than "
            "coresident with parents) is NOT in this candidate's deltas "
            "(coresident_parent initial rosters are byte-carried False) and is "
            "disclosed, not chased."
        ),
        "byte_carry_from_candidate1": (
            "Everything not touched by the three deltas is byte-carried from "
            "transport_deployment_v1: the family-C fingerprint procedures "
            "(transport_career_panel + the #115/#117 ledgers, run on the base "
            "marginals verbatim), the family-B M4-simulated DI margins (now "
            "scored report-only vs retained_anchors, gating nothing), the "
            "K=20 stream bases (9100/9200/9300), the household-disjoint "
            "holdout rule, the regenerated_surface conformance rule, and every "
            "25-59 earnings draw. A byte-carry regression vs "
            "runs/gate_w1_candidate1_v1.json is recorded in the artifact."
        ),
    }
)


# --------------------------------------------------------------------------
# Deployed-generator bundle -- v1's plus the two new train-fitted objects.
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DeployedGeneratorsV2:
    """Candidate-1 generators plus the Q1 initial-state model and the Q2
    boundary marginals (both train-fitted on the full PSID sample, frozen)."""

    base: td.DeployedGenerators
    #: Q1: {(band_label, sex): np.ndarray over MARITAL_STATES} entry-state probs
    initial_state_model: dict[tuple[str, str], np.ndarray]
    #: Q2: {(band_label, sex): CellMarginal} boundary earnings marginals
    boundary_marginals: dict[tuple[str, str], Any]
    fit_provenance: dict[str, Any]
    fit_vs_raw: dict[str, Any]

    # convenience passthroughs to the byte-carried base bundle
    @property
    def earnings_marginals(self) -> dict:
        return self.base.earnings_marginals

    @property
    def age_bin_fn(self):
        return self.base.age_bin_fn

    @property
    def fitted_ft(self):
        return self.base.fitted_ft

    @property
    def fitted_hc(self):
        return self.base.fitted_hc

    @property
    def m4_prevalence(self):
        return self.base.m4_prevalence

    @property
    def m4_bands(self):
        return self.base.m4_bands


# --------------------------------------------------------------------------
# Q1 -- initial-state model (P(state | entry-age band, sex) on PSID entries).
# --------------------------------------------------------------------------
def _band_label(age: int) -> str | None:
    for lo, hi in INITIAL_STATE_BANDS:
        if lo <= age <= hi:
            return dfm._band_label(lo, hi)
    return None


def fit_initial_state_model(
    marital_panel: transitions.MaritalPanel,
    demographic_panel: pd.DataFrame,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    """Fit ``P(marital_state | entry-age band, sex)`` on PSID ENTRY states.

    Each PSID person's ENTRY state = their reconstructed ``marital_state`` at
    the FIRST demographic wave they are observed in (their real entry age),
    bucketed by entry-age band x sex, weighted. Returns the model and its
    fit-vs-raw provenance (the weighted-fit shares vs the unweighted-raw
    counts, per band x sex).
    """
    py = marital_panel.person_years.dropna(subset=["marital_state"]).copy()
    first_wave = demographic_panel.groupby("person_id")["period"].min()
    py = py.assign(_first_wave=py["person_id"].map(first_wave))
    entry = py[py["year"] == py["_first_wave"]].copy()
    entry["_band"] = entry["age"].astype(int).map(_band_label)
    entry = entry[entry["_band"].notna()]

    model: dict[tuple[str, str], np.ndarray] = {}
    raw: dict[str, Any] = {}
    for (band, sex), g in entry.groupby(["_band", "sex"]):
        w = g["weight"].to_numpy(dtype=np.float64)
        st = g["marital_state"].to_numpy(dtype=object)
        wsum = w.sum()
        fit = np.array(
            [w[st == s].sum() for s in MARITAL_STATES], dtype=np.float64
        )
        fit = fit / fit.sum() if fit.sum() > 0 else fit
        model[(band, sex)] = fit
        n = len(g)
        raw_counts = np.array(
            [(st == s).sum() for s in MARITAL_STATES], dtype=np.float64
        )
        raw_share = raw_counts / n if n else raw_counts
        raw[f"{band}|{sex}"] = {
            "n_entrants": int(n),
            "weighted_fit_share": {
                s: float(fit[i]) for i, s in enumerate(MARITAL_STATES)
            },
            "unweighted_raw_share": {
                s: float(raw_share[i]) for i, s in enumerate(MARITAL_STATES)
            },
            "max_abs_fit_minus_raw": (
                float(np.max(np.abs(fit - raw_share))) if wsum > 0 else None
            ),
        }
    return model, raw


def _seed_band_for_entry(entry_age: int) -> str:
    """Entry-age band used to seed a person entering at ``entry_age``."""
    return _band_label(entry_age) or dfm._band_label(*INITIAL_STATE_BANDS[0])


def build_seeded_marital_panel(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    model: dict[tuple[str, str], np.ndarray],
    seed: int,
) -> transitions.MaritalPanel:
    """Build a MaritalPanel whose ENTRY person-year carries a model-seeded
    marital state at BASE_ENTRY_AGE (Q1). The certified simulator reads that
    entry row as the initial condition and evolves it to the censor year."""
    age_i = age.astype(int)
    birth_year = (REF_YEAR - age).astype(int)
    sexes = np.where(is_female, "female", "male")
    # entry age: BASE_ENTRY_AGE for adults in the gated bands; the person's own
    # age (>= 18) for the very young (never-married, unscored bands).
    entry_age = np.where(age_i >= BASE_ENTRY_AGE, BASE_ENTRY_AGE, age_i)
    entry_age = np.maximum(entry_age, 18)
    start_year = birth_year + entry_age

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, ENTRY_STATE_STREAM_SALT])
    )
    mstate = np.array(["never_married"] * len(person_id), dtype=object)
    seedable = age_i >= BASE_ENTRY_AGE
    for i in np.nonzero(seedable)[0]:
        band = _seed_band_for_entry(int(entry_age[i]))
        probs = model.get((band, sexes[i]))
        if probs is None or not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        mstate[i] = MARITAL_STATES[
            int(rng.choice(len(MARITAL_STATES), p=probs))
        ]

    duration = np.where(mstate == "married", 0, np.nan)
    years_since = np.where(np.isin(mstate, ["divorced", "widowed"]), 0, np.nan)
    person_years = pd.DataFrame(
        {
            "person_id": person_id,
            "year": start_year.astype(int),
            "marital_state": mstate,
            "marriage_duration": pd.array(duration, dtype="Int64"),
            "years_since_dissolution": pd.array(years_since, dtype="Int64"),
        }
    )
    attrs = pd.DataFrame(
        {
            "person_id": person_id,
            "birth_year": birth_year,
            "sex": sexes,
            "start_exposure_year": start_year.astype(int),
            "censor_year": REF_YEAR,
            "weight": weight,
        }
    )
    return transitions.MaritalPanel(
        person_years=person_years, events=pd.DataFrame(), attrs=attrs
    )


def regenerate_marital_v2(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    fitted_ft: Any,
    model: dict[tuple[str, str], np.ndarray],
    seed: int,
) -> pd.Series:
    """Terminal marital status per person via CANDIDATE_16 (ft.simulate) from
    the Q1 model-seeded entry state."""
    panel = build_seeded_marital_panel(
        person_id, age, is_female, weight, model, seed
    )
    hold = {int(x) for x in person_id}
    sim_panel, _ = ft.simulate(panel, hold, fitted_ft, seed)
    py = sim_panel.person_years
    term = py.loc[
        py["year"] == py.groupby("person_id")["year"].transform("max")
    ]
    return term.set_index("person_id")["marital_state"]


# --------------------------------------------------------------------------
# Q2 -- boundary earnings extension (byte-carries v1's 25-59 draws).
# --------------------------------------------------------------------------
def regenerate_earnings_v2(
    ages: np.ndarray,
    is_female: np.ndarray,
    rng: np.random.Generator,
    base_marginals: dict,
    boundary_marginals: dict[tuple[str, str], Any],
    age_bin_fn,
    period: int = TERMINAL_PERIOD,
) -> np.ndarray:
    """Draw earnings: boundary ages 18-24 / 60-69 from the sex-specific
    boundary marginals, every other age from the certified base marginal
    (byte-identical to v1). The single ``u`` draw is preserved."""
    ages = np.asarray(ages, dtype=np.float64)
    fem = np.asarray(is_female, dtype=bool)
    out = np.zeros(len(ages), dtype=np.float64)
    u = rng.random(len(ages))  # SAME single draw as v1's regenerate_earnings
    handled = np.zeros(len(ages), dtype=bool)

    def _apply(idx: np.ndarray, cell) -> None:
        if cell is None or len(idx) == 0:
            return
        ub = u[idx]
        pos = ub >= cell.p0
        if cell.p0 < 1.0 and pos.any():
            pr = (ub[pos] - cell.p0) / (1.0 - cell.p0)
            out[idx[pos]] = cell.quantile(pr)

    for lo, hi, label in BOUNDARY_RANGES:
        in_range = (ages >= lo) & (ages <= hi)
        for sex in BOUNDARY_SEXES:
            idx = np.nonzero(in_range & (fem == (sex == "female")))[0]
            _apply(idx, boundary_marginals.get((label, sex)))
            handled[idx] = True

    # every non-boundary age via the certified base marginal (v1 path)
    rest = ~handled
    bins = age_bin_fn(ages)
    for b in np.unique(bins[rest]) if rest.any() else np.array([], int):
        idx = np.nonzero(rest & (bins == b))[0]
        _apply(idx, base_marginals.get((int(b), int(period))))
    return out


def fit_boundary_marginals(
    earnings_panel: pd.DataFrame,
    person_sex: pd.Series,
    period: int = TERMINAL_PERIOD,
) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """Fit boundary CellMarginals for ages 18-24 / 60-69 per sex at ``period``.

    Uses the certified gate-1 CellMarginal construction (weighted p0 +
    positive plotting-position quantile map) on the PSID family-earnings panel
    restricted to the boundary ages and the terminal period, joined to
    person-constant sex. Returns the marginals and their fit-vs-raw record
    (the fitted implied participation 1 - p0 vs the raw weighted PSID boundary
    participation)."""
    from run_gate1_candidate5b import CellMarginal, _plotting_positions

    df = earnings_panel.copy()
    df = df[(df["period"] == period) & (df["weight"] > 0)]
    df = df.assign(_sex=df["person_id"].map(person_sex))
    df = df[df["_sex"].isin(BOUNDARY_SEXES)]

    marginals: dict[tuple[str, str], Any] = {}
    raw: dict[str, Any] = {}
    for lo, hi, label in BOUNDARY_RANGES:
        band = df[(df["age"] >= lo) & (df["age"] <= hi)]
        for sex in BOUNDARY_SEXES:
            g = band[band["_sex"] == sex]
            earn = g["earnings"].to_numpy(dtype=np.float64)
            wt = g["weight"].to_numpy(dtype=np.float64)
            is_pos = earn > 0
            w_total = float(wt.sum())
            p0 = float(wt[~is_pos].sum() / w_total) if w_total > 0 else 1.0
            if is_pos.any():
                wtil, ys = _plotting_positions(earn[is_pos], wt[is_pos])
                cell = CellMarginal(
                    p0, wtil, ys, int(is_pos.sum()), float(wt[is_pos].sum())
                )
            else:
                cell = CellMarginal(
                    p0,
                    np.empty(0),
                    np.empty(0),
                    0,
                    0.0,
                )
            marginals[(label, sex)] = cell
            raw[f"{label}|{sex}"] = {
                "n_person_years": int(len(g)),
                "fit_participation_1_minus_p0": float(1.0 - p0),
                "raw_weighted_participation": (
                    float(wt[is_pos].sum() / w_total) if w_total > 0 else None
                ),
                "fit_positive_median": (
                    float(cell.quantile(np.array([0.5]))[0])
                    if cell.n_pos > 0
                    else None
                ),
            }
    return marginals, raw


# --------------------------------------------------------------------------
# Fit the v2 generator bundle (base + the two train-fitted deltas), frozen.
# --------------------------------------------------------------------------
def fit_generators(m4_artifact_path: str) -> DeployedGeneratorsV2:
    """Fit the byte-carried base generators plus the Q1 initial-state model
    and the Q2 boundary marginals, once on the full PSID sample."""
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_gate1_baseline as g1base

    from populace_dynamics.data import marriage
    from populace_dynamics.models.family_transitions import (
        evaluation as fteval,
    )

    base = td.fit_generators(m4_artifact_path)
    prov: dict[str, Any] = dict(base.fit_provenance)

    # Q1 -- initial-state model on PSID entry states.
    t = time.time()
    src = fteval._load_sources()
    model, q1_raw = fit_initial_state_model(src.panel, src.demographic_panel)
    prov["q1_initial_state_model"] = {
        "source": "PSID first-observed-wave marital_state by entry-age band x sex",
        "n_cells": len(model),
        "bands": [dfm._band_label(lo, hi) for lo, hi in INITIAL_STATE_BANDS],
        "base_entry_age": BASE_ENTRY_AGE,
        "fit_seconds": round(time.time() - t, 1),
    }

    # Q2 -- boundary earnings marginals per sex at the terminal period.
    t = time.time()
    raw_panel = g1base.family_earnings_panel()
    person_sex = (
        marriage.marriage_history()
        .dropna(subset=["sex"])
        .groupby("person_id")["sex"]
        .first()
    )
    boundary, q2_raw = fit_boundary_marginals(raw_panel, person_sex)
    prov["q2_boundary_marginals"] = {
        "source": "run_gate1_baseline.family_earnings_panel, ages 18-24/60-69 x sex, period 2022",
        "n_cells": len(boundary),
        "ranges": [f"{lo}-{hi}" for lo, hi, _ in BOUNDARY_RANGES],
        "sex_covariate": True,
        "fit_seconds": round(time.time() - t, 1),
    }

    return DeployedGeneratorsV2(
        base=base,
        initial_state_model=model,
        boundary_marginals=boundary,
        fit_provenance=prov,
        fit_vs_raw={
            "q1_entry_state_model": q1_raw,
            "q2_boundary_support": q2_raw,
        },
    )


# --------------------------------------------------------------------------
# Q3 -- child materialization (minor rows from the maternal ft.simulate births).
# --------------------------------------------------------------------------
def regenerate_household_size_v2(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    fitted_hc: Any,
    model: dict[tuple[str, str], np.ndarray],
    seed: int,
) -> tuple[pd.Series, transitions.MaritalPanel]:
    """Terminal household size per adult via CANDIDATE_9, driven by the SAME
    Q1-seeded marital panel (the coresidence repair). Returns the terminal
    hh_size Series and the seeded marital panel (reused for child rows)."""
    hh_panel = _synthetic_household_panel(person_id, age, is_female, weight)
    mpanel = build_seeded_marital_panel(
        person_id, age, is_female, weight, model, seed
    )
    hold = {int(x) for x in person_id}
    out = hc.simulate(hh_panel, mpanel, fitted_hc, hold, seed)
    pwo = out.person_waves
    term = pwo.loc[
        pwo["year"] == pwo.groupby("person_id")["year"].transform("max")
    ]
    return term.set_index("person_id")["hh_size"], mpanel


def _synthetic_household_panel(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
):
    """The v1 household roster panel (empty coresidence rosters, waves from age
    18 to the terminal year) -- byte-carried from candidate 1."""
    from populace_dynamics.data import household_composition as hcd

    sexes = np.where(is_female, "female", "male")
    frames = []
    ages_i = age.astype(int)
    for i in range(len(person_id)):
        a0 = ages_i[i]
        entry_age = max(td.HH_ENTRY_AGE, hcd.START_AGE)
        first_year = REF_YEAR - (a0 - entry_age)
        years = np.arange(first_year, REF_YEAR + 1, td.HH_CADENCE)
        if len(years) == 0 or years[-1] != REF_YEAR:
            years = np.append(years, REF_YEAR)
        wave_ages = a0 - (REF_YEAR - years)
        keep = wave_ages >= hcd.START_AGE
        years, wave_ages = years[keep], wave_ages[keep]
        if len(years) == 0:
            years, wave_ages = np.array([REF_YEAR]), np.array([a0])
        frames.append(
            pd.DataFrame(
                {
                    "person_id": person_id[i],
                    "year": years.astype(int),
                    "age": wave_ages.astype(int),
                    "sex": sexes[i],
                    "weight": weight[i],
                }
            )
        )
    pw = pd.concat(frames, ignore_index=True)
    pw["band"] = pw["age"].map(hcd._band_of)
    for flag in hcd.CORESIDENCE_LINKS:
        pw[flag] = False
    pw["multigen"] = False
    pw["hh_size"] = 1
    pw = hcd._add_transitions(pw)
    attrs = pw[["person_id"]].drop_duplicates().reset_index(drop=True)
    return hcd.HouseholdCompositionPanel(person_waves=pw, attrs=attrs)


def materialize_children(
    seeded_mpanel: transitions.MaritalPanel,
    fitted_hc: Any,
    mother_hh_size: pd.Series,
    mother_weight: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Minor child person-rows from the maternal ft.simulate births.

    Re-runs the SAME ft.simulate(seeded_mpanel, ..., seed) call compose_base
    makes internally (deterministic, identical maternal births), keeps
    coresident minors (child_leave_years_refit with leave_year > REF_YEAR and
    age < 18) attached to MOTHERS ONLY, and emits one row per minor carrying
    its mother's terminal hh_size and weight, marital_status NaN, age < 18."""
    hold = set(int(x) for x in seeded_mpanel.attrs["person_id"])
    _sim, sim_births = ft.simulate(
        seeded_mpanel, hold, fitted_hc.family_transitions, seed
    )
    if len(sim_births) == 0:
        return _empty_child_frame()
    maternal = sim_births[["parent_person_id", "birth_year"]].copy()
    maternal["parent_person_id"] = maternal["parent_person_id"].astype("int64")
    maternal["birth_year"] = maternal["birth_year"].astype("int64")
    leave_rng = np.random.default_rng(
        np.random.SeedSequence([seed, DELTA_STREAM_TAG_V6])
    )
    leaves = child_leave_years_refit(
        maternal,
        fitted_hc.parental_exit,
        fitted_hc.child_exit_single_year,
        leave_rng,
    )
    child_age = REF_YEAR - leaves["birth_year"].to_numpy(dtype=np.int64)
    leave_year = leaves["leave_year"].to_numpy(dtype=np.float64)
    keep = (
        (child_age >= 0)
        & (child_age < CHILD_MAX_AGE)
        & (leave_year > REF_YEAR)
    )
    kids = leaves.loc[keep]
    if len(kids) == 0:
        return _empty_child_frame()
    parent = kids["parent_person_id"].to_numpy(dtype=np.int64)
    hh = mother_hh_size.reindex(parent).to_numpy()
    wt = mother_weight.reindex(parent).to_numpy()
    ages = child_age[keep]
    ok = ~pd.isna(hh) & ~pd.isna(wt)
    n = int(ok.sum())
    if n == 0:
        return _empty_child_frame()
    sex_rng = np.random.default_rng(
        np.random.SeedSequence([seed, CHILD_SEX_STREAM_SALT])
    )
    return pd.DataFrame(
        {
            "person_id": np.arange(n, dtype=np.int64) + CHILD_ID_OFFSET,
            "weight": np.asarray(wt, dtype=np.float64)[ok],
            "age": np.asarray(ages, dtype=np.float64)[ok],
            "is_female": sex_rng.random(n) < 0.5,
            "earnings": np.zeros(n, dtype=np.float64),
            "marital_status": np.array([np.nan] * n, dtype=object),
            "hh_size": np.asarray(hh)[ok],
            "coresident_spouse": np.zeros(n, dtype=bool),
        }
    )


def _empty_child_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.array([], dtype=np.int64),
            "weight": np.array([], dtype=np.float64),
            "age": np.array([], dtype=np.float64),
            "is_female": np.array([], dtype=bool),
            "earnings": np.array([], dtype=np.float64),
            "marital_status": np.array([], dtype=object),
            "hh_size": np.array([], dtype=np.float64),
            "coresident_spouse": np.array([], dtype=bool),
        }
    )


# --------------------------------------------------------------------------
# Family A -- the regenerated all-person surface (adults + materialized minors).
# --------------------------------------------------------------------------
def regenerate_person_frame_v2(
    slice_df: pd.DataFrame,
    gens: DeployedGeneratorsV2,
    k: int,
    stream_base: int,
) -> pd.DataFrame:
    """Regenerate the scored family-A surface for one draw k on a slice, with
    the three deltas: Q1 seeded marital + coresident, Q2 boundary earnings, Q3
    child rows unioned so the all-person hh_size_share is scoreable."""
    seed = stream_base + k
    rng = np.random.default_rng(seed)
    adults = slice_df[slice_df["age"] >= ADULT_MIN_AGE].reset_index(drop=True)
    pid = adults["person_id"].to_numpy()
    age = adults["age"].to_numpy(dtype=np.float64)
    fem = adults["is_female"].to_numpy(dtype=bool)
    wt = adults["weight"].to_numpy(dtype=np.float64)

    # Q2 -- boundary earnings (25-59 byte-identical to v1).
    earn = regenerate_earnings_v2(
        age,
        fem,
        rng,
        gens.earnings_marginals,
        gens.boundary_marginals,
        gens.age_bin_fn,
    )
    # Q1 -- seeded marital + coresident.
    marital = regenerate_marital_v2(
        pid, age, fem, wt, gens.fitted_ft, gens.initial_state_model, seed
    )
    marital_arr = marital.reindex(pid).to_numpy(dtype=object)
    coresident = marital_arr == "married"

    # Q3 -- seeded-mpanel household size + materialized minor child rows.
    hh_size, seeded_mpanel = regenerate_household_size_v2(
        pid, age, fem, wt, gens.fitted_hc, gens.initial_state_model, seed
    )
    hh_arr = hh_size.reindex(pid).to_numpy()
    adult_frame = pd.DataFrame(
        {
            "person_id": pid,
            "weight": wt,
            "age": age,
            "is_female": fem,
            "earnings": earn,
            "marital_status": marital_arr,
            "hh_size": hh_arr,
            "coresident_spouse": coresident,
        }
    )
    child_frame = materialize_children(
        seeded_mpanel,
        gens.fitted_hc,
        hh_size,
        pd.Series(wt, index=pid),
        seed,
    )
    return pd.concat([adult_frame, child_frame], ignore_index=True)


def family_a_score(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV2,
    floor: dict,
    tolerances: dict[str, float],
    gated_cells: list[str],
    *,
    seeds: tuple[int, ...] = GATE_SEEDS,
    k_draws: int = K_DRAWS,
    progress: bool = False,
) -> dict[str, Any]:
    """Score family A per gate seed (household-disjoint holdout, K=20 draws)
    with the v2 regenerated all-person surface. Same floor pricing, statistic,
    conjunction, and dispersion disclosures as candidate 1."""
    universe = persons["household_id"].to_numpy()
    per_seed_floor = {
        s["seed"]: s["cells"] for s in floor["noise_floor_per_seed"]
    }
    n_cell = len(gated_cells)
    cube = np.full((k_draws, n_cell, len(seeds)), np.nan)

    per_seed_out = []
    for si, seed in enumerate(seeds):
        t = time.time()
        side_a = set(td.holdout_side_a_households(universe, seed).tolist())
        hold = persons[persons["household_id"].isin(side_a)].reset_index(
            drop=True
        )
        draw_rates = np.full((k_draws, n_cell), np.nan)
        for k in range(k_draws):
            regen = regenerate_person_frame_v2(
                hold, gens, k, FAMILY_A_STREAM_BASE
            )
            cells = dfm.reference_moments(regen, weighted=True)
            for ci, cell in enumerate(gated_cells):
                if cell in cells:
                    draw_rates[k, ci] = cells[cell]["rate"]
            if progress:
                print(
                    f"  seed {seed} draw {k} {time.time()-t:.0f}s", flush=True
                )
        cube[:, :, si] = draw_rates

        rate_a = np.array(
            [per_seed_floor[seed][c]["rate_a"] for c in gated_cells]
        )
        rbar = np.nanmean(draw_rates, axis=0)
        undefined = int(np.isnan(draw_rates).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.abs(np.log(rbar / rate_a))
        tols = np.array([tolerances[c] for c in gated_cells])
        cell_pass = scores <= tols
        seed_pass = bool(cell_pass.all())
        across_sd = np.nanstd(draw_rates, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_draw_abs_ln = np.abs(np.log(draw_rates / rate_a[None, :]))
        max_per_draw = np.nanmax(per_draw_abs_ln, axis=0)
        per_seed_out.append(
            {
                "seed": int(seed),
                "n_holdout_households": len(side_a),
                "n_holdout_persons": int(len(hold)),
                "seed_pass": seed_pass,
                "n_cells_pass": int(cell_pass.sum()),
                "n_cells_fail": int((~cell_pass).sum()),
                "undefined_draw_cells": undefined,
                "worst_cells": sorted(
                    [
                        {
                            "cell": gated_cells[ci],
                            "rbar": float(rbar[ci]),
                            "rate_a": float(rate_a[ci]),
                            "score": float(scores[ci]),
                            "tolerance": float(tols[ci]),
                            "pass": bool(cell_pass[ci]),
                        }
                        for ci in range(n_cell)
                    ],
                    key=lambda d: -d["score"],
                )[:12],
                "per_cell": {
                    gated_cells[ci]: {
                        "rbar": float(rbar[ci]),
                        "rate_a": float(rate_a[ci]),
                        "score": float(scores[ci]),
                        "tolerance": float(tols[ci]),
                        "pass": bool(cell_pass[ci]),
                        "across_draw_sd": float(across_sd[ci]),
                        "max_per_draw_abs_ln": float(max_per_draw[ci]),
                    }
                    for ci in range(n_cell)
                },
                "elapsed_seconds": round(time.time() - t, 1),
            }
        )
        if progress:
            print(
                f"seed {seed}: {'PASS' if seed_pass else 'FAIL'} "
                f"({int(cell_pass.sum())}/{n_cell}) {time.time()-t:.0f}s",
                flush=True,
            )

    n_seed_pass = sum(1 for s in per_seed_out if s["seed_pass"])
    return {
        "gated_cells": gated_cells,
        "cube_shape": list(cube.shape),
        "cube": cube.tolist(),
        "per_seed": per_seed_out,
        "n_seed_pass": n_seed_pass,
        "family_a_pass": n_seed_pass >= 4,
    }


# --------------------------------------------------------------------------
# Family B -- byte-carried M4 simulation, scored REPORT-ONLY vs retained_anchors.
# --------------------------------------------------------------------------
def family_b_report(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV2,
    retained_anchors: dict[str, Any],
    report_reasons: dict[str, Any],
    *,
    k_draws: int = K_DRAWS,
    pe_us_dir: str | None = None,
) -> dict[str, Any]:
    """Compute the 10 family-B cells (M4-simulated DI margins) vs the amended
    contract's retained_anchors, DISCLOSED report-only -- family B gates
    NOTHING after amendment 1 (gated_cells is empty). Byte-carries the v1 M4
    simulation; only the surface is relabeled report-only."""
    res = td.family_b_score(
        persons,
        gens,  # DeployedGeneratorsV2 exposes m4_prevalence/m4_bands passthroughs
        {"gated_cells": retained_anchors},
        k_draws=k_draws,
        pe_us_dir=pe_us_dir,
    )
    res["reported_not_gated"] = True
    res["gates_nothing_after_amendment_1"] = True
    res["report_reasons"] = report_reasons
    res["n_report_cells"] = res.pop("n_cells", len(res.get("per_cell", {})))
    res["n_report_cells_within_tolerance"] = res.pop(
        "n_cells_pass", sum(1 for c in res["per_cell"].values() if c["pass"])
    )
    # family B contributes NOTHING to the gate; drop the pass label.
    res["family_b_pass"] = None
    res["contributes_to_gate"] = False
    return res


# --------------------------------------------------------------------------
# Family C -- byte-carried verbatim from candidate 1 (base marginals).
# --------------------------------------------------------------------------
def family_c(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV2,
    family_c_contract: dict,
) -> dict[str, Any]:
    """Re-run both compression fingerprints via the candidate-1 procedure
    verbatim (transport_career_panel on the BASE marginals + the #115/#117
    ledgers). Byte-carried: the boundary marginals are NOT used here."""
    return td.family_c(persons, gens.base, family_c_contract)
