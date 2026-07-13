"""W1 transport deployment v3 -- the three forensics-2-proven levers.

Third candidate of the audited transport ladder. Composes the certified
PSID-estimated generators onto the certified populace CPS frame (bundle
``us-4.18.8``, sha ``c2065b64...``) and scores them against the ``gate_w1``
contract, binding the gated partition from ``gates.yaml`` at RUN time so the
candidate automatically scores whatever surface is locked when its registered
one-shot eventually runs (before amendment 2: the 53 family-A + 2 family-C
gated cells; after the amendment-2 flip: the residual 47 family-A joints + the
C2 fingerprint -- the script never hard-codes either).

Three deltas vs candidate 2 (:mod:`transport_deployment_v2`), each an adjudicated
forensics-2 lever (``runs/gate_w1_forensics2_v1.json``). Everything else is
byte-carried from v2/v1 (K=20, the 9100/9200/9300 stream bases, the
household-disjoint holdout, the regenerated-surface rule, the family-C
procedures verbatim, the family-B M4 simulation report-only):

* **Lever 1 -- CPS-anchored entry-level marital model** (forensics-2 Q6). The
  synthetic-panel marital chain seeds each frame adult's marital ENTRY state at
  :data:`BASE_ENTRY_AGE` (25) from the FRAME's OWN CPS young-adult married share
  by sex -- read from the pinned frame's ``deployment_frame.reference_moments``
  entry-band (25-34) cell -- then evolves the certified CANDIDATE_16 hazards to
  the current age. This is the contract-permitted CPS-anchored ENTRY the Q6
  ``contract_adjudication.determination`` blesses ("a CPS-anchored ENTRY model
  ... is CONTRACT-PERMITTED"); v2 seeded from the PSID first-wave stock instead.
  Back-solving the entry so a band's TERMINAL reproduces its own ``rate_a`` --
  and reading any gated 65+ / older-terminal moment as an anchor -- is the
  PROHIBITED identity-in-disguise the same field names, so
  :func:`build_cps_entry_anchor` reads ONLY the entry band and asserts at
  construction that no 65+/terminal moment is read. The measured LIMIT (Q6): the
  entry lever fixes the 25-34 overshoot but CANNOT raise the 65+ undershoot (a
  hazard-vintage miss); candidate 3 does not chase 65+ (amendment 2 demotes it).

* **Lever 2 -- interior sex covariate on the earnings marginals** (forensics-2
  Q8, 4/4 clean, no collateral). gate-1 fits the interior 25-59 earnings
  marginals with NO sex covariate (one ``(age_bin, period)`` marginal for both
  sexes); v1/v2 byte-carry that. v3 extends the Q2 boundary treatment INWARD:
  per-``(interior band, sex)`` marginals at the terminal period 2022 over
  :data:`INTERIOR_BANDS`, exactly where Q8 validated the split
  (``q8_interior_sex_covariate.finding``: "clears 4 of 4 byte-carried F/M cells
  ... splitting the interior marginals by sex does not perturb any
  currently-passing interior earnings cell"). The single ``u`` draw is preserved.

* **Lever 3 -- co-designed coresident_parent roster + fertility window**
  (forensics-2 Q7). The size-1 overshoot is the young-lone-adult residual: v2
  seeds the household roster's ``coresident_parent`` FALSE (young adults never
  attach to a parental home) AND truncates fertility at the base-25 marital
  entry. Q7 finds two levers -- (a) a train-fitted coresident_parent initial
  roster (seeded True at wave-0 per the 15-24 stock ~0.69 F / 0.76 M, evolved by
  the certified parental-exit hazard) and (b) the fuller fertility window (a
  15-entry marital panel so ``ft.simulate`` fires the 15-24 maternal ages) --
  each alone "necessary but INSUFFICIENT", and warns they "must be co-designed"
  because the fuller window trades against the marital seeding. v3 CO-DESIGNS
  them on ONE shared child ledger: the roster evolution and the materialized
  children BOTH consume the same certified ``parental_exit`` hazard and the same
  15-entry maternal births, so a coresident child not-yet-left IS a coresident
  young adult and the materialized minors are exactly the children
  ``compose_base`` counts -- consistent, not independently layered
  (``q7_coresident_parent_fertility.coupling_caveat``: "The marital cells are
  Q6's domain; Q7 scores hh_size only" -- so the co-designed hh_size panel uses
  the 15-entry window while the marital/coresident cells keep lever 1's
  CPS-anchored 25-entry panel; the coupling is disclosed, not chased).

The 18-24 participation cells and the C1 fingerprint are AMENDMENT-2 questions
(``q9_concept_cells``: a measured >=15pp population-concept delta and a robust
non-reversal), NOT candidate levers -- v3 spends no lever on either and carries
them report-only, exactly as ``candidate3_design_implications.q9`` directs.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.data import household_composition as hcd
from populace_dynamics.data import transitions
from populace_dynamics.models import family_transitions as ft
from populace_dynamics.models import household_composition as hc
from populace_dynamics.models import transport_deployment_v1 as td1
from populace_dynamics.models import transport_deployment_v2 as td2

# --------------------------------------------------------------------------
# Byte-carried protocol constants (identical to candidates 1 and 2).
# --------------------------------------------------------------------------
REF_YEAR = td1.REF_YEAR
CPS_ID_OFFSET = td1.CPS_ID_OFFSET
CHILD_ID_OFFSET = td2.CHILD_ID_OFFSET
TERMINAL_PERIOD = td1.TERMINAL_PERIOD
GATE_SEEDS = td1.GATE_SEEDS
K_DRAWS = td1.K_DRAWS
#: Candidate 3 uses a DISTINCT family-A draw stream base (``5200 + k``) so its
#: registered one-shot draws are independent of candidates 1/2 (``9100 + k``):
#: each candidate is a separately registered one-shot, and reusing a prior draw
#: stream would correlate the runs. Family B (9200) and family C (9300) stay
#: byte-carried -- family B is report-only and family C is untouched by the
#: levers, so their internal streams are inherited verbatim from v1.
FAMILY_A_STREAM_BASE = 5200
FAMILY_B_STREAM_BASE = td1.FAMILY_B_STREAM_BASE
FAMILY_C_TRANSITORY_STREAM = td1.FAMILY_C_TRANSITORY_STREAM
ADULT_MIN_AGE = td1.ADULT_MIN_AGE
DI_ANCHOR_BANDS = td1.DI_ANCHOR_BANDS
HH_CADENCE = td1.HH_CADENCE
CHILD_MAX_AGE = td2.CHILD_MAX_AGE  # 18: a materialized child row is a minor
MARITAL_STATES = td2.MARITAL_STATES

# --------------------------------------------------------------------------
# Lever 1 -- CPS-anchored entry-level marital model (Q6).
# --------------------------------------------------------------------------
#: Enter the synthetic marital chain at the youngest gated marital band's lower
#: edge; the certified hazards evolve the seeded state to the current age.
BASE_ENTRY_AGE = td2.BASE_ENTRY_AGE  # 25
#: The CPS ENTRY-anchor band -- the youngest gated marital band. The entry
#: anchor reads the frame's own married share HERE and ONLY here. Q6:
#: "for the youngest band the entry age ~= terminal age, so a CPS-25-34-anchored
#: entry ... must anchor the entry LEVEL at age 25, not the 25-34 terminal" --
#: which is exactly what seeding at BASE_ENTRY_AGE and regenerating via hazards
#: does (across_draw_sd > 0, never the identity).
ENTRY_ANCHOR_BAND = "25-34"
#: Band tokens a CPS entry anchor must NEVER read: the 65+ and older-terminal
#: marital bands. Reading one back-solves the entry from a scored terminal
#: state -- the PROHIBITED identity-in-disguise
#: (``q6_marital_calibration_frame.contract_adjudication.determination``).
PROHIBITED_ANCHOR_BANDS: tuple[str, ...] = ("35-44", "45-54", "55-64", "65+")
#: Reused entry-state categorical draw salt (distinct from the A/B/C streams).
ENTRY_STATE_STREAM_SALT = td2.ENTRY_STATE_STREAM_SALT

# --------------------------------------------------------------------------
# Lever 2 -- interior sex covariate (Q8). Boundary (Q2) is byte-carried from v2.
# --------------------------------------------------------------------------
#: The in-support interior earnings bands the sex covariate splits -- exactly
#: the bands Q8 fit and validated (== ``deployment_frame.DISPERSION_BANDS``:
#: 25-34 / 35-44 / 45-54 / 55-61). Ages 18-24 and 60-69 stay on the v2 boundary
#: marginals (Q2); the boundary is applied FIRST, so the 55-61 band's 55-59
#: reroute to the interior sex marginal while 60-61 stay on the 60-69 boundary,
#: exactly as v2 mixed them.
INTERIOR_BANDS: tuple[tuple[int, int], ...] = dfm.DISPERSION_BANDS
INTERIOR_SEXES: tuple[str, ...] = ("female", "male")
BOUNDARY_RANGES = td2.BOUNDARY_RANGES
BOUNDARY_SEXES = td2.BOUNDARY_SEXES

# --------------------------------------------------------------------------
# Lever 3 -- co-designed coresident_parent roster + fertility window (Q7).
# --------------------------------------------------------------------------
#: Enter the co-designed household + fertility panels at the certified
#: fertility support floor (START_AGE 15) so (a) the coresident_parent roster
#: seeds the 15-24 stock at wave-0 and (b) ft.simulate fires the 15-24 maternal
#: ages. Both share this ONE entry age -- the shared child ledger.
FERTILITY_ENTRY_AGE = hcd.START_AGE  # 15
#: The young-adult coresident_parent stock the wave-0 roster is seeded from
#: (train-fitted; the certified parental-exit hazard evolves it out).
ROSTER_SEED_BAND = "15-24"
#: Dedicated stream for the wave-0 coresident_parent Bernoulli seed (distinct
#: from the entry-state / child-sex / A-B-C streams and the certified internal
#: occupancy streams).
ROSTER_SEED_STREAM_SALT = 0xB0A5

# Forensics-2 artifact fields the levers bind to (cited in-code above and echoed
# into the run artifact's spec_resolutions for the referee).
SPEC_RESOLUTIONS: dict[str, str] = dict(td2.SPEC_RESOLUTIONS)
SPEC_RESOLUTIONS.update(
    {
        "delta_lever1_cps_anchored_entry": (
            "Lever 1 (marital + coresident): the synthetic marital panel's ENTRY "
            "state is seeded at BASE_ENTRY_AGE=25 from the FRAME's OWN CPS "
            "young-adult married share by sex -- read from "
            "deployment_frame.reference_moments at the ENTRY band 25-34 "
            "(marital_share.married.25-34|{sex}) -- then evolved by the certified "
            "CANDIDATE_16 hazards. This REPLACES v2's PSID first-wave initial-state "
            "model with the CPS-anchored ENTRY that "
            "runs/gate_w1_forensics2_v1.json q6_marital_calibration_frame."
            "contract_adjudication.determination adjudicates CONTRACT-PERMITTED "
            "('a CPS-anchored ENTRY model ... is CONTRACT-PERMITTED ... it is "
            "structurally identical to the PSID-entry model ... only the "
            "calibration target differs'). It anchors the entry LEVEL at age 25 "
            "(not the 25-34 terminal), regenerates per draw (across_draw_sd > 0, "
            "never the person's own A_MARITL), and build_cps_entry_anchor asserts "
            "NO gated 65+/older-terminal moment is read -- the PROHIBITED "
            "back-solve / identity-in-disguise the same determination names. "
            "MEASURED LIMIT (Q6): fixes the 25-34 overshoot, cannot raise the 65+ "
            "undershoot (a hazard-vintage miss, not an entry-level miss); v3 does "
            "not chase 65+ (amendment 2 demotes the 65+ quad)."
        ),
        "delta_lever2_interior_sex_covariate": (
            "Lever 2 (earnings): the interior 25-59 earnings marginals gain a sex "
            "covariate -- per-(interior band, sex) CellMarginals fit at the "
            "terminal period 2022 over INTERIOR_BANDS "
            "(deployment_frame.DISPERSION_BANDS: 25-34/35-44/45-54/55-61), the "
            "boundary (Q2) treatment extended INWARD exactly where "
            "runs/gate_w1_forensics2_v1.json q8_interior_sex_covariate validated "
            "it (finding: 'clears 4 of 4 byte-carried F/M cells ... no collateral "
            "... splitting the interior marginals by sex does not perturb any "
            "currently-passing interior earnings cell'). gate-1 fit these bands "
            "with NO sex covariate (one (age_bin, period) marginal for both "
            "sexes), overstating female participation; the split routes each "
            "interior (band, sex) to its own marginal. The single u = rng.random(n) "
            "draw is preserved (only the marginal each u maps through is now "
            "sex-specific); the boundary 18-24/60-69 marginals are byte-carried "
            "from v2 and applied first, so 60-61 stay on the 60-69 boundary."
        ),
        "delta_lever3_codesigned_roster_fertility": (
            "Lever 3 (hh_size): the coresident_parent roster and the fertility "
            "window are CO-DESIGNED on ONE shared child ledger, not layered "
            "independently. (a) The household roster seeds coresident_parent TRUE "
            "at wave-0 (age 15) with the train-fitted 15-24 stock by sex (~0.69 F "
            "/ 0.76 M), which the certified parental-exit hazard evolves out "
            "(evolve_absorbing_exit) so young adults still home at 2024 count "
            "their parent(s) in hh_size -- the size-1 lever. (b) The marital / "
            "fertility panel enters at FERTILITY_ENTRY_AGE=15 (never-married) so "
            "ft.simulate fires the 15-24 maternal ages -- the large-size lever. "
            "CONSISTENCY: BOTH the roster's coresident_parent evolution and the "
            "materialized children draw the SAME certified parental_exit hazard, "
            "and the child rows are materialized from the SAME 15-entry maternal "
            "births compose_base counts (identical panel + seed + "
            "SeedSequence([seed, DELTA_STREAM_TAG_V6]) leave draw), so a "
            "coresident child not-yet-left IS a coresident young adult and the "
            "materialized minors (age < 18) are exactly compose_base's counted "
            "children -- disjoint from the 18-24 coresident young adults, no "
            "double-count. runs/gate_w1_forensics2_v1.json "
            "q7_coresident_parent_fertility.finding: each lever alone is "
            "'necessary but INSUFFICIENT'; jointly they move every hh_size cell "
            "toward the frame but clear only size-2 -- v3 does not claim to land "
            "the household-size family. coupling_caveat: the 15-entry window "
            "reverts the 25-entry marital seeding, so the co-designed hh_size "
            "panel uses the 15-entry window while the marital/coresident SCORING "
            "cells keep lever 1's CPS-anchored 25-entry panel (Q6's domain) -- the "
            "coupling is disclosed, not chased."
        ),
        "amendment2_concept_cells_carried_report_only": (
            "The 18-24 participation cells and the C1 (ppi_nra) fingerprint are "
            "AMENDMENT-2 questions, not candidate-3 levers "
            "(runs/gate_w1_forensics2_v1.json q9_concept_cells: the 18-24 miss is "
            "a MEASURED >=15pp population-concept delta -- PSID head/spouse ~0.86 "
            "vs CPS all-person ~0.64 -- and C1's non-reversal is robust). v3 spends "
            "no lever on either; they are scored only if still gated in gates.yaml "
            "when the one-shot runs (the partition is bound at run time), and are "
            "carried report-only-with-disclosure exactly as "
            "candidate3_design_implications.q9 directs. The gates.yaml-bound "
            "partition means the amendment-2 flip (demoting the 18-24 pair, the "
            "65+ quad, and C1) is picked up automatically with no code change."
        ),
    }
)


# --------------------------------------------------------------------------
# Deployed-generator bundle -- v1 base + Q2 boundary + Q8 interior + Q7 roster.
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DeployedGeneratorsV3:
    """The certified generators plus the three v3 train-fitted objects.

    ``base`` is the byte-carried v1 bundle (earnings marginals, CANDIDATE_16
    family transitions, CANDIDATE_9 household composition, gate_m4 prevalence).
    ``boundary_marginals`` are v2's Q2 out-of-support sex marginals (18-24 /
    60-69). ``interior_marginals`` are the NEW Q8 in-support sex marginals
    (25-34 / 35-44 / 45-54 / 55-61). ``coresident_parent_rates`` are the NEW Q7
    train-fitted wave-0 roster stock by (band, sex). Lever 1's CPS entry anchor
    is NOT here -- it is derived from the FRAME at score time, never the
    PSID fit.
    """

    base: td1.DeployedGenerators
    #: Q2: {(band_label, sex): CellMarginal} boundary earnings marginals.
    boundary_marginals: dict[tuple[str, str], Any]
    #: Q8: {(band_label, sex): CellMarginal} interior earnings marginals.
    interior_marginals: dict[tuple[str, str], Any]
    #: Q7: {(band_label, sex): float} train coresident_parent stock.
    coresident_parent_rates: dict[tuple[str, str], float]
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
# Lever 1 -- the CPS-anchored entry-level marital model.
# --------------------------------------------------------------------------
def _anchor_band_of(cell_key: str) -> str:
    """The age band token of a ``marital_share.<status>.<band>|<sex>`` key."""
    # marital_share.married.25-34|female -> "25-34"
    return cell_key.split(".")[-1].split("|")[0]


def _assert_entry_anchor_key(cell_key: str) -> None:
    """Guard: an entry anchor reads ONLY the ENTRY band, never a 65+/terminal.

    Enforces the Q6 ``contract_adjudication.determination`` boundary: seeding the
    entry state from a CPS young-adult aggregate is permitted, but reading a
    gated 65+ or older-terminal moment back-solves the entry from the scored
    surface -- the prohibited identity-in-disguise.
    """
    band = _anchor_band_of(cell_key)
    if band in PROHIBITED_ANCHOR_BANDS or "65+" in cell_key:
        raise AssertionError(
            f"entry anchor may not read a gated 65+/terminal moment "
            f"({cell_key!r}); Q6 prohibits back-solving the entry from a scored "
            "terminal state (identity-in-disguise)"
        )
    if band != ENTRY_ANCHOR_BAND:
        raise AssertionError(
            f"entry anchor must read the ENTRY band {ENTRY_ANCHOR_BAND!r}, "
            f"not {band!r} ({cell_key!r})"
        )


def build_cps_entry_anchor(
    moments: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """``P(marital_state | entry age 25, sex)`` from the FRAME's CPS moments.

    Reads the pinned frame's own ``deployment_frame.reference_moments`` at the
    ENTRY band (25-34) married share by sex -- the CPS young-adult marital
    participation -- and forms the age-25 entry categorical {married: p,
    never_married: 1-p, divorced: 0, widowed: 0} (nobody is yet divorced/widowed
    at the age-25 entry -- duration / years-since are fresh). Asserts at
    construction that ONLY the entry band is read and no gated 65+/terminal
    moment is touched (Q6 back-solve prohibition). The returned probabilities
    seed a per-draw categorical (regenerated, not the identity).
    """
    # Construction guard: the configured entry band must itself be an ENTRY
    # band, never one of the prohibited older/terminal bands.
    if ENTRY_ANCHOR_BAND in PROHIBITED_ANCHOR_BANDS:
        raise AssertionError(
            f"ENTRY_ANCHOR_BAND {ENTRY_ANCHOR_BAND!r} is a prohibited "
            "terminal band"
        )
    anchor: dict[str, np.ndarray] = {}
    married_i = MARITAL_STATES.index("married")
    never_i = MARITAL_STATES.index("never_married")
    for sex in INTERIOR_SEXES:
        key = f"marital_share.married.{ENTRY_ANCHOR_BAND}|{sex}"
        _assert_entry_anchor_key(key)  # no 65+/terminal readback
        if key not in moments:
            raise KeyError(
                f"frame reference_moments missing entry anchor cell {key!r}"
            )
        p_married = float(np.clip(moments[key]["rate"], 0.0, 1.0))
        probs = np.zeros(len(MARITAL_STATES), dtype=np.float64)
        probs[married_i] = p_married
        probs[never_i] = 1.0 - p_married
        anchor[sex] = probs
    return anchor


def build_cps_seeded_marital_panel(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    entry_anchor: dict[str, np.ndarray],
    seed: int,
) -> transitions.MaritalPanel:
    """Marital panel whose ENTRY person-year carries the CPS-anchored state.

    Each gated-band adult (age >= BASE_ENTRY_AGE) enters at BASE_ENTRY_AGE=25
    with a state drawn from the frame's CPS young-adult anchor by sex; the very
    young (below the gated marital bands) enter never-married at their own age
    (their marital cells are report-only). The certified simulator reads the
    entry row as the initial condition and evolves it to the censor year.
    """
    age_i = age.astype(int)
    birth_year = (REF_YEAR - age).astype(int)
    sexes = np.where(is_female, "female", "male")
    entry_age = np.where(age_i >= BASE_ENTRY_AGE, BASE_ENTRY_AGE, age_i)
    entry_age = np.maximum(entry_age, ADULT_MIN_AGE)
    start_year = birth_year + entry_age

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, ENTRY_STATE_STREAM_SALT])
    )
    mstate = np.array(["never_married"] * len(person_id), dtype=object)
    seedable = age_i >= BASE_ENTRY_AGE
    for i in np.nonzero(seedable)[0]:
        probs = entry_anchor.get(sexes[i])
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


def regenerate_marital_v3(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    fitted_ft: Any,
    entry_anchor: dict[str, np.ndarray],
    seed: int,
) -> pd.Series:
    """Terminal marital status per person via CANDIDATE_16 from the CPS anchor."""
    panel = build_cps_seeded_marital_panel(
        person_id, age, is_female, weight, entry_anchor, seed
    )
    hold = {int(x) for x in person_id}
    sim_panel, _ = ft.simulate(panel, hold, fitted_ft, seed)
    py = sim_panel.person_years
    term = py.loc[
        py["year"] == py.groupby("person_id")["year"].transform("max")
    ]
    return term.set_index("person_id")["marital_state"]


# --------------------------------------------------------------------------
# Lever 2 -- interior sex-covariate earnings marginals.
# --------------------------------------------------------------------------
def fit_interior_marginals(
    earnings_panel: pd.DataFrame,
    person_sex: pd.Series,
    period: int = TERMINAL_PERIOD,
) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """Fit per-(interior band, sex) CellMarginals at ``period`` -- the Q8 split.

    Uses the certified gate-1 CellMarginal construction (weighted p0 + positive
    plotting-position quantile map) on the PSID family-earnings panel restricted
    to each INTERIOR band and the terminal period, joined to person-constant
    sex. Structurally identical to v2's boundary fit, extended INWARD to the
    in-support bands (this is the ONLY earnings change vs v2, which byte-carried
    the pooled no-sex-covariate interior marginals).
    """
    from run_gate1_candidate5b import CellMarginal, _plotting_positions

    df = earnings_panel.copy()
    df = df[(df["period"] == period) & (df["weight"] > 0)]
    df = df.assign(_sex=df["person_id"].map(person_sex))
    df = df[df["_sex"].isin(INTERIOR_SEXES)]

    marginals: dict[tuple[str, str], Any] = {}
    raw: dict[str, Any] = {}
    for lo, hi in INTERIOR_BANDS:
        label = dfm._band_label(lo, hi)
        band = df[(df["age"] >= lo) & (df["age"] <= hi)]
        for sex in INTERIOR_SEXES:
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
                cell = CellMarginal(p0, np.empty(0), np.empty(0), 0, 0.0)
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


def regenerate_earnings_v3(
    ages: np.ndarray,
    is_female: np.ndarray,
    rng: np.random.Generator,
    base_marginals: dict,
    boundary_marginals: dict[tuple[str, str], Any],
    interior_marginals: dict[tuple[str, str], Any],
    age_bin_fn,
    period: int = TERMINAL_PERIOD,
) -> np.ndarray:
    """Draw earnings: boundary (18-24 / 60-69) and interior (25-61) both per
    sex, base marginal for any residual age (70+). The single ``u`` draw is
    preserved; boundary is applied FIRST so 60-61 stay on the 60-69 marginal.
    """
    ages = np.asarray(ages, dtype=np.float64)
    fem = np.asarray(is_female, dtype=bool)
    out = np.zeros(len(ages), dtype=np.float64)
    u = rng.random(len(ages))  # SAME single draw as v1/v2's earnings regen
    handled = np.zeros(len(ages), dtype=bool)

    def _apply(idx: np.ndarray, cell) -> None:
        if cell is None or len(idx) == 0:
            return
        ub = u[idx]
        pos = ub >= cell.p0
        if cell.p0 < 1.0 and pos.any():
            pr = (ub[pos] - cell.p0) / (1.0 - cell.p0)
            out[idx[pos]] = cell.quantile(pr)

    # Boundary (Q2, out-of-support) FIRST -- byte-carried from v2.
    for lo, hi, label in BOUNDARY_RANGES:
        in_range = (ages >= lo) & (ages <= hi)
        for sex in BOUNDARY_SEXES:
            idx = np.nonzero(in_range & (fem == (sex == "female")) & ~handled)[
                0
            ]
            _apply(idx, boundary_marginals.get((label, sex)))
            handled[idx] = True

    # Interior (Q8, in-support) per sex -- the v3 covariate.
    for lo, hi in INTERIOR_BANDS:
        label = dfm._band_label(lo, hi)
        in_range = (ages >= lo) & (ages <= hi)
        for sex in INTERIOR_SEXES:
            idx = np.nonzero(in_range & (fem == (sex == "female")) & ~handled)[
                0
            ]
            _apply(idx, interior_marginals.get((label, sex)))
            handled[idx] = True

    # Residual ages (e.g. 70+) via the certified base marginal (v1 path).
    rest = ~handled
    if rest.any():
        bins = age_bin_fn(ages)
        for b in np.unique(bins[rest]):
            idx = np.nonzero(rest & (bins == b))[0]
            _apply(idx, base_marginals.get((int(b), int(period))))
    return out


# --------------------------------------------------------------------------
# Lever 3 -- co-designed coresident_parent roster + fertility window.
# --------------------------------------------------------------------------
def fit_coresident_parent_rates(
    hh_person_waves: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Train coresident_parent stock by ``(band, sex)`` (weighted).

    The wave-0 roster seed reads the youngest band (:data:`ROSTER_SEED_BAND`,
    15-24); the full table is recorded for provenance. Fit once on the full
    PSID household panel, frozen.
    """
    pw = hh_person_waves[["band", "sex", "weight", "coresident_parent"]].copy()
    pw = pw[pw["sex"].isin(hcd.SEXES)]
    rates: dict[tuple[str, str], float] = {}
    raw: dict[str, Any] = {}
    for lo, hi in hcd.COMPOSITION_AGE_BANDS:
        band = hcd.band_label(lo, hi)
        g_band = pw[pw["band"] == band]
        for sex in hcd.SEXES:
            g = g_band[g_band["sex"] == sex]
            w = g["weight"].to_numpy(dtype=np.float64)
            cp = g["coresident_parent"].to_numpy(dtype=bool)
            wsum = float(w.sum())
            rate = float(w[cp].sum() / wsum) if wsum > 0 else 0.0
            rates[(band, sex)] = rate
            raw[f"{band}|{sex}"] = {
                "n_person_waves": int(len(g)),
                "weighted_coresident_parent_rate": rate,
            }
    return rates, raw


def seed_coresident_parent(
    is_female: np.ndarray,
    rates: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Bernoulli wave-0 coresident_parent seed at the 15-24 train stock by sex.

    A per-person coin at the ``ROSTER_SEED_BAND`` (15-24) rate for the person's
    sex; regenerates per draw (across_draw_sd > 0), never a fixed roster.
    """
    fem = np.asarray(is_female, dtype=bool)
    prob = np.where(
        fem,
        rates.get((ROSTER_SEED_BAND, "female"), 0.0),
        rates.get((ROSTER_SEED_BAND, "male"), 0.0),
    )
    return rng.random(len(fem)) < prob


def _synthetic_household_panel_v3(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    coresident_parent_rates: dict[tuple[str, str], float],
    seed: int,
) -> hcd.HouseholdCompositionPanel:
    """v3 household roster: waves from FERTILITY_ENTRY_AGE (15), wave-0
    coresident_parent seeded from the 15-24 train stock (Q7 lever a).

    Mirrors v1/v2's panel builder but (1) enters at START_AGE=15 so the 15-24
    coresidence window exists and (2) seeds coresident_parent TRUE at each
    person's FIRST wave per :func:`seed_coresident_parent`. ``evolve_absorbing_
    exit`` reads only the wave-0 flag and evolves it out via the certified
    parental-exit hazard, so seeding wave-0 is the whole roster lever.
    """
    sexes = np.where(is_female, "female", "male")
    ages_i = age.astype(int)
    frames = []
    for i in range(len(person_id)):
        a0 = ages_i[i]
        entry_age = max(FERTILITY_ENTRY_AGE, hcd.START_AGE)
        first_year = REF_YEAR - (a0 - entry_age)
        years = np.arange(first_year, REF_YEAR + 1, HH_CADENCE)
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

    # Wave-0 coresident_parent seed (Q7 lever a). One seed per person, applied
    # to that person's earliest wave; evolve_absorbing_exit reads only wave-0.
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, ROSTER_SEED_STREAM_SALT])
    )
    seeded = seed_coresident_parent(is_female, coresident_parent_rates, rng)
    seed_by_person = dict(
        zip(person_id.tolist(), seeded.tolist(), strict=True)
    )
    first_year_of = pw.groupby("person_id")["year"].transform("min")
    is_first_wave = (pw["year"] == first_year_of).to_numpy()
    pw_seed = (
        pw["person_id"]
        .map(seed_by_person)
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    pw.loc[is_first_wave & pw_seed, "coresident_parent"] = True

    pw = hcd._add_transitions(pw)
    attrs = pw[["person_id"]].drop_duplicates().reset_index(drop=True)
    return hcd.HouseholdCompositionPanel(person_waves=pw, attrs=attrs)


def build_fertility_window_marital_panel(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
) -> transitions.MaritalPanel:
    """15-entry never-married marital panel -- the fertility window (Q7 lever b).

    Entering the marital panel at FERTILITY_ENTRY_AGE=15 (empty person-years =>
    never-married default) makes ``ft.simulate`` fire the 15-24 maternal ages
    the certified fertility supports. This panel feeds BOTH the household
    composition (compose_base's child_counts) and the child materialization --
    the ONE shared child ledger. It is NOT the marital SCORING panel (lever 1's
    CPS-anchored 25-entry panel is); the coupling is disclosed in
    ``delta_lever3_codesigned_roster_fertility``.
    """
    birth_year = (REF_YEAR - age).astype(int)
    sexes = np.where(is_female, "female", "male")
    attrs = pd.DataFrame(
        {
            "person_id": person_id,
            "birth_year": birth_year,
            "sex": sexes,
            "start_exposure_year": birth_year + FERTILITY_ENTRY_AGE,
            "censor_year": REF_YEAR,
            "weight": weight,
        }
    )
    empty = pd.DataFrame(
        {
            "person_id": pd.array([], dtype="int64"),
            "year": pd.array([], dtype="int64"),
            "marital_state": pd.array([], dtype="object"),
            "marriage_duration": pd.array([], dtype="Int64"),
            "years_since_dissolution": pd.array([], dtype="Int64"),
        }
    )
    return transitions.MaritalPanel(
        person_years=empty, events=pd.DataFrame(), attrs=attrs
    )


def regenerate_household_size_v3(
    person_id: np.ndarray,
    age: np.ndarray,
    is_female: np.ndarray,
    weight: np.ndarray,
    fitted_hc: Any,
    coresident_parent_rates: dict[tuple[str, str], float],
    seed: int,
) -> tuple[pd.Series, transitions.MaritalPanel]:
    """Terminal household size per adult via CANDIDATE_9 with the co-designed
    roster (seeded coresident_parent) + fertility window (15-entry panel).

    Returns the terminal hh_size Series and the 15-entry fertility panel (reused
    verbatim for the child rows, so they share the ledger)."""
    hh_panel = _synthetic_household_panel_v3(
        person_id, age, is_female, weight, coresident_parent_rates, seed
    )
    fertility_mpanel = build_fertility_window_marital_panel(
        person_id, age, is_female, weight
    )
    hold = {int(x) for x in person_id}
    out = hc.simulate(hh_panel, fertility_mpanel, fitted_hc, hold, seed)
    pwo = out.person_waves
    term = pwo.loc[
        pwo["year"] == pwo.groupby("person_id")["year"].transform("max")
    ]
    return term.set_index("person_id")["hh_size"], fertility_mpanel


def classify_still_home_children(
    births_with_leaves: pd.DataFrame, ref_year: int = REF_YEAR
) -> dict[str, np.ndarray]:
    """Partition still-home children (leave_year > ref_year) by the age-18 line.

    The co-design's disjointness invariant made explicit and testable: a child
    still home at ``ref_year`` is a materialized MINOR row iff its age < 18, and
    a coresident YOUNG ADULT (the roster population) iff its age >= 18. The two
    sets are disjoint and partition the still-home children -- no person is both
    a materialized minor and a coresident young adult, so the roster and the
    fertility window cannot double-count. Expects columns ``birth_year`` and
    ``leave_year``.
    """
    if len(births_with_leaves) == 0:
        empty = np.array([], dtype=np.int64)
        return {"minors": empty, "coresident_young_adults": empty}
    birth_year = births_with_leaves["birth_year"].to_numpy(dtype=np.int64)
    leave_year = births_with_leaves["leave_year"].to_numpy(dtype=np.float64)
    child_age = ref_year - birth_year
    still_home = (leave_year > ref_year) & (child_age >= 0)
    idx = np.nonzero(still_home)[0]
    minors = idx[child_age[idx] < CHILD_MAX_AGE]
    young_adults = idx[child_age[idx] >= CHILD_MAX_AGE]
    return {"minors": minors, "coresident_young_adults": young_adults}


def materialize_children_v3(
    fertility_mpanel: transitions.MaritalPanel,
    fitted_hc: Any,
    mother_hh_size: pd.Series,
    mother_weight: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Minor child rows from the 15-entry fertility panel's maternal births.

    Byte-identical to v2's :func:`materialize_children` machinery -- it re-runs
    the SAME ``ft.simulate(fertility_mpanel, ..., seed)`` call ``compose_base``
    makes internally and the SAME ``child_leave_years_refit`` leave draw
    (``SeedSequence([seed, DELTA_STREAM_TAG_V6])``), so the emitted minors are
    exactly the children ``compose_base`` counts in each mother's hh_size (the
    co-design's shared ledger). The ONLY difference from v2 is the panel is the
    15-entry fertility window (v2 passed the 25-entry seeded panel).
    """
    return td2.materialize_children(
        fertility_mpanel, fitted_hc, mother_hh_size, mother_weight, seed
    )


# --------------------------------------------------------------------------
# The regenerated all-person family-A surface (adults + materialized minors).
# --------------------------------------------------------------------------
def regenerate_person_frame_v3(
    slice_df: pd.DataFrame,
    gens: DeployedGeneratorsV3,
    entry_anchor: dict[str, np.ndarray],
    k: int,
    stream_base: int,
) -> pd.DataFrame:
    """Regenerate one draw's family-A surface with the three levers.

    Lever 1: CPS-anchored seeded marital + coresident (from ``entry_anchor``).
    Lever 2: boundary + interior sex-covariate earnings. Lever 3: co-designed
    roster + fertility hh_size, with the materialized minor rows unioned so the
    all-person hh_size_share is scoreable. Restricted to the adult universe;
    the child rows carry ``marital_status = NaN`` and ``age < 18`` so ONLY
    hh_size_share moves.
    """
    seed = stream_base + k
    rng = np.random.default_rng(seed)
    adults = slice_df[slice_df["age"] >= ADULT_MIN_AGE].reset_index(drop=True)
    pid = adults["person_id"].to_numpy()
    age = adults["age"].to_numpy(dtype=np.float64)
    fem = adults["is_female"].to_numpy(dtype=bool)
    wt = adults["weight"].to_numpy(dtype=np.float64)

    # Lever 2 -- boundary + interior sex-covariate earnings.
    earn = regenerate_earnings_v3(
        age,
        fem,
        rng,
        gens.earnings_marginals,
        gens.boundary_marginals,
        gens.interior_marginals,
        gens.age_bin_fn,
    )
    # Lever 1 -- CPS-anchored seeded marital + coresident.
    marital = regenerate_marital_v3(
        pid, age, fem, wt, gens.fitted_ft, entry_anchor, seed
    )
    marital_arr = marital.reindex(pid).to_numpy(dtype=object)
    coresident = marital_arr == "married"

    # Lever 3 -- co-designed roster + fertility hh_size and minor child rows.
    hh_size, fertility_mpanel = regenerate_household_size_v3(
        pid, age, fem, wt, gens.fitted_hc, gens.coresident_parent_rates, seed
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
    child_frame = materialize_children_v3(
        fertility_mpanel,
        gens.fitted_hc,
        hh_size,
        pd.Series(wt, index=pid),
        seed,
    )
    return pd.concat([adult_frame, child_frame], ignore_index=True)


# --------------------------------------------------------------------------
# Fit the v3 generator bundle (base + Q2 boundary + Q8 interior + Q7 roster).
# --------------------------------------------------------------------------
def fit_generators(m4_artifact_path: str) -> DeployedGeneratorsV3:
    """Fit the byte-carried base generators plus the Q2 boundary marginals, the
    Q8 interior marginals, and the Q7 coresident_parent roster rates -- once on
    the full PSID sample, frozen. Lever 1's CPS entry anchor is NOT fit here (it
    is derived from the frame at score time)."""
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_gate1_baseline as g1base

    from populace_dynamics.data import marriage

    base = td1.fit_generators(m4_artifact_path)
    prov: dict[str, Any] = dict(base.fit_provenance)

    # Q2 + Q8 -- boundary and interior earnings marginals per sex.
    t = time.time()
    raw_panel = g1base.family_earnings_panel()
    person_sex = (
        marriage.marriage_history()
        .dropna(subset=["sex"])
        .groupby("person_id")["sex"]
        .first()
    )
    boundary, q2_raw = td2.fit_boundary_marginals(raw_panel, person_sex)
    prov["q2_boundary_marginals"] = {
        "source": (
            "run_gate1_baseline.family_earnings_panel, ages 18-24/60-69 x "
            "sex, period 2022"
        ),
        "n_cells": len(boundary),
        "ranges": [f"{lo}-{hi}" for lo, hi, _ in BOUNDARY_RANGES],
        "sex_covariate": True,
        "byte_carried_from": "transport_deployment_v2",
        "fit_seconds": round(time.time() - t, 1),
    }
    t = time.time()
    interior, q8_raw = fit_interior_marginals(raw_panel, person_sex)
    prov["q8_interior_marginals"] = {
        "source": (
            "run_gate1_baseline.family_earnings_panel, interior bands "
            "25-34/35-44/45-54/55-61 x sex, period 2022"
        ),
        "n_cells": len(interior),
        "bands": [dfm._band_label(lo, hi) for lo, hi in INTERIOR_BANDS],
        "sex_covariate": True,
        "fit_seconds": round(time.time() - t, 1),
    }

    # Q7 -- coresident_parent roster stock from the PSID household panel.
    t = time.time()
    hc_sources = hc.load_sources()
    roster_rates, q7_raw = fit_coresident_parent_rates(
        hc_sources["hh"].person_waves
    )
    prov["q7_coresident_parent_roster"] = {
        "source": "PSID household person-waves coresident_parent by band x sex",
        "seed_band": ROSTER_SEED_BAND,
        "seed_rate_female": roster_rates.get((ROSTER_SEED_BAND, "female")),
        "seed_rate_male": roster_rates.get((ROSTER_SEED_BAND, "male")),
        "fertility_entry_age": FERTILITY_ENTRY_AGE,
        "fit_seconds": round(time.time() - t, 1),
    }

    return DeployedGeneratorsV3(
        base=base,
        boundary_marginals=boundary,
        interior_marginals=interior,
        coresident_parent_rates=roster_rates,
        fit_provenance=prov,
        fit_vs_raw={
            "q2_boundary_support": q2_raw,
            "q8_interior_support": q8_raw,
            "q7_coresident_parent_roster": q7_raw,
        },
    )


# --------------------------------------------------------------------------
# Family A -- the regenerated all-person surface, gated cells bound at run time.
# --------------------------------------------------------------------------
def family_a_score(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV3,
    floor: dict,
    tolerances: dict[str, float],
    gated_cells: list[str],
    *,
    seeds: tuple[int, ...] = GATE_SEEDS,
    k_draws: int = K_DRAWS,
    progress: bool = False,
) -> dict[str, Any]:
    """Score family A per gate seed (household-disjoint holdout, K=20 draws)
    with the v3 regenerated all-person surface. The CPS entry anchor (lever 1)
    is derived ONCE from the full frame's own reference_moments. Same floor
    pricing, statistic, conjunction, and dispersion disclosures as candidates
    1/2; ``gated_cells`` is whatever the caller bound from gates.yaml."""
    # Lever 1 -- CPS young-adult entry anchor from the frame's own moments.
    entry_anchor = build_cps_entry_anchor(
        dfm.reference_moments(persons, weighted=True)
    )

    universe = persons["household_id"].to_numpy()
    per_seed_floor = {
        s["seed"]: s["cells"] for s in floor["noise_floor_per_seed"]
    }
    n_cell = len(gated_cells)
    cube = np.full((k_draws, n_cell, len(seeds)), np.nan)

    per_seed_out = []
    for si, seed in enumerate(seeds):
        t = time.time()
        side_a = set(td1.holdout_side_a_households(universe, seed).tolist())
        hold = persons[persons["household_id"].isin(side_a)].reset_index(
            drop=True
        )
        draw_rates = np.full((k_draws, n_cell), np.nan)
        for k in range(k_draws):
            regen = regenerate_person_frame_v3(
                hold, gens, entry_anchor, k, FAMILY_A_STREAM_BASE
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
        "cps_entry_anchor": {
            sex: {
                MARITAL_STATES[i]: float(probs[i])
                for i in range(len(MARITAL_STATES))
            }
            for sex, probs in entry_anchor.items()
        },
        "cube_shape": list(cube.shape),
        "cube": cube.tolist(),
        "per_seed": per_seed_out,
        "n_seed_pass": n_seed_pass,
        "family_a_pass": n_seed_pass >= 4,
    }


# --------------------------------------------------------------------------
# Family B -- byte-carried M4 simulation, scored REPORT-ONLY (gates nothing).
# --------------------------------------------------------------------------
def family_b_report(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV3,
    retained_anchors: dict[str, Any],
    report_reasons: dict[str, Any],
    *,
    k_draws: int = K_DRAWS,
    pe_us_dir: str | None = None,
) -> dict[str, Any]:
    """The 10 family-B M4-simulated DI margins vs the amended contract's
    retained_anchors, DISCLOSED report-only (family B gates nothing). Byte-
    carries v2's family_b_report, which byte-carries v1's M4 simulation."""
    return td2.family_b_report(
        persons,
        gens,  # DeployedGeneratorsV3 exposes m4_prevalence/m4_bands passthroughs
        retained_anchors,
        report_reasons,
        k_draws=k_draws,
        pe_us_dir=pe_us_dir,
    )


# --------------------------------------------------------------------------
# Family C -- byte-carried verbatim from candidate 1 (base marginals).
# --------------------------------------------------------------------------
def family_c(
    persons: pd.DataFrame,
    gens: DeployedGeneratorsV3,
    family_c_contract: dict,
) -> dict[str, Any]:
    """Re-run both compression fingerprints via the candidate-1 procedure
    verbatim (transport_career_panel on the BASE marginals + the #115/#117
    ledgers). Byte-carried: the levers do not touch family C. The SCRIPT decides
    which fingerprints GATE, binding family_c.gate_partition from gates.yaml.
    """
    return td1.family_c(persons, gens.base, family_c_contract)
