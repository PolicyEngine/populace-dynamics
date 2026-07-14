"""Ceremony step 1 for ``gate_m6``: the temporal-holdout TRUTH-SIDE floors.

This builds ``runs/m6_holdout_floors_v1.json`` -- the deployment-scale
sampling null the proposed ``gate_m6`` (``docs/design/m6_projection_engine.md``
sec. 4, sec. 9) prices its tolerances against. It runs NO projection engine,
scores NO candidate, and writes NO gate: it computes the held-out PSID truth
(2015-2022) and the real-vs-real half-split floor of every proposed cell, then
partitions the surface into the gated family-A flows and the report-only
margins per the design's shock (sec. 4.1), attrition (sec. 4.4), mixed-k /
power (sec. 4.6) and measurability dispositions.

PSID wave facts (verified against the readers): interviews are biennial ODD
years from 1999 (``panels.demographic_panel`` periods ..., 2013, 2015, 2017,
2019, 2021, 2023); income REFERENCE years are even (``family.family_earnings``
..., 2014, 2016, 2018, 2020, 2022). So the boundary ``T* = 2014`` realizes its
seed state at the **2015 interview** (reference year 2014), and reference year
2021 is unobserved (odd) -- exactly the dual-axis pin of sec. 4.1.

Frozen once written (``artifacts.write_new`` refuses overwrite). Run from the
repository root with PSID staged at ``~/PolicyEngine/psid-data``::

    .venv-wt/bin/python scripts/build_m6_holdout_floors.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics import artifacts
from populace_dynamics.data import deaths, disability, family, panels
from populace_dynamics.harness.m6_cells import (
    AUTOCORR_ABS_CAP,
    AUTOCORR_LAGS,
    DISABILITY_BANDS,
    EARN_ANCHOR_YEAR,
    EARN_COHORTS,
    FLOOR_SEEDS,
    GATE_SEEDS,
    GATEABLE_METRICS,
    GATED_EARN_YEARS,
    GATED_FLOW_YEARS,
    GATED_START_WAVES,
    K_FLOW,
    K_STOCK,
    MARGIN_K,
    MARITAL_AT_RISK,
    MARITAL_BANDS,
    METRIC_CAP,
    METRIC_CAP_SOURCE,
    MIN_EVENTS_FOR_GATE,
    MIN_GATED_CELLS_FOR_POWER,
    MOBILITY_HORIZONS,
    MORTALITY_ATTRITION_BANDS,
    MORTALITY_BANDS,
    ROUNDING,
    SEED_WAVE,
    SEXES,
    SHOCK_EARN_YEARS,
    SHOCK_FLOW_YEARS,
    SPLIT_FRACTION,
    T_MAX,
    T_MAX_SOURCE,
    T_STAR,
    WEAK_POWER_P_GATE_FLOOR,
    _normal_cdf,
    _rate,
    _score,
    _sha256_bytes,
    _tol,
    _wquantile,
    band_label,
    band_of,
    build_anchor_frame,
    disability_cells,
    disability_pairs,
    earnings_cells,
    earnings_frame,
    marital_cells,
    marital_tables,
    marital_tables_from_panel,
    mortality_cells,
    mortality_slices,
    oc_4of5,
    presence_by_wave,
    run_floor,
)  # noqa: F401 -- the floor wrappers import these names from this module

__all__ = [
    "AUTOCORR_ABS_CAP",
    "AUTOCORR_LAGS",
    "DISABILITY_BANDS",
    "EARN_ANCHOR_YEAR",
    "EARN_COHORTS",
    "FLOOR_SEEDS",
    "GATED_EARN_YEARS",
    "GATED_FLOW_YEARS",
    "GATED_START_WAVES",
    "GATEABLE_METRICS",
    "GATE_SEEDS",
    "K_FLOW",
    "K_STOCK",
    "MARGIN_K",
    "MARITAL_AT_RISK",
    "MARITAL_BANDS",
    "METRIC_CAP",
    "METRIC_CAP_SOURCE",
    "MIN_EVENTS_FOR_GATE",
    "MIN_GATED_CELLS_FOR_POWER",
    "MOBILITY_HORIZONS",
    "MORTALITY_ATTRITION_BANDS",
    "MORTALITY_BANDS",
    "ROUNDING",
    "SEED_WAVE",
    "SEXES",
    "SHOCK_EARN_YEARS",
    "SHOCK_FLOW_YEARS",
    "SPLIT_FRACTION",
    "T_MAX",
    "T_MAX_SOURCE",
    "T_STAR",
    "WEAK_POWER_P_GATE_FLOOR",
    "_normal_cdf",
    "_rate",
    "_score",
    "_sha256_bytes",
    "_tol",
    "_wquantile",
    "band_label",
    "band_of",
    "build_anchor_frame",
    "disability_cells",
    "disability_pairs",
    "earnings_cells",
    "earnings_frame",
    "marital_cells",
    "marital_tables",
    "marital_tables_from_panel",
    "mortality_cells",
    "mortality_slices",
    "oc_4of5",
    "presence_by_wave",
    "run_floor",
]

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m6_holdout_floors_v1.json"
SCHEMA_VERSION = "m6_holdout_floors.v1"
RUN = "m6_holdout_floors_v1"


def cell_family(key: str) -> str:
    head = key.split(".")[0]
    if head == "death":
        return "mortality"
    if head in ("first_marriage", "divorce", "widowhood", "remarriage"):
        return "marital"
    if head in ("incidence", "recovery"):
        return "disability"
    if head == "prevalence":
        return "disability_stock"
    if head == "married_share":
        return "marital_stock"
    return "earnings"


#: Design split unit per family (sec. 4.5 / 4.7): demographic FLOWS with
#: within-household correlation (mortality, marital -- widowhood/divorce couple
#: a household; mortality shares household frailty) split HOUSEHOLD-disjoint
#: (couple-disjoint subsumed); disability + earnings split PERSON-disjoint.
HOUSEHOLD_SPLIT_FAMILIES = frozenset({"mortality", "marital", "marital_stock"})


def split_col_for_family(fam: str) -> str:
    return "household_id" if fam in HOUSEHOLD_SPLIT_FAMILIES else "person_id"


def partition(
    floor: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Gated vs report-only, with a machine reason per report-only cell."""
    tolerances: dict[str, float] = {}
    gated: list[str] = []
    report_only: dict[str, str] = {}
    for key, fl in floor.items():
        m = meta[key]
        metric = m["metric"]
        fam = m["family"]
        k = K_STOCK if m.get("quantity_type") == "stock" else K_FLOW
        raw_tol = _tol(fl["mean"], fl["sd"], k)
        tolerances[key] = raw_tol
        # report-only dispositions (in disposition precedence order):
        if fam.endswith("_stock"):
            report_only[key] = "stock_report_only_margin"
            continue
        if metric == "report_only_horizon":
            report_only[key] = "horizon_exceeds_holdout_span"
            continue
        if metric == "report_only_shape":
            report_only[key] = "dimensionless_shape_no_power_cap"
            continue
        if key.split(".")[0] == "death" and key.split("|")[0].split(".")[
            1
        ] in (MORTALITY_ATTRITION_BANDS):
            report_only[key] = "attrition_confounded_truth"
            continue
        if metric not in GATEABLE_METRICS:
            report_only[key] = "non_gateable_metric"
            continue
        if fl["n_defined_seeds"] < len(FLOOR_SEEDS):
            report_only[key] = "undefined_on_some_seed"
            continue
        if fl["min_events_weaker_half"] < MIN_EVENTS_FOR_GATE:
            report_only[key] = "below_20_events_weaker_half"
            continue
        if raw_tol > METRIC_CAP[metric] + 1e-12:
            report_only[key] = "tolerance_above_power_cap"
            continue
        gated.append(key)
    return {
        "gated": sorted(gated),
        "report_only": dict(sorted(report_only.items())),
        "tolerances": tolerances,
        "n_gated": len(gated),
        "n_report_only": len(report_only),
    }


def split_width_diagnostic(
    household_floor: dict[str, dict[str, float]],
    person_floor: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Person-disjoint vs household-disjoint floor WIDTHS on the household-split
    cells (the ceremony watch-item, PR #170 comment 4953958949): a household-ID
    rule that biased the split would inflate/deflate the household floor sigma
    relative to person. Ratio ~1 means the T*-anchored household rule does not
    bias the family-A floor."""
    per_cell: dict[str, Any] = {}
    ratios: list[float] = []
    for key in sorted(set(household_floor) & set(person_floor)):
        hh = household_floor[key]["realized_sigma"]
        pp = person_floor[key]["realized_sigma"]
        ratio = float(hh / pp) if pp > 0 else None
        per_cell[key] = {
            "household_sigma": round(hh, 6),
            "person_sigma": round(pp, 6),
            "household_over_person": (
                None if ratio is None else round(ratio, 4)
            ),
        }
        if ratio is not None:
            ratios.append(ratio)
    arr = np.array(ratios) if ratios else np.array([1.0])
    return {
        "note": (
            "Per household-split family-A cell, realized_sigma under the "
            "HOUSEHOLD-disjoint split (the design unit) over the PERSON-disjoint "
            "split. Ratio > 1 is the expected mild inflation from positive "
            "within-household correlation (fewer effective independent units); "
            "a ratio far from 1 would flag a biasing household-ID rule."
        ),
        "n_cells": len(ratios),
        "mean_household_over_person": round(float(arr.mean()), 4),
        "median_household_over_person": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# End-of-window STOCK cells (report-only margin, sec. 4.6 gate_m4 >=3 sigma
# style). Occupancy levels at the last gated wave (2019); their half-split
# floor sigma is published as the margin basis, never |ln|-level-gated.
# --------------------------------------------------------------------------
def disability_occupancy(
    status: pd.DataFrame,
    death_records: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Disabled occupancy person-rows at the 2019 gated wave, F6-weighted."""
    with_sex = disability.attach_sex(status, death_records)
    occ = with_sex[with_sex.period == 2019].copy()
    occ = occ.merge(
        anchor[["person_id", "weight"]],
        on="person_id",
        how="inner",
        suffixes=("_raw", ""),
    )
    occ["band"] = occ.age.map(lambda a: band_of(a, DISABILITY_BANDS))
    occ = occ[occ.band.notna() & occ.sex.isin(SEXES)]
    return occ.reset_index(drop=True)


def occupancy_cells(occ: pd.DataFrame) -> dict[str, dict[str, Any]]:
    g = (
        occ.assign(w=occ.weight, wd=occ.weight * occ.disabled.astype(float))
        .groupby(["band", "sex"], observed=True)
        .agg(w=("w", "sum"), wd=("wd", "sum"), n=("w", "size"))
    )
    out: dict[str, dict[str, Any]] = {}
    for (band, sex), r in g.iterrows():
        out[f"prevalence.{band}|{sex}"] = {
            "rate": _rate(r.wd, r.w),
            "n_events": int(r.n),
            "n_at_risk": int(r.n),
        }
    return out


def marital_share_cells(py: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Married-share stock at the 2019 gated wave, by band x sex."""
    at19 = py[(py.year == 2019) & (py.window == "gated")]
    g = (
        at19.assign(
            w=at19.weight, wm=at19.weight * (at19.marital_state == "married")
        )
        .groupby(["band", "sex"], observed=True)
        .agg(w=("w", "sum"), wm=("wm", "sum"), n=("w", "size"))
    )
    out: dict[str, dict[str, Any]] = {}
    for (band, sex), r in g.iterrows():
        out[f"married_share.{band}|{sex}"] = {
            "rate": _rate(r.wm, r.w),
            "n_events": int(r.n),
            "n_at_risk": int(r.n),
        }
    return out


def _meta_from_reference(
    ref: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-cell metric / family / quantity_type from a full-sample cell dict."""
    meta: dict[str, dict[str, Any]] = {}
    for key, val in ref.items():
        fam = cell_family(key)
        if "rate" in val:
            metric = "log_ratio"
        else:
            metric = val.get("metric", "log_ratio")
        quantity = "stock" if fam.endswith("_stock") else "flow"
        meta[key] = {
            "family": fam,
            "metric": metric,
            "quantity_type": quantity,
        }
    return meta


def build_artifact(verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    status = disability.read_disability_status()
    ep = family.family_earnings_panel()

    anchor = build_anchor_frame(demo)
    present = presence_by_wave(demo)
    n_fallback = 0  # every holdout person has a gated-wave interview by build

    # Family cell tables (presence-conditioned, windowed, F6-weighted).
    sl = mortality_slices(demo, death_records, anchor)
    ev, py = marital_tables(death_records, anchor, present)
    dpairs = disability_pairs(status, death_records, anchor)
    earn = earnings_frame(ep, anchor)
    occ = disability_occupancy(status, death_records, anchor)

    sl_g, ev_g, py_g = (
        sl[sl.window == "gated"],
        ev[ev.window == "gated"],
        py[py.window == "gated"],
    )
    dp_g = dpairs[dpairs.window == "gated"]

    def c_mort(ids: set[int]) -> dict[str, Any]:
        return mortality_cells(sl_g[sl_g.person_id.isin(ids)])

    def c_marital(ids: set[int]) -> dict[str, Any]:
        return marital_cells(
            ev_g[ev_g.person_id.isin(ids)], py_g[py_g.person_id.isin(ids)]
        )

    def c_disab(ids: set[int]) -> dict[str, Any]:
        return disability_cells(dp_g[dp_g.person_id.isin(ids)])

    def c_earn(ids: set[int]) -> dict[str, Any]:
        return earnings_cells(earn[earn.person_id.isin(ids)])

    def c_prev(ids: set[int]) -> dict[str, Any]:
        return occupancy_cells(occ[occ.person_id.isin(ids)])

    def c_mshare(ids: set[int]) -> dict[str, Any]:
        return marital_share_cells(py[py.person_id.isin(ids)])

    all_ids = set(anchor.person_id.to_numpy().tolist())
    reference = {
        **c_mort(all_ids),
        **c_marital(all_ids),
        **c_disab(all_ids),
        **c_earn(all_ids),
        **c_prev(all_ids),
        **c_mshare(all_ids),
    }
    meta = _meta_from_reference(reference)

    # Primary floors on the design split unit per family.
    if verbose:
        print("floor: mortality (household + person diag)...")
    mort_hh, mort_hh_seed = run_floor(anchor, c_mort, "household_id")
    mort_pp, _ = run_floor(anchor, c_mort, "person_id")
    if verbose:
        print("floor: marital (household + person diag)...")
    mar_hh, mar_hh_seed = run_floor(anchor, c_marital, "household_id")
    mar_pp, _ = run_floor(anchor, c_marital, "person_id")
    if verbose:
        print("floor: disability (person)...")
    dis_floor, dis_seed = run_floor(anchor, c_disab, "person_id")
    if verbose:
        print("floor: earnings (person)...")
    earn_floor, earn_seed = run_floor(anchor, c_earn, "person_id")
    if verbose:
        print("floor: stocks (prevalence person / married_share household)...")
    prev_floor, _ = run_floor(anchor, c_prev, "person_id")
    mshare_floor, _ = run_floor(anchor, c_mshare, "household_id")

    # Shock-window diagnostics (sec. 4.1): reference + half-split floor of the
    # 2020-2022 exogenous-shock cells, COMPUTED and published but PARTITIONED
    # OUT of the gated set (machine reason exogenous_shock_outside_model_class).
    sl_s, ev_s, py_s = (
        sl[sl.window == "shock"],
        ev[ev.window == "shock"],
        py[py.window == "shock"],
    )
    dp_s = dpairs[dpairs.window == "shock"]

    def cs_mort(ids: set[int]) -> dict[str, Any]:
        return mortality_cells(sl_s[sl_s.person_id.isin(ids)])

    def cs_marital(ids: set[int]) -> dict[str, Any]:
        return marital_cells(
            ev_s[ev_s.person_id.isin(ids)], py_s[py_s.person_id.isin(ids)]
        )

    def cs_disab(ids: set[int]) -> dict[str, Any]:
        return disability_cells(dp_s[dp_s.person_id.isin(ids)])

    def cs_earn(ids: set[int]) -> dict[str, Any]:
        return earnings_cells(
            earn[earn.person_id.isin(ids)],
            level_years=SHOCK_EARN_YEARS,
            change_years=(2018,) + SHOCK_EARN_YEARS,
        )

    if verbose:
        print("floor: shock-window diagnostics (report-only)...")
    shock_ref = {
        **cs_mort(all_ids),
        **cs_marital(all_ids),
        **cs_disab(all_ids),
        **cs_earn(all_ids),
    }
    shock_floor: dict[str, dict[str, float]] = {}
    for compute, col in (
        (cs_mort, "household_id"),
        (cs_marital, "household_id"),
        (cs_disab, "person_id"),
        (cs_earn, "person_id"),
    ):
        fl, _ = run_floor(anchor, compute, col)
        shock_floor.update(fl)

    # Authoritative floor = the design-unit split per family.
    floor: dict[str, dict[str, float]] = {}
    floor.update(mort_hh)
    floor.update(mar_hh)
    floor.update(dis_floor)
    floor.update(earn_floor)
    floor.update(prev_floor)
    floor.update(mshare_floor)

    part = partition(floor, meta)
    tolerances = part["tolerances"]

    # OC on the draw-noise-free half-normal basis (sec. 4.9). Reported over
    # THREE surfaces: the certifiable FLOW family-A primary (mortality +
    # marital + disability -- the "dynamics-drift flows" a PASS certifies,
    # sec. 4.6), the gated EARNINGS moments (a distinct deployment-scale
    # sub-family), and the COMBINED gated surface (the precedent's one-OC
    # convention).
    flow_gated = [
        k
        for k in part["gated"]
        if meta[k]["family"] in ("mortality", "marital", "disability")
    ]
    earn_gated = [k for k in part["gated"] if meta[k]["family"] == "earnings"]
    oc_flows = oc_4of5(floor, tolerances, flow_gated)
    oc_earnings = oc_4of5(floor, tolerances, earn_gated)
    oc = oc_4of5(floor, tolerances, part["gated"])

    # The split-width diagnostic on the household-split (mortality + marital)
    # family-A cells.
    hh_floor = {**mort_hh, **mar_hh}
    pp_floor = {**mort_pp, **mar_pp}
    diag = split_width_diagnostic(hh_floor, pp_floor)

    # Weak-power OC-before-lock check (both directions, sec. 4.9 / sec. 9).
    n_gated = part["n_gated"]
    n_flow_gated = len(flow_gated)
    at_cap = [
        k
        for k in part["gated"]
        if abs(tolerances[k] - METRIC_CAP[meta[k]["metric"]]) <= 1e-9
    ]
    # near-unpassable: the certifiable surface's faithful OC drops below the
    # named floor. near-vacuous: fewer than MIN_GATED_CELLS gated FLOW cells
    # (the primary certifiable surface -- earnings padding cannot certify the
    # dynamics-drift flows), OR every gated tolerance pinned at its cap.
    clears_lower = oc["p_gate_pass_4_of_5"] >= WEAK_POWER_P_GATE_FLOOR
    clears_flow_power = n_flow_gated >= MIN_GATED_CELLS_FOR_POWER
    clears_not_all_capped = len(at_cap) < n_gated
    clears = bool(clears_lower and clears_flow_power and clears_not_all_capped)

    if verbose:
        print(
            f"gated={n_gated} (flow={n_flow_gated} earn={len(earn_gated)}) "
            f"report_only={part['n_report_only']} "
            f"p_gate_combined={oc['p_gate_pass_4_of_5']} "
            f"p_gate_flows={oc_flows['p_gate_pass_4_of_5']} "
            f"clears_weak_power={clears}"
        )

    per_seed = {
        "mortality": mort_hh_seed,
        "marital": mar_hh_seed,
        "disability": dis_seed,
        "earnings": earn_seed,
    }

    families = {}
    for key in floor:
        families.setdefault(meta[key]["family"], []).append(key)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_truth_side_only": True,
        "no_projection_no_candidate": True,
        "referee_round": (
            "gate_m6 floors ceremony step 1; design PR #170 (1d83a22) "
            "VERIFIED comment 4953958949; adjudication issue #42 4953722912"
        ),
        "component": (
            "M6 temporal-holdout gate (gate_m6): the PSID held-out truth "
            "(2015-2022) + real-vs-real half-split floor for the proposed "
            "family-A dynamics-drift flows and the report-only margins."
        ),
        "purpose": (
            "Price gate_m6's tolerances against the deployment-scale sampling "
            "null of the held-out observables, on the post-demotion surface "
            "(shock sec. 4.1, attrition sec. 4.4, mixed-k / power sec. 4.6). "
            "Builds NO engine and scores NO candidate; gates.yaml is untouched."
        ),
        "gates_yaml_untouched": True,
        "design_pins": {
            "boundary_T_star": T_STAR,
            "seed_wave": SEED_WAVE,
            "seed_wave_note": (
                "PSID interviews are biennial ODD years; T*=2014 realizes its "
                "seed state at the 2015 interview (reference year 2014)."
            ),
            "gated_flow_event_years": list(GATED_FLOW_YEARS),
            "shock_flow_event_years": list(SHOCK_FLOW_YEARS),
            "gated_earnings_reference_years": list(GATED_EARN_YEARS),
            "shock_earnings_reference_years": list(SHOCK_EARN_YEARS),
            "reference_year_2021_unobserved": True,
            "mixed_k": {"flow": K_FLOW, "stock": K_STOCK, "margin": MARGIN_K},
            "rounding": ROUNDING,
            "t_max": T_MAX,
            "t_max_source": T_MAX_SOURCE,
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "floor_seeds": [FLOOR_SEEDS[0], FLOOR_SEEDS[-1]],
            "gate_seeds": list(GATE_SEEDS),
            "conjunction": "4 of 5 seeds",
            "k20_estimator_note": (
                "the candidate side (not built here) is the K=20-draw mean on "
                "numpy.random.default_rng(5200 + k), scored once |ln(rbar/"
                "rate_a)| against side-A truth (sec. 4.7)."
            ),
        },
        "presence_conditioning": {
            "rule": (
                "Every truth cell is computed on the realized-wave-presence "
                "set (PSID sequence 1-20, positive weight), SYMMETRICALLY on "
                "both halves and (for the candidate, not built here) both "
                "sides (sec. 4.4). demographic_panel / read_disability_status "
                "/ family_earnings_panel are present-only by construction; "
                "marital events + at-risk person-years are conditioned on "
                "presence at the biennial interview OPENING the interval that "
                "contains the event year."
            ),
            "per_family": {
                "mortality": "present at the interval-start interview (slices "
                "credited only from an observed wave); conditioning-not-"
                "leakage (PSID retention is unmodelled).",
                "marital": "present at the opening biennial interview of the "
                "event's interval; conditioning-not-leakage.",
                "disability": "reproduction mode -- realized support preserved "
                "(read_disability_status present-only); a PASS certifies a "
                "TEMPORAL gate_m4 extension, NOT M6 forward disability "
                "(sec. 4.4, finding 7).",
                "earnings": "present with a valid positive-weight earnings "
                "observation at the reference-year wave.",
            },
        },
        "f6_weight_rule": {
            "definition": (
                "Every gated rate, every truth cell, and the floor use the "
                "person's realized start-of-holdout START-WAVE cross-sectional "
                "PSID weight, held FIXED across the window (F6, sec. 4.7). The "
                "per-year calibrated panel weight is NOT used (it would put a "
                "different weight on the two sides and bias every ratio); it "
                "is a report-only alternative."
            ),
            "start_wave": (
                "the earliest gated interview wave (2015 / 2017 / 2019, >= the "
                "2015 T* seed) at which the person is present."
            ),
        },
        "household_id_weight_carriage_rule": {
            "finding": "13 (sec. 9); the ceremony watch-item of comment 4953958949",
            "household_id": (
                "the PSID family interview number (ER30001-scoped, "
                "panels.demographic_panel `interview`) at the person's realized "
                "start-of-holdout wave, held FIXED across the window -- "
                "consistent with the F6 start-wave anchoring."
            ),
            "divorce_splitoff_rule": (
                "a within-window divorce / splitoff does NOT reassign "
                "household_id; both ex-partners keep their fixed start-wave "
                "household (conservative: keeps a correlated ex-couple on ONE "
                "side of the split, so the split cannot understate correlation)."
            ),
            "entrant_child_rule": (
                "synthetic births + immigrant cohorts are the family-B closed-"
                "panel entrants (sec. 4.8) and are OUT of the family-A floor; "
                "children present at the start wave carry that wave's household."
            ),
            "fallback": (
                "a person with no gated-wave interview number falls back to a "
                "singleton household (their own person_id)."
            ),
            "n_singleton_fallback": n_fallback,
            "split_bias_diagnostic": (
                "runs comparison: person-disjoint vs household-disjoint floor "
                "widths on the household-split family-A cells (see "
                "split_width_diagnostic); a rule that biased the split would "
                "move the ratio far from 1."
            ),
        },
        "split_units": {
            "note": (
                "correlation-respecting half split (sec. 4.7): the demographic "
                "flows with within-household correlation split HOUSEHOLD-"
                "disjoint (couple-disjoint subsumed); disability + earnings "
                "split PERSON-disjoint (sec. 4.5)."
            ),
            "household_disjoint_families": sorted(HOUSEHOLD_SPLIT_FAMILIES),
            "person_disjoint_families": [
                "disability",
                "earnings",
                "disability_stock",
            ],
            "machine": "populace_dynamics.harness.panel.split_panel_by_person",
            "split_fraction": SPLIT_FRACTION,
        },
        "data": {
            "n_holdout_persons": int(anchor.person_id.nunique()),
            "n_households": int(anchor.household_id.nunique()),
            "psid_population": (
                "populace_dynamics.data.panels.demographic_panel + deaths "
                "(ER32050) + read_disability_status + family_earnings_panel + "
                "marriage.marriage_history"
            ),
            "psid_wave_calendar": "biennial odd-year interviews 1999-2023; "
            "even-year income reference years",
        },
        "reference_moments": reference,
        "floor": {
            "method": (
                "100-seed real-vs-real half split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction 0.5, seeds 0-99) of the presence-conditioned, F6-"
                "weighted holdout truth; floor statistic |ln(rate_a/rate_b)| "
                "for rate / log_ratio cells, |value_a - value_b| for the "
                "typed abs_gap cells. (mean, sd, realized_sigma) per cell "
                "derive round(mean + k*sd, 3) capped at the metric's power cap."
            ),
            "cells": floor,
        },
        "metric_caps": {
            m: {"cap": METRIC_CAP[m], "source": METRIC_CAP_SOURCE[m]}
            for m in METRIC_CAP
        },
        "cell_meta": meta,
        "partition": {
            "gated": part["gated"],
            "report_only": part["report_only"],
            "n_gated": part["n_gated"],
            "n_report_only": part["n_report_only"],
            "by_family_gated": {
                fam: sorted(
                    k for k in part["gated"] if meta[k]["family"] == fam
                )
                for fam in sorted({meta[k]["family"] for k in part["gated"]})
            },
        },
        "tolerances": {k: tolerances[k] for k in part["gated"]},
        "tolerances_all_cells": tolerances,
        "faithful_candidate_oc": {
            "combined": oc,
            "family_a_flows": oc_flows,
            "earnings_subfamily": oc_earnings,
            "p_seed_pass": oc["p_seed_pass"],
            "p_gate_pass_4_of_5": oc["p_gate_pass_4_of_5"],
        },
        "oc_before_lock": {
            "weak_power_p_gate_floor": WEAK_POWER_P_GATE_FLOOR,
            "weak_power_floor_source": (
                "bottom of the lived precedent p_gate band (gate_2a 0.9685 / "
                "2b 0.9678 / 2c 0.988 / m4 0.9689 / w1 0.9481-0.9623), rounded "
                "down to 0.90."
            ),
            "min_gated_flow_cells": MIN_GATED_CELLS_FOR_POWER,
            "n_gated_cells": n_gated,
            "n_gated_flow_cells": n_flow_gated,
            "n_gated_earnings_cells": len(earn_gated),
            "n_gated_tolerances_at_cap": len(at_cap),
            "p_gate_combined": oc["p_gate_pass_4_of_5"],
            "p_gate_flows": oc_flows["p_gate_pass_4_of_5"],
            "p_gate_earnings": oc_earnings["p_gate_pass_4_of_5"],
            "clears_lower_bound_not_unpassable": bool(clears_lower),
            "clears_flow_surface_power": bool(clears_flow_power),
            "clears_not_all_tolerances_capped": bool(clears_not_all_capped),
            "clears_weak_power_threshold": clears,
            "ceremony_may_proceed": clears,
            "interpretation": (
                "Both directions (sec. 4.9, sec. 9). Near-unpassable: the "
                "combined faithful p_gate < the named floor. Near-vacuous: "
                "fewer than the minimum gated FLOW cells (the mortality / "
                "marital / disability primary that certifies the dynamics-"
                "drift claim per sec. 4.6 -- earnings-moment cells cannot "
                "certify the flow claim), OR every gated tolerance pinned at "
                "its power cap. If ANY fails the ceremony PAUSES for surface "
                "redesign rather than locking a near-vacuous gate."
            ),
            "pause_finding": (
                (
                    "The 8-year biennial holdout at age-band x sex granularity "
                    "yields ~0.1 half-split log-ratio noise per flow cell, so at "
                    "the design-pinned k=3 most flow tolerances land just above "
                    "the ln(1.5) cap and demote on power; the post-demotion FLOW "
                    "surface supports too few gated cells to certify the dynamics-"
                    "drift claim. The earnings moments gate cleanly (large N), but "
                    "cannot stand in for the flows. RECOMMENDATION for the surface "
                    "redesign the pause triggers: coarsen the sparse marital / "
                    "mortality strata (e.g. sex-pool the low-frequency transitions "
                    "or widen the oldest bands), and/or extend the flow event "
                    "window, then rebuild the floor -- not a threshold widening."
                )
                if not clears
                else "surface clears; ceremony may proceed to lock"
            ),
        },
        "split_width_diagnostic": diag,
        "shock_window_diagnostics": {
            "machine_reason": "exogenous_shock_outside_model_class",
            "note": (
                "The 2020-2022 COVID / inflation shock is outside the engine's "
                "model class (no macro / epidemic channel) and hit mortality "
                "(2020-21 excess deaths) and marital rates (2020 marriage "
                "collapse) at least as hard as earnings (sec. 4.1). These "
                "cells are a first-class PUBLISHED diagnostic -- how far a "
                "mechanical engine misses a pandemic, quantified against held-"
                "out truth -- but are PARTITIONED OUT of every gated / report-"
                "only-margin set above. Dual axis: earnings reference years "
                f"{list(SHOCK_EARN_YEARS)} (2021 unobserved), flow event years "
                f"{list(SHOCK_FLOW_YEARS)}."
            ),
            "reference": shock_ref,
            "floor": shock_floor,
            "n_cells": len(shock_ref),
        },
        "report_reasons": sorted(set(part["report_only"].values())),
        "families": {fam: sorted(keys) for fam, keys in families.items()},
        "per_seed": per_seed,
        "revision_pins": {
            "artifact_schema_version": SCHEMA_VERSION,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = build_artifact(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, artifact, sidecar=True)
    print(f"wrote {ARTIFACT_PATH}")
    print(f"sha256 {_sha256_bytes(ARTIFACT_PATH)}")


if __name__ == "__main__":
    main()
