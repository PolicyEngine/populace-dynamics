"""W2 interim seam diagnostic: full tax-benefit incidence of the caregiver
credit on certified populace (reported, not gated).

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It is
the ADR-0001 **W2 interim product**, demonstrated end to end: dynamics-computed
benefit deltas for a committed reform encoding, fed as inputs to a standard
policyengine simulation on the certified populace file, yielding full
current-year tax-and-benefit incidence of a Social Security reform with program
interactions included. What this diagnostic certifies is the **seam**, not the
levels.

Frozen spec: issue #42 comment 4950247511 (registration URL below). Where this
module and the registration disagree, the registration wins.

The seam (ADR-0001 W2)
======================
policyengine-us takes ``social_security`` as an **uprated survey input** -- no
benefit formula exists upstream (docs/adr/0001-populace-axiom-seam-ownership.md,
finding from the survivor-plumbing build #80). Integration is therefore
replacing (here: perturbing) that survey input with the modelled value; all
downstream incidence -- taxation of benefits, means-tested program interactions,
poverty, MTRs -- flows through existing policyengine machinery unchanged.

Three steps (frozen spec)
=========================
1. **Observed frame.** Per-person annual benefit deltas under the committed
   caregiver-credit (Biden plan) encoding and the Phase-A conventions, reused
   VERBATIM from the anchor replication ``scripts/replication_caregiver.py``
   (its :class:`CaregiverStudy` + :func:`score_population`; Biden =
   ``PLANS[0]``: credit 1/2 average wage, child younger than 12, 5-year cap).
   The per-person quantity is the reformed-minus-baseline monthly PIA gain; the
   annual delta is that times twelve.
2. **Transport (interim).** Cell-mean deltas by age band x sex x AIME-proxy
   decile, applied to populace persons with positive ``social_security`` in the
   matching cells (recipients only; the extensive margin is out of scope and
   named). Because the caregiver replication is **scale-invariant** -- the
   transported AIME, credit level, and bend points all scale linearly with
   NAWI[2048], so only shares/ratios are meaningful and the absolute
   2050-transport dollar level is immaterial -- the transported cell statistic
   is the scale-free **ratio of weighted totals** ``sum(w*gain)/sum(w*base_PIA)``
   within each cell (a proportional benefit increase, robust to the
   low-base-PIA denominator blow-up of per-person ratios). Each populace
   recipient's dollar delta is that cell proportion times their certified
   ``social_security``. This cell-mean mapping is NOT the W1-certified
   covariate transport; that named INTERIM-transport delta rides every scored
   claim below.
3. **policyengine.** The certified populace **default** bundle via the managed
   ``policyengine.py`` runner (``pe.us.managed_microsimulation()``), baseline
   vs ``social_security + delta``, current year (2026). Outputs: gross benefit
   cost; NET fiscal cost after the income-tax clawback (taxation of benefits)
   and the enumerated means-tested program offsets (SNAP, SSI, Medicaid, ACA
   PTC, TANF); SPM poverty (65+ and overall); decile incidence.

Named interim-transport deltas (carried on every scored claim)
==============================================================
* cell-mean proportional mapping, not the W1-certified covariate transport;
* the AIME proxy on the populace frame is the recipient's own
  ``social_security`` rank (monotone in PIA/AIME, contaminated by claiming and
  auxiliary benefits);
* the observed anchor is a single birth cohort (1943-1957 -> ages 69-83 in
  2026); recipient age bands outside that support draw the (sex, decile)
  marginal;
* recipients only (positive ``social_security``); the extensive margin (new
  entitlements) is out of scope;
* the certified default is the sparse-57k build, which zeroes untargeted engine
  inputs (100% take-up, no SSI asset test, inert imputed Medicaid), so the
  means-tested offsets move in the right DIRECTION but their magnitudes are
  interim; and
* current-year static incidence -- no behavioural response, no claiming change,
  no demographic projection (gate-2/3 territory).

5-seed floors on the delta cells (person-disjoint half-splits of the observed
frame) name the sampling scale; one run; artifact
``runs/w2_seam_caregiver_v1.json``; ``reported_not_gated: true``; publishes
regardless of how the seam lands.

Environment
===========
Steps 1-2 (the SSA oracle + PSID) run in the repository venv; step 3 (the
managed pe-us microsim) runs under the policyengine.py venv, and this script
launches the two microsim passes as separate subprocesses (one sim per process
-- the full sparse-57k pass peaks tens of GB, so baseline and reform must not
co-reside). Run from the repository root with the PSID files staged::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    export POPULACE_DYNAMICS_PE_PYTHON=$HOME/PolicyEngine/policyengine.py/.venv/bin/python
    .venv/bin/python scripts/w2_seam_caregiver.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATH = ROOT / "runs" / "w2_seam_caregiver_v1.json"
ARTIFACT_SCHEMA_VERSION = "w2_seam_caregiver.v1"
RUN_NAME = "w2_seam_caregiver_v1"

#: This diagnostic's frozen-spec registration (issue #42 comment 4950247511).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4950247511"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4950247511"
#: The committed anchor replication whose Biden encoding is reused verbatim.
ANCHOR_REPLICATION_ARTIFACT = ROOT / "runs" / "replication_caregiver_v1.json"
ADR = "docs/adr/0001-populace-axiom-seam-ownership.md"

#: Current policy year of the certified populace default bundle.
YEAR = 2026
#: Observed anchor cohort reaches these ages in ``YEAR`` (born 1943-1957).
COHORT_BIRTH_LO, COHORT_BIRTH_HI = 1943, 1957
#: Number of AIME-proxy deciles.
N_DECILES = 10
#: Minimum observed-frame persons for a full (band, sex, decile) cell to be
#: used before falling back to the (sex, decile) marginal.
MIN_CELL_N = 5
#: Person-disjoint half-split seeds for the delta-cell sampling floor.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: Winsorization of the transported proportion is NOT applied; the ratio-of-
#: means statistic is already robust. This constant documents that choice.
APPLY_WINSOR = False

#: Enumerated means-tested program-offset variables (pe-us names), in order.
OFFSET_PROGRAMS: tuple[str, ...] = (
    "snap",
    "ssi",
    "medicaid",
    "aca_ptc",
    "tanf",
)
#: Tax-side variables (the taxation-of-benefits clawback and memo lines).
TAX_VARS: tuple[str, ...] = (
    "income_tax",
    "state_income_tax",
    "employee_payroll_tax",
)
#: Every aggregate the pe-us passes report (weighted totals, dollars).
AGG_VARS: tuple[str, ...] = (
    "social_security",
    "household_net_income",
    *TAX_VARS,
    *OFFSET_PROGRAMS,
)
#: Movement threshold for "a program moved" (weighted dollars).
MOVED_ABS_DOLLARS = 1e8  # $0.1B


# ======================================================================
# Pure helpers (numpy only; no heavy imports -- unit-testable off-machine)
# ======================================================================
def age_band(age: float) -> str:
    """Coarse retiree age band shared by both frames.

    The observed anchor cohort reaches ages 69-83 in ``YEAR``; the ``62-74``
    band therefore also absorbs younger populace recipients (survivors, early
    claimers) as a named downward extrapolation, and ``80+`` absorbs the oldest.
    """
    if age < 75:
        return "62-74"
    if age < 80:
        return "75-79"
    return "80+"


AGE_BANDS: tuple[str, ...] = ("62-74", "75-79", "80+")


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    """Weighted mean; 0.0 for empty or zero-weight input."""
    w = np.asarray(weight, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0 or x.size == 0:
        return 0.0
    return float((w * x).sum() / total)


def weighted_decile_index(
    values: np.ndarray, weight: np.ndarray, n: int = N_DECILES
) -> np.ndarray:
    """Assign each element to a weighted decile group ``0..n-1``.

    Midpoint plotting-position convention (identical to the committed
    ``reform_delta_diagnostic._weighted_decile_groups``): sort by value, place
    each atom at the midpoint of its normalized cumulative weight, and floor
    ``n`` times that position into ``0..n-1``. A mass point (e.g. the lowest
    benefits) fills the lowest groups by cumulative weight.
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ws = w[order]
    cumulative = np.cumsum(ws)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    if total <= 0.0:
        return np.zeros(x.shape[0], dtype=np.int64)
    positions = (cumulative - 0.5 * ws) / total
    group_sorted = np.clip((positions * n).astype(int), 0, n - 1)
    groups = np.empty(x.shape[0], dtype=np.int64)
    groups[order] = group_sorted
    return groups


def ratio_of_means(
    gain: np.ndarray, base: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    """Scale-free proportional benefit increase = sum(w*gain)/sum(w*base).

    Returns ``(proportion, weighted_base_total)``. The proportion is 0.0 when
    the weighted baseline total is non-positive (a degenerate cell).
    """
    w = np.asarray(weight, dtype=np.float64)
    num = float((w * np.asarray(gain, dtype=np.float64)).sum())
    den = float((w * np.asarray(base, dtype=np.float64)).sum())
    return (num / den if den > 0.0 else 0.0), den


def lookup_cell_prop(
    tables: dict[str, Any], band: str, sex: str, decile: int
) -> tuple[float, str]:
    """Cell proportion with the named fallback chain.

    ``(band, sex, decile)`` if the observed cell has >= :data:`MIN_CELL_N`
    persons and a positive weighted baseline; else the ``(sex, decile)``
    marginal; else the ``(decile)`` marginal; else the overall proportion.
    Returns ``(proportion, source_level)``.
    """
    cell = tables["full"].get((band, sex, decile))
    if cell is not None and cell["n"] >= MIN_CELL_N and cell["wbase"] > 0.0:
        return cell["prop"], "cell"
    marg = tables["sex_decile"].get((sex, decile))
    if marg is not None and marg["n"] >= MIN_CELL_N and marg["wbase"] > 0.0:
        return marg["prop"], "sex_decile"
    dd = tables["decile"].get(decile)
    if dd is not None and dd["wbase"] > 0.0:
        return dd["prop"], "decile"
    return tables["overall"]["prop"], "overall"


def net_fiscal_cost(
    gross_benefit_cost: float,
    income_tax_clawback: float,
    program_deltas: dict[str, float],
) -> dict[str, float]:
    """Net fiscal cost = gross - income-tax clawback - means-tested savings.

    ``program_deltas`` are reform-minus-baseline program totals (means-tested
    programs fall, so their deltas are typically negative); the means-tested
    saving is minus their sum. Returns the components and the net/gross ratio.
    """
    means_tested_savings = -float(sum(program_deltas.values()))
    net = gross_benefit_cost - income_tax_clawback - means_tested_savings
    return {
        "gross_benefit_cost": gross_benefit_cost,
        "income_tax_clawback": income_tax_clawback,
        "means_tested_savings": means_tested_savings,
        "net_fiscal_cost": net,
        "net_over_gross": (
            net / gross_benefit_cost if gross_benefit_cost > 0.0 else None
        ),
    }


def poverty_rate(
    in_poverty: np.ndarray, weight: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Weighted share in poverty (optionally within ``mask``)."""
    flag = np.asarray(in_poverty, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        flag, w = flag[m], w[m]
    total = float(w.sum())
    return float((flag * w).sum() / total) if total > 0.0 else 0.0


# ======================================================================
# Step 1 + 2 helpers (observed frame + cell tables) -- repo venv
# ======================================================================
def build_cell_tables(df: Any) -> dict[str, Any]:
    """Ratio-of-means proportion tables at cell / (sex,decile) / decile / all.

    ``df`` has columns ``weight``, ``gain`` (monthly reformed-minus-baseline
    PIA), ``base`` (baseline monthly PIA), ``band``, ``sex``, ``dec``.
    """

    def rom(sub: Any) -> dict[str, Any]:
        prop, wbase = ratio_of_means(
            sub["gain"].to_numpy(np.float64),
            sub["base"].to_numpy(np.float64),
            sub["weight"].to_numpy(np.float64),
        )
        return {"prop": prop, "n": int(len(sub)), "wbase": wbase}

    full = {
        (b, s, int(d)): rom(sub)
        for (b, s, d), sub in df.groupby(["band", "sex", "dec"])
    }
    sex_decile = {
        (s, int(d)): rom(sub) for (s, d), sub in df.groupby(["sex", "dec"])
    }
    decile = {int(d): rom(sub) for d, sub in df.groupby("dec")}
    return {
        "full": full,
        "sex_decile": sex_decile,
        "decile": decile,
        "overall": rom(df),
    }


def transport_gross(
    tables: dict[str, Any],
    band: np.ndarray,
    sex: np.ndarray,
    decile: np.ndarray,
    social_security: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-recipient dollar delta and weighted gross cost for a cell table.

    ``delta_i = cell_prop(band_i, sex_i, decile_i) * social_security_i`` over the
    recipient arrays; gross = ``sum(weight * delta)``.
    """
    prop = np.array(
        [
            lookup_cell_prop(tables, b, s, int(d))[0]
            for b, s, d in zip(band, sex, decile, strict=True)
        ],
        dtype=np.float64,
    )
    delta = prop * np.asarray(social_security, dtype=np.float64)
    gross = float((np.asarray(weight, dtype=np.float64) * delta).sum())
    return delta, gross


def build_observed_frame() -> Any:
    """Step 1: the Phase-A observed frame with per-person Biden deltas + demo.

    Reuses the committed anchor replication byte-for-byte (its
    :class:`CaregiverStudy` and :func:`score_population`), then joins the
    person's sex (PSID cross-year individual file ``ER32000``) and age in
    ``YEAR`` (from the study's implied birth year). Returns a DataFrame with
    ``person_id, weight, base (base_pia), gain (monthly Biden PIA gain),
    annual_delta, base_aime, sex, age, band, dec``.
    """
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from replication_caregiver import (  # noqa: E402
        CaregiverStudy,
        score_population,
    )
    from replication_ppi_mermin import build_transport  # noqa: E402

    from populace_dynamics.data import psid  # noqa: E402
    from populace_dynamics.ss.params import (  # noqa: E402
        load_ssa_parameters,
    )

    params = load_ssa_parameters()
    transport = build_transport(params)
    study = CaregiverStudy(params, transport)
    scored = score_population(study, params, transport)

    birth_year = pd.Series(study.birth_year)
    out = pd.DataFrame(
        {
            "person_id": scored["person_id"].to_numpy(),
            "weight": scored["weight"].to_numpy(np.float64),
            "base": scored["base_pia"].to_numpy(np.float64),
            "gain": scored["gain_Biden"].to_numpy(np.float64),
            "base_aime": scored["base_aime"].to_numpy(np.float64),
        }
    )
    out["annual_delta"] = out["gain"] * 12.0
    out["birth_year"] = out["person_id"].map(birth_year).to_numpy()
    out["age"] = (YEAR - out["birth_year"]).astype(float)

    # Sex from the PSID cross-year individual file (time-invariant ER32000).
    labels = psid.parse_sps_labels(psid.product_sps_path("ind2023er"))
    sex_matches = psid.find_variables(labels, r"^SEX OF INDIVIDUAL")
    sex_var = sorted(sex_matches)[0]
    ind = psid.read_psid("ind2023er", columns=["ER30001", "ER30002", sex_var])
    ind_pid = ind["ER30001"].to_numpy() * 1000 + ind["ER30002"].to_numpy()
    sex_map = dict(zip(ind_pid.tolist(), ind[sex_var].tolist(), strict=False))
    out["sex_code"] = out["person_id"].map(sex_map)
    out["sex"] = out["sex_code"].map({1: "male", 2: "female"})
    if out["sex"].isna().any():
        missing = int(out["sex"].isna().sum())
        raise RuntimeError(
            f"{missing} career-frame persons lack a resolvable ER32000 sex; "
            "the transport cells require sex on every observed person"
        )
    out["band"] = out["age"].map(age_band)
    out["dec"] = weighted_decile_index(
        out["base_aime"].to_numpy(np.float64),
        out["weight"].to_numpy(np.float64),
    )
    out["sex_var"] = sex_var
    out["ss_oracle_revision"] = getattr(params, "pe_us_revision", None)
    return out


def build_delta_cells_and_floors(
    observed: Any, populace: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Step 2: primary cell table, per-recipient delta, cheap gross + floors.

    ``populace`` carries the recipient arrays ``band``, ``sex``, ``decile``,
    ``social_security``, ``weight`` (positive-``social_security`` persons only)
    and ``person_index`` (their row index in the full baseline person order).
    Returns the per-person delta array over ALL populace persons plus the cell
    diagnostics and the 5-seed delta-cell sampling floors.
    """
    import pandas as pd

    tables = build_cell_tables(observed)

    rec_band = populace["band"]
    rec_sex = populace["sex"]
    rec_dec = populace["decile"]
    rec_ss = populace["social_security"]
    rec_w = populace["weight"]

    # Primary transport + per-recipient fallback source accounting.
    props, sources = [], []
    for b, s, d in zip(rec_band, rec_sex, rec_dec, strict=True):
        p, src = lookup_cell_prop(tables, b, s, int(d))
        props.append(p)
        sources.append(src)
    props = np.array(props, dtype=np.float64)
    rec_delta = props * rec_ss
    gross_primary = float((rec_w * rec_delta).sum())

    # Scatter recipient deltas into the full-person delta array.
    full_delta = np.zeros(int(populace["n_persons"]), dtype=np.float64)
    full_delta[populace["person_index"]] = rec_delta

    # 5-seed person-disjoint half-split floor on the transported gross, and on
    # the per-cell proportion (the delta cells are where sampling enters).
    from populace_dynamics.harness import panel as hpanel  # noqa: E402

    per_seed = []
    for seed in SEEDS:
        left, right = hpanel.split_panel_by_person(
            observed, "person_id", fraction=0.5, seed=seed
        )
        t_left = build_cell_tables(left)
        t_right = build_cell_tables(right)
        _, gross_left = transport_gross(
            t_left, rec_band, rec_sex, rec_dec, rec_ss, rec_w
        )
        _, gross_right = transport_gross(
            t_right, rec_band, rec_sex, rec_dec, rec_ss, rec_w
        )
        per_seed.append(
            {
                "seed": seed,
                "gross_side_a": gross_left,
                "gross_side_b": gross_right,
                "gross_abs_gap": abs(gross_left - gross_right),
            }
        )
    gross_gaps = [s["gross_abs_gap"] for s in per_seed]
    gross_floor = {
        "mean": float(np.mean(gross_gaps)),
        "sd": float(np.std(gross_gaps, ddof=1)),
        "max": float(np.max(gross_gaps)),
        "as_share_of_gross": (
            float(np.mean(gross_gaps)) / gross_primary
            if gross_primary > 0
            else None
        ),
        "per_seed": per_seed,
    }

    # Per-cell proportion floor across seeds (delta-cell sampling scale).
    cell_gap_summary = _cell_prop_floor(observed)

    # Cell coverage diagnostics.
    source_counts = pd.Series(sources).value_counts().to_dict()
    populated = sum(
        1
        for c in tables["full"].values()
        if c["n"] >= MIN_CELL_N and c["wbase"] > 0.0
    )
    return {
        "tables": tables,
        "full_delta": full_delta,
        "gross_primary": gross_primary,
        "recipient_prop": props,
        "cell_table_summary": {
            "n_cells_defined": len(tables["full"]),
            "n_cells_full_grid": len(AGE_BANDS) * 2 * N_DECILES,
            "n_cells_usable_min_n": populated,
            "min_cell_n": MIN_CELL_N,
            "fallback_source_counts_recipients": {
                k: int(v) for k, v in source_counts.items()
            },
            "overall_proportion": tables["overall"]["prop"],
        },
        "gross_floor_5seed": gross_floor,
        "cell_proportion_floor_5seed": cell_gap_summary,
    }


def _cell_prop_floor(observed: Any) -> dict[str, Any]:
    """Per-cell |A-B| proportion gap across the 5 seeds, summarized."""
    from populace_dynamics.harness import panel as hpanel  # noqa: E402

    cells = sorted(
        {
            (b, s, int(d))
            for b, s, d in zip(
                observed["band"], observed["sex"], observed["dec"], strict=True
            )
        }
    )
    per_cell_gaps: dict[tuple, list[float]] = {c: [] for c in cells}
    for seed in SEEDS:
        left, right = hpanel.split_panel_by_person(
            observed, "person_id", fraction=0.5, seed=seed
        )
        tl = build_cell_tables(left)["full"]
        tr = build_cell_tables(right)["full"]
        for c in cells:
            pl = tl.get(c, {"prop": 0.0})["prop"]
            pr = tr.get(c, {"prop": 0.0})["prop"]
            per_cell_gaps[c].append(abs(pl - pr))
    cell_means = [float(np.mean(g)) for g in per_cell_gaps.values()]
    return {
        "n_cells": len(cells),
        "mean_abs_gap_over_cells": float(np.mean(cell_means)),
        "max_abs_gap_over_cells": float(np.max(cell_means)),
        "note": (
            "per-cell mean |side_a - side_b| proportion across 5 "
            "person-disjoint half-splits of the observed frame; the sampling "
            "scale of the transported delta cells"
        ),
    }


# ======================================================================
# Step 3: policyengine passes (subprocess, policyengine.py venv)
# ======================================================================
def _pe_stage_baseline(out_prefix: str) -> None:
    """Baseline pass: dump per-person/per-household arrays + weighted aggregates."""
    import policyengine as pe  # noqa: E402

    sim = pe.us.managed_microsimulation()
    ss = sim.calculate("social_security", YEAR).values.astype(np.float64)
    age = sim.calculate("age", YEAR).values.astype(np.float64)
    is_male = sim.calculate("is_male", YEAR).values.astype(bool)
    pweight = sim.calculate("person_weight", YEAR).values.astype(np.float64)
    pov = sim.calculate(
        "spm_unit_is_in_spm_poverty", YEAR, map_to="person"
    ).values.astype(np.float64)
    hh_decile = sim.calculate("household_income_decile", YEAR).values.astype(
        np.int64
    )
    hh_net = sim.calculate("household_net_income", YEAR).values.astype(
        np.float64
    )
    hh_weight = sim.calculate("household_weight", YEAR).values.astype(
        np.float64
    )
    aggregates = {v: float(sim.calculate(v, YEAR).sum()) for v in AGG_VARS}
    dataset_id = _dataset_identifier(sim)
    np.savez(
        out_prefix + ".npz",
        social_security=ss,
        age=age,
        is_male=is_male,
        person_weight=pweight,
        poverty=pov,
        hh_decile=hh_decile,
        hh_net=hh_net,
        hh_weight=hh_weight,
    )
    meta = {
        "aggregates": aggregates,
        "poverty": _poverty_block(pov, pweight, age),
        "counts": {
            "n_persons": int(ss.size),
            "n_households": int(hh_net.size),
            "n_recipients": int((ss > 0).sum()),
            "pop_total": float(pweight.sum()),
            "pop_65plus": float(pweight[age >= 65].sum()),
        },
        "dataset": dataset_id,
        "pe_us_version": _pe_us_version(),
    }
    Path(out_prefix + ".json").write_text(json.dumps(meta, indent=2))


def _pe_stage_reform(out_prefix: str, delta_path: str) -> None:
    """Reform pass: social_security = baseline + delta; dump reform outputs."""
    import policyengine as pe  # noqa: E402

    payload = np.load(delta_path)
    delta = payload["delta"].astype(np.float64)
    baseline_ss = payload["baseline_ss"].astype(np.float64)

    sim = pe.us.managed_microsimulation()
    base = sim.calculate("social_security", YEAR).values.astype(np.float64)
    if base.size != delta.size:
        raise RuntimeError(
            f"reform person count {base.size} != delta length {delta.size}; "
            "the certified dataset changed between passes (seam alignment)"
        )
    if not np.allclose(base, baseline_ss, rtol=0, atol=1e-6):
        raise RuntimeError(
            "reform baseline social_security does not match the baseline "
            "pass row-for-row; the managed dataset row order is not stable "
            "across processes (seam alignment failure)"
        )
    sim.set_input("social_security", YEAR, base + delta)

    age = sim.calculate("age", YEAR).values.astype(np.float64)
    pweight = sim.calculate("person_weight", YEAR).values.astype(np.float64)
    pov = sim.calculate(
        "spm_unit_is_in_spm_poverty", YEAR, map_to="person"
    ).values.astype(np.float64)
    hh_net = sim.calculate("household_net_income", YEAR).values.astype(
        np.float64
    )
    aggregates = {v: float(sim.calculate(v, YEAR).sum()) for v in AGG_VARS}
    np.savez(out_prefix + ".npz", hh_net=hh_net)
    meta = {
        "aggregates": aggregates,
        "poverty": _poverty_block(pov, pweight, age),
        "counts": {"n_households": int(hh_net.size)},
    }
    Path(out_prefix + ".json").write_text(json.dumps(meta, indent=2))


def _poverty_block(
    pov: np.ndarray, pweight: np.ndarray, age: np.ndarray
) -> dict[str, float]:
    return {
        "spm_poverty_rate_overall": poverty_rate(pov, pweight),
        "spm_poverty_rate_65plus": poverty_rate(pov, pweight, age >= 65),
    }


def _dataset_identifier(sim: Any) -> dict[str, Any]:
    ds = getattr(sim, "dataset", None)
    name = getattr(ds, "name", None) or getattr(ds, "label", None)
    return {
        "repr": str(ds)[:300] if ds is not None else None,
        "name": str(name) if name is not None else None,
        "note": "certified populace default bundle (pe.us.managed_microsimulation)",
    }


def _pe_us_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("policyengine-us")
    except Exception:
        return None


# ======================================================================
# Orchestration (repo venv): observed -> baseline -> transport -> reform
# ======================================================================
def _pe_python() -> str:
    env = os.environ.get("POPULACE_DYNAMICS_PE_PYTHON")
    if env:
        return env
    default = Path.home() / "PolicyEngine/policyengine.py/.venv/bin/python"
    return str(default)


def _run_pe_stage(stage: str, out_prefix: str, delta_path: str | None) -> None:
    cmd = [
        _pe_python(),
        str(Path(__file__).resolve()),
        "--pe-stage",
        stage,
        "--out",
        out_prefix,
    ]
    if delta_path is not None:
        cmd += ["--delta", delta_path]
    subprocess.run(cmd, check=True)


def _load_stage(out_prefix: str) -> tuple[dict[str, Any], Any]:
    meta = json.loads(Path(out_prefix + ".json").read_text())
    arrays = np.load(out_prefix + ".npz")
    return meta, arrays


def _decile_incidence(
    hh_decile: np.ndarray,
    hh_weight: np.ndarray,
    hh_net_base: np.ndarray,
    hh_net_reform: np.ndarray,
) -> dict[str, Any]:
    """Household net-income change by baseline household income decile."""
    change = hh_net_reform - hh_net_base
    total_gain = float((hh_weight * change).sum())

    def share(mask: np.ndarray) -> float:
        return (
            100.0 * float((hh_weight[mask] * change[mask]).sum()) / total_gain
            if total_gain != 0
            else 0.0
        )

    rows = []
    for d in range(1, N_DECILES + 1):
        m = hh_decile == d
        w = hh_weight[m]
        wtot = float(w.sum())
        agg_base = float((w * hh_net_base[m]).sum())
        rows.append(
            {
                "decile": d,
                "n_households": int(m.sum()),
                "mean_change_per_household": (
                    float((w * change[m]).sum()) / wtot if wtot > 0 else 0.0
                ),
                "pct_change_of_net_income": (
                    100.0 * float((w * change[m]).sum()) / agg_base
                    if agg_base > 0
                    else 0.0
                ),
                "share_of_aggregate_gain_pct": share(m),
            }
        )
    # pe-us assigns household_income_decile = -1 to negative-net-income
    # households; they sit outside the 1..10 structure but still receive a
    # share of the gain, so account for them explicitly (the decile shares +
    # this block partition the total gain).
    oor = (hh_decile < 1) | (hh_decile > N_DECILES)
    out_of_range = {
        "n_households": int(oor.sum()),
        "weighted_households": float(hh_weight[oor].sum()),
        "share_of_aggregate_gain_pct": share(oor),
        "note": (
            "households with household_income_decile outside 1..10 "
            "(negative net income; pe-us assigns -1), outside the decile grid"
        ),
    }
    pct = [r["pct_change_of_net_income"] for r in rows]
    return {
        "by_decile": rows,
        "out_of_range": out_of_range,
        "bottom_half_share_of_gain_pct": float(
            sum(r["share_of_aggregate_gain_pct"] for r in rows[:5])
        ),
        "bottom_quintile_share_of_gain_pct": float(
            sum(r["share_of_aggregate_gain_pct"] for r in rows[:2])
        ),
        "pct_change_monotone_decreasing_top_vs_bottom": bool(
            pct[0] >= pct[-1]
        ),
        "note": (
            "deciles are baseline household_income_decile (1=poorest); "
            "incidence as % of baseline net income is the bottom-loaded shape, "
            "the aggregate-gain share is diluted by the recipient-only transport"
        ),
    }


def _program_offsets(
    base_agg: dict[str, float], reform_agg: dict[str, float]
) -> dict[str, Any]:
    deltas = {p: reform_agg[p] - base_agg[p] for p in OFFSET_PROGRAMS}
    moved = {p: d for p, d in deltas.items() if abs(d) >= MOVED_ABS_DOLLARS}
    return {
        "program_deltas": deltas,
        "programs_that_moved": sorted(moved, key=lambda p: deltas[p]),
        "moved_threshold_dollars": MOVED_ABS_DOLLARS,
        "snap_falls": bool(deltas["snap"] < 0.0),
        "baseline_levels": {p: base_agg[p] for p in OFFSET_PROGRAMS},
        "reform_levels": {p: reform_agg[p] for p in OFFSET_PROGRAMS},
    }


def build_expectations(
    fiscal: dict[str, Any],
    offsets: dict[str, Any],
    poverty: dict[str, Any],
    decile: dict[str, Any],
) -> dict[str, Any]:
    """The registration's four pre-registered bands, recomputed with verdicts."""
    nog = fiscal["net_over_gross"]
    net_in_band = bool(nog is not None and 0.75 <= nog <= 0.92)
    pov_red_overall = poverty["reduction_overall_pp"]
    pov_red_65 = poverty["reduction_65plus_pp"]
    concentrates_65 = bool(pov_red_65 >= pov_red_overall and pov_red_65 > 0)
    bottom_half = bool(decile["bottom_half_share_of_gain_pct"] > 50.0)
    failures = []
    if not net_in_band:
        failures.append(
            f"net/gross={nog:.3f} outside the pre-registered 0.75-0.92 band"
        )
    if not offsets["snap_falls"]:
        failures.append("SNAP did not fall")
    if not concentrates_65:
        failures.append("SPM poverty reduction did not concentrate at 65+")
    if not bottom_half:
        failures.append(
            "aggregate net-income gain did not concentrate in the bottom half"
        )
    return {
        "net_cost_75_92pct_of_gross": {
            "statement": (
                "net fiscal cost lands 75-92% of gross benefit cost "
                "(taxation of benefits + means-tested offsets claw back 8-25%)"
            ),
            "net_over_gross": nog,
            "held": net_in_band,
        },
        "snap_spending_falls": {
            "statement": "SNAP spending falls",
            "snap_delta": offsets["program_deltas"]["snap"],
            "held": offsets["snap_falls"],
        },
        "poverty_reduction_concentrates_65plus": {
            "statement": "the poverty reduction concentrates at 65+",
            "reduction_overall_pp": pov_red_overall,
            "reduction_65plus_pp": pov_red_65,
            "held": concentrates_65,
        },
        "decile_gains_bottom_half": {
            "statement": (
                "decile gains concentrate in the bottom half (the anchor's "
                "52-62% bottom-quintile band, diluted by the recipient-only "
                "transport)"
            ),
            "bottom_half_share_of_gain_pct": decile[
                "bottom_half_share_of_gain_pct"
            ],
            "bottom_quintile_share_of_gain_pct": decile[
                "bottom_quintile_share_of_gain_pct"
            ],
            "held": bottom_half,
        },
        "all_expectations_held": len(failures) == 0,
        "seam_failure_modes": failures,
    }


def named_interim_deltas() -> list[str]:
    return [
        "cell-mean proportional mapping (ratio of weighted totals "
        "sum(w*gain)/sum(w*base_PIA) per age-band x sex x AIME-proxy-decile "
        "cell), NOT the W1-certified covariate transport",
        "the AIME proxy on the populace frame is the recipient's own "
        "social_security rank decile (monotone in PIA/AIME; contaminated by "
        "claiming-age adjustment and auxiliary benefits)",
        "the observed anchor is a single 1943-1957 birth cohort (ages 69-83 "
        "in the policy year); recipient age bands outside that support draw "
        "the (sex, decile) marginal proportion",
        "recipients only (positive social_security); the extensive margin "
        "(new entitlements from the credit) is out of scope",
        "the certified populace default is the sparse-57k build, which zeroes "
        "untargeted engine inputs (100% program take-up, no SSI asset test, "
        "inert imputed Medicaid), so the means-tested offsets move in the "
        "correct DIRECTION but their magnitudes are interim, not calibrated",
        "current-year static incidence: no behavioural response, no claiming "
        "change, no demographic projection (gate-2/3 territory)",
    ]


def _sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def run(verbose: bool = True, workdir: str | None = None) -> dict[str, Any]:
    """Execute the full W2 seam diagnostic and return the artifact dict."""
    import pandas as pd  # noqa: F401  (ensures the repo venv is in use)

    started = time.time()
    chosen = workdir or os.environ.get("POPULACE_DYNAMICS_W2_WORKDIR")
    if chosen:
        tmp = Path(chosen)
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="w2_seam_"))

    # Step 1: observed frame (SSA oracle + PSID), repo venv.
    if verbose:
        print("[1/4] observed frame (caregiver Biden deltas + sex/age) ...")
    observed = build_observed_frame()
    sex_var = str(observed["sex_var"].iloc[0])
    ss_oracle_revision = observed["ss_oracle_revision"].iloc[0]
    if verbose:
        print(f"      {len(observed)} career-frame persons scored")

    # Step 3a: baseline managed pe-us pass (subprocess, pe venv).
    if verbose:
        print("[2/4] baseline policyengine pass (certified default) ...")
    base_prefix = str(tmp / "baseline")
    _run_pe_stage("baseline", base_prefix, None)
    base_meta, base_arr = _load_stage(base_prefix)

    ss = base_arr["social_security"]
    age = base_arr["age"].astype(np.float64)
    is_male = base_arr["is_male"].astype(bool)
    pweight = base_arr["person_weight"].astype(np.float64)
    n_persons = int(ss.size)

    # Step 2: interim transport -> per-person delta on the populace frame.
    if verbose:
        print("[3/4] interim cell-mean transport -> per-person delta ...")
    rec_idx = np.nonzero(ss > 0.0)[0]
    rec_ss = ss[rec_idx].astype(np.float64)
    rec_w = pweight[rec_idx].astype(np.float64)
    rec_age = age[rec_idx]
    rec_band = np.array([age_band(a) for a in rec_age])
    rec_sex = np.where(is_male[rec_idx], "male", "female")
    rec_dec = weighted_decile_index(rec_ss, rec_w)
    populace = {
        "band": rec_band,
        "sex": rec_sex,
        "decile": rec_dec,
        "social_security": rec_ss,
        "weight": rec_w,
        "person_index": rec_idx,
        "n_persons": n_persons,
    }
    step2 = build_delta_cells_and_floors(observed, populace)
    full_delta = step2["full_delta"]

    delta_path = str(tmp / "delta.npz")
    np.savez(delta_path, delta=full_delta, baseline_ss=ss)

    # Step 3b: reform pass (subprocess, pe venv).
    if verbose:
        print("[4/4] reform policyengine pass (social_security + delta) ...")
    reform_prefix = str(tmp / "reform")
    _run_pe_stage("reform", reform_prefix, delta_path)
    reform_meta, reform_arr = _load_stage(reform_prefix)

    base_agg = base_meta["aggregates"]
    reform_agg = reform_meta["aggregates"]

    gross = reform_agg["social_security"] - base_agg["social_security"]
    income_tax_clawback = reform_agg["income_tax"] - base_agg["income_tax"]
    offsets = _program_offsets(base_agg, reform_agg)
    fiscal = net_fiscal_cost(
        gross, income_tax_clawback, offsets["program_deltas"]
    )
    fiscal["gross_from_transport_sum"] = step2["gross_primary"]
    fiscal["gross_engine_vs_transport_abs_gap"] = abs(
        gross - step2["gross_primary"]
    )
    fiscal["memo_state_income_tax_delta"] = (
        reform_agg["state_income_tax"] - base_agg["state_income_tax"]
    )
    fiscal["memo_payroll_tax_delta"] = (
        reform_agg["employee_payroll_tax"] - base_agg["employee_payroll_tax"]
    )
    fiscal["memo_household_net_income_delta"] = (
        reform_agg["household_net_income"] - base_agg["household_net_income"]
    )

    poverty = {
        "baseline": base_meta["poverty"],
        "reform": reform_meta["poverty"],
        "reduction_overall_pp": 100.0
        * (
            base_meta["poverty"]["spm_poverty_rate_overall"]
            - reform_meta["poverty"]["spm_poverty_rate_overall"]
        ),
        "reduction_65plus_pp": 100.0
        * (
            base_meta["poverty"]["spm_poverty_rate_65plus"]
            - reform_meta["poverty"]["spm_poverty_rate_65plus"]
        ),
    }

    decile = _decile_incidence(
        base_arr["hh_decile"].astype(np.int64),
        base_arr["hh_weight"].astype(np.float64),
        base_arr["hh_net"].astype(np.float64),
        reform_arr["hh_net"].astype(np.float64),
    )

    expectations = build_expectations(fiscal, offsets, poverty, decile)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "adr": ADR,
        "purpose": (
            "ADR-0001 W2 interim product end to end: dynamics-computed "
            "caregiver-credit (Biden) benefit deltas, transported by an "
            "interim cell-mean mapping onto the certified populace default "
            "file, fed to the managed policyengine runner as a "
            "social_security perturbation -> full current-year tax-and-benefit "
            "incidence of a Social Security reform, program interactions "
            "included. Certifies the SEAM, not the levels. Reads no gate, "
            "changes no gate; publishes regardless of outcome."
        ),
        "what_is_certified": (
            "the SEAM: that dynamics-computed benefit deltas flow through the "
            "existing policyengine machinery (taxation of benefits, "
            "means-tested program interactions, SPM poverty, decile incidence) "
            "and land inside the pre-registered qualitative bands -- NOT the "
            "population dollar LEVELS, which every named interim-transport "
            "delta below qualifies"
        ),
        "interim_transport_delta_named_on_every_claim": named_interim_deltas(),
        "reform_encoding": {
            "plan": "Biden caregiver credit (2020 primary)",
            "source": (
                "scripts/replication_caregiver.py PLANS[0], reused verbatim: "
                "credit tops up to 1/2 the economy-wide average wage in years "
                "with a child younger than 12, capped at 5 benefit-maximising "
                "years, AIME/PIA recomputed on the Phase-A career frame"
            ),
            "credit_fraction": 0.5,
            "child_age_limit_exclusive": 12,
            "year_cap": 5,
            "anchor_replication_artifact": str(
                ANCHOR_REPLICATION_ARTIFACT.relative_to(ROOT)
            ),
            "anchor_replication_registration": "#42 comment 4911453454",
        },
        "step1_observed_frame": {
            "n_career_frame": int(len(observed)),
            "n_gainers": int((observed["gain"] > 1e-9).sum()),
            "weighted_mean_annual_delta_gainers": weighted_mean(
                observed.loc[observed["gain"] > 1e-9, "annual_delta"].to_numpy(
                    np.float64
                ),
                observed.loc[observed["gain"] > 1e-9, "weight"].to_numpy(
                    np.float64
                ),
            ),
            "sex_source": (
                f"PSID cross-year individual file ind2023er {sex_var} "
                "(SEX OF INDIVIDUAL), time-invariant, joined on "
                "ER30001*1000+ER30002"
            ),
            "weighted_sex_shares": {
                s: float(
                    observed.loc[observed["sex"] == s, "weight"].sum()
                    / observed["weight"].sum()
                )
                for s in ("male", "female")
            },
            "conventions": (
                "the Phase-A career-selection frame and the full 42 USC "
                "415(a)/(b) AIME/PIA chain transported to a common 2050 cohort, "
                "reused byte-for-byte from replication_caregiver; the per-person "
                "quantity is the reformed-minus-baseline monthly pre-415(g) PIA "
                "gain, annualized x12 for reporting"
            ),
        },
        "step2_transport": {
            "cell_definition": "age band x sex x AIME-proxy decile",
            "age_bands": list(AGE_BANDS),
            "n_deciles": N_DECILES,
            "cell_statistic": (
                "ratio of weighted totals sum(w*gain)/sum(w*base_PIA) within "
                "the cell (scale-free proportional benefit increase), applied "
                "to each populace recipient's certified social_security"
            ),
            "aime_proxy_observed": "transported baseline AIME (base_aime)",
            "aime_proxy_populace": "recipient social_security rank decile",
            "fallback_chain": (
                "cell -> (sex, decile) -> (decile) -> overall; a full cell "
                f"needs >= {MIN_CELL_N} observed persons and a positive "
                "weighted baseline"
            ),
            "winsorization_applied": APPLY_WINSOR,
            **step2["cell_table_summary"],
        },
        "step3_policyengine": {
            "runner": "pe.us.managed_microsimulation() (managed policyengine.py)",
            "dataset": base_meta["dataset"],
            "policy_year": YEAR,
            "pe_us_version": base_meta["pe_us_version"],
            "seam_mechanism": (
                "social_security is an uprated survey INPUT in pe-us (no "
                "upstream formula); the reform sets social_security = baseline "
                "+ per-person delta via set_input and recomputes all downstream "
                "incidence; baseline and reform run in separate processes and "
                "the reform asserts row-for-row baseline alignment"
            ),
            "counts": base_meta["counts"],
        },
        "fiscal": fiscal,
        "program_offsets": offsets,
        "poverty": poverty,
        "decile_incidence": decile,
        "delta_cell_floors_5seed": {
            "gross_cost_floor": step2["gross_floor_5seed"],
            "cell_proportion_floor": step2["cell_proportion_floor_5seed"],
            "seeds": list(SEEDS),
            "note": (
                "the delta cells are where sampling enters; the floor is the "
                "person-disjoint half-split |A-B| scale of the observed frame, "
                "propagated to the transported gross cost (no re-run of the "
                "microsim -- the gross cost is a weighted sum over the same "
                "populace recipients)"
            ),
        },
        "registered_expectations": expectations,
        "revision_pins": {
            "populace_dynamics_sha": _sha(ROOT),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "ss_oracle_pe_us_revision": ss_oracle_revision,
            "pe_us_version": base_meta["pe_us_version"],
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(artifact)
    return artifact


def _print_summary(artifact: dict[str, Any]) -> None:
    f = artifact["fiscal"]
    e = artifact["registered_expectations"]
    print("\n=== W2 seam: caregiver-credit full tax-benefit incidence ===")
    print(f"gross benefit cost:      ${f['gross_benefit_cost'] / 1e9:8.2f}B")
    print(f"income-tax clawback:     ${f['income_tax_clawback'] / 1e9:8.2f}B")
    print(f"means-tested savings:    ${f['means_tested_savings'] / 1e9:8.2f}B")
    print(f"NET fiscal cost:         ${f['net_fiscal_cost'] / 1e9:8.2f}B")
    nog = f["net_over_gross"]
    print(f"net / gross:              {nog:.3f}" if nog else "net/gross: n/a")
    print(
        "program offsets moved:   "
        + ", ".join(
            f"{p} {artifact['program_offsets']['program_deltas'][p] / 1e9:+.2f}B"
            for p in artifact["program_offsets"]["programs_that_moved"]
        )
    )
    p = artifact["poverty"]
    print(
        f"SPM poverty overall: {p['baseline']['spm_poverty_rate_overall']:.4f}"
        f" -> {p['reform']['spm_poverty_rate_overall']:.4f} "
        f"({p['reduction_overall_pp']:+.3f}pp)"
    )
    print(
        f"SPM poverty 65+:     {p['baseline']['spm_poverty_rate_65plus']:.4f}"
        f" -> {p['reform']['spm_poverty_rate_65plus']:.4f} "
        f"({p['reduction_65plus_pp']:+.3f}pp)"
    )
    d = artifact["decile_incidence"]
    print(
        f"bottom-half share of gain: "
        f"{d['bottom_half_share_of_gain_pct']:.1f}%"
    )
    for name, blk in e.items():
        if isinstance(blk, dict) and "held" in blk:
            print(f"  [{'PASS' if blk['held'] else 'FAIL'}] {name}")
    print(f"ALL EXPECTATIONS HELD: {e['all_expectations_held']}")
    if e["seam_failure_modes"]:
        print("seam failure modes: " + "; ".join(e["seam_failure_modes"]))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


def main() -> None:
    from populace_dynamics.artifacts import write_new  # noqa: E402

    artifact = run(verbose=True)
    # Round-trip through _json_default so numpy scalars survive write_new.
    payload = json.loads(json.dumps(artifact, default=_json_default))
    write_new(ARTIFACT_PATH, payload)
    print(f"\nwrote {ARTIFACT_PATH}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="W2 seam caregiver diagnostic"
    )
    parser.add_argument(
        "--pe-stage", choices=("baseline", "reform"), default=None
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--delta", default=None)
    args = parser.parse_args()
    if args.pe_stage == "baseline":
        _pe_stage_baseline(args.out)
    elif args.pe_stage == "reform":
        _pe_stage_reform(args.out, args.delta)
    else:
        main()


if __name__ == "__main__":
    _cli()
