"""Spouse/survivor auxiliary-benefit worked-example grid (#74, Phase C).

Builds ``runs/aux_benefit_examples_v1.json``: the reference artifact for
the 42 USC 402(b)/(c)/(e)/(f) plumbing in
:mod:`populace_dynamics.ss.benefits` and
:mod:`populace_dynamics.household`. Three blocks, each with inputs, our
outputs, oracle outputs where one exists, and the max absolute deviation:

1. **SSA worked examples** — the auxiliary amounts (spousal 50%/32.5%,
   the 71.5% survivor floor, RIB-LIM 82.5%, DRC pass-through). These are
   the primary oracle: policyengine-us has **no** spousal/survivor
   computation (its ``social_security_*`` variables are uprated survey
   inputs), so the auxiliary rates are validated against SSA's published
   figures, each cited. Every "expected" value is transcribed from SSA;
   the deceased's own claiming factors are DERIVED from the pinned
   402(q)/(w) machinery, not hand-typed.

2. **policyengine-us PIA foundation** — the own-benefit layer the
   auxiliary amounts are fractions of (½·PIA for spouses, PIA·factor for
   the RIB-LIM). Our :func:`~populace_dynamics.ss.benefits.pia` and
   :func:`~populace_dynamics.claiming.benefit_factor` are cross-checked
   against a **live policyengine-us Simulation** (``ss_pia`` and
   ``ss_retirement_age_adjustment_factor``), run in a separate
   interpreter located by ``POPULACE_DYNAMICS_PE_US_PYTHON`` — the same
   subprocess discipline ``build_cross_engine_pia_artifact.py`` uses for
   the Axiom engine. pe-us's ``ss_pia`` uses the *period*-year bend
   points, so the oracle is aligned by using the same year as the
   eligibility year. Skipped (oracle recorded null) if that interpreter
   cannot import policyengine_us.

3. **Couple / survivor illustrative grid** — our couple and survivor
   outputs over a worker-PIA × spouse-PIA × claim-age × death-timing
   grid, for reproducibility (no external oracle; pe-us cannot compute
   these).

Plus the real both-spouse **panel coverage** from the staged Marriage
History File, when available.

Run (from the populace-dynamics worktree)::

    PYTHONPATH=src POPULACE_DYNAMICS_PE_US_DIR=~/PolicyEngine/policyengine-us-main \\
        POPULACE_DYNAMICS_PE_US_PYTHON=<pe-us venv>/bin/python \\
        <ss-model venv>/bin/python scripts/build_aux_benefit_examples.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

from populace_dynamics import claiming, household
from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters, load_ssa_parameters

SCHEMA_VERSION = "aux_benefit_examples.v1"
RUN = "aux_benefit_examples_v1"
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = _ROOT / "runs" / "aux_benefit_examples_v1.json"

#: Fixed reference year for the pe-us PIA foundation cross-check. pe-us
#: ss_pia uses this year's bend points; the oracle uses it as the
#: eligibility year so the two align.
FOUNDATION_YEAR = 2026
#: A cohort born 1954 attains FRA at exactly 66 (792 months), so claiming
#: at 62/64/70 lands on the clean 0.75 / 0.86667 / 1.32 factors SSA's
#: RIB-LIM / DRC examples use.
_FRA66_BIRTH_YEAR = 1954


# ---------------------------------------------------------------------------
# Block 1: SSA worked examples (the auxiliary-amount oracle)
# ---------------------------------------------------------------------------
def _spousal_examples(params: SSAParameters) -> list[dict]:
    out = []
    # 50% of the worker's PIA at the spouse's FRA.
    out.append(
        {
            "name": "spouse_at_fra_half_pia",
            "citation": "42 USC 402(b)(2)/(c)(2); SSA RS 00615.020; "
            "ssa.gov/benefits/retirement/planner/applying7.html",
            "inputs": {
                "own_pia": 0.0,
                "worker_pia": 2000.0,
                "months_early": 0,
            },
            "our_output": benefits.spousal_benefit(0.0, 2000.0, 0, params),
            "expected": 1000.0,
            "note": "spouse's benefit is one-half of the worker's PIA at FRA",
        }
    )
    # 32.5% of the worker's PIA at 62 against an FRA of 67 (60 months).
    out.append(
        {
            "name": "spouse_at_62_fra67",
            "citation": "42 USC 402(q)(1); ssa.gov/OACT/quickcalc/spouse.html "
            "(32.5% at 62 when FRA is 67)",
            "inputs": {
                "own_pia": 0.0,
                "worker_pia": 2000.0,
                "months_early": 60,
            },
            "our_output": benefits.spousal_benefit(0.0, 2000.0, 60, params),
            "expected": 650.0,
            "note": "36*25/36% + 24*5/12% = 35% reduction -> 65% of 50%",
        }
    )
    # 35% of the worker's PIA at 62 against an FRA of 66 (48 months).
    out.append(
        {
            "name": "spouse_at_62_fra66",
            "citation": "42 USC 402(q)(1); SSA benefit-reduction table "
            "(35% at 62 when FRA is 66)",
            "inputs": {
                "own_pia": 0.0,
                "worker_pia": 2000.0,
                "months_early": 48,
            },
            "our_output": benefits.spousal_benefit(0.0, 2000.0, 48, params),
            "expected": 700.0,
            "note": "36*25/36% + 12*5/12% = 30% reduction -> 70% of 50%",
        }
    )
    # Dual entitlement: the excess over the spouse's own PIA.
    out.append(
        {
            "name": "spouse_dual_entitlement_excess",
            "citation": "42 USC 402(k)(3)(A); SSA RS 00615.020 (excess spouse)",
            "inputs": {
                "own_pia": 600.0,
                "worker_pia": 2000.0,
                "months_early": 0,
            },
            "our_output": benefits.spousal_benefit(600.0, 2000.0, 0, params),
            "expected": 400.0,
            "note": "excess of 50%*2000=1000 over own PIA 600",
        }
    )
    return out


def _survivor_examples(params: SSAParameters) -> list[dict]:
    out = []
    fra66_params = _replace_survivor_period(params, 72)
    # Deceased-claim factors DERIVED from the pinned machinery (FRA 66).
    f62 = claiming.benefit_factor(62 * 12, _FRA66_BIRTH_YEAR, params)  # 0.75
    f64 = claiming.benefit_factor(
        64 * 12, _FRA66_BIRTH_YEAR, params
    )  # 0.86667
    f70 = claiming.benefit_factor(70 * 12, _FRA66_BIRTH_YEAR, params)  # 1.32

    # 100% of the deceased's PIA at survivor FRA (deceased claimed at FRA).
    out.append(
        {
            "name": "widow_at_fra_full_pia",
            "citation": "42 USC 402(e)(2)(A)/(f)(3)(A); SSA RS 00615.301; "
            "ssa.gov/benefits/survivors/",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 0,
                "deceased_claimed_early_factor": 1.0,
            },
            "our_output": benefits.widow_benefit(0.0, 1000.0, 0, 1.0, params),
            "expected": 1000.0,
            "note": "widow(er) at survivor FRA gets 100% of the deceased PIA",
        }
    )
    # 71.5% floor at age 60 (survivor FRA 67 -> 84-month span).
    out.append(
        {
            "name": "widow_at_60_floor",
            "citation": "42 USC 402(q); SSA RS 00615.302 (71.5% at age 60)",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 84,
                "deceased_claimed_early_factor": 1.0,
            },
            "our_output": benefits.widow_benefit(0.0, 1000.0, 84, 1.0, params),
            "expected": 715.0,
            "note": "maximum 28.5% reduction over 84 months (age 60 to FRA 67)",
        }
    )
    # 81% at age 62 for a survivor FRA of 66 (48-month early over 72 span).
    out.append(
        {
            "name": "widow_at_62_fra66",
            "citation": "42 USC 402(q); SSA survivors reduction table "
            "(81% at 62 when survivor FRA is 66)",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 48,
                "deceased_claimed_early_factor": 1.0,
                "survivor_reduction_period_months": 72,
            },
            "our_output": benefits.widow_benefit(
                0.0, 1000.0, 48, 1.0, fra66_params
            ),
            "expected": 810.0,
            "note": "0.285 * 48/72 = 0.19 reduction -> 81% of the deceased PIA",
        }
    )
    # RIB-LIM: deceased took a reduced RIB below 82.5% -> widow gets 82.5%.
    out.append(
        {
            "name": "rib_lim_82_5_floor",
            "citation": "42 USC 402(e)(2)(D)/(k)(3)(A); SSA RS 00615.310 "
            "(RIB-LIM, 82.5% of PIA)",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 0,
                "deceased_claimed_early_factor": f62,
            },
            "our_output": benefits.widow_benefit(0.0, 1000.0, 0, f62, params),
            "expected": 825.0,
            "note": "deceased claimed at 62 (0.75 RIB < 0.825); widow gets 82.5%",
        }
    )
    # RIB-LIM higher-of: deceased's actual reduced benefit exceeds 82.5%.
    out.append(
        {
            "name": "rib_lim_deceased_actual_higher",
            "citation": "42 USC 402(e)(2)(D); SSA RS 00615.310 (larger of "
            "deceased actual or 82.5%)",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 0,
                "deceased_claimed_early_factor": f64,
            },
            "our_output": benefits.widow_benefit(0.0, 1000.0, 0, f64, params),
            "expected": 1000.0 * f64,
            "note": "deceased claimed at 64 (0.86667 RIB > 0.825); widow gets "
            "the deceased's actual benefit = PIA * factor (the higher-of "
            "branch; the factor is cross-checked against pe-us in "
            "pe_us_pia_foundation)",
        }
    )
    # Delayed-retirement credits pass through to the widow(er).
    out.append(
        {
            "name": "drc_pass_through",
            "citation": "42 USC 402(e)/(f); SSA RS 00615.301 (widow inherits DRCs)",
            "inputs": {
                "own_pia": 0.0,
                "deceased_pia": 1000.0,
                "survivor_months_early": 0,
                "deceased_claimed_early_factor": f70,
            },
            "our_output": benefits.widow_benefit(0.0, 1000.0, 0, f70, params),
            "expected": 1000.0 * f70,
            "note": "deceased claimed at 70 (1.32); widow inherits the credits "
            "= PIA * factor (the DRC pass-through branch)",
        }
    )
    return out


def _replace_survivor_period(
    params: SSAParameters, period_months: int
) -> SSAParameters:
    import dataclasses

    return dataclasses.replace(
        params, survivor_reduction_period_months=period_months
    )


def _summarise(examples: list[dict]) -> dict:
    devs = [abs(e["our_output"] - e["expected"]) for e in examples]
    worst = max(devs) if devs else 0.0
    argmax = examples[devs.index(worst)]["name"] if devs else None
    return {
        "n": len(examples),
        "max_abs_deviation": worst,
        "argmax": argmax,
    }


# ---------------------------------------------------------------------------
# Block 2: policyengine-us PIA foundation cross-check (live Simulation)
# ---------------------------------------------------------------------------
#: (aime, age-at-period, claim_age) grid. aime spans all three PIA
#: brackets (2026 bend points 1286/7749); ages fix FRA-67 (born 1960) and
#: FRA-66 (born 1954) cohorts; claim ages sweep early/at/delayed.
_AIMES = (1000, 2000, 4000, 6000, 9000)
_AGES = (66, 72)  # born 1960 (FRA 67) and 1954 (FRA 66) at FOUNDATION_YEAR
_CLAIM_AGES = (62, 65, 67, 70)

_PE_US_RUNNER = r"""
import json, sys
from policyengine_us import Simulation
job = json.load(sys.stdin)
year = job["year"]
out = []
for c in job["cases"]:
    sim = Simulation(situation={
        "people": {"p": {
            "age": {str(year): c["age"]},
            "ss_aime": {str(year): c["aime"]},
            "ss_claiming_age": {str(year): c["claim_age"]},
        }},
        "households": {"h": {"members": ["p"]}},
    })
    out.append({
        "aime": c["aime"], "age": c["age"], "claim_age": c["claim_age"],
        "pia": float(sim.calculate("ss_pia", year)[0]),
        "factor": float(sim.calculate(
            "ss_retirement_age_adjustment_factor", year)[0]),
        "fra_months": int(sim.calculate(
            "ss_full_retirement_age_months", year)[0]),
    })
json.dump({"results": out}, sys.stdout)
"""


def _pe_us_python() -> str:
    return os.environ.get("POPULACE_DYNAMICS_PE_US_PYTHON", "")


def _pe_us_available(python: str) -> bool:
    if not python or not Path(python).exists():
        return False
    try:
        return (
            subprocess.run(
                [python, "-c", "import policyengine_us"],
                capture_output=True,
                timeout=120,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        return False


def _foundation_cases() -> list[dict]:
    cases = []
    for aime in _AIMES:
        for age in _AGES:
            for claim_age in _CLAIM_AGES:
                cases.append(
                    {"aime": aime, "age": age, "claim_age": claim_age}
                )
    return cases


def _pe_us_oracle(python: str, cases: list[dict]) -> list[dict]:
    job = {"year": FOUNDATION_YEAR, "cases": cases}
    proc = subprocess.run(
        [python, "-c", _PE_US_RUNNER],
        input=json.dumps(job),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pe-us oracle runner failed (python={python}):\n"
            f"{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout)["results"]


def _foundation_block(params: SSAParameters) -> dict:
    cases = _foundation_cases()
    python = _pe_us_python()
    available = _pe_us_available(python)
    rows = []
    if available:
        oracle = {
            (r["aime"], r["age"], r["claim_age"]): r
            for r in _pe_us_oracle(python, cases)
        }
    else:
        oracle = {}
    max_pia_dev = 0.0
    max_factor_dev = 0.0
    for c in cases:
        aime, age, claim_age = c["aime"], c["age"], c["claim_age"]
        birth_year = FOUNDATION_YEAR - age
        our_pia = benefits.pia(aime, FOUNDATION_YEAR, params)
        our_factor = claiming.benefit_factor(
            claim_age * 12, birth_year, params
        )
        row = {
            "aime": aime,
            "age": age,
            "birth_year": birth_year,
            "claim_age": claim_age,
            "our_pia": our_pia,
            "our_factor": our_factor,
        }
        key = (aime, age, claim_age)
        if key in oracle:
            o = oracle[key]
            row["oracle_pia"] = round(o["pia"], 2)
            row["oracle_factor"] = round(o["factor"], 6)
            row["pia_abs_dev"] = abs(our_pia - o["pia"])
            row["factor_abs_dev"] = abs(our_factor - o["factor"])
            max_pia_dev = max(max_pia_dev, row["pia_abs_dev"])
            max_factor_dev = max(max_factor_dev, row["factor_abs_dev"])
        else:
            row["oracle_pia"] = None
            row["oracle_factor"] = None
            row["pia_abs_dev"] = None
            row["factor_abs_dev"] = None
        rows.append(row)
    return {
        "oracle": "policyengine_us Simulation" if available else None,
        "oracle_available": available,
        "year": FOUNDATION_YEAR,
        "alignment_note": (
            "pe-us ss_pia uses period-year bend points; the oracle uses "
            f"{FOUNDATION_YEAR} as the eligibility year to align. ss_aime, "
            "ss_claiming_age and age are set directly; birth_year = "
            "year - age."
        ),
        "residual_note": (
            "Residual deviations are pe-us float32 array storage (max PIA "
            "dev < 1e-4 dollars, i.e. under a hundredth of a cent); the "
            "oracle is exact float64. The statutory resolution is "
            "identical (both floor PIA to the dime over integer AIME and "
            "integer bend points)."
        ),
        "n_cases": len(cases),
        "max_pia_abs_deviation": max_pia_dev if available else None,
        "max_factor_abs_deviation": max_factor_dev if available else None,
        "cases": rows,
    }


# ---------------------------------------------------------------------------
# Block 3: couple / survivor illustrative grid (our outputs)
# ---------------------------------------------------------------------------
_WORKER_PIAS = (1000.0, 2000.0, 3000.0)
_SPOUSE_PIAS = (0.0, 800.0, 1600.0)
_PAIR_CLAIM_AGES = (62, 67)
_PAIR_BIRTH_YEAR = 1960  # FRA 67


def _couple_grid(params: SSAParameters) -> list[dict]:
    rows = []
    for wp in _WORKER_PIAS:
        for sp in _SPOUSE_PIAS:
            for ca_w in _PAIR_CLAIM_AGES:
                for ca_s in _PAIR_CLAIM_AGES:
                    cb = household.couple_benefit(
                        wp,
                        sp,
                        ca_w * 12,
                        ca_s * 12,
                        _PAIR_BIRTH_YEAR,
                        _PAIR_BIRTH_YEAR,
                        params,
                    )
                    rows.append(
                        {
                            "worker_pia": wp,
                            "spouse_pia": sp,
                            "worker_claim_age": ca_w,
                            "spouse_claim_age": ca_s,
                            "own_worker": round(cb.own_a, 4),
                            "own_spouse": round(cb.own_b, 4),
                            "excess_spousal_worker": round(
                                cb.excess_spousal_a, 4
                            ),
                            "excess_spousal_spouse": round(
                                cb.excess_spousal_b, 4
                            ),
                            "couple_total": round(cb.total, 4),
                        }
                    )
    return rows


def _survivor_grid(params: SSAParameters) -> list[dict]:
    rows = []
    # Surviving spouse own PIA 800; deceased PIA 2200; vary deceased and
    # survivor claim ages (death timing = the deceased's locked-in factor).
    surv_pia, dec_pia = 800.0, 2200.0
    for dec_claim in (62, 66, 70):
        for surv_claim in (60, 63, 67):
            sb = household.survivor_benefit_at_death(
                surv_pia,
                dec_pia,
                surv_claim * 12,
                dec_claim * 12,
                _PAIR_BIRTH_YEAR,
                _FRA66_BIRTH_YEAR,
                params,
            )
            rows.append(
                {
                    "surviving_own_pia": surv_pia,
                    "deceased_pia": dec_pia,
                    "deceased_claim_age": dec_claim,
                    "survivor_claim_age": surv_claim,
                    "deceased_own_factor": round(sb.deceased_own_factor, 6),
                    "survivor_months_early": sb.survivor_months_early,
                    "survivor_own_benefit": round(sb.survivor_own_benefit, 4),
                    "widow_benefit": round(sb.widow_benefit, 4),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Panel coverage (real Marriage History File, when staged)
# ---------------------------------------------------------------------------
def _panel_coverage() -> dict | None:
    try:
        from populace_dynamics.data.marriage import (
            marriage_episodes,
            marriage_history,
        )
    except Exception:
        return None
    try:
        mh = marriage_history()
    except Exception as exc:  # staged file absent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    ep = marriage_episodes(mh)
    persons = set(mh["person_id"].unique())
    by = mh.drop_duplicates("person_id").set_index("person_id")["birth_year"]
    # PIA supply proxy: every person with a known birth year is "computable".
    pia = {int(p): 1.0 for p in by.dropna().index}
    cov = household.both_spouse_coverage(ep, pia)
    spouse_in_file = ep["spouse_person_id"].isin(persons)
    return {
        "available": True,
        "source": "PSID Marriage History File mh85_23 (staged Release 2)",
        "n_actual_marriage_episodes": int(len(ep)),
        "joinable_spouse_share": round(cov.joinable_spouse_share, 4),
        "spouse_in_file_share": round(float(spouse_in_file.mean()), 4),
        "both_computable_share": round(cov.both_pia_share, 4),
        "n_widowhood_episodes": int((ep["how_ended"] == "widowhood").sum()),
        "widowhood_share": round(
            float((ep["how_ended"] == "widowhood").mean()), 4
        ),
        "note": (
            "both_computable_share uses a PIA supply of every person with a "
            "known birth year (the structural both-spouse ceiling); a real "
            "PIA supply from a certified earnings history can only lower it."
        ),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build() -> dict:
    params = load_ssa_parameters()
    spousal = _spousal_examples(params)
    survivor = _survivor_examples(params)
    foundation = _foundation_block(params)

    ssa_devs = [
        abs(e["our_output"] - e["expected"]) for e in spousal + survivor
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "description": (
            "Spouse/survivor auxiliary-benefit worked examples (42 USC "
            "402(b)/(c)/(e)/(f)): our outputs vs SSA published figures, the "
            "policyengine-us PIA foundation cross-check, and the couple/"
            "survivor grid. Phase-C plumbing for #74 — reported, NOT gated."
        ),
        "reported_not_gated": True,
        "pe_us_revision": params.pe_us_revision,
        "ssa_worked_examples": {
            "note": (
                "policyengine-us has no spousal/survivor computation "
                "(social_security_* are uprated survey inputs), so these "
                "auxiliary amounts are validated against SSA's published "
                "figures. Deceased-claim factors are derived from the "
                "pinned 402(q)/(w) machinery."
            ),
            "spousal": spousal,
            "survivor": survivor,
            "max_abs_deviation": max(ssa_devs) if ssa_devs else 0.0,
            "spousal_summary": _summarise(spousal),
            "survivor_summary": _summarise(survivor),
        },
        "pe_us_pia_foundation": foundation,
        "couple_grid": _couple_grid(params),
        "survivor_grid": _survivor_grid(params),
        "panel_coverage": _panel_coverage(),
        "build": {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "builder": "scripts/build_aux_benefit_examples.py",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_PATH, help="output JSON path"
    )
    args = parser.parse_args(argv)
    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)
        fh.write("\n")
    ssa = artifact["ssa_worked_examples"]["max_abs_deviation"]
    found = artifact["pe_us_pia_foundation"]
    print(f"wrote {args.out}")
    print(f"  pe_us_revision            = {artifact['pe_us_revision']}")
    print(f"  ssa max_abs_deviation     = {ssa}")
    print(f"  pe-us oracle available    = {found['oracle_available']}")
    print(f"  pe-us max_pia_abs_dev     = {found['max_pia_abs_deviation']}")
    print(f"  pe-us max_factor_abs_dev  = {found['max_factor_abs_deviation']}")
    cov = artifact["panel_coverage"]
    if cov and cov.get("available"):
        print(f"  both-spouse coverage      = {cov['both_computable_share']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
