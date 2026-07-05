"""Cross-engine PIA agreement artifact.

Proves the Axiom rules engine (Rust, lifetime execution) and the
``populace_dynamics.ss`` Python oracle compute *identical* Social Security
primary insurance amounts (PIAs) from raw nominal earnings through the full
statutory chain of 42 USC 415.

Both sides are driven from the SAME parameters, which the oracle loads from
policyengine-us (:func:`populace_dynamics.ss.params.load_ssa_parameters`). The
engine-side RuleSpec module is generated *programmatically* from those loaded
parameters here — no parameter value is ever hand-typed — so the only thing the
artifact tests is engine-versus-oracle statutory resolution, not a re-keyed
copy of the numbers.

What the engine module exercises (every one is required by the pre-declared
merge trigger for axiom-rules-engine#68 and rulespec-us#541):

* an over-periods reduction (``sum_top_n_over_periods`` — 415(b) highest-n
  selection over the worker's own period axis);
* a DERIVED person-varying ``n`` (the benefit-computation-year count, built
  from per-person-constant inputs) feeding that reduction;
* per-period parameters resolved INSIDE the reduction (the wage base and NAWI,
  each a step function, applied to the earnings of the period being reduced);
* invariant-bound per-person inputs (year attained 21 / 62 / 60 and the NAWI at
  the indexing year), supplied identically in every period and bound outside or
  inside the reduction;
* period-indexed bend points resolved at the LAST supplied period — the
  eligibility (age-62) year — outside the reduction;
* the 415(g) round-down-to-the-next-lower-dime floor.

Semantic-fidelity decisions (see module docstring notes in the generated YAML
and the artifact JSON ``notes`` field):

* The oracle's :func:`~populace_dynamics.ss.benefits.aime` hard-codes 35
  computation years and divides by ``35 * 12 = 420``. For the engine's DERIVED
  ``n`` to equal that 35, the elapsed-years count must be
  ``year_attained_62 - max(1950, year_attained_21) - 1`` (= 40 for both
  cohorts), the statutory 415(b)(2)(B) count of years *after* attaining 21 (or
  1950) and *before* attaining 62 — 40 elapsed years, minus 5 dropout years,
  yields 35. Both cohorts here (born 1958 and 1964) attain 21 well after 1950
  and so have exactly 40 elapsed / 35 computation years, matching the oracle's
  constant exactly.
* The oracle's dime floor is ``floor(amount * 10 + 1e-9) / 10``; the engine uses
  ``floor(pia_raw * 10) / 10`` with no epsilon. The epsilon is inert here: AIME
  is always an integer (the oracle floors it) and the bend points are integer
  dollars, so ``amount * 10`` never lands in the sub-1e-9 zone just below an
  integer. This is verified by an exhaustive integer-AIME sweep in
  ``tests/ss/test_cross_engine.py`` and confirmed end to end by exact agreement.

Run (from the populace-dynamics worktree)::

    PYTHONPATH=src POPULACE_DYNAMICS_PE_US_DIR=~/PolicyEngine/policyengine-us-main \\
        <ss-model venv>/bin/python scripts/build_cross_engine_pia_artifact.py

The engine extension is imported from a separate maturin-built venv; its path
is configured by ``AXIOM_ENGINE_PYTHON`` (default: the axiom-engine-67 worktree
venv). The script shells out to that interpreter to run the engine side, so a
single Python process need not import both stacks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters, load_ssa_parameters

# ---------------------------------------------------------------------------
# Cohorts and worker design
# ---------------------------------------------------------------------------

#: eligibility (age-62) year -> birth year. Both cohorts attain 21 after 1950,
#: so both have exactly 40 elapsed / 35 computation years (see module docstring).
COHORTS: dict[int, int] = {2020: 1958, 2026: 1964}

#: Workers per cohort (the brief asks for ~120 each).
WORKERS_PER_COHORT = 120

#: Deterministic seed for the pseudo-random career shapes.
SEED = 20200626

_INDEXING_AGE = 60
_FIRST_EARNING_AGE = 22
_ELIGIBILITY_AGE = 62


def _cohort_years(birth_year: int) -> list[int]:
    """Calendar years ages 22..62 inclusive (41 years; last = eligibility)."""
    return list(
        range(
            birth_year + _FIRST_EARNING_AGE, birth_year + _ELIGIBILITY_AGE + 1
        )
    )


@dataclass(frozen=True)
class Worker:
    """One synthetic career."""

    worker_id: str
    cohort: int  # eligibility year
    birth_year: int
    shape: str
    history: dict[int, float]  # calendar year -> nominal earnings


class _Lcg:
    """A tiny deterministic PRNG (numpy-free, so the oracle side needs no
    numpy). Linear congruential (glibc constants); good enough for spreading
    career shapes reproducibly."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * (self._next() / 0x7FFFFFFF)

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive integer in [lo, hi]."""
        return lo + self._next() % (hi - lo + 1)


def _flat_level_for_aime(
    target_aime: float, birth_year: int, params: SSAParameters
) -> float:
    """Flat nominal earnings level whose oracle AIME is nearest ``target_aime``.

    Bisects the oracle's monotone (flat-level → AIME) map. Used only to place
    the bend-point straddlers in each PIA bracket; the exact AIME that results
    is whatever the oracle and engine agree on.
    """
    years = _cohort_years(birth_year)

    def flat_aime(level: float) -> float:
        return benefits.aime(
            {y: float(level) for y in years}, birth_year, params
        )

    lo, hi = 0.0, 1.0e6
    # Grow hi until it brackets the target (AIME is capped by the wage base, so
    # a very high target may be unreachable; the loop then returns the max).
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if flat_aime(mid) < target_aime:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 2)


def _build_workers(
    cohort: int, birth_year: int, params: SSAParameters
) -> list[Worker]:
    """~120 workers for one cohort, spanning the required career shapes.

    Deterministic given ``SEED`` and the cohort. Shapes: flat low/median/high,
    rising, declining, spiky, zero-earning spells, wage-base-clipped in many
    years, bend-point crossers, and the min/max edges (all zeros; astronomical).
    """
    years = _cohort_years(birth_year)
    n_years = len(years)
    y0 = years[0]
    wb = {y: params.wage_base_for(y) for y in years}
    rng = _Lcg(SEED ^ (cohort * 2654435761))
    workers: list[Worker] = []

    def add(shape: str, history: dict[int, float]) -> None:
        idx = len(workers)
        workers.append(
            Worker(
                worker_id=f"{cohort}-{idx:03d}",
                cohort=cohort,
                birth_year=birth_year,
                shape=shape,
                history={y: float(v) for y, v in history.items()},
            )
        )

    # --- Deterministic edge cases (always present, exact) --------------------
    add("all_zeros", {y: 0.0 for y in years})  # min edge: AIME 0, PIA 0
    add(
        "astronomical", {y: 1.0e12 for y in years}
    )  # max edge: clipped to wage base every year
    add("flat_low", {y: 18_000.0 for y in years})
    add("flat_median", {y: 52_000.0 for y in years})
    add("flat_high", {y: 250_000.0 for y in years})  # clipped most years
    # Rising and declining linear careers.
    add(
        "rising",
        {y: 12_000.0 + 3_000.0 * (y - y0) for y in years},
    )
    add(
        "declining",
        {y: max(0.0, 90_000.0 - 1_800.0 * (y - y0)) for y in years},
    )
    # Spiky: alternating high spikes and low troughs.
    add(
        "spiky",
        {y: (180_000.0 if (y - y0) % 3 == 0 else 6_000.0) for y in years},
    )
    # Zero-earning spells: a 7-year gap mid-career.
    gap = set(range(y0 + 15, y0 + 22))
    add(
        "zero_spell",
        {y: (0.0 if y in gap else 44_000.0) for y in years},
    )
    # Wage-base-clipped in essentially every year (well above the base).
    add(
        "clipped_always",
        {y: wb[y] * 5.0 for y in years},
    )
    # Careers landing in each of the three 90/32/15 PIA brackets: pick a target
    # AIME below the first bend (90%-only), between the bends (into the 32%
    # bracket), and above the second bend (into the 15% bracket), then invert
    # the oracle to the flat nominal earnings level that yields it. Inverting
    # against the oracle (rather than a heuristic) guarantees each bracket is
    # actually exercised, including the low-earnings 90%-only bracket.
    first_bend, second_bend = params.bend_points(cohort)
    for label, target_aime in (
        ("bend_below_first", first_bend * 0.6),
        ("bend_between", 0.5 * (first_bend + second_bend)),
        ("bend_above_second", second_bend + 800.0),
    ):
        level = _flat_level_for_aime(target_aime, birth_year, params)
        add(label, {y: level for y in years})

    # --- Pseudo-random careers filling out to WORKERS_PER_COHORT -------------
    families = [
        "flat_low",
        "flat_median",
        "flat_high",
        "rising",
        "declining",
        "spiky",
        "zero_spell",
        "clipped_always",
    ]
    while len(workers) < WORKERS_PER_COHORT:
        fam = families[rng.randint(0, len(families) - 1)]
        base = rng.uniform(8_000.0, 220_000.0)
        slope = rng.uniform(-2_500.0, 4_000.0)
        history: dict[int, float] = {}
        # Optional random zero spell.
        spell_start = rng.randint(0, n_years - 1)
        spell_len = rng.randint(0, 6) if rng.uniform() < 0.35 else 0
        spell = set(range(spell_start, min(spell_start + spell_len, n_years)))
        for k, y in enumerate(years):
            if k in spell:
                history[y] = 0.0
                continue
            if fam == "flat_low":
                v = base * 0.35
            elif fam == "flat_median":
                v = base
            elif fam == "flat_high":
                v = base * 2.0
            elif fam == "rising":
                v = max(0.0, base + slope * k)
            elif fam == "declining":
                v = max(0.0, base + abs(slope) * (n_years - k))
            elif fam == "spiky":
                v = base * (3.0 if k % 3 == 0 else 0.2)
            elif fam == "zero_spell":
                v = base
            else:  # clipped_always
                v = wb[y] * rng.uniform(1.5, 6.0)
            # A little idiosyncratic noise, deterministic.
            v *= rng.uniform(0.9, 1.1)
            history[y] = round(v, 2)
        add(f"rand_{fam}", history)

    return workers[:WORKERS_PER_COHORT]


# ---------------------------------------------------------------------------
# Oracle side
# ---------------------------------------------------------------------------


def _oracle_results(
    worker: Worker, params: SSAParameters
) -> tuple[int, float]:
    """(aime, pia) from the populace_dynamics oracle for one worker."""
    aime_value = benefits.aime(worker.history, worker.birth_year, params)
    pia_value = benefits.pia(aime_value, worker.cohort, params)
    return aime_value, pia_value


# ---------------------------------------------------------------------------
# Engine-side RuleSpec module — generated from the loaded parameters
# ---------------------------------------------------------------------------


def _f64(x: float) -> str:
    """Shortest decimal string that round-trips to the same IEEE-754 double.

    Python's ``repr`` is round-trip-exact for floats, and the engine parses the
    formula literal into an f64, so ``repr`` guarantees the engine binds the
    identical bit pattern the oracle used. (Confirmed empirically by zero AIME
    difference across all workers.)
    """
    return repr(float(x))


def _versions_by_year(
    value_of: Callable[[int], float], years: list[int]
) -> str:
    """A ``versions:`` block, one step per year (a step function resolved per
    period inside a reduction and at the last supplied period outside it)."""
    lines = []
    for y in years:
        lines.append(f"      - effective_from: '{y}-01-01'")
        lines.append(f"        formula: '{_f64(value_of(y))}'")
    return "\n".join(lines)


def _const_param(name: str, dtype: str, formula: str) -> str:
    return (
        f"  - name: {name}\n"
        f"    kind: parameter\n"
        f"    dtype: {dtype}\n"
        f"    versions:\n"
        f"      - effective_from: '1979-01-01'\n"
        f"        formula: '{formula}'"
    )


def build_engine_module(params: SSAParameters) -> str:
    """Emit the RuleSpec/v1 module text from the oracle's loaded parameters.

    NEVER hand-types a parameter value: NAWI, the wage base, and the bend
    points all come from ``params`` (which read policyengine-us); the bend
    points derive inside the loader per 415(a)(1)(B) from NAWI, so they remain
    pe-us-sourced.
    """
    # Union of calendar years spanned by ages 22..62 across both cohorts.
    year_lo = min(_cohort_years(b)[0] for b in COHORTS.values())
    year_hi = max(_cohort_years(b)[-1] for b in COHORTS.values())
    all_years = list(range(year_lo, year_hi + 1))
    elig_years = sorted(COHORTS.keys())

    nawi_v = _versions_by_year(lambda y: params.nawi[y], all_years)
    wb_v = _versions_by_year(lambda y: params.wage_base_for(y), all_years)
    bp1_v = _versions_by_year(lambda y: params.bend_points(y)[0], elig_years)
    bp2_v = _versions_by_year(lambda y: params.bend_points(y)[1], elig_years)
    f1, f2, f3 = params.pia_factors

    # The indexed, wage-base-capped earnings of the period being reduced.
    # Years at/after the indexing year (birth+60) enter nominal; earlier years
    # scale by NAWI(indexing year) / NAWI(that year). wage_base and nawi resolve
    # PER PERIOD; year_attained_60 and nawi_at_indexing_year are invariant.
    indexed_expr = (
        "if calendar_year >= year_attained_60: "
        "min(raw_earnings, wage_base) "
        "else: "
        "min(raw_earnings, wage_base) * nawi_at_indexing_year / nawi"
    )
    pia_raw_expr = (
        "pia_factor_first * min(aime, bend_point_first) "
        "+ pia_factor_second "
        "* max(0, min(aime, bend_point_second) - bend_point_first) "
        "+ pia_factor_third * max(0, aime - bend_point_second)"
    )

    return f"""\
format: rulespec/v1
module:
  summary: |-
    Cross-engine PIA agreement fixture (42 USC 415). Generated programmatically
    from policyengine-us parameters by
    scripts/build_cross_engine_pia_artifact.py in
    PolicyEngine/populace-dynamics; every value is oracle-sourced. Pre-declared
    merge trigger for TheAxiomFoundation/axiom-rules-engine#68 and
    rulespec-us#541. Exercises: sum_top_n_over_periods with a derived
    person-varying n; wage base and NAWI as per-period step functions inside the
    reduction; invariant-bound per-person inputs; bend points resolved at the
    last supplied period; and the 415(g) dime floor. n is the 415(b)(2)(B)
    computation-year count (elapsed = year62 - max(1950, year21) - 1; both
    cohorts here have 40 elapsed / 35 computation years, matching the oracle's
    hard-coded 35). The dime floor omits the oracle's inert 1e-9 epsilon.
rules:
  - name: nawi
    kind: parameter
    dtype: Money
    versions:
{nawi_v}
  - name: wage_base
    kind: parameter
    dtype: Money
    versions:
{wb_v}
  - name: bend_point_first
    kind: parameter
    dtype: Money
    versions:
{bp1_v}
  - name: bend_point_second
    kind: parameter
    dtype: Money
    versions:
{bp2_v}
{_const_param("pia_factor_first", "Rate", _f64(f1))}
{_const_param("pia_factor_second", "Rate", _f64(f2))}
{_const_param("pia_factor_third", "Rate", _f64(f3))}
{_const_param("months_per_year", "Integer", "12")}
{_const_param("dropout_years", "Integer", "5")}
{_const_param("minimum_computation_years", "Integer", "2")}
  - name: elapsed_years
    kind: derived
    entity: Worker
    dtype: Integer
    period: Year
    source: 42 USC 415(b)(2)(B)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          year_attained_62 - max(1950, year_attained_21) - 1
  - name: computation_year_count
    kind: derived
    entity: Worker
    dtype: Integer
    period: Year
    source: 42 USC 415(b)(2)(A)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          max(minimum_computation_years, elapsed_years - dropout_years)
  - name: earnings_total
    kind: derived
    entity: Worker
    dtype: Money
    period: Year
    source: 42 USC 415(b)(1)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          sum_top_n_over_periods({indexed_expr}, computation_year_count)
  - name: aime
    kind: derived
    entity: Worker
    dtype: Money
    period: Year
    source: 42 USC 415(b)(1)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          floor(earnings_total / (months_per_year * computation_year_count))
  - name: pia_raw
    kind: derived
    entity: Worker
    dtype: Money
    period: Year
    source: 42 USC 415(a)(1)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          {pia_raw_expr}
  - name: pia
    kind: derived
    entity: Worker
    dtype: Money
    period: Year
    source: 42 USC 415(a)(1), 415(g)
    versions:
      - effective_from: '1979-01-01'
        formula: |-
          floor(pia_raw * 10) / 10
"""


# ---------------------------------------------------------------------------
# Engine-side runner — executed in the axiom engine venv via a subprocess
# ---------------------------------------------------------------------------

#: Script text run by the engine interpreter. Reads a JSON job on stdin
#: ({module, cohorts:[{elig, birth, years, workers:[{id, history:{year:earn}}]}]})
#: and writes {results:{id:{aime,pia}}} on stdout. Kept as a string so the
#: parent process (in the oracle venv) never imports the engine.
_ENGINE_RUNNER = r"""
import json, os, sys, tempfile
import numpy as np
from axiom_rules_engine import CompiledDenseProgram

job = json.load(sys.stdin)
path = os.path.join(tempfile.mkdtemp(), "pia.yaml")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(job["module"])
prog = CompiledDenseProgram.from_file(path, entity="Worker")

results = {}
for cohort in job["cohorts"]:
    years = cohort["years"]
    birth = cohort["birth"]
    y21 = birth + 21
    y62 = birth + 62
    y60 = birth + 60
    nawi_iy = cohort["nawi_at_indexing_year"]
    workers = cohort["workers"]
    n = len(workers)
    if n == 0:
        continue
    periods = [("calendar_year", f"{y}-01-01", f"{y}-12-31") for y in years]
    batches = []
    for y in years:
        ys = str(y)
        batches.append({
            "raw_earnings": np.array(
                [float(w["history"].get(ys, 0.0)) for w in workers]
            ),
            "calendar_year": np.array([float(y)] * n),
            "year_attained_21": np.array([float(y21)] * n),
            "year_attained_62": np.array([float(y62)] * n),
            "year_attained_60": np.array([float(y60)] * n),
            "nawi_at_indexing_year": np.array([float(nawi_iy)] * n),
        })
    res = prog.execute_lifetime_f64(
        periods=periods, batches=batches, outputs=["aime", "pia"]
    )
    aime_out = res["outputs"]["aime"]
    pia_out = res["outputs"]["pia"]
    for i, w in enumerate(workers):
        results[w["id"]] = {
            "aime": float(aime_out[i]),
            "pia": float(pia_out[i]),
        }

json.dump({"results": results}, sys.stdout)
"""


def _default_engine_python() -> str:
    return os.environ.get(
        "AXIOM_ENGINE_PYTHON",
        str(
            Path(
                "~/.claude-worktrees/axiom-engine-67/.venv/bin/python"
            ).expanduser()
        ),
    )


def _engine_results(
    module_text: str,
    workers_by_cohort: dict[int, list[Worker]],
    params: SSAParameters,
    engine_python: str,
) -> dict[str, dict[str, float]]:
    """Run the engine side in its own interpreter; return {id:{aime,pia}}."""
    cohorts_payload = []
    for cohort, workers in workers_by_cohort.items():
        birth = COHORTS[cohort]
        years = _cohort_years(birth)
        cohorts_payload.append(
            {
                "elig": cohort,
                "birth": birth,
                "years": years,
                "nawi_at_indexing_year": params.nawi[birth + _INDEXING_AGE],
                "workers": [
                    {
                        "id": w.worker_id,
                        "history": {str(y): v for y, v in w.history.items()},
                    }
                    for w in workers
                ],
            }
        )
    job = {"module": module_text, "cohorts": cohorts_payload}
    proc = subprocess.run(
        [engine_python, "-c", _ENGINE_RUNNER],
        input=json.dumps(job),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "engine runner failed "
            f"(python={engine_python}):\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout)["results"]


def _engine_revision(engine_python: str) -> str:
    """The axiom-rules-engine git short SHA (from the worktree of the venv).

    Walks up from the interpreter path WITHOUT resolving symlinks — the venv
    ``bin/python`` is a symlink into a shared uv Python install, so resolving it
    would leave the engine worktree entirely. ``git -C`` on the venv's parent
    directories finds the enclosing worktree (``<worktree>/.venv/bin/python`` →
    worktree root two levels above ``bin``).
    """
    candidates = list(Path(engine_python).parents)
    for root in candidates:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "log", "-1", "--format=%h"],
                capture_output=True,
                text=True,
                check=True,
            )
            sha = out.stdout.strip()
            if sha:
                return sha
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return "unknown"


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------

#: A worker whose PIA differs by more than this (dollars) is a real semantic
#: divergence to diagnose, not float noise.
AGREEMENT_TOLERANCE_DOLLARS = 0.005


def build_artifact(params: SSAParameters, engine_python: str) -> dict:
    """Run both engines over all workers and assemble the artifact dict."""
    workers_by_cohort = {
        cohort: _build_workers(cohort, birth, params)
        for cohort, birth in COHORTS.items()
    }
    module_text = build_engine_module(params)
    engine_out = _engine_results(
        module_text, workers_by_cohort, params, engine_python
    )

    rows = []
    max_abs_diff = 0.0
    n_exact = 0
    per_cohort: dict[str, dict[str, float]] = {}

    for cohort, workers in workers_by_cohort.items():
        oracle_pias = []
        engine_pias = []
        for w in workers:
            aime_o, pia_o = _oracle_results(w, params)
            eng = engine_out[w.worker_id]
            aime_e = eng["aime"]
            pia_e = eng["pia"]
            diff = abs(pia_e - pia_o)
            max_abs_diff = max(max_abs_diff, diff)
            # Exact-to-cent: round both to cents and compare.
            if round(pia_o, 2) == round(pia_e, 2):
                n_exact += 1
            oracle_pias.append(pia_o)
            engine_pias.append(pia_e)
            rows.append(
                {
                    "id": w.worker_id,
                    "cohort": cohort,
                    "shape": w.shape,
                    "aime_oracle": aime_o,
                    "aime_engine": aime_e,
                    "pia_oracle": round(pia_o, 2),
                    "pia_engine": round(pia_e, 2),
                    "abs_diff": round(diff, 6),
                }
            )
        per_cohort[str(cohort)] = {
            "n": len(workers),
            "mean_pia_oracle": round(sum(oracle_pias) / len(oracle_pias), 4),
            "mean_pia_engine": round(sum(engine_pias) / len(engine_pias), 4),
        }

    n_workers = sum(len(w) for w in workers_by_cohort.values())
    return {
        "run": "pia_cross_engine_v1",
        "description": (
            "Cross-engine PIA agreement: the Axiom rules engine (Rust, lifetime "
            "execution) versus the populace_dynamics.ss Python oracle, over "
            "synthetic earnings histories driven through the full 42 USC 415 "
            "chain. Every parameter is oracle-sourced (policyengine-us); the "
            "engine RuleSpec module is generated programmatically. Pre-declared "
            "merge trigger for axiom-rules-engine#68 and rulespec-us#541."
        ),
        "n_workers": n_workers,
        "cohorts": {
            str(c): {"eligibility_year": c, "birth_year": b}
            for c, b in COHORTS.items()
        },
        "worker_design": (
            f"{WORKERS_PER_COHORT} workers per cohort (eligibility "
            f"{sorted(COHORTS)}), deterministic seed {SEED}. Shapes: all-zeros "
            "and astronomical edges; flat low/median/high; rising; declining; "
            "spiky; zero-earning spells; wage-base-clipped in many years; three "
            "bend-point straddlers (below first, between, above second); and "
            "pseudo-random careers drawn from those families with optional zero "
            "spells. Each worker's history is nominal earnings for ages 22-62 "
            "(41 calendar years, final year = the age-62 eligibility year, so "
            "the eligibility-year bend points resolve at the last supplied "
            "period)."
        ),
        "engine_revision": _engine_revision(engine_python),
        "pe_us_revision": params.pe_us_revision,
        "seed": SEED,
        "agreement_tolerance_dollars": AGREEMENT_TOLERANCE_DOLLARS,
        "max_abs_diff_dollars": round(max_abs_diff, 6),
        "n_exact_to_cent": n_exact,
        "exercises": [
            "sum_top_n_over_periods (over-periods reduction, 415(b))",
            "derived person-varying n (computation-year count) feeding the "
            "reduction",
            "per-period parameters inside the reduction (wage base, NAWI as "
            "step functions)",
            "invariant-bound per-person inputs (year attained 21/62/60, NAWI at "
            "indexing year)",
            "period-indexed bend points resolved at the last supplied period",
            "415(g) round-down-to-next-lower-dime floor",
        ],
        "notes": (
            "n = 415(b)(2)(B) computation-year count; elapsed = year62 - "
            "max(1950, year21) - 1 (= 40 for both cohorts) minus 5 dropout "
            "years = 35, matching the oracle's hard-coded 35. The engine's "
            "dime floor omits the oracle's 1e-9 epsilon; verified inert for "
            "integer AIME x integer bend points by an exhaustive sweep in "
            "tests/ss/test_cross_engine.py."
        ),
        "per_cohort": per_cohort,
        "workers": rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "runs" / ("pia_cross_engine_v1.json")
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="artifact JSON path (default: runs/pia_cross_engine_v1.json)",
    )
    parser.add_argument(
        "--engine-python",
        default=_default_engine_python(),
        help="python interpreter with axiom_rules_engine built "
        "(default: $AXIOM_ENGINE_PYTHON or the axiom-engine-67 venv)",
    )
    parser.add_argument(
        "--pe-us-dir",
        type=Path,
        default=None,
        help="policyengine-us checkout (default: "
        "$POPULACE_DYNAMICS_PE_US_DIR)",
    )
    args = parser.parse_args(argv)

    params = load_ssa_parameters(args.pe_us_dir)
    artifact = build_artifact(params, str(args.engine_python))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(
        f"Wrote {args.output}\n"
        f"  n_workers            = {artifact['n_workers']}\n"
        f"  engine_revision      = {artifact['engine_revision']}\n"
        f"  pe_us_revision       = {artifact['pe_us_revision']}\n"
        f"  max_abs_diff_dollars = {artifact['max_abs_diff_dollars']}\n"
        f"  n_exact_to_cent      = {artifact['n_exact_to_cent']}"
        f" / {artifact['n_workers']}"
    )
    if artifact["max_abs_diff_dollars"] > AGREEMENT_TOLERANCE_DOLLARS:
        worst = max(artifact["workers"], key=lambda r: r["abs_diff"])
        print(
            "  WARNING: max abs diff exceeds tolerance "
            f"{AGREEMENT_TOLERANCE_DOLLARS}; worst worker: {worst}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
