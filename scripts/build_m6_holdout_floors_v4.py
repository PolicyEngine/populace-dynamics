#!/usr/bin/env python3
"""Build the registered closed-domain M6 floors-v4 truth artifact.

Resolution B of the ratified candidate-2 program changes one floor input: the
six already-gated earnings cells are repriced on the population the harness
actually scores, ``realized support intersect the 2014 earnings domain``.
Everything else is carried from the frozen v3 artifact after its SHA-256 is
verified: the five flow floors, the 11-cell surface, reducers, split units,
metric caps, seeds, and 4-of-5 conjunction.

The domain is derived through the registered <=2014 forward-earnings refit.
The script never constructs a projection, reads a candidate run, or consumes a
candidate score. For every floor seed 0..99 it splits the FULL holdout anchor
first and intersects each half with the fitted domain inside the unchanged
``earnings_cells`` reducer (the section-2.8.3a/F7 order).

Run only in the dedicated fitting environment documented by
``scripts/run_gate_m6_candidate1.py``. The default output and its environment
sidecar are exclusive-create one-shot files. ``--out`` exists only so a referee
can reproduce the primary artifact at a fresh path and compare its SHA-256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import populace.fit.model as populace_fit_model
import populace.fit.qrf as populace_qrf
import populace.frame as populace_frame
import registered_m6_inputs

from populace_dynamics import artifacts, evaluation
from populace_dynamics.engine import earnings_domain, forward_earnings, refit
from populace_dynamics.engine.earnings_domain import (
    earnings_domain_person_ids,
)
from populace_dynamics.engine.refit import refit_earnings_chained_generator
from populace_dynamics.harness import (
    m6_cells,
    m6_inputs,
    m6_runner,
    m6_scoring,
    moments,
)
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.harness.m6_cells import (
    FLOOR_SEEDS,
    METRIC_CAP,
    MIN_EVENTS_FOR_GATE,
    MIN_GATED_CELLS_FOR_POWER,
    SPLIT_FRACTION,
    WEAK_POWER_P_GATE_FLOOR,
    _tol,
    earnings_cells,
    oc_4of5,
    run_floor,
)
from populace_dynamics.harness.m6_runner import resolve_m6_contract
from populace_dynamics.harness.m6_scoring import EARNINGS_CELL_NAMES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m6_holdout_floors_v4.json"
SCHEMA_VERSION = "m6_holdout_floors.v4"
RUN = "m6_holdout_floors_v4"

PROGRAM_COMMIT = "051b4494ecce9345da14d68488bb2833ed476d22"
VERIFICATION_COMMENT = "5001901052"
V3_PATH = ROOT / "runs" / "m6_holdout_floors_v3.json"
V3_SHA256 = "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77"
S2_DISCLOSURE_PATH = (
    "docs/amendments/gate_m6_amendment_1_closed_domain_floors.md"
)
EXPECTED_DOMAIN_PERSONS = 13_561
EXPECTED_DOMAIN_EARNINGS_ROWS = 45_606
EXPECTED_FULL_GATED_EARNINGS_PERSONS = 13_163
EXPECTED_LATER_ENTRANT_GATED_PERSONS = 2_722
EXPECTED_LATER_ENTRANT_BY_COHORT = {"older": 590, "prime": 2_199}
POPULACE_SOURCE_COMMIT = "af02c917fcb3c50816bf3af9c2b64509e928889a"
POPULACE_REPO = Path(populace_qrf.__file__).resolve().parents[5]

_CODE_PATHS = (
    Path(__file__).resolve(),
    Path(registered_m6_inputs.__file__).resolve(),
    Path(artifacts.__file__).resolve(),
    Path(evaluation.__file__).resolve(),
    Path(earnings_domain.__file__).resolve(),
    Path(forward_earnings.__file__).resolve(),
    Path(refit.__file__).resolve(),
    Path(m6_cells.__file__).resolve(),
    Path(m6_inputs.__file__).resolve(),
    Path(m6_runner.__file__).resolve(),
    Path(m6_scoring.__file__).resolve(),
    Path(moments.__file__).resolve(),
    Path(hpanel.__file__).resolve(),
)

_POPULACE_CODE_PATHS = (
    Path(populace_qrf.__file__).resolve(),
    Path(populace_fit_model.__file__).resolve(),
    Path(populace_frame.__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lines_sha256(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_clean_tracked_source() -> None:
    """Bind the derivation to committed code while ignoring new output files."""
    for command in (
        ("git", "diff", "--quiet"),
        ("git", "diff", "--cached", "--quiet"),
    ):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "floors-v4 requires a clean tracked source tree; commit the "
                "builder before the one-shot derivation"
            )
    untracked_source = subprocess.run(
        (
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "scripts",
            "src",
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if untracked_source:
        raise RuntimeError(
            "floors-v4 source directories contain untracked files: "
            f"{untracked_source}"
        )
    for path in _CODE_PATHS:
        relative = path.relative_to(ROOT).as_posix()
        tracked = subprocess.run(
            ("git", "ls-files", "--error-unmatch", relative),
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"derivation source is not tracked: {relative}")
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != _sha256(path):
            raise RuntimeError(
                f"derivation source is not byte-identical to HEAD: {relative}"
            )


def _code_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in _CODE_PATHS:
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"derivation code resolved outside this worktree: {path}"
            ) from error
        manifest[relative] = _sha256(path)
    return manifest


def _populace_source_manifest() -> dict[str, Any]:
    """Bind the two editable fitting packages to one clean Populace commit."""
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=POPULACE_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != POPULACE_SOURCE_COMMIT:
        raise RuntimeError(
            "Populace fitting-stack commit drifted: "
            f"{commit} != {POPULACE_SOURCE_COMMIT}"
        )
    package_status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "packages/populace-fit",
            "packages/populace-frame",
        ),
        cwd=POPULACE_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if package_status:
        raise RuntimeError("Populace fitting-stack package sources are dirty")
    files: dict[str, str] = {}
    for path in _POPULACE_CODE_PATHS:
        relative = path.relative_to(POPULACE_REPO).as_posix()
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative}"),
            cwd=POPULACE_REPO,
            check=True,
            capture_output=True,
        ).stdout
        digest = _sha256(path)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(
                "Populace fitting source is not byte-identical to HEAD: "
                f"{relative}"
            )
        files[relative] = digest
    return {
        "repository": "PolicyEngine/populace",
        "commit": commit,
        "package_sources_clean": True,
        "files_sha256": files,
    }


def _support_by_period(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for period, rows in frame.groupby("period", sort=True):
        out[str(int(period))] = {
            "n_rows": int(len(rows)),
            "n_persons": int(rows.person_id.nunique()),
        }
    return out


def _domain_support(
    anchor: pd.DataFrame,
    earnings: pd.DataFrame,
    domain: frozenset[int],
) -> dict[str, Any]:
    domain_earnings = earnings[earnings.person_id.isin(domain)].copy()
    gated = earnings[earnings.period.isin((2016, 2018))].copy()
    gated_domain = gated[gated.person_id.isin(domain)].copy()
    outside = gated[~gated.person_id.isin(domain)].copy()
    domain_lines = [str(int(value)) for value in domain]
    key_lines = [
        f"{int(row.person_id)}|{int(row.period)}"
        for row in domain_earnings[["person_id", "period"]].itertuples(
            index=False
        )
    ]
    outside_by_cohort = {
        cohort: int(
            outside.loc[outside.cohort == cohort, "person_id"].nunique()
        )
        for cohort in sorted(outside.cohort.dropna().unique())
    }
    return {
        "definition": (
            "truth anchor person ids intersect fitted realized-2014 "
            "earnings keys intersect fitted u_w keys"
        ),
        "scored_support": (
            "realized positive-weight valid-earnings support at 2016/2018 "
            "intersected symmetrically with the 2014 domain"
        ),
        "split_order": (
            "split the full anchor by person, then intersect each half with "
            "the domain inside the reducer"
        ),
        "n_full_anchor_persons": int(anchor.person_id.nunique()),
        "n_domain_persons": int(len(domain)),
        "n_domain_earnings_rows": int(len(domain_earnings)),
        "n_full_gated_earnings_persons": int(gated.person_id.nunique()),
        "n_domain_gated_earnings_persons": int(
            gated_domain.person_id.nunique()
        ),
        "n_later_entrant_gated_persons": int(outside.person_id.nunique()),
        "later_entrant_persons_by_cohort": outside_by_cohort,
        "domain_person_ids_sha256": _lines_sha256(domain_lines),
        "domain_earnings_person_period_keys_sha256": _lines_sha256(key_lines),
        "domain_earnings_by_period": _support_by_period(domain_earnings),
    }


def _all_seed_disclosure(
    detail: list[dict[str, Any]],
    anchor: pd.DataFrame,
    domain: frozenset[int],
    domain_earnings: pd.DataFrame,
) -> list[dict[str, Any]]:
    by_seed = {int(row["seed"]): row for row in detail}
    disclosure: list[dict[str, Any]] = []
    persons = anchor[["person_id", "household_id"]].drop_duplicates(
        "person_id"
    )
    for seed in FLOOR_SEEDS:
        left, right = hpanel.split_panel_by_person(
            persons,
            "person_id",
            fraction=SPLIT_FRACTION,
            seed=seed,
        )
        ids_a = set(int(value) for value in left.person_id) & domain
        ids_b = set(int(value) for value in right.person_id) & domain
        cells = {
            cell: by_seed[seed]["cells"][cell] for cell in EARNINGS_CELL_NAMES
        }
        disclosure.append(
            {
                "seed": seed,
                "full_anchor_persons_a": int(len(left)),
                "full_anchor_persons_b": int(len(right)),
                "domain_persons_a": int(len(ids_a)),
                "domain_persons_b": int(len(ids_b)),
                "domain_earnings_rows_a": int(
                    domain_earnings.person_id.isin(ids_a).sum()
                ),
                "domain_earnings_rows_b": int(
                    domain_earnings.person_id.isin(ids_b).sum()
                ),
                "cells": cells,
            }
        )
    return disclosure


def _derive_domain_earnings_floor(
    inputs: Any,
    domain: frozenset[int],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    anchor = inputs.truth.anchor
    domain_earnings = inputs.truth.earnings[
        inputs.truth.earnings.person_id.isin(domain)
    ].copy()

    def compute(full_half_ids: set[object]) -> dict[str, Any]:
        domain_half = set(full_half_ids) & domain
        return earnings_cells(
            domain_earnings[domain_earnings.person_id.isin(domain_half)]
        )

    floor, detail = run_floor(
        anchor,
        compute,
        "person_id",
        retained_seeds=FLOOR_SEEDS,
    )
    selected = {cell: floor[cell] for cell in EARNINGS_CELL_NAMES}
    if any(
        record["n_defined_seeds"] != len(FLOOR_SEEDS)
        for record in selected.values()
    ):
        raise RuntimeError("a closed-domain earnings floor cell is undefined")
    return selected, detail


def _plain_refit_provenance(earnings_refit: Any) -> dict[str, Any]:
    return {
        "seed": int(earnings_refit.seed),
        "spec_registration": earnings_refit.spec_registration,
        "adapter_spec_sha256": earnings_refit.adapter_spec_sha256,
        "provenance": asdict(earnings_refit.provenance),
        "n_realized_2014_keys": int(
            len(earnings_refit.generator.realized_earn_2014_by_person)
        ),
        "n_u_w_keys": int(len(earnings_refit.generator.u_w_by_person)),
    }


def _stable_input_provenance(inputs: Any) -> dict[str, Any]:
    """Remove a site-packages Git-walk artifact from SSA provenance.

    ``load_ssa_parameters`` asks Git for the revision at its parameter root.
    For an installed distribution that root is below this worktree, so Git
    incorrectly reports the enclosing worktree HEAD. The factory separately
    proves metadata version 1.752.2 and binds the parameter directory to that
    distribution; record that stable fact instead of the unrelated Git hash.
    """
    provenance = inputs.provenance.to_artifact()
    provenance["external_details"][
        "ssa_revision"
    ] = "installed_distribution_no_source_git_revision"
    provenance["external_details"][
        "ssa_distribution"
    ] = "policyengine-us==1.752.2"
    provenance["external_details"][
        "ssa_parameter_dir_binding"
    ] = "metadata_versioned_distribution_asserted"
    return provenance


def build_v4() -> dict[str, Any]:
    resolved = resolve_m6_contract(ROOT)
    if resolved.floor_path != V3_PATH.relative_to(ROOT).as_posix():
        raise RuntimeError("the live contract does not point to frozen v3")
    if resolved.floor_sha256 != V3_SHA256 or _sha256(V3_PATH) != V3_SHA256:
        raise RuntimeError("the frozen v3 floor SHA-256 does not match")
    v3 = copy.deepcopy(dict(resolved.floor_artifact))
    populace_source = _populace_source_manifest()

    inputs = registered_m6_inputs.build_inputs()
    earnings_refit = refit_earnings_chained_generator(
        inputs.refit_inputs.earnings_panel,
        inputs.refit_inputs.ssa_params.nawi,
        seed=inputs.refit_inputs.earnings_seed,
    )
    provenance = earnings_refit.provenance
    if provenance.boundary_year != 2014:
        raise RuntimeError("earnings domain refit boundary is not 2014")
    if provenance.max_year["earnings_reference_year"] > 2014:
        raise RuntimeError("post-2014 earnings entered the domain refit")
    if (
        provenance.certified_full_window_artifacts_read
        or provenance.certified_full_window_artifacts_written
    ):
        raise RuntimeError("the domain refit touched a full-window artifact")
    fitted_domain = earnings_domain_person_ids(earnings_refit.generator)
    anchor_ids = frozenset(
        int(value) for value in inputs.truth.anchor.person_id
    )
    domain = frozenset(int(value) for value in fitted_domain) & anchor_ids
    support = _domain_support(
        inputs.truth.anchor, inputs.truth.earnings, domain
    )
    if support["n_domain_persons"] != EXPECTED_DOMAIN_PERSONS:
        raise RuntimeError(
            "closed-domain person count drifted from the registered finding: "
            f"{support['n_domain_persons']} != {EXPECTED_DOMAIN_PERSONS}"
        )
    if support["n_domain_earnings_rows"] != EXPECTED_DOMAIN_EARNINGS_ROWS:
        raise RuntimeError(
            "closed-domain earnings-row count drifted from the registered "
            f"finding: {support['n_domain_earnings_rows']} != "
            f"{EXPECTED_DOMAIN_EARNINGS_ROWS}"
        )
    if (
        support["n_full_gated_earnings_persons"]
        != EXPECTED_FULL_GATED_EARNINGS_PERSONS
    ):
        raise RuntimeError("full gated-earnings support count drifted")
    if (
        support["n_later_entrant_gated_persons"]
        != EXPECTED_LATER_ENTRANT_GATED_PERSONS
    ):
        raise RuntimeError("later-entrant gated support count drifted")
    if (
        support["later_entrant_persons_by_cohort"]
        != EXPECTED_LATER_ENTRANT_BY_COHORT
    ):
        raise RuntimeError("later-entrant cohort support counts drifted")

    earnings_floor, detail = _derive_domain_earnings_floor(inputs, domain)
    rules = resolved.contract.by_cell
    earnings_tolerances: dict[str, float] = {}
    raw_tolerances: dict[str, float] = {}
    at_cap: list[str] = []
    for cell in EARNINGS_CELL_NAMES:
        rule = rules[cell]
        raw = _tol(
            earnings_floor[cell]["mean"],
            earnings_floor[cell]["sd"],
            rule.k,
        )
        tolerance = min(raw, METRIC_CAP[rule.metric])
        raw_tolerances[cell] = raw
        earnings_tolerances[cell] = tolerance
        if raw >= METRIC_CAP[rule.metric] - 1e-12:
            at_cap.append(cell)

    # Re-run the registered report-only audit and require exact equality with
    # the ceremony calculation. This binds the new artifact to the harness's
    # existing F7 order without using any projection or candidate output.
    audit = m6_scoring.recompute_domain_earnings_floor(
        inputs.truth.anchor,
        inputs.truth.earnings,
        domain,
        resolved.contract,
        frozen_floor_artifact=v3,
    )
    for cell in EARNINGS_CELL_NAMES:
        if audit["per_cell"][cell]["domain_floor"] != earnings_floor[cell]:
            raise RuntimeError(f"domain floor audit disagrees for {cell}")
        if (
            audit["per_cell"][cell]["domain_capped_tolerance"]
            != earnings_tolerances[cell]
        ):
            raise RuntimeError(f"domain tolerance audit disagrees for {cell}")

    gated = list(v3["partition"]["gated"])
    flow_cells = [cell for cell in gated if cell not in EARNINGS_CELL_NAMES]
    floor_all = copy.deepcopy(v3["floor"]["cells"])
    floor_all.update(earnings_floor)
    gated_floor = {cell: floor_all[cell] for cell in gated}
    tolerances = {
        cell: (
            earnings_tolerances[cell]
            if cell in earnings_tolerances
            else v3["tolerances"][cell]
        )
        for cell in gated
    }
    combined_oc = oc_4of5(gated_floor, tolerances, gated)
    flow_oc = oc_4of5(
        {cell: gated_floor[cell] for cell in flow_cells},
        {cell: tolerances[cell] for cell in flow_cells},
        flow_cells,
    )
    earnings_oc = oc_4of5(
        earnings_floor,
        earnings_tolerances,
        list(EARNINGS_CELL_NAMES),
    )
    if flow_oc != v3["faithful_candidate_oc"]["family_a_flows"]:
        raise RuntimeError("the carried v3 flow OC changed")
    locked_domain_combined_oc = oc_4of5(
        gated_floor,
        {cell: v3["tolerances"][cell] for cell in gated},
        gated,
    )
    oc_comparison = {
        "ratified_v3_full_support": v3["faithful_candidate_oc"]["combined"],
        "v3_tolerances_on_closed_domain_earnings": audit["oc"][
            "locked_tolerances_on_domain"
        ],
        "v3_tolerances_on_closed_domain_combined": (locked_domain_combined_oc),
        "v4_closed_domain_earnings": earnings_oc,
        "v4_closed_domain_combined": combined_oc,
        "p_gate_delta_locked_domain_vs_v3_full_support": (
            locked_domain_combined_oc["p_gate_pass_4_of_5"]
            - v3["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"]
        ),
        "p_gate_delta_v4_vs_v3_full_support": (
            combined_oc["p_gate_pass_4_of_5"]
            - v3["faithful_candidate_oc"]["combined"]["p_gate_pass_4_of_5"]
        ),
        "program_provisional_rounded_subfamily_arithmetic": {
            "operative": False,
            "p_gate": 0.9019,
            "note": (
                "The fresh ceremony uses exact individual-cell OC; it does "
                "not multiply rounded subfamily p_seed values."
            ),
        },
    }

    near_unfailable = [
        cell
        for cell, record in earnings_oc["per_cell"].items()
        if record["cell_pass_prob"] == 1.0
    ]
    near_tautological = earnings_oc["p_gate_pass_4_of_5"] == 1.0 and bool(
        near_unfailable
    )
    vacuity = bool(at_cap) or near_tautological
    near_unpassable = (
        combined_oc["p_gate_pass_4_of_5"] < WEAK_POWER_P_GATE_FLOOR
    )
    clears_flow_surface = len(flow_cells) >= MIN_GATED_CELLS_FOR_POWER
    below_min_support = [
        cell
        for cell in EARNINGS_CELL_NAMES
        if earnings_floor[cell]["min_events_weaker_half"] < MIN_EVENTS_FOR_GATE
    ]
    all_at_cap = [
        cell
        for cell in gated
        if cell in at_cap
        or abs(tolerances[cell] - METRIC_CAP[rules[cell].metric]) <= 1e-12
    ]
    ceremony_may_proceed = bool(
        not near_unpassable
        and not vacuity
        and clears_flow_surface
        and not below_min_support
    )

    domain_earnings = inputs.truth.earnings[
        inputs.truth.earnings.person_id.isin(domain)
    ].copy()
    seed_disclosure = _all_seed_disclosure(
        detail,
        inputs.truth.anchor,
        domain,
        domain_earnings,
    )
    per_cell = {
        cell: {
            "metric": rules[cell].metric,
            "k": rules[cell].k,
            "rounding": rules[cell].rounding,
            "v3_tolerance": v3["tolerances"][cell],
            "v4_raw_tolerance": raw_tolerances[cell],
            "v4_tolerance": earnings_tolerances[cell],
            "tolerance_delta": (
                earnings_tolerances[cell] - v3["tolerances"][cell]
            ),
            "metric_cap": METRIC_CAP[rules[cell].metric],
            "at_metric_cap": cell in at_cap,
            "floor": earnings_floor[cell],
        }
        for cell in EARNINGS_CELL_NAMES
    }
    core = {
        "domain_support": support,
        "earnings_per_cell": per_cell,
        "floor_seed_disclosure": seed_disclosure,
        "floor_cells": floor_all,
        "tolerances": tolerances,
        "faithful_candidate_oc": {
            "combined": combined_oc,
            "family_a_flows": flow_oc,
            "earnings_subfamily": earnings_oc,
        },
        "oc_comparison": oc_comparison,
        "two_directional_power_check": {
            "weak_power_floor": WEAK_POWER_P_GATE_FLOOR,
            "near_unpassable": near_unpassable,
            "domain_tolerances_at_metric_cap": at_cap,
            "near_unfailable_cells": near_unfailable,
            "near_tautological_oc": near_tautological,
            "vacuity": vacuity,
            "clears_flow_surface_power": clears_flow_surface,
            "minimum_weaker_half_support": MIN_EVENTS_FOR_GATE,
            "cells_below_minimum_support": below_min_support,
            "clears_minimum_support": not below_min_support,
            "ceremony_may_proceed": ceremony_may_proceed,
        },
    }

    artifact = v3
    artifact.pop("elapsed_seconds", None)
    artifact["coarsening_ladder"]["v4_disposition"] = (
        "Historical v3 selection record; not rerun by v4. Every flow floor "
        "and partition byte is carried from SHA-verified v3."
    )
    artifact["earnings_decompounding_ladder"]["v4_disposition"] = (
        "Historical v3 surface-selection record; not rerun or repruned by "
        "v4. Only the six retained earnings cells are repriced."
    )
    artifact.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run": RUN,
            "reported_truth_side_only": True,
            "no_projection_no_candidate": True,
            "candidate_blind": (
                "The numerical derivation reads only frozen v3, the locked "
                "protocol, staged truth, and the <=2014 fitted domain. It "
                "reads no candidate artifact, projection, or candidate score."
            ),
            "supersedes_note": (
                "v4 prospectively reprices only the six retained earnings "
                "floors on realized support intersected with the fitted 2014 "
                "domain. Frozen v3 remains immutable and historical; its five "
                "flow floors and every non-floor protocol byte are carried."
            ),
            "lineage": {
                "program_commit": PROGRAM_COMMIT,
                "verification_comment": VERIFICATION_COMMENT,
                "v3_artifact": V3_PATH.relative_to(ROOT).as_posix(),
                "v3_sha256": V3_SHA256,
                "v3_read_only": True,
                "v3_floor_method": v3["floor"]["method"],
            },
            "governance": {
                "operative": False,
                "status": "DERIVED_NOT_OPERATIVE_UNTIL_PROSPECTIVE_LOCK",
                "gates_yaml_edited_by_builder": False,
                "v3_historical_contract_unchanged": True,
                "requires_before_registration_8": [
                    "adversarial byte reproduction",
                    "narrow prospective section-2.8.4 amendment",
                    "ratifying gate-contract lock to the verified v4 SHA-256",
                ],
            },
            "input_boundary": {
                "candidate_artifacts_read": False,
                "projection_outputs_read": False,
                "post_2014_values_used_for_domain_fit": False,
                "numerical_inputs": [
                    "frozen v3 flow floor and locked protocol",
                    "registered staged PSID truth tables",
                    "registered <=2014 forward-earnings fit state",
                ],
                "s2_movement_disclosure_used_in_derivation": False,
                "s2_movement_disclosure_location": (S2_DISCLOSURE_PATH),
            },
            "derivation_code_sha256": _code_manifest(),
            "fitting_stack_source": populace_source,
            "fitting_environment": {
                package: importlib.metadata.version(package)
                for package in (
                    "populace-fit",
                    "populace-frame",
                    "policyengine-us",
                    "scikit-learn",
                    "numpy",
                    "pandas",
                    "scipy",
                )
            },
            "input_provenance": _stable_input_provenance(inputs),
            "input_provenance_normalization": (
                "The installed policyengine-us parameter directory has no "
                "source Git repository. The loader's enclosing-worktree Git "
                "walk is discarded; version and parameter-directory identity "
                "remain factory-asserted."
            ),
            "domain_refit": _plain_refit_provenance(earnings_refit),
            "domain_support": support,
            "closed_domain_earnings": {
                "per_cell": per_cell,
                "all_100_floor_seeds": seed_disclosure,
                "registered_self_check": audit,
            },
            "floor": {
                "method": (
                    "Five flow floor records are byte-carried from frozen v3. "
                    "For the six retained earnings cells, split the FULL "
                    "holdout anchor 50/50 by person at seeds 0..99, intersect "
                    "each half with the fitted 2014 earnings domain inside "
                    "the unchanged earnings_cells reducer, then derive "
                    "round(mean + 3*sample_sd, 3) with registered caps."
                ),
                "cells": floor_all,
            },
            "tolerances": tolerances,
            "faithful_candidate_oc": {
                "combined": combined_oc,
                "family_a_flows": flow_oc,
                "earnings_subfamily": earnings_oc,
                "p_seed_pass": combined_oc["p_seed_pass"],
                "p_gate_pass_4_of_5": combined_oc["p_gate_pass_4_of_5"],
            },
            "oc_comparison": oc_comparison,
            "two_directional_power_check": core["two_directional_power_check"],
            "oc_before_lock": {
                "weak_power_p_gate_floor": WEAK_POWER_P_GATE_FLOOR,
                "min_gated_flow_cells": MIN_GATED_CELLS_FOR_POWER,
                "n_gated_cells": len(gated),
                "n_gated_flow_cells": len(flow_cells),
                "n_gated_earnings_cells": len(EARNINGS_CELL_NAMES),
                "n_gated_tolerances_at_cap": len(all_at_cap),
                "n_domain_earnings_tolerances_at_cap": len(at_cap),
                "p_gate_combined": combined_oc["p_gate_pass_4_of_5"],
                "p_gate_flows": flow_oc["p_gate_pass_4_of_5"],
                "p_gate_earnings": earnings_oc["p_gate_pass_4_of_5"],
                "clears_lower_bound_not_unpassable": not near_unpassable,
                "clears_flow_surface_power": clears_flow_surface,
                "clears_minimum_support": not below_min_support,
                "cells_below_minimum_support": below_min_support,
                "clears_not_all_tolerances_capped": len(all_at_cap)
                < len(gated),
                "clears_no_domain_tolerance_at_cap": not at_cap,
                "clears_weak_power_threshold": ceremony_may_proceed,
                "ceremony_may_proceed": ceremony_may_proceed,
                "outcome": (
                    "surface clears; ceremony may proceed to prospective lock"
                    if ceremony_may_proceed
                    else "surface does not clear; do not lock"
                ),
            },
            "derivation_core_sha256": _json_sha256(core),
            "candidate_blindness_proof": {
                "derivation_core_sha256": _json_sha256(core),
                "candidate_artifact_or_projection_read": False,
                "candidate_score_used_in_floor_tolerance_or_oc": False,
                "pre_registered_trigger": (
                    "Section 2.8.3a registered the exact-domain truth-only "
                    "floor trigger before this ceremony."
                ),
                "separation": (
                    "The S2 movement table is governance disclosure in the "
                    "separate amendment, not data or code in this numerical "
                    "builder. It cannot change a floor, tolerance, or OC."
                ),
                "resolution_b_s2_disclosure": {
                    "used_in_derivation": False,
                    "location": S2_DISCLOSURE_PATH,
                    "authority": (
                        "docs/design/m6_candidate2_program.md section 6.1; "
                        "verification comment 5001901052"
                    ),
                    "retrospective_v4_application_prohibited": True,
                    "historical_candidate_1_contract": "immutable v3",
                },
                "persistence_margin_conclusion": (
                    "The still-wide mobility and autocorrelation failures show "
                    "that floor movement cannot manufacture a persistence pass."
                ),
            },
            "revision_pins": {
                "artifact_schema_version": SCHEMA_VERSION,
                "v3_sha256": V3_SHA256,
                "floor_seed_range": [FLOOR_SEEDS[0], FLOOR_SEEDS[-1]],
                "full_anchor_split_before_domain_intersection": True,
            },
        }
    )
    return artifact


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_PATH,
        help=(
            "Fresh exclusive output path; defaults to the registered v4 path."
        ),
    )
    return command


def main() -> int:
    args = parser().parse_args()
    _assert_clean_tracked_source()
    artifact = build_v4()
    primary = (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    artifacts.write_new(args.out, primary, sidecar=True)
    print(f"wrote {args.out}")
    print(f"sha256 {_sha256(args.out)}")
    print("tolerances " + json.dumps(artifact["tolerances"], sort_keys=True))
    oc = artifact["faithful_candidate_oc"]
    print(
        "OC combined "
        f"p_seed={oc['combined']['p_seed_pass']:.4f} "
        f"p_gate={oc['combined']['p_gate_pass_4_of_5']:.4f}; "
        "earnings "
        f"p_seed={oc['earnings_subfamily']['p_seed_pass']:.4f} "
        f"p_gate={oc['earnings_subfamily']['p_gate_pass_4_of_5']:.4f}"
    )
    if not artifact["oc_before_lock"]["ceremony_may_proceed"]:
        print("PAUSE: the two-directional power check blocks a v4 lock")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
