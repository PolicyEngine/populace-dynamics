"""Gate-1 candidate 5a': segment splicing.

The SIXTH pre-registered model run of PolicyEngine/populace-dynamics and
the second non-generative candidate, a refinement of the single-donor
splice (candidate 5a, pull request 50) toward MINT's actual practice. The
candidate-5a' spec is registered, frozen before the run, in issue #42's
candidate-5a' comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4892604375);
every rule -- segmentation, boundary, match, fallback cascade, scaling --
is pinned there and implemented LITERALLY. No threshold is hardcoded, no
rule is tuned against holdout scores, and the run is one shot. The
outcome publishes whether it passes or fails.

Why segment splicing (from the candidate-5a' registration). The
single-donor run failed on three named artifacts; this variant addresses
each by construction. Segments drawn from multiple donors mix careers
(against whole-career over-persistence); splices are indexed by CALENDAR
PERIOD rather than by age (against the offset-age-grid signature); and
level adjustment happens PER SEGMENT at its boundary (against a single
global anchor-noise conversion). It is otherwise the field-standard
donor-splice benchmark.

The candidate, per the frozen spec (each rule implemented literally). Per
seed (locked protocol, paired splits, seeds 0-4), for each holdout
person, walking BACKWARD from the anchor (chronologically last observed
period, kept at its REAL value):

1. **Segmentation.** The person's observed periods before the anchor are
   grouped, in backward order, into consecutive segments of up to 3
   observed periods each (the last segment may be shorter).
2. **Boundary.** Each segment's boundary is the person's next-later
   observed period -- already filled by a previous (later) segment, or
   the anchor itself for the first segment. The boundary value ``b`` is
   the target's value there (real at the anchor, spliced otherwise), so
   the segments form a backward chain: each scales to the value the
   later segment already produced at the shared boundary.
3. **Donor match (deterministic).** Candidate donors for a segment are
   the seed's train persons observed at ALL of the segment's calendar
   periods AND at the boundary's calendar period, with age at the
   boundary period within +/-2 years of the target's age there. Among
   candidates, select the donor minimizing ``|donor earnings at the
   boundary period - b|``; ties broken by smaller absolute boundary-age
   difference, then smaller ``person_id``.
4. **Fallback cascade.** If no candidate exists, widen the age window in
   +/-2-year steps (+/-4, +/-6, ...) up to +/-10; if still empty, shorten
   the segment by one period (dropping its EARLIEST period, which
   re-groups into the following, earlier segment, and preserving the
   boundary) and retry from the original +/-2 window; a one-period
   segment with a same-period boundary requirement always has candidates
   in practice (rates reported).
5. **Splice and scale.** The segment's generated values are the donor's
   earnings at the segment's own calendar periods, multiplied by
   ``b / (donor's earnings at the boundary period)`` clipped to
   ``[0.2, 5]`` when both are positive; if either is zero or negative the
   segment copies unscaled. Donor zeros copy as zeros.

Determinism. No RNG is needed: the run is fully deterministic given the
seed's split. The gate seed enters ONLY through ``split_panel_by_person``.
Because no model is fit, this candidate needs NO populace-fit and no
fitting of any kind; it runs under the repo's ``.venv``. The protocol
machinery -- the filter-first load, the person-disjoint 0.2 split per
seed, the two locked views, ``panel_scorecard`` scoring, the battery vs
the committed ``battery_reference`` with locked definitions, the
thresholds read from ``gates.yaml`` at runtime, the seed-level
conjunction (>=4/5 both blocks), and the battery-reference bit-exact
precheck -- is IMPORTED from the merged baseline runner
(:mod:`run_gate1_baseline`, pull request 40), byte-for-byte the prior
runs'. Only the generation (segmentation + period-indexed splicing) is
local.

Run from the repository root with the PSID family files staged, using
the repo ``.venv`` (no populace-fit needed for this deterministic
candidate):

    .venv/bin/python scripts/run_gate1_candidate5a2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol machinery is IMPORTED from the merged baseline runner so
# that the filtered-panel load, the person-disjoint split, the view
# construction, the battery definitions, the geometry / battery checks,
# the threshold loading, and the battery-reference reproduction are
# byte-for-byte identical to every prior gate-1 run. Only the generation
# (deterministic segmentation + period-indexed splicing) is local. The
# baseline module defers its populace-fit import to its fit path, so this
# import succeeds under the repo ``.venv``; this candidate fits no model
# and never triggers that path.
from run_gate1_baseline import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    SEEDS,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    load_filtered_panel,
    load_gate1_thresholds,
    reproduce_battery_reference,
    split_holdout_train,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_splice_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate1_splice.v2"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4892604375"
)

#: The frozen level-adjustment clip bounds (reported, applied literally).
SCALE_CLIP_LO, SCALE_CLIP_HI = 0.2, 5.0
#: The frozen segmentation cap and age-window cascade bounds.
SEGMENT_MAX = 3
AGE_WINDOW_INIT = 2.0
AGE_WINDOW_MAX = 10.0


# --------------------------------------------------------------------------
# Anchors (chronologically last observed period per person)
# --------------------------------------------------------------------------
def anchor_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per person: their chronologically LAST observed period.

    Anchor = the person's maximum ``period`` in the filtered panel, with
    that row's ``earnings``, ``age``, ``weight``. This is the SAME anchor
    definition every prior candidate uses, stated once here as a
    person-level table. Deterministic (periods are unique per
    person-period, so the ``idxmax`` is unique).
    """
    idx = panel.groupby("person_id")["period"].idxmax()
    cols = ["person_id", "period", "earnings", "age", "weight"]
    return panel.loc[idx, cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Donor index (per-period person->row lookups for the period-indexed match)
# --------------------------------------------------------------------------
def build_donor_index(
    train: pd.DataFrame,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[int, int]]]:
    """Per-calendar-period donor arrays and person->position maps.

    Returns ``(by_period, pos)`` where

    * ``by_period[p]`` holds parallel ``pid`` / ``earn`` / ``age`` arrays
      for the train persons observed at period ``p``, sorted ascending by
      ``person_id`` (so the deterministic tie-break by ``person_id`` reads
      off a stable order), and
    * ``pos[p][person_id]`` is that person's row position within
      ``by_period[p]`` -- an O(1) "observed at ``p``?" test and an O(1)
      earnings/age lookup.

    Ages and earnings are unique within a person-period, so the maps are
    well defined. The splice needs exact-period lookups (unlike the
    age-indexed 5a splice); this index provides them.
    """
    donor = train[["person_id", "period", "earnings", "age"]]
    by_period: dict[int, dict[str, np.ndarray]] = {}
    pos: dict[int, dict[int, int]] = {}
    for p, g in donor.groupby("period"):
        gg = g.sort_values("person_id")
        pid = gg["person_id"].to_numpy()
        by_period[int(p)] = {
            "pid": pid,
            "earn": gg["earnings"].to_numpy(dtype=np.float64),
            "age": gg["age"].to_numpy(dtype=np.float64),
        }
        pos[int(p)] = {int(v): i for i, v in enumerate(pid)}
    return by_period, pos


def match_donor(
    seg_periods: list[int],
    boundary_period: int,
    boundary_age: float,
    boundary_value: float,
    by_period: dict[int, dict[str, np.ndarray]],
    pos: dict[int, dict[int, int]],
) -> tuple[int | None, int]:
    """Select the matched donor person for one segment (deterministic).

    Candidate donors are train persons observed at EVERY period in
    ``seg_periods`` AND at ``boundary_period``, with boundary-period age
    within an age window of ``boundary_age``. The window starts at +/-2
    and widens in +/-2 steps up to +/-10. Among the non-empty cell, the
    donor with the smallest ``|donor boundary-period earnings -
    boundary_value|`` wins; ties break by smaller ``|donor boundary age -
    boundary_age|``, then smaller ``person_id``.

    Returns ``(donor_person_id, widen_count)``. ``donor_person_id`` is
    ``None`` iff even the +/-10 window at the boundary period holds no
    donor observed at all the segment periods (the caller then shortens
    the segment). ``widen_count`` is the number of +/-2 widenings applied
    beyond the initial +/-2 window (0 if the initial window sufficed).
    """
    if boundary_period not in by_period:
        return None, 0
    b = by_period[boundary_period]
    cand_pid = b["pid"]
    keep = np.ones(len(cand_pid), dtype=bool)
    for p in seg_periods:
        pmap = pos.get(p)
        if not pmap:
            return None, 0
        keep &= np.fromiter(
            (int(pid) in pmap for pid in cand_pid),
            dtype=bool,
            count=len(cand_pid),
        )
        if not keep.any():
            return None, 0
    idx = np.nonzero(keep)[0]
    c_pid = cand_pid[idx]
    c_bage = b["age"][idx]
    c_bearn = b["earn"][idx]

    half = AGE_WINDOW_INIT
    widen = 0
    while True:
        mask = np.abs(c_bage - boundary_age) <= half
        if mask.any():
            break
        if half >= AGE_WINDOW_MAX:
            return None, widen
        half += AGE_WINDOW_INIT
        widen += 1

    s_pid = c_pid[mask]
    s_bage = c_bage[mask]
    s_bearn = c_bearn[mask]
    d_earn = np.abs(s_bearn - boundary_value)
    d_age = np.abs(s_bage - boundary_age)
    # lexsort: last key primary. Primary = |earnings diff|, then
    # |boundary-age diff|, then person_id (all ascending).
    order = np.lexsort((s_pid, d_age, d_earn))
    return int(s_pid[order[0]]), widen


# --------------------------------------------------------------------------
# Generation (segmentation + period-indexed splice + per-segment scaling)
# --------------------------------------------------------------------------
def generate_candidate(
    holdout: pd.DataFrame,
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Segment-spliced candidate panel over the holdout persons.

    For each holdout person: keep the anchor (chronologically last
    observed period) at its REAL earnings, group the earlier observed
    periods backward into segments of up to 3, and splice each segment
    from a period-matched donor scaled to the segment's boundary value
    (the frozen 5a' rules). Segments are processed from the anchor
    backward, so a segment's boundary value is the already-generated
    value of the later segment at their shared boundary period.

    Returns ``(candidate, diagnostics)`` where ``candidate`` holds exactly
    the holdout persons on exactly their observed periods (the locked
    candidate-panel pin: only ``earnings`` is generated;
    ``person_id`` / ``period`` / ``age`` / ``weight`` copy from the
    holdout), and ``diagnostics`` carries the reported-not-gated
    distributions (segment count / length, age-window widening, segment
    shortening, scaling-clip, donor reuse, boundary match error).
    """
    holdout_ids = holdout["person_id"].unique()
    holdout_anchor = (
        all_anchor[all_anchor.person_id.isin(holdout_ids)]
        .sort_values("person_id")
        .reset_index(drop=True)
    )

    by_period, pos = build_donor_index(train)

    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    hp_pid = hp["person_id"].to_numpy()
    hp_period = hp["period"].to_numpy()
    hp_age = hp["age"].to_numpy(dtype=np.float64)

    row_pos: dict[tuple[int, int], int] = {}
    periods_of: dict[int, list[int]] = {}
    for i in range(len(hp)):
        row_pos[(int(hp_pid[i]), int(hp_period[i]))] = i
        periods_of.setdefault(int(hp_pid[i]), []).append(int(hp_period[i]))

    # Diagnostics accumulators.
    seg_counts: list[int] = []
    seg_lengths: list[int] = []
    n_widened_segments = 0
    widen_steps_total = 0
    max_widen_steps = 0
    shorten_events = 0
    n_segments = 0
    n_segments_scaled = 0
    n_segments_clipped = 0
    n_segments_unscaled = 0
    n_spliced_observations = 0
    match_errors: list[float] = []
    scale_ratios: list[float] = []
    n_unmatched_one_period_segments = 0
    donors_of_person: dict[int, list[int]] = {}
    all_donor_ids: set[int] = set()

    # Iterate persons in ``person_id`` order (the anchor table is sorted)
    # for a deterministic, reproducible pass. Each person is spliced
    # independently, so order is immaterial to the result -- but pinned.
    for row in holdout_anchor.itertuples():
        pid = int(row.person_id)
        obs = sorted(periods_of[pid])  # ascending observed periods
        pre = obs[:-1]  # everything before the anchor (obs[-1])
        donors_of_person[pid] = []
        if not pre:
            seg_counts.append(0)
            continue

        remaining = list(pre)  # ascending; segments taken from the top
        n_person_segments = 0
        while remaining:
            seg_len = min(SEGMENT_MAX, len(remaining))
            donor_pid: int | None = None
            seg: list[int] = []
            boundary_period = 0
            boundary_value = 0.0
            # Shorten-on-failure loop: try the latest ``seg_len`` remaining
            # periods; if unmatched even at +/-10, drop to ``seg_len - 1``
            # (the earliest of these periods re-groups into the following,
            # earlier segment) and retry from the +/-2 window.
            while True:
                seg = remaining[len(remaining) - seg_len :]
                boundary_period = obs[obs.index(seg[-1]) + 1]
                bp_row = row_pos[(pid, boundary_period)]
                boundary_age = float(hp_age[bp_row])
                boundary_value = float(gen_earn[bp_row])
                donor_pid, widen = match_donor(
                    seg,
                    boundary_period,
                    boundary_age,
                    boundary_value,
                    by_period,
                    pos,
                )
                if donor_pid is not None:
                    if widen > 0:
                        n_widened_segments += 1
                        widen_steps_total += widen
                        max_widen_steps = max(max_widen_steps, widen)
                    break
                if seg_len == 1:
                    n_unmatched_one_period_segments += 1
                    break
                seg_len -= 1
                shorten_events += 1

            # Consume this segment's periods (``seg`` / ``boundary_period``
            # hold the resolved segment from the loop above).
            remaining = remaining[: len(remaining) - seg_len]
            n_person_segments += 1
            n_segments += 1
            seg_lengths.append(seg_len)

            if donor_pid is None:
                # A one-period segment with no period-matched donor: leave
                # the holdout values (does not occur on the staged panel;
                # the rate is reported).
                continue

            donors_of_person[pid].append(donor_pid)
            all_donor_ids.add(donor_pid)
            donor_boundary = float(
                by_period[boundary_period]["earn"][
                    pos[boundary_period][donor_pid]
                ]
            )
            match_errors.append(abs(donor_boundary - boundary_value))

            if boundary_value > 0 and donor_boundary > 0:
                raw_ratio = boundary_value / donor_boundary
                scale = float(np.clip(raw_ratio, SCALE_CLIP_LO, SCALE_CLIP_HI))
                n_segments_scaled += 1
                scale_ratios.append(raw_ratio)
                if raw_ratio < SCALE_CLIP_LO or raw_ratio > SCALE_CLIP_HI:
                    n_segments_clipped += 1
            else:
                scale = 1.0  # no scaling: donor values copy raw.
                n_segments_unscaled += 1

            for p in seg:
                donor_val = float(by_period[p]["earn"][pos[p][donor_pid]])
                gen_earn[row_pos[(pid, p)]] = donor_val * scale  # zeros stay
                n_spliced_observations += 1
        seg_counts.append(n_person_segments)

    out = hp.copy()
    out["earnings"] = gen_earn
    candidate = out[["person_id", "period", "earnings", "age", "weight"]]

    diagnostics = _splice_diagnostics(
        n_holdout_persons=int(len(holdout_anchor)),
        seg_counts=seg_counts,
        seg_lengths=seg_lengths,
        n_segments=n_segments,
        n_widened_segments=n_widened_segments,
        widen_steps_total=widen_steps_total,
        max_widen_steps=max_widen_steps,
        shorten_events=shorten_events,
        n_unmatched_one_period_segments=n_unmatched_one_period_segments,
        n_segments_scaled=n_segments_scaled,
        n_segments_clipped=n_segments_clipped,
        n_segments_unscaled=n_segments_unscaled,
        n_spliced_observations=n_spliced_observations,
        match_errors=match_errors,
        scale_ratios=scale_ratios,
        donors_of_person=donors_of_person,
        all_donor_ids=all_donor_ids,
    )
    return candidate, diagnostics


def _dist(values: list[int]) -> dict[int, int]:
    """A sorted ``{value: count}`` histogram over a list of ints."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=int)
    uniq, counts = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts, strict=True)}


def _splice_diagnostics(
    *,
    n_holdout_persons: int,
    seg_counts: list[int],
    seg_lengths: list[int],
    n_segments: int,
    n_widened_segments: int,
    widen_steps_total: int,
    max_widen_steps: int,
    shorten_events: int,
    n_unmatched_one_period_segments: int,
    n_segments_scaled: int,
    n_segments_clipped: int,
    n_segments_unscaled: int,
    n_spliced_observations: int,
    match_errors: list[float],
    scale_ratios: list[float],
    donors_of_person: dict[int, list[int]],
    all_donor_ids: set[int],
) -> dict[str, Any]:
    """Assemble the reported-not-gated splice diagnostics block."""
    # Donor reuse: distinct donors PER PERSON, and PER HOLDOUT (how many
    # holdout persons share each donor).
    distinct_per_person = [len(set(v)) for v in donors_of_person.values()]
    n_segments_per_person = [len(v) for v in donors_of_person.values()]
    holdouts_per_donor: dict[int, int] = {}
    for donors in donors_of_person.values():
        for d in set(donors):
            holdouts_per_donor[d] = holdouts_per_donor.get(d, 0) + 1
    reuse_series = pd.Series(list(holdouts_per_donor.values()), dtype=int)
    reuse_dist = _dist(reuse_series.tolist()) if len(reuse_series) else {}

    match_arr = np.asarray(match_errors, dtype=np.float64)
    match_pcts: dict[str, float] = {}
    if match_arr.size:
        for q in (50, 90, 99):
            match_pcts[f"p{q}"] = float(np.percentile(match_arr, q))

    return {
        "n_holdout_persons": n_holdout_persons,
        "n_segments": int(n_segments),
        "segment_count": {
            "distribution": _dist(seg_counts),
            "mean_per_person": (
                float(np.mean(seg_counts)) if seg_counts else 0.0
            ),
            "max_per_person": int(max(seg_counts)) if seg_counts else 0,
            "note": (
                "number of spliced segments per holdout person (0 = the "
                "person has only an anchor, so nothing is spliced); each "
                "segment is up to 3 observed pre-anchor periods"
            ),
        },
        "segment_length": {
            "distribution": _dist(seg_lengths),
            "mean": float(np.mean(seg_lengths)) if seg_lengths else 0.0,
            "note": (
                "observed periods per segment (<= 3; the last segment of a "
                "person, and any shortened segment, may be shorter)"
            ),
        },
        "age_window_widening": {
            "n_widened_segments": int(n_widened_segments),
            "n_segments": int(n_segments),
            "rate_over_segments": (
                float(n_widened_segments / n_segments) if n_segments else 0.0
            ),
            "widen_steps_total": int(widen_steps_total),
            "max_widen_steps": int(max_widen_steps),
            "note": (
                "a segment is 'widened' when its boundary-age window grew "
                "beyond the initial +/-2 (in +/-2 steps up to +/-10) before "
                "a period-matched donor was found; rate over all segments"
            ),
        },
        "segment_shortening": {
            "shorten_events": int(shorten_events),
            "n_segments": int(n_segments),
            "rate_over_segments": (
                float(shorten_events / n_segments) if n_segments else 0.0
            ),
            "n_unmatched_one_period_segments": int(
                n_unmatched_one_period_segments
            ),
            "note": (
                "shorten_events counts single-period drops (a segment "
                "shrank by one period because even the +/-10 window held no "
                "donor observed at all its periods); "
                "n_unmatched_one_period_segments counts one-period segments "
                "that still found no period-matched donor (0 in practice on "
                "the staged panel; those keep the holdout values)"
            ),
        },
        "scaling_clip": {
            "n_segments_scaled": int(n_segments_scaled),
            "n_segments_unscaled": int(n_segments_unscaled),
            "n_segments_clipped": int(n_segments_clipped),
            "rate_over_scaled": (
                float(n_segments_clipped / n_segments_scaled)
                if n_segments_scaled
                else 0.0
            ),
            "rate_over_segments": (
                float(n_segments_clipped / n_segments) if n_segments else 0.0
            ),
            "raw_ratio_min": (
                float(np.min(scale_ratios)) if scale_ratios else None
            ),
            "raw_ratio_max": (
                float(np.max(scale_ratios)) if scale_ratios else None
            ),
            "clip_bounds": [SCALE_CLIP_LO, SCALE_CLIP_HI],
            "note": (
                "a segment is scaled when its boundary value and the "
                "donor's earnings at the boundary period are both positive; "
                "clipped when the raw ratio boundary_value/donor_boundary "
                "falls outside [0.2, 5]; unscaled otherwise (donor values "
                "copy raw, zeros stay zeros)"
            ),
        },
        "boundary_match_error": {
            "n_scaled_or_copied_segments": int(len(match_errors)),
            "mean": float(np.mean(match_arr)) if match_arr.size else 0.0,
            "max": float(np.max(match_arr)) if match_arr.size else 0.0,
            "percentiles": match_pcts,
            "note": (
                "|donor earnings at the boundary period - boundary value| "
                "for each spliced segment; the match rule minimizes this, "
                "so it is the residual gap of the closest period- and "
                "age-eligible donor (reported, not gated)"
            ),
        },
        "donor_reuse": {
            "n_distinct_donors": int(len(all_donor_ids)),
            "distinct_donors_per_person": {
                "mean": (
                    float(np.mean(distinct_per_person))
                    if distinct_per_person
                    else 0.0
                ),
                "max": (
                    int(max(distinct_per_person)) if distinct_per_person else 0
                ),
                "distribution": _dist(distinct_per_person),
            },
            "segments_per_person": {
                "mean": (
                    float(np.mean(n_segments_per_person))
                    if n_segments_per_person
                    else 0.0
                ),
            },
            "holdouts_per_donor": {
                "max": (int(reuse_series.max()) if len(reuse_series) else 0),
                "distribution": reuse_dist,
            },
            "note": (
                "distinct_donors_per_person: how many different donors a "
                "holdout person's segments draw from (>1 means a person's "
                "career is stitched from multiple donors, unlike the "
                "single-donor 5a); holdouts_per_donor.distribution maps "
                "'number of holdout persons that reused one donor' -> "
                "'number of donors reused that many times'"
            ),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_seed(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    view_specs: dict[str, Any],
    views_cfg: dict[str, Any],
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Segment, splice, and score candidate 5a' for a single gate seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)
    candidate, diagnostics = generate_candidate(holdout, train, all_anchor)

    # --- geometry: score candidate vs holdout on both locked views ---
    geometry_by_view: dict[str, Any] = {}
    geometry_seed_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_seed_pass = geometry_seed_pass and view_pass
        cand_windows, _ = hpanel.project_panel(candidate, view)
        n_windows[vname] = int(len(cand_windows))
        geometry_by_view[vname] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "thresholds": views_cfg[vname]["geometry"],
            "checks": checks,
            "view_pass": bool(view_pass),
        }

    # --- battery: on the CANDIDATE panel, vs committed reference ---
    battery_values = compute_battery(candidate)
    battery_checks = check_battery(
        battery_values, battery_reference, battery_tol
    )
    battery_seed_pass = all(c["pass"] for c in battery_checks.values())

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_windows": n_windows,
        "splice_diagnostics": diagnostics,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        d = diagnostics
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"segs/person={d['segment_count']['mean_per_person']:.2f} "
            f"widen_rate={d['age_window_widening']['rate_over_segments']:.4f} "
            f"shorten={d['segment_shortening']['shorten_events']} "
            f"clip_rate={d['scaling_clip']['rate_over_segments']:.4f} "
            f"distinct_donors/person="
            f"{d['donor_reuse']['distinct_donors_per_person']['mean']:.2f} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-5a' run."""
    started = time.time()
    thresholds = load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    views_cfg = thresholds["views"]
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }

    battery_ref_artifact = json.loads(
        (ROOT / BATTERY_REFERENCE_RUN).read_text()
    )
    battery_reference = battery_ref_artifact["battery_reference"]

    panel = load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Identical battery-reference bit-exact precheck as every prior run:
    # the battery code path must reproduce every committed reference value
    # to float precision before any candidate is scored.
    repro = reproduce_battery_reference(panel)
    if verbose:
        print(
            "battery_reference reproduced exactly: "
            f"{repro['all_committed_values_reproduced_exactly']}"
        )
    if not repro["all_committed_values_reproduced_exactly"]:
        raise RuntimeError(
            "Battery code path does not reproduce the committed "
            "battery_reference to float precision; refusing to proceed "
            "with a divergent definition."
        )

    # Anchors on the FULL filtered panel (a person's last observed period
    # is a property of the panel, computed once and sliced per split).
    all_anchor = anchor_rows(panel)

    view_specs = {
        "psid_family_earnings_pairs": build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed.append(
            run_seed(
                seed,
                panel,
                all_anchor,
                view_specs,
                views_cfg,
                battery_reference,
                battery_tol,
                verbose,
            )
        )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_splice_v2",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 5a': segment splicing. For each held-out "
            "person the anchor (chronologically last observed period) is "
            "held at its real value and the earlier observed periods are "
            "grouped backward into segments of up to 3; each segment is "
            "spliced from a train-person donor matched deterministically "
            "at the segment's boundary (the next-later observed period, "
            "filled by the later segment or the anchor): candidate donors "
            "are observed at every segment period and the boundary period "
            "with boundary-age within a +/-2-year window (widened in +/-2 "
            "steps to +/-10, else the segment shortens by one period), and "
            "the donor minimizing |donor boundary earnings - boundary "
            "value| wins (ties -> smaller boundary-age gap, then smaller "
            "person_id). The segment's donor earnings at its own calendar "
            "periods are scaled by boundary_value / donor_boundary_earnings "
            "(clipped to [0.2, 5]) when both are positive, else copied "
            "unscaled; donor zeros copy as zeros. Fully deterministic given "
            "the split -- no RNG, no model fit, no populace-fit. Registered "
            "frozen before the run in issue #42 (see spec_registration). "
            "Candidate scored against the held-out PSID family earnings "
            "panel geometry (two locked views) and the locked moment "
            "battery, per the locked seed-level conjunction in gates.yaml "
            "(pull request 39). Protocol machinery imported byte-for-byte "
            "from the baseline runner (pull request 40)."
        ),
        "model": {
            "class": "deterministic segment splicing (period-indexed)",
            "stochastic": False,
            "populace_fit_used": False,
            "runtime_environment": (
                "repo .venv (scikit-learn 1.9, pandas); no populace-fit"
            ),
            "donor_pool": (
                "the seed's 80% train persons' observed trajectories in "
                "the locked filtered panel"
            ),
            "segmentation": {
                "scope": (
                    "the holdout person's observed periods before the "
                    "anchor, grouped in backward order into consecutive "
                    "segments of up to 3 observed periods (the last may be "
                    "shorter)"
                ),
                "segment_max": SEGMENT_MAX,
            },
            "boundary": {
                "definition": (
                    "each segment's boundary is the person's next-later "
                    "observed period (already filled by a later segment, or "
                    "the anchor for the first segment)"
                ),
                "value": (
                    "the target's value at the boundary period -- real at "
                    "the anchor, spliced otherwise (a backward chain: each "
                    "segment scales to the value the later segment produced "
                    "at the shared boundary)"
                ),
            },
            "match_rule": {
                "cell": (
                    "train persons observed at ALL of the segment's "
                    "calendar periods AND at the boundary period, with age "
                    "at the boundary period within +/-2 years of the "
                    "target's age there"
                ),
                "selection": (
                    "smallest |donor earnings at the boundary period - "
                    "boundary value|"
                ),
                "tie_breaks": [
                    "smaller |donor boundary age - target boundary age|",
                    "smaller donor person_id",
                ],
            },
            "fallback_cascade": {
                "widen": (
                    "if no candidate, widen the age window in +/-2 steps "
                    "(+/-4, +/-6, ...) up to +/-10"
                ),
                "shorten": (
                    "if still empty, shorten the segment by one period "
                    "(drop its earliest period, which re-groups into the "
                    "following earlier segment; the boundary is preserved) "
                    "and retry from the original +/-2 window"
                ),
                "one_period_floor": (
                    "a one-period segment with a same-period boundary "
                    "requirement always has candidates in practice (rate "
                    "reported)"
                ),
            },
            "splice_and_scale": {
                "value": (
                    "the segment's generated values are the donor's "
                    "earnings at the segment's own calendar periods"
                ),
                "scale": (
                    "multiplied by boundary_value / donor earnings at the "
                    "boundary period, clipped to [0.2, 5], when both are "
                    "positive"
                ),
                "otherwise": (
                    "if either is zero or negative the segment copies "
                    "unscaled; donor zeros copy as zeros"
                ),
                "clip_bounds": [SCALE_CLIP_LO, SCALE_CLIP_HI],
            },
            "anchor": (
                "chronologically last observed period held at real earnings"
            ),
            "candidate_panel_pin": (
                "exactly the holdout persons on exactly their observed "
                "periods; only earnings generated; anchor keeps real "
                "value; person_id/period/age/weight copied from holdout"
            ),
            "determinism": (
                "fully deterministic given the split; the gate seed enters "
                "only through split_panel_by_person"
            ),
        },
        "protocol": {
            "filter": (
                f"age {AGE_MIN}-{AGE_MAX}, reference years "
                f"{PERIOD_MIN}-{PERIOD_MAX}, positive weights (applied "
                "before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel, 'person_id', fraction=0.2, seed=s); the drawn 20% "
                "is the holdout, the complement is the training set / "
                "donor pool (imported from the baseline runner)"
            ),
            "seeds": list(SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "panel_scorecard(candidate, holdout, view, seed=s) per "
                "locked view; battery on the candidate panel vs committed "
                "battery_reference (imported from the baseline runner)"
            ),
            "pass_rule": (
                "seed passes geometry iff every locked threshold on every "
                "locked view holds; seed passes battery iff every locked "
                "tolerance holds; gate passes iff >=4/5 seeds pass geometry "
                "AND >=4/5 seeds pass battery"
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
            }
            for s in per_seed
        ],
        "splice_diagnostics_context": {
            "note": (
                "Reported-not-gated diagnostics per seed: segment count "
                "and length distributions, age-window widening and segment "
                "shortening rates, scaling-clip rate, donor reuse (distinct "
                "donors per person and per holdout), and the boundary-value "
                "match-error distribution. None of these enters the "
                "geometry or battery pass/fail; the gate rule names only "
                "those two families."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "segment_count": s["splice_diagnostics"]["segment_count"],
                    "segment_length": s["splice_diagnostics"][
                        "segment_length"
                    ],
                    "age_window_widening": s["splice_diagnostics"][
                        "age_window_widening"
                    ],
                    "segment_shortening": s["splice_diagnostics"][
                        "segment_shortening"
                    ],
                    "scaling_clip_rate_over_segments": s["splice_diagnostics"][
                        "scaling_clip"
                    ]["rate_over_segments"],
                    "boundary_match_error": s["splice_diagnostics"][
                        "boundary_match_error"
                    ],
                    "donor_reuse": s["splice_diagnostics"]["donor_reuse"],
                }
                for s in per_seed
            ],
        },
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "gate_1_pass": bool(gate_pass),
            "rule": ">=4/5 seeds geometry AND >=4/5 seeds battery",
        },
        "revision_pins": _revision_pins(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5)"
        )
    return artifact


def _revision_pins() -> dict[str, Any]:
    """Repo/populace SHAs and schema version for provenance."""
    import subprocess

    def _sha(cwd: Path) -> str | None:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode()
                .strip()
            )
        except Exception:
            return None

    populace_root = Path("~/PolicyEngine/populace").expanduser()
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "populace_repo_sha": _sha(populace_root),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml_locked": True,
    }


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
