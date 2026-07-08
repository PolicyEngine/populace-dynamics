"""Bind every locked gate-1 threshold to its stated derivation.

Round-2 review finding 5b: derivations lived only in YAML comments, so
an artifact rebuild that shifted a floor could silently break the
stated rationale. Each view's ``derivations`` block in gates.yaml is
machine-checkable data: threshold == round(floor mean + k * floor sd)
at the stated rounding. These tests run everywhere — they touch only
committed files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _gate_1() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _floor_stats(run_path: str, key: str) -> tuple[float, float]:
    artifact = json.loads((ROOT / run_path).read_text())
    stats = artifact["noise_floor_seeds_0_4"][key]
    return stats["mean"], stats["sd"]


def _derive(run_path: str, key: str, k: float, rounding: int) -> float:
    mean, sd = _floor_stats(run_path, key)
    return round(mean + k * sd, rounding)


def _view_cases():
    for view_name, view in _gate_1()["views"].items():
        derivations = view["derivations"]
        for threshold_name, rule in derivations["rules"].items():
            yield (
                view_name,
                threshold_name,
                derivations["floor_run"],
                rule,
                view["geometry"][threshold_name],
            )


@pytest.mark.parametrize(
    "view_name, threshold_name, floor_run, rule, locked",
    list(_view_cases()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_locked_threshold_matches_stated_derivation(
    view_name, threshold_name, floor_run, rule, locked
):
    rounding = rule.get("rounding", 2)
    if isinstance(rule["k"], list):
        lower = _derive(floor_run, rule["key"], rule["k"][0], rounding)
        upper = _derive(floor_run, rule["key"], rule["k"][1], rounding)
        assert [lower, upper] == pytest.approx(locked), (
            f"{view_name}.{threshold_name}: derived [{lower}, {upper}] "
            f"!= locked {locked}"
        )
    else:
        derived = _derive(floor_run, rule["key"], rule["k"], rounding)
        assert derived == pytest.approx(locked), (
            f"{view_name}.{threshold_name}: derived {derived} "
            f"!= locked {locked}"
        )


def test_every_locked_geometry_threshold_has_a_derivation():
    for view in _gate_1()["views"].values():
        assert set(view["geometry"]) == set(view["derivations"]["rules"])


def test_battery_tolerances_have_committed_references():
    thresholds = _gate_1()
    reference = json.loads(
        (ROOT / "runs/noise_floor_psid_family_9822.json").read_text()
    )["battery_reference"]
    aliases = {"mobility_diagonal": "mobility_diagonal_mean"}
    for name in thresholds["battery"]:
        stat = name.removesuffix("_tolerance")
        stat = aliases.get(stat, stat)
        assert stat in reference, f"no committed reference for {stat}"


def test_zero_persistence_is_one_minus_exit_rate():
    """The lock annotates these as ONE constraint; pin the identity."""
    reference = json.loads(
        (ROOT / "runs/noise_floor_psid_family_9822.json").read_text()
    )["battery_reference"]
    assert math.isclose(
        reference["zero_persistence"],
        1.0 - reference["exit_rate"],
        rel_tol=0,
        abs_tol=1e-12,
    )


# --------------------------------------------------------------------------
# Amendment proposal binding (gate_1.amendment_proposed)
#
# The amendment proposal is an inert public OBJECT: it changes no locked
# value and no model reads it. These tests bind its arithmetic so a
# referee round (and any later ratification) inherits a proposal whose
# stated derivations already hold. They run everywhere -- they touch only
# committed files (gates.yaml plus the anchor artifact). All are no-ops
# (skip) until the amendment_proposed subsection exists.
# --------------------------------------------------------------------------
def _amendment() -> dict | None:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"].get("amendment_proposed")


def _benefit_space_change() -> dict | None:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    ratified = gates["gates"]["gate_1"]["thresholds"].get("benefit_space")
    if ratified is not None:
        return ratified
    amendment = _amendment()
    if amendment is None:
        return None
    for change in amendment["changes"]:
        if change.get("type") == "add_gated_block":
            return change
    return None


def _anchor_stats(run_path: str, key: str) -> tuple[float, float]:
    artifact = json.loads((ROOT / run_path).read_text())
    stats = artifact["noise_floor_seeds_0_4"][key]
    return stats["mean"], stats["sd"]


def test_amendment_ks_threshold_matches_committed_anchor():
    """Proposed KS band == round(anchor mean + k * anchor sd) at rounding.

    The one floor-DERIVED band in the proposal (every other band is the
    a-priori paper criterion). Mirrors the locked-geometry derivation
    test: the stated ``k`` and ``rounding`` must reproduce the value.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derivations = change["derivations"]
    run_path = derivations["anchor_run"]
    rule = derivations["rules"]["weighted_ks_max"]
    rounding = rule.get("rounding", 4)
    mean, sd = _anchor_stats(run_path, rule["key"])
    derived = round(mean + rule["k"] * sd, rounding)
    locked = change["metrics"]["weighted_ks_max"]["value"]
    assert derived == pytest.approx(
        locked
    ), f"proposed KS band derived {derived} != stated {locked}"


def test_amendment_gated_metrics_carry_derivation_or_a_priori_source():
    """Every proposed GATED metric is justified.

    Each metric in the proposed benefit_space block either derives from
    the committed anchor (a ``derivations.rules`` entry) or cites an
    a-priori source (an ``a_priori_source`` field). No proposed gated
    number is left unexplained.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derived_metrics = set(change["derivations"]["rules"])
    for name, spec in change["metrics"].items():
        has_derivation = name in derived_metrics
        has_a_priori = bool(spec.get("a_priori_source"))
        assert has_derivation or has_a_priori, (
            f"proposed gated metric {name} carries neither a derivation "
            f"nor an a_priori_source citation"
        )


def test_amendment_derived_metrics_are_not_also_flat_a_priori():
    """The floor-derived KS band is not mislabeled as an a-priori band.

    Derived and a-priori are mutually exclusive justifications; a metric
    with a ``derivations.rules`` entry must NOT also claim an a-priori
    source (that would hide that its value tracks the floor, not a fixed
    criterion).
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derived_metrics = set(change["derivations"]["rules"])
    for name in derived_metrics:
        assert not change["metrics"][name].get("a_priori_source"), (
            f"{name} is floor-derived; it must not also cite an "
            f"a_priori_source"
        )


def test_amendment_anchor_stats_recompute_from_stored_per_seed_values():
    """The anchor artifact's pooled stats recompute from its per-seed data.

    An artifact rebuild that shifted the floor must not silently keep
    stale pooled numbers: every headline the proposal leans on (the KS
    mean/sd it derives from, the abs mean/median/Q0 gap magnitudes, the
    pooled Q0 magnitude, and the per-decile 5%-clip counts) recomputes
    from ``per_seed``.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    run_path = change["derivations"]["anchor_run"]
    art = json.loads((ROOT / run_path).read_text())
    per_seed = art["per_seed"]
    floor = art["floor_seeds_0_4"]

    def dist(row: dict, path: list[str]):
        node = row["distribution"]
        for key in path:
            node = node[key]
        return node

    # KS block (the derived one) recomputes, and the headline
    # noise_floor_seeds_0_4 mirror equals it.
    ks_vals = [dist(r, ["ks_distance"]) for r in per_seed]
    assert floor["ks_distance"]["values"] == pytest.approx(ks_vals)
    assert floor["ks_distance"]["mean"] == pytest.approx(
        sum(ks_vals) / len(ks_vals)
    )
    n = len(ks_vals)
    mean_ks = sum(ks_vals) / n
    var_ks = sum((v - mean_ks) ** 2 for v in ks_vals) / (n - 1)
    assert floor["ks_distance"]["sd"] == pytest.approx(var_ks**0.5)
    assert art["noise_floor_seeds_0_4"]["ks_distance"] == (
        floor["ks_distance"]
    )

    # Absolute percent-gap magnitudes recompute from the signed per-seed
    # distribution gaps.
    abs_mean = [abs(dist(r, ["mean", "pct_diff"])) for r in per_seed]
    abs_median = [abs(dist(r, ["median", "pct_diff"])) for r in per_seed]
    assert floor["abs_mean_pct_diff"]["values"] == pytest.approx(abs_mean)
    assert floor["abs_median_pct_diff"]["values"] == pytest.approx(abs_median)

    # Q0: the pooled magnitude the gate is scored on equals the absolute
    # across-seed mean of the signed per-seed Q0 gaps.
    q0_signed = [r["q0_zero_anchor"]["mean_pct_diff"] for r in per_seed]
    pooled_abs_q0 = abs(sum(q0_signed) / len(q0_signed))
    assert floor["pooled_abs_q0_mean_pct_diff"] == pytest.approx(pooled_abs_q0)

    # Per-decile 5%-clip counts (the reported-not-gated evidence for d1,
    # d2) recompute from the signed per-seed decile gaps.
    for dkey, count in floor["decile_seeds_clipping_5pct"].items():
        vals = floor["signed_decile_pct_diff"][dkey]["values"]
        assert count == sum(abs(v) > 5.0 for v in vals), dkey


def test_amendment_reported_not_gated_deciles_are_denominator_fragile():
    """d1/d2 are reported-not-gated because the FLOOR itself clips 5%.

    The amendment reports-not-gates exactly the deciles whose
    real-vs-real floor exceeds the +/-5% band on a majority of seeds
    (denominator fragility), and gates exactly the deciles whose floor
    clips no seed. This binds that the reported/gated partition matches
    the committed anchor, not a hand-picked list.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    run_path = change["derivations"]["anchor_run"]
    art = json.loads((ROOT / run_path).read_text())
    clip = art["floor_seeds_0_4"]["decile_seeds_clipping_5pct"]
    n_seeds = art["n_persons"] and len(
        art["floor_seeds_0_4"]["ks_distance"]["values"]
    )
    majority = n_seeds // 2 + 1

    reported = set(change["reported_not_gated"])
    gated = set(change["metrics"]["decile_pct_diff_max"]["deciles_gated"])
    assert reported.isdisjoint(gated)

    # Every reported-not-gated decile clips 5% on a majority of seeds.
    for dkey in reported:
        assert clip[dkey] >= majority, (
            f"{dkey} is reported-not-gated but its real-vs-real floor "
            f"clips only {clip[dkey]}/{n_seeds} seeds"
        )
    # Every gated decile clips 5% on no seed.
    for dkey in gated:
        assert clip[dkey] == 0, (
            f"{dkey} is gated but its real-vs-real floor clips "
            f"{clip[dkey]} seeds"
        )


# --------------------------------------------------------------------------
# Amendment proposal 2 binding (gate_1.amendment_proposed, proposal_number 2)
#
# Reworked per the adversarial referee's AMEND verdict (PR #67). The
# pairs-view c2st gate moves from per-seed on the locked 5 to the
# across-seed mean over 20 pre-registered seeds (fix B), guarded by a
# VERSION-MATCHED per-seed catastrophe cap (fix C), and the amendment
# rescues NO committed verdict (fix A). Like amendment 1 this proposal is
# an inert public OBJECT: it changes no locked value and no model reads it.
# These tests bind its arithmetic (the version-matched cap, the unchanged
# mean line, the 20-seed set, the operating-characteristic table, the
# considered-and-rejected lines) and verify the illustrative retroactive
# disclosure -- applied: false, no committed verdict changed -- against a
# fresh recomputation from the committed run artifacts. All are no-ops
# (skip) until the amendment_2 subsection exists. They touch only committed
# files.
# --------------------------------------------------------------------------
PAIRS_VIEW = "psid_family_earnings_pairs"


def _amendment_2() -> dict | None:
    amendment = _amendment()
    if amendment is None or amendment.get("proposal_number") != 2:
        return None
    return amendment


def _amend2_change(change_id: int) -> dict | None:
    amendment = _amendment_2()
    if amendment is None:
        return None
    for change in amendment["changes"]:
        if change.get("id") == change_id:
            return change
    return None


def _amend2_rejected(entry_id: str) -> dict | None:
    amendment = _amendment_2()
    if amendment is None:
        return None
    for entry in amendment.get("considered_and_rejected", []):
        if entry.get("id") == entry_id:
            return entry
    return None


def _floor_at_path(run_path: str, path: list[str]) -> tuple[float, float]:
    """Return (mean, sd) from a nested node that stores them directly."""
    node = json.loads((ROOT / run_path).read_text())
    for key in path:
        node = node[key]
    return node["mean"], node["sd"]


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_amendment2_cap_derivation_binds_to_version_matched_floor():
    """The per-seed cap == round(matched floor mean + 8*sd, 3) == 0.554.

    Fix C: the cap derives from the VERSION-MATCHED (sklearn 1.8.0) floor
    -- the 20-seed floor_c2st_distribution in the diagnostics artifact --
    not the committed 1.9.0 floor. The stated k/rounding and the floor at
    the stated path must reproduce 0.554 from the committed artifact, not
    the version-mismatched 0.547.
    """
    change = _amend2_change(2)
    if change is None:
        pytest.skip("no amendment_proposed proposal 2 with a cap change")
    deriv = change["proposed"]["derivation"]
    mean, sd = _floor_at_path(deriv["floor_run"], deriv["floor_path"])
    rule = deriv["rule"]
    derived = round(mean + rule["k"] * sd, rule.get("rounding", 3))
    stated = change["proposed"]["value"]
    assert derived == pytest.approx(
        stated
    ), f"cap derived {derived} != stated {stated}"
    assert stated == 0.554


def test_amendment2_mean_rule_keeps_ratified_line_new_estimator():
    """The mean line holds the ratified 0.53 with the ratified derivation.

    Change 1 re-reads the SAME line as an across-seed mean; its value and
    its floor/key/k derivation equal the ratified per-seed pairs-c2st
    derivation (unchanged number). The amendment must RECORD that only the
    estimator changed: per_seed -> across_seed_mean, same value.
    """
    change = _amend2_change(1)
    if change is None:
        pytest.skip("no amendment_proposed proposal 2 with a mean change")
    proposed = change["proposed"]
    rule = proposed["derivation"]["rule"]
    derived = _derive(
        proposed["derivation"]["floor_run"],
        rule["key"],
        rule["k"],
        rule.get("rounding", 2),
    )
    assert derived == pytest.approx(proposed["value"])
    assert proposed["value"] == 0.53

    ratified = _gate_1()["views"][PAIRS_VIEW]["derivations"]
    ratified_rule = ratified["rules"]["c2st_auc_max"]
    assert proposed["derivation"]["floor_run"] == ratified["floor_run"]
    assert rule["key"] == ratified_rule["key"]
    assert rule["k"] == ratified_rule["k"]
    # The amendment records an estimator change, not a new number.
    assert change["currently_locked"]["statistic"] == "per_seed"
    assert proposed["statistic"] == "across_seed_mean"
    assert change["currently_locked"]["value"] == proposed["value"]


def test_amendment2_mean_gates_over_twenty_preregistered_seeds():
    """Fix B: the mean is over exactly seeds 0-19, not the 5 locked."""
    change = _amend2_change(1)
    if change is None:
        pytest.skip("no amendment_proposed proposal 2 with a mean change")
    assert change["proposed"]["seed_set"] == list(range(20))


def test_amendment2_operating_characteristics_recompute():
    """Every OC pass-probability recomputes from the stated normal approx.

    Fix B evidence, as data: pass probability at each hypothetical true
    mean under the per-seed >=4/5 rule, the rejected mean-of-5, and the
    proposed mean-of-20, all at the 0.53 line. Recomputed with math.erf
    (no scipy) and rounded to 2 decimals, they must equal the stored
    table. The proposed mean-of-20 must SHARPEN the operating
    characteristic at the line (referee finding 2): 0.50 at 0.530, and
    TIGHTER than the per-seed rule above the line where a mean-of-5 would
    LOOSEN it.
    """
    change = _amend2_change(1)
    if change is None:
        pytest.skip("no amendment_proposed proposal 2 with a mean change")
    oc = change["operating_characteristics"]
    sd = oc["per_seed_sd"]
    line = oc["line"]
    means = oc["true_means"]

    def per_seed_ge_4_of_5(mu, n):
        p = _normal_cdf((line - mu) / sd)
        return p**5 + 5 * p**4 * (1 - p)

    def mean_le_line(mu, n):
        se = sd / math.sqrt(n)
        return _normal_cdf((line - mu) / se)

    fns = {
        "per_seed_ge_4_of_5": per_seed_ge_4_of_5,
        "mean_le_line": mean_le_line,
    }
    for name, spec in oc["estimators"].items():
        fn = fns[spec["statistic"]]
        recomputed = [round(fn(mu, spec["n"]), 2) for mu in means]
        assert recomputed == pytest.approx(spec["pass_prob"]), (
            f"OC {name}: recomputed {recomputed} != stored "
            f"{spec['pass_prob']}"
        )

    idx_line = means.index(0.530)
    idx_above = means.index(0.533)
    ps = oc["estimators"]["per_seed_4_of_5"]["pass_prob"]
    m20 = oc["estimators"]["mean_of_20_proposed"]["pass_prob"]
    m5 = oc["estimators"]["mean_of_5_rejected"]["pass_prob"]
    assert m20[idx_line] == 0.50  # 50% at the line by symmetry
    assert m20[idx_above] < ps[idx_above]  # mean-of-20 sharper (tighter)
    assert m5[idx_above] > ps[idx_above]  # mean-of-5 would loosen


def test_amendment2_considered_line_binds():
    """The rejected tighter mean-of-5 line reproduces exactly (0.5194).

    Even the line the amendment DECLINES to propose is machine-checked:
    round(floor mean + k * sd / sqrt(n), rounding). Kept in
    considered_and_rejected; a rounded-input hand computation reads
    ~0.5196, so the referee weighs the exact committed-artifact value.
    """
    entry = _amend2_rejected("mean_of_five_standard_error_line")
    if entry is None:
        pytest.skip("no amendment 2 rejected mean-of-5 line")
    rule = entry["derivation"]["rule"]
    mean, sd = _floor_stats(entry["derivation"]["floor_run"], rule["key"])
    derived = round(
        mean + rule["k"] * sd / math.sqrt(rule["divide_by_sqrt_n"]),
        rule["rounding"],
    )
    assert derived == pytest.approx(entry["value"])
    assert entry["value"] == 0.5194


def test_amendment2_rejected_cap_binds_to_unmatched_floor():
    """The rejected 0.547 cap reproduces from the 1.9.0 floor sd.

    Disclosure that the original cap was off the wrong version:
    round(1.9.0 floor mean + 8*sd, 3) = 0.547, which the version-matched
    0.554 (change 2) replaces per fix C.
    """
    entry = _amend2_rejected("cap_from_1_9_0_floor_sd")
    if entry is None:
        pytest.skip("no amendment 2 rejected 1.9.0 cap")
    rule = entry["derivation"]["rule"]
    derived = _derive(
        entry["derivation"]["floor_run"],
        rule["key"],
        rule["k"],
        rule.get("rounding", 3),
    )
    assert derived == pytest.approx(entry["value"])
    assert entry["value"] == 0.547
    # It must differ from the adopted, version-matched cap.
    assert _amend2_change(2)["proposed"]["value"] != entry["value"]


def test_amendment2_no_self_rescue_clause_present():
    """Fix A: the verbatim no-self-rescue principle is recorded."""
    amendment = _amendment_2()
    if amendment is None:
        pytest.skip("no amendment_proposed proposal 2")
    clause = amendment.get("no_self_rescue", "")
    assert (
        "committed run verdict changes under a rule proposed after" in clause
    )
    assert "applies only to runs registered after its ratification" in clause


def test_amendment2_illustrative_block_is_not_applied():
    """Fix A: the retroactive table is disclosure only -- applied: false.

    No committed verdict changes: every run's committed_verdict is FAIL,
    the old applied-recomputation key is gone, and candidate 10 -- the
    sole would-flip -- stays FAIL.
    """
    amendment = _amendment_2()
    if amendment is None:
        pytest.skip("no amendment_proposed proposal 2")
    # The old applied verdict-recomputation key must not exist.
    assert "verdict_recomputation" not in amendment
    block = amendment["illustrative_retroactive_application"]
    assert block["applied"] is False
    for entry in block["runs"]:
        assert entry["committed_verdict"] == "FAIL", entry["run"]
    assert block["would_flip_if_applied"]["runs"] == ["candidate 10 (PR #64)"]


def test_amendment2_path_to_pass_requires_fresh_registration():
    """Fix A: a pass needs a fresh registration on seeds 0-19."""
    amendment = _amendment_2()
    if amendment is None:
        pytest.skip("no amendment_proposed proposal 2")
    p = amendment["path_to_pass"]
    assert "FRESH" in p or "fresh" in p
    assert "runs/c10_diagnostics_v1.json" in p
    assert "0.5234" in p


def _recompute_amended_verdict(entry: dict, mean_line: float, cap: float):
    """Recompute the four amended sub-verdicts from a committed artifact.

    Pairs mean/max/battery come from the run's own gate artifact; pooled
    Q0 comes from that artifact's benefit_space_gated block (candidates
    9-10) or the committed downstream-relevance artifact (candidate 7).
    """
    art = json.loads((ROOT / entry["artifact"]).read_text())
    per_seed = art["per_seed"]
    c2st = [s["geometry"][PAIRS_VIEW]["scores"]["c2st_auc"] for s in per_seed]
    pairs_mean = sum(c2st) / len(c2st)
    pairs_max = max(c2st)
    n_battery = sum(bool(s["battery_pass"]) for s in per_seed)

    q0_source = entry.get("q0_source", "not_scored")
    pooled_q0 = None
    if q0_source == "benefit_space_gated":
        pooled_q0 = art["benefit_space_gated"]["pooled_q0_gate"][
            "pooled_q0_mean_pct_diff"
        ]
    elif q0_source == "downstream_relevance":
        q0_art = json.loads((ROOT / entry["q0_artifact"]).read_text())
        pooled_q0 = q0_art["candidate_pooled"]["by_anchor_quintile"]["Q0"][
            "mean_pct_diff"
        ]

    return {
        "pairs_mean": pairs_mean,
        "pairs_max": pairs_max,
        "mean_rule_pass": pairs_mean <= mean_line,
        "cap_pass": pairs_max <= cap,
        "battery_seeds_pass": n_battery,
        "battery_gate_pass": n_battery >= 4,
        "pooled_q0": pooled_q0,
        "q0_pass": None if pooled_q0 is None else abs(pooled_q0) <= 5,
        "old_gate_1_pass": art["verdict"]["gate_1_pass"],
    }


def test_amendment2_illustrative_recomputation_matches_artifacts():
    """Every disclosed number recomputes from the committed artifacts.

    The load-bearing honesty section, now disclosure-only (applied:
    false): for each committed run the disclosed pairs mean, pairs
    max/cap, battery pass count, and pooled Q0 (plus pass/fail) equal a
    fresh recomputation from the run's committed artifact under the
    amended rule (mean line 0.53, version-matched cap 0.554). The
    would-be verdict is consistent -- a PASS satisfies all four
    sub-verdicts; a FAIL is backed by >= 1 failing sub-verdict -- the
    sole would-flip is candidate 10, and NO committed verdict moves.
    """
    amendment = _amendment_2()
    if amendment is None:
        pytest.skip("no amendment_proposed proposal 2")
    block = amendment["illustrative_retroactive_application"]
    assert block["applied"] is False
    mean_line = block["mean_line"]
    cap = block["cap_value"]
    # The disclosure's own line/cap must equal the change derivations.
    assert mean_line == _amend2_change(1)["proposed"]["value"]
    assert cap == _amend2_change(2)["proposed"]["value"]

    flips = []
    for entry in block["runs"]:
        rc = _recompute_amended_verdict(entry, mean_line, cap)
        label = f"{entry['run']} (PR #{entry['pr']})"

        assert entry["pairs_c2st_mean"] == round(rc["pairs_mean"], 4), label
        assert entry["pairs_c2st_max"] == round(rc["pairs_max"], 4), label
        assert entry["mean_rule_pass"] == rc["mean_rule_pass"], label
        assert entry["cap_pass"] == rc["cap_pass"], label
        assert entry["battery_seeds_pass"] == rc["battery_seeds_pass"], label
        assert entry["battery_gate_pass"] == rc["battery_gate_pass"], label

        stated_q0 = entry.get("pooled_q0_mean_pct_diff")
        if rc["pooled_q0"] is None:
            assert stated_q0 is None, f"{label}: stated Q0 but none scored"
            assert entry.get("q0_pass") is None, label
        else:
            assert stated_q0 == round(rc["pooled_q0"], 4), label
            assert entry["q0_pass"] == rc["q0_pass"], label

        # Committed verdict is FAIL for every run and is NOT changed by
        # this amendment (no_self_rescue / applied: false).
        assert entry["committed_verdict"] == "FAIL", label
        assert rc["old_gate_1_pass"] is False, label

        # Would-be (illustrative) verdict consistency.
        sub_pass = [
            rc["mean_rule_pass"],
            rc["cap_pass"],
            rc["battery_gate_pass"],
        ]
        if rc["q0_pass"] is not None:
            sub_pass.append(rc["q0_pass"])
        if entry["would_be_verdict_if_applied"] == "PASS":
            assert all(
                sub_pass
            ), f"{label}: would-PASS but a sub-verdict fails"
            flips.append(label)
        else:
            assert entry["would_be_verdict_if_applied"] == "FAIL", label
            assert not all(
                sub_pass
            ), f"{label}: would-FAIL but every sub-verdict passes"

    # Exactly one committed run WOULD flip, and it is candidate 10 -- but
    # only illustratively; its committed verdict stays FAIL (asserted
    # above for every run).
    assert flips == ["candidate 10 (PR #64)"], flips
    assert flips == block["would_flip_if_applied"]["runs"]
    assert block["would_flip_if_applied"]["count"] == len(flips)


# --------------------------------------------------------------------------
# Amendment 2 RATIFIED bindings (live thresholds)
#
# Amendment 2 (proposed PR #67, referee AMEND -> fixes ae0c166 ->
# verification RATIFY AS-IS -> ratified by merge 4e06e244, flipped live
# 2026-07-07). The proposal-object tests above go dormant once the
# amendment_proposed subsection is removed; these tests bind the SAME
# guarantees at their ratified locations, so a rebuild that shifted a
# floor -- or an edit that quietly reapplied the amendment retroactively
# -- breaks loudly. They touch only committed files.
# --------------------------------------------------------------------------
def _pairs_view() -> dict:
    return _gate_1()["views"]["psid_family_earnings_pairs"]


def test_ratified_mean_rule_keeps_line_and_derivation():
    """The live 20-seed mean rule holds the ratified 0.53 derivation."""
    rule = _pairs_view()["c2st_mean_rule"]
    deriv = rule["derivations"]["rules"]["c2st_auc_mean_max"]
    derived = _derive(
        rule["derivations"]["floor_run"],
        deriv["key"],
        deriv["k"],
        deriv.get("rounding", 2),
    )
    assert derived == pytest.approx(rule["value_max"])
    assert rule["value_max"] == 0.53
    assert deriv["k"] == 4.2  # the ratified per-seed k, unchanged
    assert rule["statistic"] == "across_seed_mean"
    assert rule["seed_set"] == list(range(20))


def test_ratified_cap_binds_to_version_matched_floor():
    """The live per-seed cap == round(matched mean + 8*sd, 3) == 0.554."""
    cap = _pairs_view()["c2st_per_seed_cap"]
    deriv = cap["derivations"]
    mean, sd = _floor_at_path(deriv["floor_run"], deriv["floor_path"])
    rule = deriv["rules"]["c2st_auc_cap"]
    derived = round(mean + rule["k"] * sd, rule.get("rounding", 3))
    assert derived == pytest.approx(cap["value_max"])
    assert cap["value_max"] == 0.554
    assert derived != 0.547  # the rejected version-mismatched cap


def test_ratified_per_seed_c2st_is_superseded_not_gated():
    """c2st_auc left the per-seed geometry; the supersession is recorded."""
    view = _pairs_view()
    assert "c2st_auc_max" not in view["geometry"]
    assert "c2st_auc_max" not in view["derivations"]["rules"]
    assert view["per_seed_rule_superseded"] == ["c2st_auc"]
    # geometry <-> derivations equality still holds after the removal.
    assert set(view["geometry"]) == set(view["derivations"]["rules"])


def test_ratified_operating_characteristics_recompute():
    """Every live OC pass-probability recomputes via the normal approx."""
    oc = _pairs_view()["c2st_mean_rule"]["operating_characteristics"]
    sd, line = oc["per_seed_sd"], oc["line"]
    for name, est in oc["estimators"].items():
        for mu, stored in zip(oc["true_means"], est["pass_prob"], strict=True):
            p = _normal_cdf((line - mu) / sd)
            if est["statistic"] == "per_seed_ge_4_of_5":
                prob = p**5 + 5 * p**4 * (1 - p)
            else:
                se = sd / math.sqrt(est["n"])
                prob = _normal_cdf((line - mu) / se)
            assert round(prob, 2) == pytest.approx(stored), (name, mu)
    # The sharpening claim: above the line the 20-seed mean is
    # STRICTER than the old per-seed rule; the rejected mean-of-5
    # would have been looser.
    at_533 = {
        name: est["pass_prob"][-1] for name, est in oc["estimators"].items()
    }
    assert at_533["mean_of_20_proposed"] < at_533["per_seed_4_of_5"]
    assert at_533["per_seed_4_of_5"] < at_533["mean_of_5_rejected"]


def test_ratified_amendment_is_prospective_no_verdict_changed():
    """no_self_rescue: no PRE-amendment-2 run's verdict changed at the flip.

    The strongest live form of no_self_rescue is backward-looking: every
    gate-run artifact registered BEFORE amendment 2 -- i.e. scored under the
    per-seed pairs c2st rule, not the 20-seed mean rule -- still records
    gate_1_pass false, candidate 10 (run 12) included. A retroactive
    application would have flipped candidate 10; the flip did not touch it.

    A run registered AFTER the 2026-07-07 ratification is scored under the
    20-seed mean rule + per-seed cap (its verdict carries ``c2st_mean_rule_pass``)
    and MAY pass -- amendment_history says a first pass must come from exactly
    such a fresh one-shot registration (candidate 11 / run 13 is the first).
    Those prospective runs are outside this backward-looking guarantee, so the
    invariant is asserted on the pre-amendment set only.
    """
    runs = sorted(ROOT.glob("runs/gate1_*.json"))
    pre_amendment = [
        p
        for p in runs
        if json.loads(p.read_text())["verdict"].get("c2st_mean_rule_pass")
        is None
    ]
    # The committed pre-amendment runs are untouched by the flip: all 12 stand
    # FAIL (candidate 10 among them). Post-amendment runs are excluded.
    assert len(pre_amendment) == 12
    for path in pre_amendment:
        verdict = json.loads(path.read_text())["verdict"]
        assert verdict["gate_1_pass"] is False, path.name


def test_ratified_standing_rules_and_history_record():
    """no_self_rescue + version pin stand; history entry 2 is complete."""
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate_1 = gates["gates"]["gate_1"]
    rules = gate_1["amendment_rules"]
    assert "No candidate's committed run verdict changes" in (
        rules["no_self_rescue"]
    )
    assert "scikit-learn" in rules["classifier_version_pin"]
    entry = gate_1["amendment_history"][1]
    assert entry["id"] == "2026-07-07-mean-based-classifier-gating"
    for key in ("referee_round", "ratified", "flipped_live", "content"):
        assert entry[key], key
    assert "PROSPECTIVE ONLY" in entry["content"]
    assert "4904161939" in entry["referee_round"]["review"]
    assert "4905067301" in entry["referee_round"]["verification"]


# --------------------------------------------------------------------------
# Gate 2 DRAFT threshold binding (gate_2.thresholds, status draft, v2)
#
# The gate-2 pre-lock evidence fills the gate_2 stub with a DRAFT thresholds
# block (locked: false, status: draft_pending_referee_round) whose per-cell
# |ln ratio| tolerances are machine-bound to the committed 100-seed
# half-split floor runs/gate2_floors_v2.json exactly as the locked gate-1
# geometry thresholds bind to theirs: tolerance == round(floor mean + k *
# floor sd, rounding). v2 applies the round-1 referee amendments (PR #79
# comment 4910467957): the option-(a) protocol, the 100-seed floor, the
# T_max = ln(1.5) power cap with pre-registered aggregate supersession, the
# joint/sequence cells, the governance inheritance, and the scope map.
# These run everywhere (committed files only) and SKIP the moment gate_2
# leaves the draft status -- a ratification round replaces them with locked
# bindings, just as amendment 1/2's proposal-object tests went dormant.
# --------------------------------------------------------------------------
GATE2_FLOOR_RUN = "runs/gate2_floors_v2.json"
GATE2_FLOOR_KEY = "noise_floor_seeds_0_99"


def _gate_2() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_2"]["thresholds"]


def _gate_2_if_draft() -> dict | None:
    """Retained name; post-lock the bindings run unconditionally.

    Amendment-2 lesson: derivation bindings must stay HOT after
    ratification, not go dormant with the draft status.
    """
    return _gate_2()


def _gate2_floor() -> dict:
    return json.loads((ROOT / GATE2_FLOOR_RUN).read_text())


def _gate2_derive(cell: str, k: float, rounding: int) -> float:
    stats = _gate2_floor()[GATE2_FLOOR_KEY][cell]
    return round(stats["mean"] + k * stats["sd"], rounding)


def test_gate2_locked_with_ratification_record():
    """Gate 2 is locked, with the full ceremony recorded in the contract."""
    g2 = _gate_2()
    assert g2["locked"] is True
    assert g2["status"] == "locked"
    record = g2["ratified"]
    for token in (
        "PR 79",
        "4910467957",  # round-1 adversarial review
        "4910712436",  # fixes summary
        "4910856982",  # verification LOCK AS-IS
        "no_self_rescue",
    ):
        assert token in record, token


def test_gate2_floor_run_is_a_reported_anchor():
    """The floor the locked thresholds cite reads no gate."""
    g2 = _gate_2_if_draft()
    assert g2["floor_run"] == GATE2_FLOOR_RUN
    floor = _gate2_floor()
    assert floor["reported_anchor_not_gated"] is True
    assert floor["schema_version"] == "gate2_floors.v2"
    assert "NOT RATIFIED" in floor["draft_thresholds_note"]


def test_gate2_locked_thresholds_bind_to_floor():
    """Every locked tolerance == round(100-seed floor mean + k*sd, rounding).

    The exact machine-binding the locked gate-1 geometry thresholds carry,
    applied to each gate-2 view's ``tolerances`` against the committed
    100-seed half-split floor. Every ``tolerances`` entry has a matching
    ``derivations.rules`` entry whose ``key`` is the floor cell itself.
    """
    g2 = _gate_2_if_draft()
    for view_name, view in g2["views"].items():
        rules = view["derivations"]["rules"]
        tolerances = view["tolerances"]
        assert set(rules) == set(tolerances), view_name
        assert view["derivations"]["floor_run"] == GATE2_FLOOR_RUN
        for cell, rule in rules.items():
            assert rule["key"] == cell, f"{view_name}.{cell}"
            derived = _gate2_derive(cell, rule["k"], rule.get("rounding", 3))
            assert derived == pytest.approx(tolerances[cell]), (
                f"{view_name}.{cell}: derived {derived} != tolerance "
                f"{tolerances[cell]}"
            )


def test_gate2_locked_k_matches_gate1_precedent():
    """Every locked k is the ~4-sigma gate-1 precedent."""
    g2 = _gate_2_if_draft()
    assert g2["holdout_basis"] == ["mh85_23", "cah85_23", "MX23REL"]
    assert g2["floor_run"] == GATE2_FLOOR_RUN
    for view in g2["views"].values():
        for rule in view["derivations"]["rules"].values():
            assert rule["k"] == 4


def test_gate2_power_cap_binds_every_gated_cell():
    """Finding 3: every gated tolerance <= T_max = ln(1.5), and the floor
    partitioned exactly on that cap (+ >=20 events + supersession)."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    assert g2["power_cap"]["t_max"] == "ln(1.5)"
    floor = _gate2_floor()
    t_max = floor["internal_noise_floor"]["t_max"]
    assert t_max == pytest.approx(math.log(1.5))
    for view in g2["views"].values():
        for cell, tol in view["tolerances"].items():
            assert tol <= t_max, f"{cell} tolerance {tol} exceeds T_max"


def test_gate2_report_only_matches_committed_floor_partition():
    """report_only / gated == the floor's derived partition (events + power
    cap + aggregate supersession), not hand-picked; the two are disjoint."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    floor = _gate2_floor()
    partition = floor["gate_partition"]
    assert set(g2["report_only"]) == set(partition["report_only"])

    gated = set()
    for view in g2["views"].values():
        gated |= set(view["tolerances"])
    assert gated == set(partition["gate_eligible"])
    assert gated.isdisjoint(set(g2["report_only"]))
    # Every gated cell carries a floor block; the partition covers all cells.
    for cell in gated:
        assert cell in floor[GATE2_FLOOR_KEY], cell


def test_gate2_aggregations_are_pre_registered_and_consistent():
    """Finding 3: the coverage-recovery aggregates match the floor, and a
    gating aggregate demotes every per-age member it spans."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    floor = _gate2_floor()
    yaml_aggs = g2["power_cap"]["aggregations"]
    art_aggs = floor["aggregations"]
    assert set(yaml_aggs) == set(art_aggs)
    report_only = set(g2["report_only"])
    gated = set()
    for view in g2["views"].values():
        gated |= set(view["tolerances"])
    for agg, spec in yaml_aggs.items():
        assert spec["gated"] == art_aggs[agg]["gated"], agg
        if spec["gated"]:
            assert agg in gated, agg
            for member in art_aggs[agg]["members"]:
                assert member in report_only, (agg, member)


def test_gate2_protocol_is_option_a_gate1_mirror():
    """Finding 1: one coherent protocol -- per-seed refit + simulate holdout,
    scored against the holdout half's own rate, 4-of-5 conjunction."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    proto = g2["protocol"]
    assert str(proto["option"]).strip().startswith("a")
    assert proto["gate_seeds"] == [0, 1, 2, 3, 4]
    for field in ("split", "candidate", "scored_against", "pass_rule"):
        assert proto[field].strip(), field
    # The finding-1 fix: the candidate is scored against the seed's HOLDOUT
    # half's OWN empirical rate (side A / rate_a), not the full-panel ref.
    assert "holdout" in proto["split"].lower()
    assert "own empirical rate" in proto["scored_against"].lower()
    assert "rate_a" in proto["scored_against"]
    assert ">=4 of 5" in proto["pass_rule"]
    # The OC recorded in the contract matches the artifact's recomputation.
    oc = proto["faithful_candidate_oc"]
    art_oc = _gate2_floor()["faithful_candidate_oc"]
    assert oc["p_seed_pass"] == pytest.approx(art_oc["p_seed_pass"])
    assert oc["p_gate_pass_4_of_5"] == pytest.approx(
        art_oc["p_gate_pass_4_of_5"]
    )
    assert oc["n_gated_cells"] == art_oc["n_gated_cells"]


def test_gate2_governance_inherits_gate1_and_pins_specifics():
    """Finding 5: registration on #42, the inherited no_self_rescue + version
    pin, the holdout-id commitment, the pinned scale, and the in-block weight
    definition all reach gate 2."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    gov = g2["governance"]
    assert "issue #42" in gov["registration"]
    amend = gov["amendment_rules"]
    assert amend["inherits"] == "gate_1"
    assert "committed run verdict changes" in amend["no_self_rescue"]
    assert "runs registered after its ratification" in amend["no_self_rescue"]
    assert "version" in amend["classifier_version_pin"].lower()
    assert "referee round" in amend["amendments_only_via"]
    assert "holdout" in gov["candidate_scale"].lower()
    assert "sha256" in gov["holdout_id_commitment"]
    # The holdout ids are actually committed in the floor artifact.
    floor = _gate2_floor()
    assert floor["holdout_ids"]["gate_seeds"] == [0, 1, 2, 3, 4]
    assert len(floor["holdout_ids"]["per_seed"]) == 5
    weight = gov["weight_definition"]
    assert "most-recent positive" in weight
    assert "no unweighted gated statistic" in weight.lower()


def test_gate2_scope_declares_provision_classes_and_gaps():
    """Finding 7: the provision-class coverage map + the declared
    marriage x earnings dependence gap + the caregiver/MX23REL precondition."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    cov = g2["scope"]["provision_class_coverage"]
    assert "COVERED" in cov["marital_and_survivor_timing"]
    assert "MX23REL" in cov["caregiver_and_child_in_care_survivor"]
    assert "NOT COVERED" in cov["caregiver_and_child_in_care_survivor"]
    joint = cov["marriage_x_earnings_joint"]
    assert "NOT COVERED" in joint
    assert "earnings" in joint
    assert g2["scope"]["remarriage_numerator_note"].strip()


def test_gate2_external_anchor_is_period_matched_and_decomposed():
    """Finding 6: the ASFR anchor is period-matched and the marriage/divorce
    anchor is concept-decomposed, both reported not gated."""
    g2 = _gate_2_if_draft()
    if g2 is None:
        pytest.skip("gate_2 is not draft")
    checks = g2["external_anchor_shape_checks"]
    assert "period_matched" in checks["asfr"]
    assert "concept_decomposition" in checks["marriage_divorce"]
    floor = _gate2_floor()
    md = floor["external_anchor"]["marriage_divorce"]
    assert md["concept_decomposition"]["person_to_couple_factor"] == 2.0
    assert "period_matched" in floor["external_anchor"]["asfr"]
