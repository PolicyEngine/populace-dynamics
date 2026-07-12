"""Gate-2c candidate 1: couple formation from certified components.

Registered on issue #42 comment 4950250151 (the FROZEN spec). First
candidate against the locked gate-2c contract (``gates.yaml`` gate_2.gate_2c,
LOCKED 2026-07-10): the 27 gated marriage x earnings joint cells, the K=20
mean-over-draws estimator, the per-seed COUPLE-DISJOINT train/holdout
protocol (split by ``attrs.component_id``), scored as ``|ln(rbar / rate_a)|``
against the frozen floor ``runs/gate2c_floors_v1.json``.

A couple-formation generator composed from certified pieces plus
train-fitted joints, exactly the registration's FIVE components:

1. **Marriage / remarriage events and timing** -- the certified tranche-2a
   family-transition registry (``family_transitions.registry.CANDIDATE_16``
   / ``REGISTRY``), refit per seed on side B and simulated on side A with
   ``family_transitions.simulator.simulate(mpanel, holdout_ids, components,
   5200 + k)``. The simulated ``first_marriage`` / ``remarriage`` events and
   the never-married / dissolved person-years become the earnings-conditional
   marriage-hazard denominators (tercile-tagged with the COMMITTED cuts).
2. **WHO marries whom** -- a train-fitted conditional assortative kernel
   ``P(spouse earnings-decile | own earnings-decile, own age band, own sex)``
   on the locked per-year indexed-earnings axis, emitted DIRECTED in BOTH
   orientations (ego -> spouse AND spouse -> ego) exactly as the contract's
   ``candidate_construction.couple_emission`` pins (a single-orientation
   emission is NON-CONFORMANT).
3. **Spouse age** -- the certified tranche-2a spousal-age-gap distributions
   (``components.spousal_age_gaps`` + ``draw_spousal_gaps``), drawn given own
   age x sex (the 2a machinery's existing convention). Recorded; no gated
   cell reads spouse age.
4. **Event-window earnings dynamics** -- the train-fitted around-event
   post/pre ratio distributions (side B's ``event_windows``), sampled as
   shift kernels around each simulated marriage / divorce event, weighted by
   the ego demographic weight (fix F). The gated cell detrends by the
   COMMITTED placebo deflator (fix E), so the emitted ``ratio`` is the raw
   (nominal) draw.
5. **Shared-earnings cutpoint cells** -- computed from the simulated couples'
   combined axis (own + spouse) by the locked ``reference_moments`` cell
   definitions.

Committed-cut provenance (``candidate_construction.cut_provenance``): the
earnings-axis tercile cut levels, the earnings-decile edges + within-decile
value pools, and the placebo drift deflators are all FIXED on the FULL REAL
earnings supply and applied to every seed and every draw -- never recomputed
on simulated output. Only the CONDITIONAL kernels (assortative + event
window) and the tranche-2a components are train-fitted (side B), with the
seed's holdout couples excluded from all fitting.

The simulated frames are built to the ``CoupleEarningsPanel`` schema and
scored through the LOCKED ``couple_earnings.reference_moments`` cell
machinery VERBATIM, so a candidate cell is the identical statistic the floor
measured on the real half. Reusing the reference's own construction
(``_build_marital``, ``_tercile_of``, ``reference_moments``) guarantees the
cell definitions match to bit precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import couple_earnings as ce
from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions.components.spousal_age_gap import (  # noqa: E501
    draw_spousal_gaps,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)
from populace_dynamics.models.family_transitions.registry import (
    CANDIDATE_16,
    REGISTRY,
    FitContext,
)
from populace_dynamics.models.family_transitions.simulator import simulate

#: Earnings-axis decile count for the assortative kernel (registration
#: component 2: "spouse rank decile | own rank decile"). Fixed on the full
#: real earnings supply (cut_provenance).
N_DECILES = 10

#: Hierarchical add-alpha smoothing strength for the assortative kernel
#: (effective pseudo-couples pulled toward the backoff prior). Load-bearing
#: judgment knob; pinned and recorded in the run artifact.
KERNEL_SMOOTHING_ALPHA = 5.0

#: Own-age bands for the assortative kernel conditioning -- reused from the
#: certified first-marriage hazard bands so the "own age band" matches the
#: locked marital surface (18-24, 25-34, 35-44, 45+).
MARRIAGE_AGE_BANDS = ce.FIRST_MARRIAGE_AGE_BANDS

#: Marriage transitions that create a couple (a directed couple record) and a
#: "marriage" event window; the certified simulator emits both at a marriage
#: start.
_MARRIAGE_TRANSITIONS = ("first_marriage", "remarriage")

#: The certified tranche-2a candidate spec + component registry (the "who
#: marries / when" core). Re-exported so the runner and tests can pin them.
CERTIFIED_SPEC = CANDIDATE_16
CERTIFIED_REGISTRY = REGISTRY

_SEX_INDEX = {"female": 0, "male": 1}
_OPPOSITE_SEX = {"female": "male", "male": "female"}


# --------------------------------------------------------------------------
# Committed (seed-independent) earnings axis -- frozen on the full real supply
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CommittedAxis:
    """The earnings-axis categories frozen on the FULL real supply.

    Everything here is fixed once (never per-seed, never per-draw,
    never recomputed on simulated output) -- the ``cut_provenance`` pin.

    Attributes:
        earn: ``{person_id: mean indexed positive-year earnings}`` for every
            admitted earnings-supply person (the locked per-year axis).
        cuts: the committed tercile cut levels ``(t33, t67)``.
        decile_edges: the 9 committed earnings-axis decile edges.
        supply_by_decile: per decile (0..9), the array of full-supply axis
            values in that decile -- the pool a drawn spouse value is sampled
            from.
        sex: ``{person_id: "female"/"male"}`` from the marriage records.
        birth_year: ``{person_id: implied birth year}``.
        person_weight: ``{person_id: demographic most-recent-positive PSID
            weight}`` (fix F; the gated-statistic weight).
    """

    earn: dict[int, float]
    cuts: tuple[float, float]
    decile_edges: np.ndarray
    supply_by_decile: tuple[np.ndarray, ...]
    sex: dict[int, str]
    birth_year: dict[int, int]
    person_weight: dict[int, float]


def build_committed_axis(
    ce_panel: ce.CoupleEarningsPanel,
    *,
    earnings_panel: pd.DataFrame,
    marriage_records: pd.DataFrame,
    params: Any,
    person_weight: pd.Series,
) -> CommittedAxis:
    """Build the frozen earnings axis from the same certified inputs the
    floor's :func:`couple_earnings.build_couple_panel` used.

    The tercile cuts come straight from the committed panel
    (``ce_panel.earn_tercile_cuts``); the decile edges and within-decile
    value pools are computed on the full admitted supply exactly as the
    terciles are (both are frozen categories, not learned parameters).
    """
    history, birth_year, _ = ce.person_earnings_histories(earnings_panel)
    earn = ce.indexed_earnings_supply(history, birth_year, params)

    values = np.fromiter(earn.values(), dtype=float)
    quantiles = np.arange(1, N_DECILES) / N_DECILES
    decile_edges = np.quantile(values, quantiles)
    # decile index of every supply value (0..9), same convention used at draw
    idx = np.searchsorted(decile_edges, values, side="right")
    supply_by_decile = tuple(
        np.sort(values[idx == d]) for d in range(N_DECILES)
    )

    sex = (
        marriage_records.drop_duplicates("person_id")
        .set_index("person_id")["sex"]
        .to_dict()
    )
    pw = {int(k): float(v) for k, v in person_weight.items()}
    return CommittedAxis(
        earn={int(k): float(v) for k, v in earn.items()},
        cuts=tuple(float(c) for c in ce_panel.earn_tercile_cuts),
        decile_edges=decile_edges,
        supply_by_decile=supply_by_decile,
        sex={int(k): str(v) for k, v in sex.items() if isinstance(v, str)},
        birth_year={int(k): int(v) for k, v in birth_year.items()},
        person_weight=pw,
    )


# --------------------------------------------------------------------------
# The fitted couple-formation model (train / side B)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupleFormationModelV1:
    """The per-seed side-B (train) fit of all five components.

    Attributes:
        components: the certified tranche-2a ``FittedFamilyTransitions``
            (marriage / remarriage / divorce events, spousal age gaps),
            refit on the seed's train half.
        assort_kernel: ``[sex(2), age_band(4), own_decile(10),
            spouse_decile(10)]`` conditional probabilities (hierarchically
            smoothed with backoff), the train-fitted assortative joint.
        event_window_pools: ``{(event_type, sex): (ratios, weights)}`` the
            train around-event post/pre ratio pools sampled as shift kernels.
        train_ids: the side-B person ids all components were fit on.
        meta: fit diagnostics recorded in the run artifact.
    """

    components: FittedFamilyTransitions
    assort_kernel: np.ndarray
    event_window_pools: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    train_ids: frozenset[int]
    meta: dict[str, Any]


def _age_band_index(ages: np.ndarray) -> np.ndarray:
    """Index (0..3) into :data:`MARRIAGE_AGE_BANDS` for each age (clipped)."""
    lowers = np.array([lo for lo, _ in MARRIAGE_AGE_BANDS])
    out = np.searchsorted(lowers, ages, side="right") - 1
    return np.clip(out, 0, len(MARRIAGE_AGE_BANDS) - 1)


def _decile_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Committed earnings-axis decile (0..9) of each value."""
    return np.clip(
        np.searchsorted(edges, values, side="right"), 0, N_DECILES - 1
    )


def _weighted_decile_counts(
    own_dec: np.ndarray,
    spouse_dec: np.ndarray,
    sex_idx: np.ndarray,
    band_idx: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Weighted couple counts ``[sex, band, own_dec, spouse_dec]``."""
    counts = np.zeros((2, len(MARRIAGE_AGE_BANDS), N_DECILES, N_DECILES))
    np.add.at(counts, (sex_idx, band_idx, own_dec, spouse_dec), weight)
    return counts


def _smooth(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Add-alpha shrink of a count vector toward a normalized prior."""
    total = counts.sum()
    return (counts + alpha * prior) / (total + alpha)


def _fit_assortative_kernel(
    train_couples: pd.DataFrame,
    axis: CommittedAxis,
    alpha: float,
) -> np.ndarray:
    """Hierarchical ``P(spouse_decile | own_decile, age_band, sex)`` with
    backoff (global -> sex -> sex x own_decile -> full), fit on the side-B
    directed couples on the committed decile axis.

    The backoff makes every conditioning cell well-defined despite the sparse
    per-(decile, band, sex) couple counts; the finest level shrinks toward
    the coarser prior with :data:`KERNEL_SMOOTHING_ALPHA` pseudo-couples.
    """
    own = train_couples["earn_own"].to_numpy(dtype=float)
    spouse = train_couples["earn_spouse"].to_numpy(dtype=float)
    sexes = train_couples["sex"].to_numpy()
    ages = train_couples["ego_age_at_marriage"].to_numpy(dtype=float)
    weight = train_couples["weight"].to_numpy(dtype=float)

    own_dec = _decile_index(own, axis.decile_edges)
    spouse_dec = _decile_index(spouse, axis.decile_edges)
    sex_idx = np.array([_SEX_INDEX[s] for s in sexes])
    band_idx = _age_band_index(ages)

    counts = _weighted_decile_counts(
        own_dec, spouse_dec, sex_idx, band_idx, weight
    )
    n_bands = len(MARRIAGE_AGE_BANDS)

    # level 0: global spouse-decile marginal (uniform prior at the root).
    global_counts = counts.sum(axis=(0, 1, 2))
    uniform = np.full(N_DECILES, 1.0 / N_DECILES)
    level0 = _smooth(global_counts, uniform, alpha)

    kernel = np.empty((2, n_bands, N_DECILES, N_DECILES))
    for s in range(2):
        sex_counts = counts[s].sum(axis=(0, 1))
        level1 = _smooth(sex_counts, level0, alpha)
        for d in range(N_DECILES):
            sd_counts = counts[s, :, d, :].sum(axis=0)
            level2 = _smooth(sd_counts, level1, alpha)
            for b in range(n_bands):
                kernel[s, b, d] = _smooth(counts[s, b, d], level2, alpha)
    return kernel


def _fit_event_window_pools(
    train_windows: pd.DataFrame,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """The side-B around-event post/pre RAW ratio pools, keyed
    ``(event_type, sex)`` -- sampled (weighted) as the shift kernel.

    The pools are the raw (nominal) ratios; the gated cell detrends by the
    COMMITTED placebo deflator, so sampling from the raw pool and detrending
    with the frozen deflator reproduces the event increment.
    """
    pools: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for event_type in ("marriage", "divorce"):
        sub = train_windows[train_windows["event_type"] == event_type]
        for sex in ce.SEXES:
            g = sub[sub["sex"] == sex]
            pools[(event_type, sex)] = (
                g["ratio"].to_numpy(dtype=float),
                g["weight"].to_numpy(dtype=float),
            )
    return pools


def _couples_with_age(
    couples: pd.DataFrame, birth_year: dict[int, float]
) -> pd.DataFrame:
    """Side-B directed couples with the ego age at marriage attached."""
    out = couples.copy()
    by = out["person_id"].map(birth_year)
    out["ego_age_at_marriage"] = out["start_year"].astype(float) - by
    return out[out["ego_age_at_marriage"].notna()].copy()


def fit_couple_model_v1(
    ce_panel: ce.CoupleEarningsPanel,
    mpanel: transitions.MaritalPanel,
    *,
    demographic_panel: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    axis: CommittedAxis,
    train_ids: set[int],
    alpha: float = KERNEL_SMOOTHING_ALPHA,
) -> CoupleFormationModelV1:
    """Refit all five components on the seed's train half (side B).

    The certified tranche-2a registry is fit on the intersection of the train
    ids with the marital panel's valid persons; the assortative and
    event-window kernels are fit on the side-B directed couples / event
    windows only, so the seed's holdout couples are excluded from every fit.
    """
    valid_ids = set(int(v) for v in mpanel.attrs["person_id"])
    train_valid = frozenset(train_ids & valid_ids)
    context = FitContext(
        panel=mpanel,
        demographic_panel=demographic_panel,
        marriage_records=marriage_records,
        birth_records=birth_records,
        marriage_order_map=marriage_order_map,
        train_ids=train_valid,
    )
    components = REGISTRY.fit(CERTIFIED_SPEC, context)

    couples = ce_panel.couples
    train_couples = _couples_with_age(
        couples[couples["person_id"].isin(train_ids)], axis.birth_year
    )
    kernel = _fit_assortative_kernel(train_couples, axis, alpha)

    windows = ce_panel.event_windows
    train_windows = windows[windows["person_id"].isin(train_ids)]
    pools = _fit_event_window_pools(train_windows)

    meta = {
        "n_train_ids": len(train_ids),
        "n_train_ids_in_marital_panel": len(train_valid),
        "n_train_directed_couples": int(len(train_couples)),
        "n_train_event_windows": int(len(train_windows)),
        "kernel_smoothing_alpha": alpha,
        "certified_spec_sha256": CERTIFIED_SPEC.sha256,
        "component_implementation_ids": dict(components.implementation_ids),
    }
    return CoupleFormationModelV1(
        components=components,
        assort_kernel=kernel,
        event_window_pools=pools,
        train_ids=frozenset(train_ids),
        meta=meta,
    )


# --------------------------------------------------------------------------
# Simulate one draw (holdout / side A)
# --------------------------------------------------------------------------
def _draw_spouse_deciles(
    kernel: np.ndarray,
    sex_idx: np.ndarray,
    band_idx: np.ndarray,
    own_dec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized inverse-CDF draw of the spouse decile per marriage."""
    probs = kernel[sex_idx, band_idx, own_dec]
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(len(own_dec))
    return np.clip((cdf < u[:, None]).sum(axis=1), 0, N_DECILES - 1)


def _draw_spouse_values(
    spouse_dec: np.ndarray,
    supply_by_decile: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a spouse earnings-axis value from the committed within-decile
    supply pool for each drawn spouse decile."""
    out = np.empty(len(spouse_dec), dtype=float)
    for d in range(N_DECILES):
        mask = spouse_dec == d
        n = int(mask.sum())
        if n == 0:
            continue
        pool = supply_by_decile[d]
        if len(pool) == 0:  # empty decile: fall back to the global supply
            pool = np.concatenate([p for p in supply_by_decile if len(p)])
        out[mask] = pool[rng.integers(0, len(pool), size=n)]
    return out


def _tercile_array(
    values: np.ndarray, cuts: tuple[float, float]
) -> np.ndarray:
    """Committed tercile (1/2/3) of each value (vectorized ``_tercile_of``)."""
    return (
        1 + (values >= cuts[0]).astype(int) + (values >= cuts[1]).astype(int)
    )


def _weighted_sample(
    ratios: np.ndarray,
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weighted-with-replacement sample of ``n`` ratios from a pool."""
    if len(ratios) == 0 or n == 0:
        return np.ones(n, dtype=float)
    total = weights.sum()
    if total <= 0:
        idx = rng.integers(0, len(ratios), size=n)
        return ratios[idx]
    cdf = np.cumsum(weights) / total
    u = rng.random(n)
    idx = np.clip(np.searchsorted(cdf, u, side="left"), 0, len(ratios) - 1)
    return ratios[idx]


def _build_simulated_couples(
    sim_panel: transitions.MaritalPanel,
    model: CoupleFormationModelV1,
    axis: CommittedAxis,
    rng_dec: np.random.Generator,
    rng_val: np.random.Generator,
    rng_gap: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Directed both-orientation couples for the simulated marriages.

    For each simulated marriage (first_marriage or remarriage) of a supply
    ego, draw the spouse earnings decile from the assortative kernel, a
    spouse axis value from the committed within-decile pool, and the spouse
    age gap from the certified 2a distributions. Emit BOTH orientations
    (ego -> spouse AND spouse -> ego), the contract's pinned symmetrized
    directed emission; assign terciles with the COMMITTED cuts.
    """
    events = sim_panel.events
    ev = events[events["transition"].isin(_MARRIAGE_TRANSITIONS)].copy()
    ev = ev[ev["person_id"].isin(axis.earn)]
    ev = ev.sort_values(["person_id", "year", "transition"]).reset_index(
        drop=True
    )
    if len(ev) == 0:
        empty = pd.DataFrame(
            columns=[
                "person_id",
                "spouse_person_id",
                "sex",
                "spouse_sex",
                "weight",
                "earn_own",
                "earn_spouse",
                "own_tercile",
                "spouse_tercile",
                "shared",
                "start_year",
                "start_decade",
            ]
        )
        return empty, {"n_marriages": 0, "spouse_age_gap_mean": None}

    ego = ev["person_id"].to_numpy()
    ego_sex = ev["sex"].to_numpy()
    ego_age = ev["age"].to_numpy(dtype=float)
    start_year = ev["year"].to_numpy()
    earn_own = np.array([axis.earn[int(p)] for p in ego], dtype=float)
    weight = np.array(
        [axis.person_weight.get(int(p), 0.0) for p in ego], dtype=float
    )
    sex_idx = np.array([_SEX_INDEX[s] for s in ego_sex])
    is_male = (sex_idx == 1).astype(float)
    band_idx = _age_band_index(ego_age)
    own_dec = _decile_index(earn_own, axis.decile_edges)

    spouse_dec = _draw_spouse_deciles(
        model.assort_kernel, sex_idx, band_idx, own_dec, rng_dec
    )
    earn_spouse = _draw_spouse_values(
        spouse_dec, axis.supply_by_decile, rng_val
    )

    # Component 3: spouse age from the certified 2a age-gap distributions
    # (gap = ego_birth - spouse_birth ~ ego_age - spouse_age); recorded only.
    indices = np.arange(len(ego))
    gaps = draw_spousal_gaps(
        rng_gap, indices, ego_age, is_male, model.components.spousal_age_gaps
    )
    spouse_age = ego_age + gaps

    own_terc = _tercile_array(earn_own, axis.cuts)
    spouse_terc = _tercile_array(earn_spouse, axis.cuts)
    shared = earn_own + earn_spouse
    start_decade = (start_year // 10 * 10).astype(int)
    spouse_sex = np.array([_OPPOSITE_SEX[s] for s in ego_sex])
    synth_id = -(np.arange(len(ego)) + 1)

    forward = pd.DataFrame(
        {
            "person_id": ego.astype(int),
            "spouse_person_id": synth_id,
            "sex": ego_sex,
            "spouse_sex": spouse_sex,
            "weight": weight,
            "earn_own": earn_own,
            "earn_spouse": earn_spouse,
            "own_tercile": own_terc,
            "spouse_tercile": spouse_terc,
            "shared": shared,
            "start_year": start_year.astype(int),
            "start_decade": start_decade,
        }
    )
    # Mirror orientation: the drawn spouse as ego, the real ego as spouse,
    # weighted by the same couple (ego) weight -- an exactly symmetrized
    # directed contingency (the contract's both-orientation emission).
    mirror = pd.DataFrame(
        {
            "person_id": synth_id,
            "spouse_person_id": ego.astype(int),
            "sex": spouse_sex,
            "spouse_sex": ego_sex,
            "weight": weight,
            "earn_own": earn_spouse,
            "earn_spouse": earn_own,
            "own_tercile": spouse_terc,
            "spouse_tercile": own_terc,
            "shared": shared,
            "start_year": start_year.astype(int),
            "start_decade": start_decade,
        }
    )
    couples = pd.concat([forward, mirror], ignore_index=True)
    diag = {
        "n_marriages": int(len(ego)),
        "n_directed_couples": int(len(couples)),
        "spouse_age_gap_mean": float(np.mean(gaps)) if len(gaps) else None,
        "spouse_age_mean": float(np.mean(spouse_age)) if len(gaps) else None,
    }
    return couples, diag


def _build_simulated_event_windows(
    sim_panel: transitions.MaritalPanel,
    model: CoupleFormationModelV1,
    axis: CommittedAxis,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Around-event post/pre ratios for the simulated marriage / divorce
    events, sampled from the train shift kernel, weighted by the ego
    demographic weight (fix F)."""
    events = sim_panel.events
    plans = (
        ("marriage", events["transition"].isin(_MARRIAGE_TRANSITIONS)),
        ("divorce", events["transition"] == "divorce"),
    )
    rows: list[pd.DataFrame] = []
    for event_type, mask in plans:
        ev = events[mask]
        ev = ev[ev["person_id"].isin(axis.earn)]
        for sex in ce.SEXES:
            g = ev[ev["sex"] == sex]
            if len(g) == 0:
                continue
            pool_r, pool_w = model.event_window_pools[(event_type, sex)]
            ratios = _weighted_sample(pool_r, pool_w, len(g), rng)
            pid = g["person_id"].to_numpy().astype(int)
            rows.append(
                pd.DataFrame(
                    {
                        "person_id": pid,
                        "sex": sex,
                        "weight": [
                            axis.person_weight.get(int(p), 0.0) for p in pid
                        ],
                        "event_type": event_type,
                        "ratio": ratios,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=["person_id", "sex", "weight", "event_type", "ratio"]
        )
    frame = pd.concat(rows, ignore_index=True)
    return frame[frame["weight"] > 0].reset_index(drop=True)


def simulate_draw_v1(
    ce_panel: ce.CoupleEarningsPanel,
    mpanel: transitions.MaritalPanel,
    model: CoupleFormationModelV1,
    axis: CommittedAxis,
    holdout_ids: set[int],
    draw_seed: int,
) -> tuple[ce.CoupleEarningsPanel, dict[str, Any]]:
    """Simulate the seed's holdout (side A) couples at draw ``draw_seed``.

    Returns a :class:`CoupleEarningsPanel` restricted to the holdout egos,
    carrying the COMMITTED tercile cuts and placebo deflators, plus a
    diagnostics dict. Score it with
    ``couple_earnings.reference_moments(sim_panel, weighted=True)`` (person
    ids ``None``: the frames are already holdout-scoped, and passing the
    holdout id set would drop the both-orientation mirror records whose
    synthetic ego ids are not holdout persons).

    RNG topology (registration: ``5200 + k`` draw stream): the certified
    tranche-2a ``simulate`` consumes ``default_rng(draw_seed)`` (its own
    main + spawned gap streams); the candidate joints draw from four
    independent children of ``SeedSequence([draw_seed, 0x2C1])`` (spouse
    decile, spouse value, spouse age gap, event window), so the joints never
    share draws with the certified marital core.
    """
    valid = set(int(v) for v in mpanel.attrs["person_id"])
    sim_ids = holdout_ids & valid
    sim_panel, _sim_births = simulate(
        mpanel, sim_ids, model.components, draw_seed
    )

    seeds = np.random.SeedSequence([draw_seed, 0x2C1]).spawn(4)
    rng_dec = np.random.default_rng(seeds[0])
    rng_val = np.random.default_rng(seeds[1])
    rng_gap = np.random.default_rng(seeds[2])
    rng_ewin = np.random.default_rng(seeds[3])

    marital_events, marital_exposure = ce._build_marital(
        sim_panel, axis.earn, axis.cuts
    )
    couples, couple_diag = _build_simulated_couples(
        sim_panel, model, axis, rng_dec, rng_val, rng_gap
    )
    event_windows = _build_simulated_event_windows(
        sim_panel, model, axis, rng_ewin
    )

    sim_ce_panel = ce.CoupleEarningsPanel(
        couples=couples,
        marital_events=marital_events,
        marital_exposure=marital_exposure,
        event_windows=event_windows,
        attrs=ce_panel.attrs,
        earn_tercile_cuts=axis.cuts,
        placebo_deflators=ce_panel.placebo_deflators,
        meta={},
    )
    diag = {
        "draw_seed": draw_seed,
        "n_sim_holdout_persons": len(sim_ids),
        "n_marital_events": int(len(marital_events)),
        "n_marital_exposure_person_years": int(len(marital_exposure)),
        "n_event_windows": int(len(event_windows)),
        **couple_diag,
    }
    return sim_ce_panel, diag
