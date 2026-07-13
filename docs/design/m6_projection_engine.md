# M6 projection engine — design and temporal-holdout gate proposal

- **Design id**: `2026-07-12-m6-projection-engine`
- **Roadmap**: #113 M6 (the projection engine), the last build before #113 M7
  (trust-fund accounting) and #113 M8 (integrated scoring). Workstreams #100
  (W1/W2/W3 seams), ADR-0001 (`docs/adr/0001-populace-axiom-seam-ownership.md`).
- **Status**: DESIGN (draft). This document seeds the `gate_m6` lock ceremony
  (design review → floor build → adversarial referee → verification →
  ratify-by-merge → lock). **It edits no `gates.yaml` cell, moves no threshold,
  builds no floor, and writes no test.** The gate block and its floor are
  authored in a later ceremony, not here.
- **Evidence base cited by path+field**: the roadmap (#113); the locked contract
  `gates.yaml` (`holdout_protocol`, `noise_floor`, `views`, `moment_battery`,
  `gates.gate_m4`, `gates.gate_w1`); the module registries
  (`src/populace_dynamics/models/family_transitions/registry.py`,
  `src/populace_dynamics/models/household_composition/registry.py`); the M4
  candidate (`src/populace_dynamics/models/disability_hazard_sim.py`); the W1
  transport candidates (`src/populace_dynamics/models/transport_deployment_v1.py`,
  `transport_deployment_v2.py`); the W2 seam (`scripts/w2_seam_caregiver.py`,
  artifact `runs/w2_seam_caregiver_v1.json`); the M2 pseudo-projection
  (`runs/m2_pseudo_projection_v1.json`, `scripts/m2_pseudo_projection.py`); the
  contract loader (`src/populace_dynamics/evaluation.py`,
  `contract.py`, `artifacts.py`); the W1 amendment
  (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`); the standing
  process addendum (#42 comment 4948637741).

## 1. Summary

M6 composes the M1–M5 certified generators into a year-by-year forward
simulation on the unified person-period panel (#113: populace persons × years,
future births and immigrant entrants as synthetic persons entering the panel).
It fits nothing new except a versioned alignment layer; its certification move is
a **temporal holdout** — fit the dynamics through year `T*`, project the held-out
window `T*+1 … T_end`, and score projected observables against their realized
PSID values under the same locked machinery every prior gate uses
(`gates.yaml` `noise_floor` half-split floors; K=20-draw mean estimator on
`numpy.random.default_rng(5200 + k)`; tolerance `round(mean + k·sd, 3)` capped at
`ln(1.5)`; seeds `0–4` with a 4-of-5 conjunction; the one-shot rule enforced by
`artifacts.write_new`).

The design record below covers the engine architecture (§2), the RNG stream
registry that extends the `5200 + k` convention to the period axis (§3), the
`gate_m6` temporal-holdout proposal with cell-level observable definitions (§4),
the per-year PolicyEngine seam (§5, the W2 pattern), the M7 interface the engine
must expose (§6, interface only), the explicit non-goals (§7), and the numbered
open decisions for adjudication before the ceremony (§8).

The one architectural claim this document defends, and the one it declines to
make: the temporal holdout is **PSID-projected vs PSID-realized**, so — unlike
`gate_w1` family B (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`) —
it gates a level match on **one instrument against itself** and needs no concept
bridge. It does **not** claim the 2100 production projection is validated: only
the `T*+1 … T_end` window is gated; everything past the last realized wave is
report-only extrapolation under the domains-of-validity doctrine (§4.8, §7).

## 2. Engine architecture

### 2.1 The unified person-period panel

The canonical object is the #113 person-period panel: one row per
`(person_id, year)`, with the full variable surface as column groups (earnings,
marital/roster, disability/claiming state, AIME/PIA, per-year calibrated weight).
Every product is a slice — the current populace cross-section is one year-slice,
the DYNASIM-class projection is the whole panel. M6 is the engine that writes the
future year-slices. It does not own the panel schema (a populace-side decision
that lands with M5, per #113); it consumes the pinned M5 frame as year 0 and
appends `T` future year-slices.

Two population strata coexist in the panel and must be kept distinct because they
have different ground truth:

- the **closed panel** — persons present at year 0, projected forward from their
  realized year-0 state. These have realized PSID futures over the holdout window
  and are the gated surface (§4).
- the **open additions** — synthetic births (from the fertility/roster module)
  and immigrant entry cohorts, materialized as new person-rows entering in their
  entry year. These have no PSID ground truth over the holdout window and are
  therefore report-only in `gate_m6` (§4.7, decision 4).

### 2.2 The wave loop and the per-period order of operations

The engine is a loop over projection years `t = 1 … T`. Within each period the
modules compose in a fixed order of operations, chosen so each module reads only
predetermined state (the year `t-1` panel plus already-resolved year-`t`
sub-states):

1. **mortality** — draw deaths for the surviving `t-1` population from the
   differential-mortality component (`data/deaths.py` basis; NCHS × PSID-band
   hazards). Decedents leave the risk set for all subsequent year-`t` steps.
2. **aging** — advance age by one year and roll calendar state (indexing series,
   wage base) for survivors.
3. **couple formation / dissolution** — first marriage, divorce, remarriage,
   widowhood, spousal-age-gap assignment, from the `family_transitions` registry
   (`CANDIDATE_16`). Widowhood interacts with step 1 (see §2.5, friction F1).
4. **fertility / roster** — births from the fertility component; new child
   person-rows attach to mothers; the roster (household membership) updates.
5. **disability** — work-limitation incidence / recovery and the near-FRA
   disability-exit split, from the M4 hazard machinery
   (`disability_hazard_sim.py`; `disability_conversion.py` for the FRA
   auto-conversion, factor `1.0`, 42 USC 402(a)/416(i)(2)(D)).
6. **earnings** — draw the year-`t` earnings increment from the gate-1-certified
   earnings generator, conditioned on the resolved demographic state.
7. **claiming** — draw claim age from the calibrated 6.B5.1 distribution
   (`claiming.claim_age_pmf(exclude_conversions=True)`), with DI→retirement
   conversions supplied by step 5, not the sampler (`disability_conversion.py`:
   the two halves partition the entrant population).
8. **household composition reconciliation** — resolve cohabitation, legal-spouse
   residual, parental-home exit, multigenerational and skip-generation occupancy,
   non-family bridge, and household size from the `household_composition` registry
   (`CANDIDATE_9`), reconciling the roster into scored household units.

The order encodes a directed acyclic graph with exactly one deliberately lagged
edge (earnings ⇄ marriage; §2.5 friction F4, decision 6). Deaths first so no
dead person is married, made disabled, or paid; earnings after the demographic
state resolves so the certified earnings conditioning set (age × sex × cohort,
and — under `couple_formation_sim_v2` — the marital state) is available; household
composition last because it is a reconciliation over the fully resolved roster,
not a generator of primitive state.

### 2.3 The module registries as the interface

The two flattened registries are the engine's module interface. Each is an
immutable `ComponentRegistry` of `ComponentDefinition`s resolved by
`(kind, implementation_id)` with frozen `params`, fitted in declared dependency
order:

- `family_transitions/registry.py` — `CANDIDATE_16`, `contract_revision
  gate_2_amendment_1`; seven kinds in fit order `initial_states → first_marriage
  → divorce → widowhood → remarriage → fertility → spousal_age_gap`
  (`ComponentRegistry.fit` enforces the exact required-kind set).
- `household_composition/registry.py` — `CANDIDATE_9`, `contract_revision
  gate_2b_locked`; ten kinds (`marital_core_adapter`, `cohabitation_overlay`,
  `legal_spouse_residual`, `parental_home_exit`, `multigenerational_occupancy`,
  `child_attribution`, `skip_generation_state`, `nonfamily_bridge`,
  `household_size`, `fertility_core_lift`).

The engine binds to registries, not to concrete simulators: a `gate_m6` run
pins the resolved `CandidateSpec.sha256` of each registry it composes, exactly as
the gate-2 runs pin `CANDIDATE_16` / `CANDIDATE_9`. This is the composition
interface — the engine is a *scheduler over registries across periods*, and the
per-period DAG (§2.2) is the schedule.

### 2.4 Where the W1 transport layer sits

W1 transport (`transport_deployment_v1.py` / `v2.py`, `gate_w1`) is **period 0**
of the projection, not a separate stage. Deployment places the certified
PSID-estimated generators onto the pinned populace frame (`gate_w1`
`deployment_frame`: bundle `us-4.18.8`, dataset `populace_us_2024`, artifact sha
`c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5`, `n_persons
166302`, `n_households 57240`) and seeds each frame adult's entry state
(`transport_deployment_v2` Q1: `P(marital state | entry-age band, sex)` at
`BASE_ENTRY_AGE`, evolved by the `CANDIDATE_16` hazards). M6 takes W1's deployed
year-0 panel as its initial condition and runs the §2.2 loop forward from it.

Two distinct initial conditions therefore exist, and the gate must not conflate
them (decision 5):

- the **deployment initial condition** (W1) — re-drawn entry state on the CPS
  frame, used for the production projection. Its year-0 fidelity is `gate_w1`'s
  burden, not `gate_m6`'s.
- the **holdout initial condition** (`gate_m6` family A) — each PSID person's
  *realized* year-`T*` state, used so the temporal holdout tests the forward
  dynamics in isolation from any transport error.

### 2.5 Registry frictions against the composition order

The registries were sealed for single-period gates; the period axis and the
mortality-linked composition order expose four concrete resistances the engine
must resolve. These are reported, not hidden.

- **F1 — endogenous vs exogenous widowhood.** `family_transitions/registry.py`
  fits `widowhood` as a standalone PSID hazard
  (`mh85_23_age7_sex_support75_untrended.v1`, `_WIDOWHOOD_PARAMS`) that reads
  `initial_states.support.by_person`, **not** a simulated spouse death. In the
  §2.2 order, mortality (step 1) already kills spouses; if the fitted widowhood
  hazard (step 3) also transitions survivors to widowed, widowhood is either
  double-counted or must be reconciled against the mortality-induced widowings.
  The certified object is the exogenous hazard; an endogenous "widow when your
  linked spouse dies" rule is a different object the registry does not expose.
  Resolution (decision 5, F1 leg): the gated closed-panel test uses the certified
  exogenous hazard (what `gate_2a` certified); the open production projection with
  linked couples needs an endogenous-widowhood reconciliation that is out of
  M6-certified scope and named as such.

- **F2 — household composition embeds a second marital core.** The
  `household_composition` registry's `marital_core_adapter` component
  (`_fit_marital`) calls `fit_family_transitions(...)` internally
  (`registry.py:251-261`). Household composition is therefore **not** a pure
  downstream consumer of the `family_transitions` registry — it wraps its own
  copy. A naive pipeline that runs `family_transitions` (step 3) and then feeds
  its state into `household_composition` (step 8) would fit and simulate the
  marital core twice, from two RNG draws, producing inconsistent marital state.
  Resolution: the engine treats the `household_composition` embedded marital core
  as the single source of truth for the scored household units, or injects the
  step-3 fitted core into `marital_core_adapter`; it must not run two independent
  marital simulations per period. This is an interface seam the M6 build has to
  add; the registries do not currently expose the injection point.

- **F3 — frozen `core_seed` has no period axis.** `household_composition`
  components seed from frozen params: `_LEGAL_PARAMS = _params(core_seed=5200,
  …)`, and cohabitation / skip-generation / child-attribution components spawn
  child streams keyed by fixed tags (`SeedSequence([5200 + k, 0xC2/0xC5/0xC7])`,
  `_CHILD_PARAMS` `persistence_fit_seed=0xC7`). A component that constructs
  `default_rng(core_seed)` internally will draw the **same** numbers every
  projection period, because the period is nowhere in its entropy. But the params
  are frozen (`MappingProxyType`) and the registry rejects any mismatch
  (`ComponentRegistry.definition` raises when `reference.params !=
  definition.params`), so the period cannot be folded into `core_seed` without
  breaking the frozen-candidate contract. Resolution (§3): thread the period at
  the **engine** level — spawn the per-`(module, period)` generator outside the
  component and inject it, rather than letting the component build its RNG from
  the frozen `core_seed`. Components that construct their own RNG internally
  resist this and must be adapted to accept an injected generator; this is the
  single largest registry-side lift for M6.

- **F4 — the earnings ⇄ marriage within-period cycle.**
  `couple_formation_sim_v2` conditions first-marriage timing on earnings terciles
  (`m(tercile | age band, sex)`), while household composition and the scored
  household units depend on marital state, and earnings (step 6) is drawn after
  marital state (step 3). Taken simultaneously these form a within-period cycle.
  Resolution (decision 6): break it with one lagged edge — couple formation at
  period `t` reads year `t-1` earnings (predetermined), earnings at `t` reads year
  `t` marital state. This matches how `couple_formation_sim_v2` reads a *carried*
  earnings axis rather than a simultaneously-drawn one, and preserves the DAG.

- **F5 — the M4 simulator preserves observed support.**
  `disability_hazard_sim.py` simulates "on the HOLDOUT half, side A, preserving
  the observed *support* — the same persons, waves, ages, sexes and START-WAVE
  weights — and re-drawing only the disability/retirement STATE." A forward
  projection must instead **synthesize** support (advance ages/waves for a
  population that has no future waves on file). The certified M4 candidate is a
  holdout-*reproduction* simulator, not a forward-*projection* simulator. The
  gated closed-panel temporal holdout (§4) can reuse the reproduction mode over
  the realized post-`T*` waves; the open production projection needs a forward
  variant. Decision 5 owns this split.

## 3. RNG discipline — the projection stream registry

### 3.1 The inherited convention

The stack scores K=20 draws of each candidate on `numpy.random.default_rng(5200 +
k)`, `k = 0…19` (`family_transitions/evaluation.py` `DRAW_SEED_BASE = 5200`;
`household_composition/evaluation.py` same; `gates.gate_m4.protocol`
`draw_stream_base: 5200`, `candidate_draw_stream: numpy.random.default_rng(5200
+ k)`). The estimator is the mean over the K draws, scored **once** as
`|ln(rbar / rate_a)|`, not the mean of per-draw scores (`evaluation.py`
`_load_scoring`; `gate_m4` `estimator`). Components spawn disjoint child streams
by tag: `SeedSequence([5200 + k, 0xC2])` (cohabitation), `0xC5`, `0xC7`, occupancy
tag `0xB2B` (`household_composition_sim_v2.py`). The simulation stream is kept
**distinct** from the split stream (`gate_m4` finding 2 "restored 5200+k from the
drifted 4100+k"; `gate_w1` candidate streams `9100`/`9200` "distinct from the
split seeds").

### 3.2 The (draw × module × period × person) spawn tree

M6 adds a period axis. The requirement is that every module × period × person
draw be individually addressable and bit-reproducible, so any single
module-period can be re-simulated without re-running the whole projection, and so
the K=20 ensemble draws stay mutually independent. The proposed registry is a
hierarchical `SeedSequence` spawn tree rooted at the inherited base:

```
root(k)              = SeedSequence([5200 + k, GATE_M6_TAG])
period(k, t)         = root(k).spawn(T+1)[t]          # ordered child per year t = 0..T
stream(k, t, module) = SeedSequence(entropy=period(k,t).entropy,
                                    spawn_key=(MODULE_TAG,))   # or period(k,t).spawn per module in fixed order
generator(k,t,module)= default_rng(stream(k, t, module))
```

`GATE_M6_TAG` is a fixed engine constant distinct from any component tag;
`MODULE_TAG` is a stable per-module constant (mortality, aging, couple_formation,
fertility, disability, earnings, claiming, household_composition — the §2.2
order), reusing the existing component child-tags (`0xC2`/`0xC5`/`0xC7`/`0xB2B`)
where a module already owns one so the single-period draws remain byte-identical
at `t = 0`. Per-person draws are consumed from `generator(k, t, module)` in
canonical `person_id`-sorted order, so person `i`'s draw at `(module, period)` is
a pure function of `(k, t, module, i)`.

### 3.3 Disjointness guarantees and the separate split stream

The spawn tree gives three disjointness properties by construction:

- **across draws** — distinct `k` root the tree at distinct entropy, so the K=20
  ensemble members are independent (the property the K-draw mean relies on).
- **across periods** — distinct `t` are distinct ordered children of one root, so
  re-running period `t` reproduces it exactly and no two periods share entropy
  (this is the fix for friction F3: the period lives in the tree, not in a frozen
  `core_seed`).
- **across modules** — distinct `MODULE_TAG` within a period are disjoint, so
  mortality's draws never collide with earnings' draws in the same year.

The **split stream stays separate** from this entire simulation tree. The
temporal fit/holdout boundary is a *time* cut, not a person split; but the
`gate_m6` **floor** (§4.6) still draws person/couple/household-disjoint half-splits
of the holdout window to price sampling noise, and those half-split seeds use
their own base (the `4xxx`-style split base the prior gates use), disjoint from
`5200 + k`. This preserves the `gate_m4` finding-2 discipline — the estimator
stream and the floor-split stream never coincide.

### 3.4 K-draw ensembles at the engine level

`gate_m6` reuses K=20 and `5200 + k` verbatim (§4.7, decision 7), so the
draw-noise-free half-normal OC basis the prior gates use
(`gate_2a.faithful_candidate_oc` `basis_note`) transfers unchanged: the 20-draw
mean approximates the draw-noise-free expected rate, the OC is computed on floor
σ only. The one new consideration is that projection variance **compounds over
periods** (a period-`t` draw feeds period `t+1`'s state), so the across-draw sd
of a late-window cell is larger than a single-period cell's. This is priced
correctly because the floor and the candidate use the *same* K and the *same*
holdout window — the compounding is in both the reference and the estimate — but
it is the reason a larger K is a live alternative (decision 7).

## 4. The temporal-holdout gate (`gate_m6`, proposed)

### 4.1 The fit/holdout split

**Proposed boundary `T* = 2014`: fit on all PSID observations dated `≤ 2014`;
project and score the held-out window `2015–2022`.**

Wave facts that constrain the split (verified against the data readers and
`gates.yaml` `views`):

- earnings — `data/family.py` head/spouse labor income, reference years
  1968–2022, **annual through 1997, biennial from 1999**. The gate-1 earnings
  view (`views.psid_family_earnings_pairs`) is `period_range [1998, 2022]`,
  `period_step 2`, `window 2` — biennial waves 1998, 2000, …, 2022 (13 waves).
- marital / fertility — `mh85_23` / `cah85_23`, collection waves 1985–2023,
  transitions reconstructed **annually** (`data/transitions.py` `MAX_YEAR =
  2023`).
- mortality — PSID death file 1968–2023 (`data/deaths.py`, `ER32050`).

At `T* = 2014` the fit uses earnings waves {1998…2014} (9 biennial waves) and the
holdout uses {2016, 2018, 2020, 2022} (4 biennial waves); demographic and
mortality flows are annual over 2015–2022 (8 years). Note that **2015 is not an
earnings wave** (biennial, even years), so "project 2015–2022" means the
2016/2018/2020/2022 earnings waves and the annual 2015–2022 demographic/mortality
years.

Justification:

- an **8-year horizon** is long enough that projection drift — the thing M6 adds
  over the M1–M5 single-period generators — actually accumulates and becomes
  visible, and short enough that the youngest fit cohorts still carry holdout
  exposure. It matches the `T+1 … T+k` language of the #113 M6 gate row.
- it leaves **4 biennial earnings holdout waves** and **8 annual
  demographic/mortality holdout years** — enough exposure for a
  deployment-scale half-split floor (§4.6) to be non-degenerate.
- it keeps the fit window entirely within the biennial-from-1999 regime, so the
  fit does not straddle the annual/biennial `period_step` change.

Named cost, owned not softened: the **2020–2022 COVID/inflation shock lands
inside the holdout** and the engine has no macro-feedback channel (§7). Holdout
earnings-moment drift in those waves partly scores a shock the mechanical engine
cannot represent. This is a domain-of-validity disclosure and the substance of
decision 2 (report-only the 2020–2022 earnings-moment cells, or end the holdout
at 2018).

### 4.2 Prohibited re-estimation (temporal seed-disjointness)

The discipline that makes the holdout honest is **temporal disjointness**, not
person disjointness. Persons overlap fit and holdout by design — the projection's
legitimate input is "today's" realized cross-section, and the test is whether the
fitted dynamics reproduce those same persons' realized futures. What is
prohibited is any post-`T*` information entering any fitter:

- no hazard, initial-state model, spousal-gap table, fertility kernel, disability
  rate, or claiming distribution may be fit, refit, tuned, knot-selected, or
  candidate-selected using **any** person-year dated after `T*`;
- no alignment-layer coefficient may be re-anchored to a post-`T*` Trustees
  vintage or a post-`T*` realized aggregate;
- no forensics question feeding a candidate revision may be selected from a
  post-`T*` holdout failure pattern (the ladder-search disclosure of §4.8 applies
  to the ≤`T*` fit-side forensics only).

Concretely: the certified `CANDIDATE_16` / `CANDIDATE_9` / M4 fits are re-run
with their estimation panels **truncated at `T*`**, and the projection re-draws
2015–2022 state from those truncated fits, seeded from each person's realized
2014 state.

### 4.3 Why PSID-vs-PSID avoids the W1 concept-bridge trap

`gate_w1` amendment 1 demoted all 10 family-B cells because the gate scored a
PSID work-limitation **point-prevalence** against an SSA disabled-worker
**beneficiary stock** across a concept bridge it never defined
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md` §2: misses of
2.9×–21.9× tolerance, `concept_delta_dominant_share 0.595`). `gate_m6` family A
does **not** repeat this: both sides of every family-A cell are the **same PSID
instrument** — projected PSID state vs realized PSID state, same definition, same
population, same severity threshold, same self-report. The seven named
`concept_deltas` of `gate_m4` (`estimand`: definition, population, severity,
transience, conversion denominator, biennial censoring, period pooling) are
**zero** between a projected PSID rate and a realized PSID rate. A level match is
therefore legitimate here, exactly where it was illegitimate in W1 family B. The
external-aggregate comparison that *does* carry concept/coverage deltas (CPS,
SSA, OCACT) is quarantined in the report-only / margin-gated family B (§4.7), not
in the certifiable family A.

### 4.4 The gated observables and cell definitions

Four observable classes, each a set of cells keyed by demographic strata,
computed on the realized post-`T*` waves. Cell naming follows the `gate_w1`
family-A convention (`observable.stratum`, e.g. `earnings_participation.25-34|male`).

- **Survival curves (mortality flow).** Cell = the realized biennial death
  hazard `q(band, sex)` = deaths ÷ start-of-interval exposure among holdout-window
  person-years, per (age-band × sex) over bands {…, 55-64, 65-74, 75-84, 85+}.
  The engine projects deaths from the mortality component; scored
  `|ln(rbar_projected / q_realized)|`. Optionally the survivor-cohort conditional
  survival `S(a | a0)` over the window, per (sex × 5-year birth cohort), as a
  report-only stock (§4.5).
- **Marital composition drift.** Two cell groups: (i) the four transition
  hazards realized in the holdout window — `first_marriage`, `divorce`,
  `widowhood`, `remarriage` — per (age-band × sex) (the flows, gated); (ii) the
  marital-state shares (never-married / married / divorced / widowed) per
  (age-band × sex) at the window end year (the stock; report-only or margin-gated
  per §4.5, because the end-window stock is largely determined by the realized
  `T*` stock plus the gated flows). Split unit: **couple-disjoint** where the
  observable is couple-correlated (the `gate_2c` "couple-disjoint where units
  correlate" lesson; `gate_w1` family-A `split_note`).
- **Earnings distribution moments.** The `gates.yaml` `moment_battery` computed on
  the holdout biennial waves: `change_moments` (mean, sd, skew, kurtosis of
  Δlog-earnings, by cohort, log), `age_profile` (quantiles 0.1/0.5/0.9 by cohort),
  `mobility_matrix` (5-bin, zero-bin, horizons 1–2 in wave units), `zero_spells`.
  `autocorrelation` (lags 1, 2, 5) is **partially unidentifiable** in an 8-year
  holdout — lag 5 in wave units is 10 years, longer than the 4-wave holdout, so
  the lag-5 cells cannot be computed and are report-only by construction (§4.5,
  decision 1). Split unit: **person-disjoint** (earnings are person-level).
- **Disability stocks/flows.** Flows (gated): work-limitation `incidence` and
  `recovery` hazards per (age-band × sex) realized in the holdout window, and the
  near-FRA disability-exit split (retirement vs return-to-work). Stock
  (report-only / margin): disabled-occupancy `prevalence` per (age-band × sex) at
  window end. Same instrument both sides (§4.3), so a level gate on the flows is
  legitimate. Split unit: **person-disjoint**.

### 4.5 Flows vs stocks, mixed-k tolerance, and the trivially-passable-stock problem

A temporal holdout has a structural asymmetry the person-disjoint gates do not:
the **end-of-window stock is nearly a deterministic function of the realized
`T*` stock plus the flows already gated**. Over 8 years a married 60-year-old is
almost certainly still married at 68; the drift being scored in a stock cell is
the small net flow, which is already gated as a flow. Double-gating the stock
adds little discriminating power and risks a **trivially-passable headline** (a
stock cell that passes because initial conditions carry it, inflating the OC).

Design consequence (decision 3): gate the **flows** (death hazard, the four
marital transitions, disability incidence/recovery) as family-A primary; treat
end-of-window **stocks** (survival curves, marital shares, disability prevalence)
as **report-only with a margin check**, not a `|ln|` level gate — the `gate_m4`
anchor-margin pattern (`gate_m4` `statistic`: "the MARGIN reading, not the bare
ordinal"). This certifies the dynamics that M6 actually adds without letting a
near-tautological stock cell carry the conjunction.

Where stocks *are* gated, adopt the `gate_m4` **mixed-k** discipline: FLOW cells
`k = 3`, STOCK cells `k = 4` (`gate_m4` `thresholds` "FLOW hazards k=3, prevalence
STOCK k=4"), tolerance `round(mean + k·sd, 3)` capped at `T_max = ln(1.5)`
(`evaluation.derive_tolerance`; the `ln(1.5)` power cap is the standing bound).

### 4.6 The floor — holdout half-split, real-vs-real, unit-correlation-respecting

The floor is the deployment-scale sampling null of the **holdout-window
observable itself**, constructed exactly as the prior gates construct theirs
(`noise_floor` `method: half_vs_half`), but split within the holdout window:

- split the holdout-contributing persons into two disjoint halves at the
  correlation-respecting unit (person / couple / household as the observable
  dictates; `gate_w1` family-A `split_note`, `gate_2c` couple-disjoint);
- compute each gated observable on each real half; score `|ln(rate_a / rate_b)|`;
- repeat over floor seeds `0–99` (`gate_m4` `floor_seeds: 0-99`); the
  (mean, sd) per cell derive the tolerance `round(mean + k·sd, 3)` capped
  `ln(1.5)`.

The candidate (the projection) is then scored `|ln(rbar_projected / rate_a)|`
where `rate_a` is side-A's **realized** holdout rate and `rbar_projected` is the
K=20-draw mean of the engine projecting side-A persons from their realized `T*`
state. This is the `gate_m4` construction with the time axis substituted for the
re-drawn-state axis: there, the candidate simulates the holdout half's observed
support and matches its empirical rate; here, the candidate simulates the holdout
half's *future* and matches its realized future rate. The floor artifact
(`runs/m6_holdout_floors_v1.json`, **not built by this document**) is authored in
the ceremony, sha-pinned, and read-never-rewritten like every prior floor.

### 4.7 Families, the one-shot rule, and report-only tiers

Mirroring `gate_w1`'s three-family structure:

- **family A — dynamics drift (floor-priced, certifiable).** The §4.4 flow cells,
  `|ln(rbar/rate_a)| ≤ round(mean + k·sd, 3)` capped `ln(1.5)`, K=20 mean on
  `5200 + k`, gate seeds `0–4`, **4-of-5 seed conjunction** (`holdout_protocol`
  `seeds: [0,1,2,3,4]`; `gate_m4` "4-of-5 seed conjunction is operative"). This is
  the surface a `gate_m6` PASS certifies.
- **family B — external alignment audit (report-only / margin-gated).** The
  projection's aggregate agreement with realized external series over the holdout
  window (CPS covered-earnings growth, NCHS mortality, SSA DI prevalence, OCACT
  single-provision triangulation) and the per-run alignment-layer interventions.
  These carry concept/coverage deltas and the macro-shock confound (§4.1), so —
  per the W1 amendment lesson — they are **report-only**, or at most margin-gated
  (bounded/ordinal), **never** `|ln|`-level-gated. The alignment layer itself is
  report-only (decision 9): a layer that hits external targets by construction
  cannot be gated against those targets (the `m2_pseudo_projection_v1.json`
  `calibration_disclosure` lesson — the reserve is calibrated so the *deltas*, not
  the level, are the test).
- **family C — drift-direction fingerprints (binary, optional).** A small set of
  pre-committed qualitative directions the projection must reproduce (e.g.
  never-married share rises across the window; disability prevalence rises with
  age; the earnings age-profile peaks in prime age), binary like `gate_w1`
  family C.

**One-shot rule.** Each candidate is registered on #42 before its run and writes
its artifact through `artifacts.write_new` (the `x`-mode exclusive create that
raises `_already_exists`; the sidecar records `environment_block()` and
`ContractRef.current()`). The `undefined_draw_rule` carries over: if any gated
cell's rate is undefined on any draw (empty simulated denominator — zero holdout
exposure for that cell on that draw), the run is invalidated, not silently
dropped (`gate_2a.fresh_run_artifact_schema.undefined_draw_rule`).

### 4.8 Operating characteristic, pass-verification, and search disclosure

The OC is computed on the draw-noise-free half-normal basis from the family-A
floor σ (`gate_2a.faithful_candidate_oc` `basis_note`), reported at the ceremony
from `runs/m6_holdout_floors_v1.json` once built; this document states no OC
number because the floor does not yet exist (stating one would be fabrication,
the posture `gate_w1_amendment_1` §4b takes for its indeterminate option-b OC).

Two standing #42 process rules (comment 4948637741) bind `gate_m6`:

- **pass-run verification** — a PASS enters the record only after an independent
  adversarial verification round reproduces it bit-exactly, mutation-checks its
  bindings, and scrutinizes its deltas against the registration. A pass without
  verification is a claim, not a result.
- **ladder-search disclosure** — the candidate ladder is an audited iterative
  search; any pass memo must state the effective search size (candidates run,
  forensics rounds, cells) and frame the certification as "the engine reproduces
  held-out `2015–2022` moments after an audited iterative search, under the
  domains-of-validity doctrine," not a first-attempt out-of-sample claim. The
  2100 production projection is **report-only extrapolation beyond the validated
  window** (decision 10, §7).

## 5. PolicyEngine integration — the W2 pattern per projection year

The seam is the ADR-0001 W2 pattern applied once per projected calendar year.
pe-us takes `social_security` as an **uprated survey input** — no benefit formula
exists upstream (`w2_seam_caregiver.py` header; ADR-0001 W2). So for projection
year `y`, the engine's per-person modelled benefit (from `ss/benefits.py`, the
AIME/PIA oracle, plus the M4 DI and 402(b)/(c)/(e)/(f) auxiliary paths) **replaces
the `social_security` column** of that year's panel slice, and a standard managed
`policyengine.py` microsimulation on that slice yields full tax-and-benefit
incidence — taxation of benefits, SNAP/SSI/Medicaid/ACA-PTC/TANF interactions,
SPM poverty, MTRs — through existing pe-us machinery unchanged
(`w2_seam_caregiver.py` `pe.us.managed_microsimulation()`, baseline vs
`social_security + delta`).

Division of ownership, unchanged from the W2 interim:

- **stays in populace-dynamics** — the stochastic dynamics (the §2 engine), the
  statutory benefit oracle (`ss/`), the person-period panel, the alignment layer,
  and the `gate_m6` scoring. The engine emits, per year `y`, the panel slice with
  a modelled `social_security` column (and AIME/PIA/claiming/DI state).
- **runs in policyengine.py** — the deterministic per-year cross-sectional
  tax-benefit incidence, one calendar year at a time, **one simulation per
  process** (the sparse-57k pass peaks tens of GB, so baseline and reform must not
  co-reside — `w2_seam_caregiver.py` launches separate subprocesses;
  `POPULACE_DYNAMICS_PE_US_DIR` / `POPULACE_DYNAMICS_PE_PYTHON`).

The seam is **per-year and stateless**: pe-us does not know it is inside a
projection — it sees a calendar-year cross-section with a modelled SS column,
exactly the W2 object. The `w2_seam_caregiver.py` interim caveats carry forward
verbatim: the certified default is the sparse-57k build, which zeroes untargeted
engine inputs (100% take-up, no SSI asset test, inert imputed Medicaid), so
means-tested offsets move in the right **direction** but their magnitudes are
interim until a dense bundle is certified. The full cross-program execution of
rulespec/pe-us rules across **all** year-slices of the panel at once is M8, not
M6 (§7).

## 6. M7 interface sketch (levels the engine must expose)

Interface only — the trust-fund arithmetic (cost/income rates, trust-fund ratio
path, 75-year actuarial balance in % of taxable payroll) is #113 M7 and is not
designed here. M6 must expose, keyed by `(year)` and aggregable by cohort from the
panel without re-running the engine, the level quantities M2 already computes on
the **closed** 1943–1957 frame (`runs/m2_pseudo_projection_v1.json`) but now on
the **open** projected panel with entrants:

- **covered / taxable earnings aggregate by year** — `Σ w · min(earnings_y,
  taxable_max_y)`, the M2 `taxable_payroll_convention` (each year's earnings
  capped at that year's historical wage base, NAWI-indexed) evaluated per
  projection year. Expose both the **capped** and the **uncapped** person
  earnings so cap-reform provisions (`cap_150k`, `elimination`) recompute the
  aggregate without re-simulation (M2 `revenue_side.cap_150k`,
  `payroll_increments`).
- **taxable-maximum interaction** — each person's year-`y` earnings and the
  year-`y` wage base, so the taxable-max bite is a per-person function M7 can
  differentiate under a reform.
- **benefit outlays by type by year** — OASI retired-worker, DI disabled-worker,
  and the 402(b)/(c)/(e)/(f) auxiliary (spousal/survivor) benefits,
  survival-weighted and placed on the calendar (the M2 `outlay_side.convention`,
  extended with DI from the M4 stock and the auxiliaries the oracle already
  computes; DI→retirement conversions at FRA per `disability_conversion.py`).
- **immigration entries by year** — the synthetic entrant cohorts (count, weight,
  imputed earnings/AIME/demographic state, covered-earnings contribution) that M2
  structurally lacks (its frame is a **closed** cohort, `n_common_frame 1549`,
  `weight_sum 33696344.0`). The open-panel entrant stream is the M6 addition M7
  needs for an open-group actuarial balance.
- **the per-year alignment-layer adjustments** — the versioned interventions
  applied each year (§4.7), so M7 reports scheduled-vs-payable on an **audited**
  alignment rather than a hidden one (#113 named-hard-part 2).
- **the OASDI rate constants** — carried as pe-us nodes where they exist (M2
  `revenue_side.oasdi_combined_rate` `combined 0.124`, source
  `gov/irs/payroll/social_security/rate`); the discount and Trustees-vintage
  assumptions M2 carries as cited constants (`balance_analogue.discount_rate
  0.029`; `tr_vintage_cite`) become M7's versioned alignment inputs.

M6 stops at exposing these levels; M2 already demonstrates the accounting they
feed (`balance_analogue`, `exhaustion_analogue` `baseline_exhaustion_year 2034.0`)
on the closed frame, and M7 lifts that accounting onto the open panel.

## 7. Explicit non-goals for M6

- **No behavioral response.** Claiming is drawn mechanically from the calibrated
  6.B5.1 distribution; there is no labor-supply, claiming-timing, or savings
  response (#113 named-hard-part 4; `m2_pseudo_projection_v1.json` "no behavioural
  response, no claiming change"). Behavioral modules plug in later behind their
  own gates, stated as a domain-of-validity, never silently.
- **No macro feedback.** Wages, prices, interest, and the wage base are exogenous
  Trustees-assumption inputs; the engine closes no general-equilibrium loop. The
  2020–2022 shock inside the holdout is not modelled (§4.1) — its residual is
  disclosed, not fitted.
- **No trust-fund accounting.** M6 exposes the §6 levels but computes no
  cost/income rate, trust-fund ratio, or actuarial balance. That is **M7**.
- **No rules-on-the-whole-panel scoring.** M6 runs the per-year W2 seam (§5); the
  simultaneous execution of rulespec/pe-us rules across every year-slice, and the
  cross-validation of upstreamed rules against the frozen local oracle (W3), is
  **M8**.
- **No new estimation.** M6 composes the M1–M5 certified generators; it fits
  nothing new except the alignment layer, which is a **versioned adjustment**
  (disclosed per run), not a certified generator. `gate_m6` certifies
  composition/projection, not re-estimation — the `gate_w1` posture ("certifies
  transport, not re-estimation").
- **No validated projection beyond the holdout window.** Only `2015–2022` is
  gated; the 2100 production path is report-only extrapolation under the
  domains-of-validity doctrine (§4.8).
- **Deferred to M8:** cross-program interaction answers ("what does this OASDI
  reform do to SNAP eligibility and state income tax in 2045") and the W3 rule
  cross-validation.

## 8. Open design decisions

Each needs adjudication before the `gate_m6` floor is built. Recommendation first,
then the alternative.

1. **Which earnings-moment cells are identifiable in an 8-year holdout.**
   *Recommend* gating `change_moments`, `age_profile`, `mobility_matrix` (horizons
   1–2), and `zero_spells`, and demoting `autocorrelation` **lag 5** to
   report-only (10 years > the 4-wave holdout span, so it is not computable; lag 1
   and lag 2 are). *Alternative*: lengthen the holdout to recover lag 5 (decision
   depends on the `T*` choice, decision 2), accepting a staler fit.

2. **The fit/holdout boundary `T*` and the COVID-era waves.** *Recommend* `T* =
   2014` (holdout 2015–2022), with the 2020–2022 earnings-moment cells
   **report-only** so the gated earnings surface is the pre-shock 2016–2018 waves
   the mechanical engine can represent. *Alternative*: `T* = 2016` (holdout
   2017–2022, excludes COVID from the gated earnings change-windows but leaves only
   ~3 biennial waves, weak power), or `T* = 2012` (holdout 2013–2022, more drift
   power, staler fit, COVID still inside).

3. **Gate flows, report-only stocks — or gate both.** *Recommend* gate the flows
   (death hazard, the four marital transitions, disability incidence/recovery) as
   family-A primary and treat end-of-window stocks (survival curves, marital
   shares, disability prevalence) as report-only with a `gate_m4`-style margin
   check, to avoid a trivially-passable stock cell inflating the OC (§4.5).
   *Alternative*: gate stocks too under the mixed-k discipline (FLOW k=3, STOCK
   k=4), accepting that several stock cells pass near-tautologically.

4. **Gated population — closed panel only, or entrant-inclusive.** *Recommend*
   gate only the **closed** panel (year-0 persons projected from realized state),
   which is clean PSID-vs-PSID (§4.3), and place synthetic births / immigrant
   entry cohorts in report-only family B (no PSID ground truth for entrants over
   the holdout). *Alternative*: gate an entrant-inclusive projection against CPS
   cross-sections — which re-imports the `gate_w1` transport concept-delta the
   temporal holdout was designed to avoid.

5. **Initial condition and the forward-vs-reproduction simulator (frictions F1,
   F5).** *Recommend* the gated closed-panel test seeds each person's **realized
   `T*` state** and, for disability, reuses the M4 reproduction mode over the
   realized post-`T*` waves and the **exogenous** certified widowhood hazard —
   testing the certified dynamics in isolation. *Alternative*: seed a **re-drawn**
   `T*` state from the initial-state models and build the forward-projection
   simulator (synthesized support, endogenous widowhood from simulated spouse
   death) now — testing dynamics + initialization jointly at higher variance, and
   requiring the F1/F5 registry adaptations before the gate rather than for
   production.

6. **Resolving the earnings ⇄ marriage within-period cycle (friction F4).**
   *Recommend* one lagged edge — couple formation at `t` reads `t-1` earnings,
   earnings at `t` reads `t` marital state — matching how
   `couple_formation_sim_v2` reads a carried earnings axis. *Alternative*: a joint
   fixed-point iteration within the period (costlier, and the 2c modifier was not
   fit as a simultaneous system, so the fixed point is not the estimated object).

7. **Ensemble size K.** *Recommend* K = 20 on `5200 + k` verbatim, preserving the
   draw-noise-free OC basis and the byte-identical `t = 0` draws. *Alternative*:
   a larger K to absorb the period-compounded projection variance (§3.4), at the
   cost of breaking the shared-basis reuse and re-deriving the OC.

8. **The `household_composition` double-marital-core seam (friction F2).**
   *Recommend* the engine treats `CANDIDATE_9`'s embedded `marital_core_adapter`
   as the single source of truth for scored household units and does not run
   `family_transitions` twice per period. *Alternative*: inject the step-3
   `CANDIDATE_16` fitted core into `marital_core_adapter` (a registry interface
   change), so the two registries share one marital simulation.

9. **The alignment layer's gate status.** *Recommend* report-only in M6 — gate
   the **un-aligned** projection's drift (family A), and disclose the alignment
   interventions per run (family B), because a layer that hits external targets by
   construction cannot be gated against those targets (the M2
   `calibration_disclosure` lesson). *Alternative*: margin-gate the alignment's
   **magnitude** (bound how far it may move a year aggregate), certifying that
   alignment is small rather than certifying it is absent.

10. **Horizon of certified claims.** *Recommend* certify only the `2015–2022`
    holdout window and label the 2100 path report-only extrapolation.
    *Alternative*: attempt longer-horizon gating against external cohort-completion
    anchors (the #113 Johnson–Smith 2065 medians) — but those are external levels
    that reintroduce the W1 concept-delta and a macro-assumption dependence, so
    they are report-only triangulation (family B) at best.

## 9. Design-parameters summary (design proposal — not a test-bound ledger)

This block summarizes the proposed parameters and the artifact fields they derive
from. Unlike a ratified amendment's consistency ledger, it is **not** bound to
committed artifacts by a test — this is a docs-only design PR that builds no floor
and writes no test. The numbers become test-bound only when
`runs/m6_holdout_floors_v1.json` and the `gates.gate_m6` block are authored in the
lock ceremony this document seeds.

```json m6-design-parameters
{
  "design_id": "2026-07-12-m6-projection-engine",
  "status": "design_draft",
  "gates_yaml_untouched_by_this_document": true,
  "fit_holdout": {
    "boundary_T_star": 2014,
    "holdout_window": "2015-2022",
    "earnings_holdout_waves": [2016, 2018, 2020, 2022],
    "earnings_wave_basis": "biennial from 1999 (data/family.py); gates.yaml views.psid_family_earnings_pairs period_range [1998,2022] step 2",
    "demographic_holdout_years": "2015-2022 annual (data/transitions.py MAX_YEAR 2023)",
    "prohibited_reestimation": "no post-T* person-year in any fitter, alignment coefficient, or forensics selection",
    "person_overlap_intended": true,
    "disjointness": "temporal, not person"
  },
  "gated_observables": {
    "survival_flow": "death hazard q(band,sex) per age-band x sex",
    "marital_flow": "first_marriage|divorce|widowhood|remarriage hazards per age-band x sex (couple-disjoint floor)",
    "earnings_moments": "moment_battery change_moments|age_profile|mobility_matrix|zero_spells (person-disjoint floor); autocorrelation lag5 report-only",
    "disability_flow": "incidence|recovery hazards + near-FRA exit split per age-band x sex (person-disjoint floor)",
    "stocks_report_only_or_margin": "survival curves, marital shares, disability prevalence (near-tautological over 8yr)"
  },
  "scoring": {
    "estimator": "mean over K=20 draws, numpy.random.default_rng(5200 + k), scored once as |ln(rbar/rate_a)|",
    "tolerance": "round(floor mean + k*sd, 3) capped at ln(1.5)",
    "mixed_k": {"flow": 3, "stock": 4},
    "gate_seeds": [0, 1, 2, 3, 4],
    "floor_seeds": "0-99",
    "conjunction": "4 of 5 seeds",
    "floor": "half_vs_half real-vs-real on holdout window, correlation-respecting split unit, runs/m6_holdout_floors_v1.json (NOT built here)"
  },
  "rng_stream_registry": {
    "root": "SeedSequence([5200 + k, GATE_M6_TAG])",
    "period_axis": "ordered spawn per year t = 0..T",
    "module_axis": "stable MODULE_TAG per §2.2 module, reusing 0xC2/0xC5/0xC7/0xB2B at t=0 for byte-identity",
    "person_axis": "consumed in canonical person_id order",
    "split_stream_separate_from_5200_plus_k": true
  },
  "families": {
    "A_certifiable": "PSID-projected vs PSID-realized dynamics-drift flows (floor-priced)",
    "B_report_only_or_margin": "external alignment audit (CPS/NCHS/SSA/OCACT) + per-run alignment interventions",
    "C_binary_optional": "drift-direction fingerprints"
  },
  "registry_frictions": {
    "F1": "exogenous widowhood hazard vs mortality-induced widowhood (family_transitions/registry.py widowhood)",
    "F2": "household_composition embeds its own marital core (_fit_marital calls fit_family_transitions)",
    "F3": "frozen core_seed=5200 has no period axis (household_composition params frozen; registry rejects mismatch)",
    "F4": "earnings <-> marriage within-period cycle (couple_formation_sim_v2 earnings terciles)",
    "F5": "disability_hazard_sim preserves observed support (reproduction, not forward projection)"
  },
  "non_goals": ["behavioral_response", "macro_feedback", "trust_fund_accounting_M7", "rules_on_whole_panel_M8", "new_estimation", "validated_projection_beyond_holdout"],
  "process_addendum_bindings": {
    "pass_run_verification": "#42 comment 4948637741",
    "ladder_search_disclosure": "#42 comment 4948637741"
  }
}
```
