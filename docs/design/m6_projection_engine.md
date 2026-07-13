# M6 projection engine — design and temporal-holdout gate proposal

- **Design id**: `2026-07-12-m6-projection-engine`
- **Roadmap**: #113 M6 (the projection engine), the last build before #113 M7
  (trust-fund accounting) and #113 M8 (integrated scoring). Workstreams #100
  (W1/W2/W3 seams), ADR-0001 (`docs/adr/0001-populace-axiom-seam-ownership.md`).
- **Status**: DESIGN (draft, revision 1). This document seeds the `gate_m6` lock
  ceremony (design review → floor build → adversarial referee → verification →
  ratify-by-merge → lock). **It edits no `gates.yaml` cell, moves no threshold,
  builds no floor, and writes no test.** The gate block and its floor are
  authored in a later ceremony, not here.
- **Revision 1** incorporates the adversarial design referee round (PR #170
  comment 4953818376, verdict MAJOR REVISION) and the ten-decision adjudication
  (comment 4953722912, whose conditions bind). The referee cleared the two hard
  gates — the P1 leakage BLOCKER is **not** triggered (§4.2 refits ≤`T*`) and the
  P2 decision-8 premise **passes** (the embedded marital core is literally the
  certified `CANDIDATE_16` object) — and returned fourteen findings, mapped to
  sections in the revision log below.
- **Evidence base cited by path+field**: the roadmap (#113); the locked contract
  `gates.yaml` (`holdout_protocol`, `noise_floor`, `views`, `moment_battery`,
  `gates.gate_m4`, `gates.gate_w1`); the module registries
  (`family_transitions/registry.py`, `household_composition/registry.py`,
  `household_composition/base_simulator.py`,
  `household_composition/components/marital_core_adapter.py`); the M4 candidate
  (`disability_hazard_sim.py`); the gate-2c candidate
  (`couple_formation_sim_v2.py`, `data/couple_earnings.py`); the gate-1 earnings
  candidate (`scripts/run_gate1_candidate11.py`); the claiming module
  (`claiming.py`); the PSID readers (`data/family.py`, `data/transitions.py`,
  `data/deaths.py`); the W1 transport candidates (`transport_deployment_v1.py`,
  `transport_deployment_v2.py`); the W2 seam (`scripts/w2_seam_caregiver.py`,
  `runs/w2_seam_caregiver_v1.json`); the M2 pseudo-projection
  (`runs/m2_pseudo_projection_v1.json`); the contract loader (`evaluation.py`,
  `contract.py`, `artifacts.py`); the W1 amendment
  (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`); the standing
  process addendum (#42 comment 4948637741).

## Revision log (finding → section)

- 1 (spec-selection-on-full-sample disclosure) → §4.2, §4.9, decision 10.
- 2 (earnings QRF in the refit enumeration) → §4.2.
- 3 (spec-vs-fit distinction; §7↔§4.2) → §4.2, §7.
- 4 (gate-2c modifier composed at a named step) → §2.2 step 3, §2.6.
- 5 + 6 (single-source-of-truth dataflow; injection line) → §2.6.
- 7 + 9 (support basis; symmetric presence-conditioning; attrition) → §4.4,
  friction F7.
- 8 (weight semantics over the projection window) → friction F6, §4.7.
- 10 (dual-axis reference-year/event-year shock-window pin) → §4.1.
- 11 (claiming-table vintage) → §2.2 step 7, §7.
- 12 (biennial boundary peek at `T*`) → §4.2.
- 13 (household-ID persistence + weight carriage) → §2.2 step 8, §9.
- 14 (compute the OC before lock; weak-power pause) → §9.

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

The design record below covers the engine architecture (§2), the composition
resolution under the plumbing/surgery rules (§2.6), the RNG stream registry (§3),
the `gate_m6` temporal-holdout proposal with cell-level observable definitions and
the support-basis/attrition treatment (§4), the per-year PolicyEngine seam (§5,
the W2 pattern), the M7 interface (§6, interface only), the explicit non-goals
(§7), the numbered open decisions with their adjudications (§8), and the
floors-ceremony plan (§9).

Two scope statements this revision makes explicit and does **not** overstate:

- The holdout is out-of-sample for fitted **parameters** but **in-sample for
  spec/structure selection**: `CANDIDATE_16`/`CANDIDATE_9`, M4, and the gate-1
  chained-QRF were structurally chosen by audited ladder searches that observed
  1998–2022 including the 2015–2022 window. A `gate_m6` pass reproduces held-out
  moments *after* an audited iterative search whose structure was chosen on the
  full window — not a structural out-of-sample claim (§4.2, §4.9, decision 10,
  per #42 comment 4948637741).
- The temporal holdout is **PSID-projected vs PSID-realized**, so — unlike
  `gate_w1` family B — it needs no concept bridge (§4.3). But the realized side is
  PSID-attrition-selected, and the gated cells condition on realized
  wave-presence; this is disclosed as conditioning-not-leakage and priced per
  cell, with mortality the worst case (§4.4).

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
  realized year-0 state. Those who remain in PSID over the holdout window have
  realized futures; **attriters do not** (friction F7, §4.4). The gated surface
  (§4) is defined on realized-wave-presence with that selection disclosed.
- the **open additions** — synthetic births (from the fertility/roster module)
  and immigrant entry cohorts, materialized as new person-rows entering in their
  entry year. These have no PSID ground truth over the holdout window and are
  therefore report-only in `gate_m6` (family B, §4.8, decision 4).

The panel carries a **per-year calibrated weight**, but the certified generators
and their floors use **start-wave weights** (`gate_m4`; `disability_hazard_sim`
"start-wave weights"). Which weight the gated rates use, and how a start-wave
weight extends across the projection window, is friction F6 (§2.5) and is pinned
in §4.7.

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
3. **couple formation / dissolution** — the single marital core: the certified
   `CANDIDATE_16` hazards (first marriage, divorce, remarriage, widowhood,
   spousal-age-gap), **with the certified gate-2c first-marriage-by-earnings
   modifier composed onto the first-marriage events** — `couple_formation_sim_v2.
   apply_first_marriage_modifier`, a no-RNG marginal-preserving reweight reading
   each person's **person-constant career-earnings-capacity tercile** (§2.6, R2).
   This step emits the period's **authoritative** marital state (`sim_years`);
   §2.6 lists every downstream reader. Widowhood interacts with step 1 (friction
   F1).
4. **fertility / roster** — births from the fertility component; new child
   person-rows attach to mothers; the roster (household membership) updates.
   Fertility reads authoritative marital state (`marital_core_adapter.
   paternal_births` selects `marital_state == "married"`), so it must run after
   step 3 — the constraint that forces the single core to sit at step 3 (§2.6).
5. **disability** — work-limitation incidence / recovery and the near-FRA
   disability-exit split, from the M4 hazard machinery (`disability_hazard_sim.py`;
   `disability_conversion.py` for the FRA auto-conversion, factor `1.0`, 42 USC
   402(a)/416(i)(2)(D)). Conditions on (age-band, sex) only; marital-blind (§2.6).
6. **earnings** — draw the year-`t` earnings increment from the gate-1-certified
   chained-QRF earnings generator, conditioned on age and the person's
   latent-permanent/anchor component (`scripts/run_gate1_candidate11.py`).
   Marital-blind (§2.6, R2).
7. **claiming** — draw claim age from the calibrated Table 6.B5.1 distribution by
   (sex, entitlement year) (`claiming.claim_age_pmf(exclude_conversions=True)`),
   with DI→retirement conversions supplied by step 5. The 6.B5.1 reference is the
   *2023* Supplement (`claiming.py:7`), a post-`T*` external table; it touches no
   gated cell but feeds report-only benefit levels, so it is frozen to a ≤`T*`
   vintage or its vintage leak is named (finding 11, §7).
8. **household composition reconciliation** — resolve cohabitation, legal-spouse
   residual, parental-home exit, multigenerational and skip-generation occupancy,
   non-family bridge, and household size from the `household_composition` registry
   (`CANDIDATE_9`), **consuming the step-3 authoritative marital state through the
   adapter's read-only mappers rather than re-simulating it** (§2.6, findings
   5/6). Household identity and per-period household weight carry across
   composition changes per the rule pinned in §9 (finding 13).

The order encodes a directed acyclic graph. Deaths first so no dead person is
married, made disabled, or paid; the single marital core at step 3 so fertility,
and the household reconciliation, read one authoritative marital state; earnings
after the demographic state resolves (though it reads none of it — §2.6);
household composition last as a reconciliation over the resolved roster.

### 2.3 The module registries as the interface

The two flattened registries are the engine's module interface. Each is an
immutable `ComponentRegistry` of `ComponentDefinition`s resolved by
`(kind, implementation_id)` with frozen `params`, fitted in declared dependency
order:

- `family_transitions/registry.py` — `CANDIDATE_16`, `contract_revision
  gate_2_amendment_1`; seven kinds in fit order `initial_states → first_marriage
  → divorce → widowhood → remarriage → fertility → spousal_age_gap`.
- `household_composition/registry.py` — `CANDIDATE_9`, `contract_revision
  gate_2b_locked`; ten kinds.

The certified gate-2c object — the first-marriage-by-earnings modifier — lives in
**neither** registry (`couple_formation_sim_v2.py`); the engine composes it
explicitly at step 3 (§2.2, §2.6, finding 4). A `gate_m6` run pins the resolved
`CandidateSpec.sha256` of each registry plus the 2c modifier's committed spec.
`CandidateSpec.sha256` hashes params/impl, **not** estimates or data
(`family_transitions/registry.py:118-124`), which is the hinge for the spec-vs-fit
distinction (§4.2): the same spec sha with a truncated-panel fit is a different
fitted artifact, not a changed certified object.

### 2.4 Where the W1 transport layer sits

W1 transport (`transport_deployment_v1.py` / `v2.py`, `gate_w1`) is **period 0**
of the projection, not a separate stage. Deployment places the certified
generators onto the pinned populace frame (`gate_w1` `deployment_frame`: bundle
`us-4.18.8`, dataset `populace_us_2024`, artifact sha
`c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5`, `n_persons
166302`, `n_households 57240`) and seeds each frame adult's entry state
(`transport_deployment_v2` Q1). M6 takes W1's deployed year-0 panel as its initial
condition and runs the §2.2 loop forward. Two distinct initial conditions exist
and the gate must not conflate them (decision 5): the **deployment initial
condition** (W1, re-drawn entry state on the CPS frame, for the production
projection) and the **holdout initial condition** (`gate_m6` family A, each PSID
person's *realized* year-`T*` state, so the temporal holdout tests forward
dynamics in isolation from transport error).

### 2.5 Registry frictions against the composition order

The registries were sealed for single-period gates; the period axis and the
mortality-linked composition order expose seven concrete resistances the engine
must resolve. These are reported, not hidden. F1–F5 verify against code (referee
"what is NOT wrong"); F6–F7 are added in revision 1 (findings 8, 9).

- **F1 — endogenous vs exogenous widowhood.** `family_transitions/registry.py`
  fits `widowhood` as a standalone PSID hazard (`_WIDOWHOOD_PARAMS
  period_trend_applied=False`, reading `initial_states.support.by_person`), not a
  simulated spouse death. With mortality first, a mortality-killed spouse and the
  fitted widowhood hazard double-count unless reconciled. The gated closed-panel
  test uses the certified exogenous hazard (what `gate_2a` certified); the open
  production projection needs an endogenous-widowhood reconciliation named out of
  M6-certified scope (decision 5, successor gate).

- **F2 — household composition embeds a second marital core.**
  `household_composition/registry.py` `marital_core_adapter._fit_marital`
  (`registry.py:251-261`) calls `fit_family_transitions`, whose body
  (`marital_core_adapter.py:36-44`) returns `ft.REGISTRY.fit(ft.CANDIDATE_16,
  context)` — the identical estimated object as the standalone certified core (P2,
  referee-verified). Candidate-9 then **re-simulates** the core internally
  (`base_simulator.py:72-77`, `ft.simulate(mpanel, holdout_ids,
  fitted.family_transitions, draw_seed)`). Running both step 3 and this internal
  simulation is a double-run with inconsistent marital state. Resolution in §2.6
  (injection, forced by R1+R2).

- **F3 — frozen `core_seed` has no period axis.** `household_composition`
  components seed from frozen params (`_LEGAL_PARAMS core_seed=5200`; child streams
  `SeedSequence([5200 + k, 0xC2/0xC5/0xC7])`, `_CHILD_PARAMS
  persistence_fit_seed=0xC7`). A component that constructs `default_rng(core_seed)`
  internally draws the same numbers every projection period, but the params are
  frozen (`MappingProxyType`) and the registry rejects any mismatch
  (`registry.py:148-153`), so the period cannot be folded into `core_seed`. The
  period is threaded at the engine level (§3), spawning the per-`(module, period)`
  generator and injecting it — engine-side plumbing, not a change to certified
  params (R1, §2.6).

- **F4 — the earnings ⇄ marriage dependency is person-constant, not a
  within-period cycle.** `couple_formation_sim_v2` conditions first-marriage
  timing on an earnings tercile, but the tercile is the person's **career-mean
  indexed positive-year earnings** — a person-constant capacity measure
  (`data/couple_earnings.py:17-45`; `couple_formation_sim_v2.py:261`), **not** a
  contemporaneous or lagged annual level. So there is no within-period annual
  cycle: the modifier reads a predetermined permanent-earnings tercile (§2.6, R2).
  This supersedes decision 6's lagged-edge framing (§8, revised decision 6).

- **F5 — the M4 simulator preserves observed support.**
  `disability_hazard_sim.simulate_draw` (lines 221-259) preserves realized support
  (persons, waves, ages, sexes, start-wave weights) and re-draws only state. It is
  a holdout-*reproduction* simulator, not a forward-*projection* one. The gated
  disability path reuses reproduction mode (§4.4); the open production projection
  needs a forward variant (decision 5's successor gate).

- **F6 — weight semantics over the projection window (finding 8).** `gate_m4` and
  `disability_hazard_sim` use **start-wave weights**; the panel (§2.1) carries a
  per-year calibrated weight. A projected-vs-realized rate must use one consistent
  weight on both sides or every gated numerator/denominator is biased. The
  projection-window convention is pinned in §4.7 (extend the `gate_m4` start-wave
  definition: the gated rates use the person's realized-`T*` start-wave weight held
  fixed across the window on both the projected and the realized side, matching the
  floor).

- **F7 — PSID attrition on the truth side (finding 9).** The realized holdout
  rates and the half-split floor are computed on survival-and-retention-selected
  persons; a non-death for an attriter is unobservable, so the realized
  death-hazard exposure is retention-selected. Treated by symmetric
  presence-conditioning with per-cell dispositions, mortality worst (§4.4).

### 2.6 Composition under the plumbing/surgery rules (R1–R3)

The adjudication (4953722912, findings 5/6) sets three rules and directs deriving
the concrete design from code, not picking a side a priori. **R1** — engine-supplied
RNG generators are plumbing; changing a certified object's fitted params or
internal code paths is surgery; feeding a certified object externally-simulated
**input state through an interface it was certified with** is plumbing; **bypassing
an internal simulation it was certified with** is surgery unless accompanied by an
explicit re-certification note. **R2** — every fitted object must be fed covariates
with the timing/conditioning it was **estimated** with. **R3** — if R1+R2 jointly
force injection into candidate-9, adopt it with a targeted re-certification note.

**R2, pinned from code — the marital/earnings timing each downstream object was
estimated with:**

- gate-1 earnings QRF: predictors are age plus a fixed latent-permanent/anchor
  component and the zero-anchor participation regime
  (`scripts/run_gate1_candidate11.py`; gate-1 view `covariate_columns: [age]`). It
  reads **no marital state** (grep for marital/married/marriage over the candidate:
  zero hits).
- M4 disability: `incidence`/`recovery`/`prevalence` by (age-band, sex),
  `exit_retirement_share` by sex (`disability_hazard_sim.py`). **No marital
  state.**
- claiming: Table 6.B5.1 shares by (sex, entitlement year) (`claiming.py:7-11`,
  `claim_age_pmf` line 262). **No marital state.**
- gate-2c first-marriage modifier: `m(tercile | age band, sex)` where the tercile
  is the person's **career-mean indexed positive-year earnings** — a
  person-CONSTANT capacity measure (`data/couple_earnings.py indexed_earnings_
  supply`; `couple_formation_sim_v2.py:261 tmap = {pid: ce._tercile_of(v,
  axis.cuts) for pid, v in axis.earn.items()}`), not an annual level.

Two consequences follow:

- **Only fertility (step 4) and household composition (step 8) read authoritative
  marital state** among the composed objects; earnings, disability, and claiming
  are marital-blind. So the single marital core can run once at step 3 and be
  consumed read-only downstream with no R2 reorder conflict.
- **The F4 "cycle" is not contemporaneous.** The 2c modifier conditions on a
  person-constant career-capacity tercile, whose projection-time realization is
  the person's permanent/anchor earnings component (predetermined at panel entry;
  the realized ≤`T*` career-capacity in the gated closed-panel test). Feeding it a
  `t-1` annual level (decision 6's lagged edge) is as much an R2 violation as
  feeding it `t`; the R2-faithful composition reads the permanent tercile, which
  has no within-period timing cycle. Decision 6 is revised accordingly (§8).

**R1+R2 force injection into candidate-9 — adopted with the R3 note.** Fertility
(step 4) reads married state (`marital_core_adapter.paternal_births` selects
`marital_state == "married"`), so the marital core must be simulated by step 3 — it
cannot wait for candidate-9 at step 8. But candidate-9 re-simulates the core
internally (`base_simulator.py:72-77`) and consumes the result through its
read-only mappers (`marital_core_adapter.spouse_from_marital` lines 66-87,
`simulated_marital_binary` lines 99-113). Running both is the F2 double-run. The
only consistent resolution is to **inject step-3's authoritative `sim_years` into
candidate-9, bypassing its internal `ft.simulate`, and let the read-only mappers
consume it.** Under R1 that is surgery (bypassing a certified internal
simulation), so per R3 it carries a targeted re-certification note. R1 is the
binding line and is strictly more conservative than finding-6's proposed
"params/impl unchanged ⇒ plumbing" reading; the conservative reading is adopted so
the floors-ceremony referee re-verifies rather than assumes.

**What the re-certification note must establish** (targeted margin check, not a
gate-2b re-ceremony):

1. the estimated core is byte-identical — `ft.REGISTRY.fit(ft.CANDIDATE_16,
   context)` in both the step-3 path and candidate-9's `_fit_marital` (P2);
2. the composition path changed: candidate-9 consumes injected step-3 `sim_years`
   (carrying the 2c marginal-preserving modifier) instead of its internal
   `ft.simulate(draw_seed)` draw;
3. the two differences leave candidate-9's actual conditioning input intact — (a)
   the draw seed differs but the distribution is identical (same estimated object,
   P2); (b) the 2c modifier **preserves the (band, sex) first-marriage marginal by
   construction** — `apply_first_marriage_modifier` enforces `sum_t m·phi_cert = 1`
   and raises on violation (`couple_formation_sim_v2.py:520-525`), and candidate-9
   reads only the married/not-married marginal via `spouse_from_marital`, not the
   within-band earnings-tercile composition the modifier shifts;
4. a pre-named distributional margin check (gate_m4 ≥3σ style) on candidate-9's
   household-composition **output** moments under injected-vs-internal marital
   state, over the gate seeds. Pass ⇒ certification transfers; fail ⇒ the
   composition depends on the bypassed internal draw and a fuller re-ceremony is
   required.

**Dataflow, once per period** (the single-source-of-truth trace finding 5 asks
for): step 3 emits authoritative `sim_years` (CANDIDATE_16 core + 2c modifier).
Readers: step 4 fertility (`paternal_births`, reads `married`); step 8 household
composition (via `spouse_from_marital`, `simulated_marital_binary`,
`father_marital_by_year` — all read-only, no `ft.simulate`). Non-readers (verified
marital-blind, R2): step 5 disability, step 6 earnings, step 7 claiming. No step
re-simulates the core.

## 3. RNG discipline — the projection stream registry

### 3.1 The inherited convention

The stack scores K=20 draws on `numpy.random.default_rng(5200 + k)`, `k = 0…19`
(`family_transitions/evaluation.py` / `household_composition/evaluation.py`
`DRAW_SEED_BASE = 5200`; `gate_m4.protocol draw_stream_base: 5200`). The estimator
is the mean over K draws, scored **once** as `|ln(rbar / rate_a)|`
(`evaluation.py _load_scoring`; `gate_m4 estimator`). Components spawn disjoint
child streams by tag: `SeedSequence([5200 + k, 0xC2])`, `0xC5`, `0xC7`, occupancy
`0xB2B` (`base_simulator.py:72-73`). The simulation stream is kept **distinct**
from the split stream (`gate_m4` finding 2; `gate_w1` streams `9100`/`9200`).

### 3.2 The (draw × module × period × person) spawn tree

M6 adds a period axis; every module × period × person draw must be addressable and
bit-reproducible. The proposed registry is a hierarchical `SeedSequence` spawn
tree rooted at the inherited base:

```
root(k)              = SeedSequence([5200 + k, GATE_M6_TAG])
period(k, t)         = root(k).spawn(T+1)[t]          # ordered child per year t = 0..T
stream(k, t, module) = period(k,t).spawn per module in the fixed §2.2 order
generator(k,t,module)= default_rng(stream(k, t, module))
```

`GATE_M6_TAG` is a fixed engine constant distinct from any component tag;
`MODULE_TAG`s reuse the existing child-tags (`0xC2`/`0xC5`/`0xC7`/`0xB2B`) at
`t = 0` so single-period draws stay byte-identical. Per-person draws are consumed
in canonical `person_id`-sorted order, so person `i`'s draw at `(module, period)`
is a pure function of `(k, t, module, i)`. The engine passes
`generator(k,t,module)` **into** each component (the F3 fix); it never lets a
component construct `default_rng(core_seed)` from a frozen param — engine-side
plumbing, no change to certified params (R1, §2.6).

### 3.3 Disjointness guarantees and the separate split stream

The spawn tree gives disjointness across draws (distinct `k` root the tree at
distinct entropy → independent K=20 members), across periods (distinct `t` are
distinct ordered children → re-running period `t` is exact, no two periods share
entropy — the F3 fix), and across modules (distinct spawn positions within a
period). The **floor split stream stays separate** from this simulation tree: the
temporal fit/holdout boundary is a time cut, but the `gate_m6` floor (§4.7) draws
person/couple/household-disjoint half-splits of the holdout window, and those
split seeds use their own base disjoint from `5200 + k` (`gate_m4` finding-2
discipline).

### 3.4 K-draw ensembles at the engine level

`gate_m6` reuses K=20 and `5200 + k` verbatim (decision 7), so the draw-noise-free
half-normal OC basis (`gate_2a.faithful_candidate_oc basis_note`) transfers
unchanged. Projection variance **compounds over periods** (a period-`t` draw feeds
`t+1`), so a late-window cell's across-draw sd is larger; this is priced correctly
because the floor and the candidate use the same K and window. If the floors build
reveals degenerate ensemble width under compounding, that is a floors-ceremony
finding to bring back, not a silent redesign (adjudication 7).

## 4. The temporal-holdout gate (`gate_m6`, proposed)

### 4.1 The fit/holdout split and the dual-axis shock-window pin

**Proposed boundary `T* = 2014`: fit on all PSID observations dated `≤ 2014`;
project and score the held-out window `2015–2022`.** Wave facts (verified against
the readers and `gates.yaml views`): earnings are biennial from 1999
(`data/family.py`: waves 1994-2023, income **reference years** 1993-2022), and the
gate-1 earnings view (`views.psid_family_earnings_pairs`) is `period_range [1998,
2022]`, `period_step 2`; marital/fertility/mortality flows are annual to
`MAX_YEAR = 2023` (`data/transitions.py`); PSID death file 1968-2023
(`data/deaths.py`).

Justification: an 8-year horizon is long enough that projection drift — the thing
M6 adds over the M1–M5 single-period generators — accumulates and becomes visible,
short enough that the youngest fit cohorts still carry holdout exposure; it keeps
the fit inside the biennial-from-1999 regime.

**Shock-window demotion, uniform across all modules (adjudication 2, finding 10).**
The 2020–2022 COVID/inflation shock is outside the engine's model class (no
macro/epidemic channel) and hit mortality (2020–21 excess deaths) and
marital-transition rates (the 2020 marriage collapse) at least as hard as
earnings. All shock-window cells across all modules are **report-only** with one
machine reason, `exogenous_shock_outside_model_class`, pinned on a **dual axis**:

- **earnings (reference-year axis):** report-only `{2020, 2022}`; **reference year
  2021 is unobserved** (odd year, no biennial interview — `data/family.py`
  biennial-from-1999); gated earnings reference years `= {2016, 2018}` (2 pre-shock
  biennial waves ⇒ ~2 change-transitions).
- **flows (event-year axis):** report-only `{2020, 2021}` (excess deaths, marriage
  collapse); gated event years `= {2015–2019}` (5 annual years).

This reconciles the earlier "4 biennial earnings holdout waves / gates 2015–2022
flows" wording to the post-demotion surface: gated `= 2015–2019 annual demographic
flows + 2016/2018 earnings waves`. The shock cells are a first-class published
diagnostic (how far a mechanical engine misses a pandemic, quantified against
held-out truth). The whole disposition is **pre-lock** — the amendment-2 lesson
(identify concept-class-unclearable cells before registration) applied before the
gate, exactly as the attrition demotions are (§4.4).

### 4.2 Prohibited re-estimation, spec-selection, and the spec-vs-fit distinction

The discipline that makes the holdout honest for **parameters** is **temporal
disjointness**: persons overlap fit and holdout by design (the projection's input
is today's realized cross-section), but no post-`T*` information may enter any
fitter. The complete refit enumeration — **including the gate-1 earnings generator**
(finding 2), the object §4.1's earnings-moment cells score:

- no hazard, initial-state model, spousal-gap table, fertility kernel, disability
  rate, claiming distribution, **or gate-1 chained-QRF earnings generator** may be
  fit, refit, tuned, knot-selected, or candidate-selected using any person-year
  dated after `T*`;
- no alignment-layer coefficient may be re-anchored to a post-`T*` Trustees vintage
  or realized aggregate;
- concretely, the certified `CANDIDATE_16` / `CANDIDATE_9` / M4 / **gate-1
  earnings-QRF** fits are re-run with their estimation panels truncated at `T*`,
  and the projection re-draws 2015–2022 state seeded from each person's realized
  2014 state.

**Spec-vs-fit (finding 3, reconciling §7's "no new estimation").** Refitting a
frozen spec on ≤2014 produces new fitted numbers — re-estimation of *estimates*,
with the *spec* unchanged. The M6 run pins the same `CandidateSpec.sha256` (which
hashes params/impl, not estimates or data — `family_transitions/registry.py:118-124`;
the registry still rejects any param mismatch, `:148-153`) but a **different**,
truncated-panel **fitted artifact**, and it neither reads nor writes the certified
full-window artifacts. §7's "no new estimation" means no new **spec**; the
truncation is a data input, not a param change.

**Spec-selection-on-full-sample (finding 1, the residual second-order leak).** The
holdout cures parameter leakage but not **structure** leakage: `CANDIDATE_16/9`,
M4 (`disability_hazard_sim`), and the gate-1 chained-QRF (`gates.yaml:122-126`)
were *structurally* chosen by audited ladder searches that observed 1998–2022,
including the 2015–2022 holdout. The temporal holdout tests refit-stability and
drift **conditional on that selection** and cannot cure it. Per the #42 comment
4948637741 search-disclosure addendum, this is a named limitation: the holdout is
out-of-sample for fitted parameters, **in-sample for spec/structure selection**;
any pass is framed as "reproduces held-out moments after an audited iterative
search whose *structure* was chosen on the full window," never a structural
out-of-sample claim (carried into decision 10's certified-claims language and
§4.9).

**Biennial boundary peek (finding 12).** Annual flows reconstructed from biennial
interviews (`data/transitions.py`) can date a reference-year-2014 transition using
the 2015 interview — a one-interview peek across the cut. Conservative rule
adopted: **any flow event whose dating requires a post-`T*` (2015+) interview is
excluded from the ≤`T*` fit** (no post-`T*` interview informs a fit-window flow),
so the fit/holdout cut is unambiguous. The alternative (bounding the peek) is
rejected as harder to audit.

### 4.3 Why PSID-vs-PSID avoids the W1 concept-bridge trap

`gate_w1` amendment 1 demoted all 10 family-B cells because the gate scored a PSID
work-limitation **point-prevalence** against an SSA disabled-worker **beneficiary
stock** across a concept bridge it never defined
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md` §2; misses 2.9×–21.9×
tolerance, `concept_delta_dominant_share 0.595`). `gate_m6` family A does not
repeat this: both sides of every family-A cell are the **same PSID instrument** —
projected PSID state vs realized PSID state, same definition, population, severity
threshold, self-report. The seven `gate_m4` `concept_deltas` are **zero** between a
projected PSID rate and a realized PSID rate. The external-aggregate comparison
that *does* carry concept/coverage deltas (CPS, SSA, OCACT) is quarantined in the
report-only / margin-gated family B (§4.8), not in family A. (The remaining
selection concern is not a concept bridge but PSID attrition — §4.4.)

### 4.4 Support basis and symmetric presence-conditioning (findings 7, 9)

A temporal holdout on PSID persons confronts two selection facts: reproduction-mode
conditioning on realized support, and PSID attrition on the truth side.

**Support basis differs by module.** `disability_hazard_sim.simulate_draw`
preserves the realized holdout support and re-draws only state; mortality and
marital cells cannot be scored that way — conditioning a death-hazard test on
realized survival-through-the-interval is circular. So:

- **disability** is scored in reproduction mode (realized support). The same
  wave-presence selection sits on both the projected and the realized rate, so the
  common selection largely cancels in the projected-vs-realized ratio; a disability
  PASS certifies a **temporal extension of `gate_m4`** (the fitted hazards
  reproduce a future PSID window), **not** M6's forward disability path (which
  synthesizes support — F5, decision 5's successor gate). The claim is scoped
  accordingly so certification is not over-transferred (finding 7 (iii)).
- **mortality and marital** cells are scored on **presence-at-start-of-interval**:
  among persons realized-present and at risk at wave `t`, the projected transition
  rate over `[t, t+1]` vs the realized rate. Presence-at-`t` is not a modeled
  outcome of the engine (PSID retention is not simulated), so conditioning on it is
  **conditioning-not-leakage**, on the same footing as age/sex conditioning.

**Symmetric presence-conditioning.** Both the projection cells and the truth cells
(and the §4.7 floor) are computed on the same realized-wave-presence set per
interval, disclosed as a conditioning choice, not a leak.

**Attrition bites worst on mortality (friction F7).** Presence-at-`t` is exogenous
to the interval's transition, but attrition *within* the interval is not fully so:
a person present at `t` who dies in `[t, t+1]` is observed as a death (`data/
deaths.py` `ER32050`), but one who **attrites** (leaves the panel, not dead) is
censored, and if attrition correlates with mortality the realized death-hazard
exposure is retention-selected. Per-cell dispositions:

- mortality/marital cells whose truth is attrition-confounded beyond repair are
  **demoted to report-only pre-lock**, machine reason `attrition_confounded_truth`
  — the amendment-2 lesson applied before the lock, as with the shock window (§4.1);
- for retained cells the residual attrition bias is disclosed; inverse-attrition
  weighting is a floors-ceremony option, not assumed;
- the **forward production engine** cannot condition on realized future presence
  (no PSID panel over the projected future), so its mortality/marital path is a
  different object; its delta from the gated presence-conditioned path is stated,
  and the forward variant is decision 5's registered successor gate.

### 4.5 The gated observables and cell definitions

Four observable classes on the post-`T*`, post-demotion surface (§4.1 shock, §4.4
attrition). Cell naming follows the `gate_w1` family-A convention
(`observable.stratum`).

- **Survival curves (mortality flow).** Cell = the realized death hazard
  `q(band, sex)` = deaths ÷ start-of-interval exposure among presence-conditioned
  holdout person-years (§4.4), per (age-band × sex) over bands {…, 55-64, 65-74,
  75-84, 85+}, event years 2015–2019. Scored `|ln(rbar_projected / q_realized)|`.
  Attrition-confounded-beyond-repair cells demoted per §4.4.
- **Marital composition drift.** Gated: the four transition hazards realized in the
  holdout window — `first_marriage`, `divorce`, `widowhood`, `remarriage` — per
  (age-band × sex), event years 2015–2019 (couple-disjoint floor). Report-only /
  margin: end-window marital-state shares (§4.6).
- **Earnings distribution moments.** The `gates.yaml moment_battery` on the gated
  earnings reference years {2016, 2018}: `change_moments` (mean, sd, skew,
  kurtosis of Δlog-earnings, by cohort, log), `age_profile` (quantiles 0.1/0.5/0.9
  by cohort), `mobility_matrix` (5-bin, zero-bin, horizons 1–2 in wave units),
  `zero_spells` (person-disjoint floor). `autocorrelation` lag 5 is **report-only**
  by measurability — a 10-year statistic is not computable inside the holdout,
  machine reason `horizon_exceeds_holdout_span`; its status is re-derived, not
  grandfathered, if `T*` moves (adjudication 1).
- **Disability stocks/flows.** Gated flows (reproduction mode, temporal gate_m4
  extension per §4.4): work-limitation `incidence` and `recovery` hazards per
  (age-band × sex), event years 2015–2019, and the near-FRA disability-exit split.
  Report-only / margin stock: disabled-occupancy `prevalence`. Person-disjoint
  floor.

### 4.6 Flows vs stocks, mixed-k, and the trivially-passable-stock problem

Over the holdout window the end-of-window **stock** is nearly a deterministic
function of the realized `T*` stock plus the gated **flows**, so double-gating a
stock adds little discriminating power and risks a trivially-passable headline.
Disposition (decision 3): gate the **flows** as family-A primary; treat
end-of-window stocks (survival curves, marital shares, disability prevalence) as
report-only with a `gate_m4`-style **≥3σ margin check**, not a `|ln|` level gate.
Where stocks are gated, adopt the `gate_m4` **mixed-k** discipline: FLOW cells
`k = 3`, STOCK cells `k = 4`, tolerance `round(mean + k·sd, 3)` capped at
`ln(1.5)` (`evaluation.derive_tolerance`).

### 4.7 The floor, weights, and presence-conditioning

The floor is the deployment-scale sampling null of the holdout-window observable,
built as the prior gates build theirs (`noise_floor method: half_vs_half`), but
split within the holdout window and **presence-conditioned symmetrically** (§4.4):

- split the presence-conditioned holdout persons into two disjoint halves at the
  correlation-respecting unit (person / couple / household as the observable
  dictates; `gate_w1` family-A `split_note`, `gate_2c` couple-disjoint);
- compute each gated observable on each real half; score `|ln(rate_a / rate_b)|`;
  repeat over floor seeds `0–99`; the (mean, sd) per cell derive the tolerance
  `round(mean + k·sd, 3)` capped `ln(1.5)`.

**Weight convention (F6, finding 8).** Both the projected and the realized side —
and the floor — use the person's **realized-`T*` start-wave weight held fixed
across the projection window**, extending the `gate_m4` start-wave definition to
the projection axis. The per-year calibrated panel weight (§2.1) is **not** used
for the gated rates (it would put a different weight on the two sides and bias
every ratio); it is a report-only alternative. The floor must be built on the same
start-wave weight it scores against.

The candidate is scored `|ln(rbar_projected / rate_a)|` where `rate_a` is side-A's
realized presence-conditioned holdout rate and `rbar_projected` is the K=20-draw
mean of the engine projecting side-A persons from their realized `T*` state. This
is the `gate_m4` construction with the time axis substituted for the re-drawn-state
axis. The floor artifact (`runs/m6_holdout_floors_v1.json`, **not built by this
document**) is authored in the ceremony, sha-pinned, read-never-rewritten.

### 4.8 Families, the one-shot rule, and report-only tiers

Mirroring `gate_w1`'s three-family structure:

- **family A — dynamics drift (floor-priced, certifiable).** The §4.5 flow cells on
  the post-demotion surface, `|ln(rbar/rate_a)| ≤ round(mean + k·sd, 3)` capped
  `ln(1.5)`, K=20 mean on `5200 + k`, gate seeds `0–4`, **4-of-5 seed
  conjunction**. This is the surface a `gate_m6` PASS certifies.
- **family B — external alignment audit + entrants (report-only / margin-gated).**
  The projection's aggregate agreement with realized external series (CPS, NCHS,
  SSA, OCACT triangulation), the per-run alignment-layer interventions and the
  per-year **maximum alignment displacement** (adjudication 9, a reported not
  gated magnitude), and the **entrant cells** (synthetic births + immigrant
  cohorts) with their named bridge requirements (adjudication 4). All report-only
  or bounded/ordinal margin, never `|ln|`-level-gated (the W1 lesson; the M2
  `calibration_disclosure` lesson for the alignment layer).
- **family C — drift-direction fingerprints (binary, optional).** A small set of
  pre-committed qualitative directions the projection must reproduce, binary like
  `gate_w1` family C.

**One-shot rule.** Each candidate is registered on #42 before its run and writes
its artifact through `artifacts.write_new`; the `undefined_draw_rule` carries over
(an undefined gated-cell rate on any draw invalidates the run —
`gate_2a.fresh_run_artifact_schema.undefined_draw_rule`).

**Report-only re-drawn-seed comparison (adjudication 5b).** `gate_m6` also
publishes, alongside the gated realized-`T*`-seed run, the same holdout cells
scored once from **re-drawn** `T*` initial states, with a pre-named margin that
seeds the successor gate's flip criteria — making initialization error visible in
the M6 record rather than deferred.

### 4.9 Operating characteristic, pass-verification, and search disclosure

The OC is computed on the draw-noise-free half-normal basis from the family-A
floor σ (`gate_2a.faithful_candidate_oc basis_note`), reported at the ceremony from
`runs/m6_holdout_floors_v1.json` once built; this document states no OC number
because the floor does not yet exist (stating one would be fabrication, the posture
`gate_w1_amendment_1` §4b takes). The post-adjudication surface is **thin** (the
shock and attrition demotions shrink N), so §9 requires the OC be checked before
lock in **both** directions (near-vacuous from wide tolerances, or unfailable).

Two standing #42 process rules (comment 4948637741) bind `gate_m6`:
**pass-run verification** (a PASS enters the record only after independent
adversarial bit-exact reproduction) and **ladder-search disclosure**. The latter is
strengthened by finding 1: the disclosure names not only the M6 gate's own ≤`T*`
forensics but the **spec-selection-on-full-sample** limitation (§4.2) — a pass is
"reproduces held-out 2015–2019 / 2016–2018 moments after an audited iterative
search whose *structure* was chosen on the full window," and the 2100 production
projection is report-only extrapolation beyond the validated window (decision 10,
§7).

## 5. PolicyEngine integration — the W2 pattern per projection year

The seam is the ADR-0001 W2 pattern applied once per projected calendar year. pe-us
takes `social_security` as an **uprated survey input** — no benefit formula exists
upstream (`w2_seam_caregiver.py` header; ADR-0001 W2). For projection year `y`, the
engine's per-person modelled benefit (from `ss/benefits.py`, plus the M4 DI and
402(b)/(c)/(e)/(f) auxiliary paths) **replaces the `social_security` column** of
that year's panel slice, and a standard managed `policyengine.py` microsimulation
yields full tax-and-benefit incidence through existing pe-us machinery unchanged
(`pe.us.managed_microsimulation()`).

Division of ownership: **stays in populace-dynamics** — the stochastic dynamics
(§2 engine), the statutory benefit oracle (`ss/`), the panel, the alignment layer,
and `gate_m6` scoring; **runs in policyengine.py** — the deterministic per-year
cross-sectional tax-benefit incidence, one calendar year at a time, **one
simulation per process** (`w2_seam_caregiver.py` launches separate subprocesses;
`POPULACE_DYNAMICS_PE_US_DIR` / `POPULACE_DYNAMICS_PE_PYTHON`). The seam is
per-year and stateless: pe-us sees a calendar-year cross-section with a modelled SS
column, exactly the W2 object. The `w2_seam_caregiver.py` sparse-57k caveats carry
forward (zeroed untargeted inputs → means-tested offsets move in the right
direction, magnitudes interim). The full cross-program execution across **all**
year-slices at once is M8, not M6 (§7).

## 6. M7 interface sketch (levels the engine must expose)

Interface only — the trust-fund arithmetic is #113 M7. M6 must expose, keyed by
`(year)` and aggregable by cohort from the panel, the level quantities M2 already
computes on the **closed** 1943–1957 frame (`runs/m2_pseudo_projection_v1.json`)
but now on the **open** projected panel with entrants:

- **covered / taxable earnings aggregate by year** — `Σ w · min(earnings_y,
  taxable_max_y)`, the M2 `taxable_payroll_convention`; expose both **capped** and
  **uncapped** person earnings so cap-reform provisions (`cap_150k`, `elimination`)
  recompute without re-simulation (M2 `revenue_side.cap_150k`, `payroll_increments`).
- **taxable-maximum interaction** — each person's year-`y` earnings and wage base.
- **benefit outlays by type by year** — OASI retired-worker, DI disabled-worker,
  402(b)/(c)/(e)/(f) auxiliary, survival-weighted on the calendar (M2
  `outlay_side.convention`, extended with M4 DI and the auxiliaries; FRA conversions
  per `disability_conversion.py`).
- **immigration entries by year** — the synthetic entrant cohorts M2 structurally
  lacks (its frame is a **closed** cohort, `n_common_frame 1549`, `weight_sum
  33696344.0`).
- **the per-year alignment-layer adjustments** — the versioned interventions
  applied each year (§4.8), so M7 reports scheduled-vs-payable on an audited
  alignment (#113 named-hard-part 2).
- **the OASDI rate constants** — pe-us nodes where they exist (M2
  `revenue_side.oasdi_combined_rate combined 0.124`); the discount / Trustees
  vintage assumptions M2 carries (`balance_analogue.discount_rate 0.029`;
  `tr_vintage_cite`) become M7's versioned alignment inputs.

M6 stops at exposing these levels; M2 demonstrates the accounting they feed
(`balance_analogue`, `exhaustion_analogue baseline_exhaustion_year 2034.0`) on the
closed frame, and M7 lifts it onto the open panel.

## 7. Explicit non-goals for M6

- **No behavioral response.** Claiming is drawn mechanically from Table 6.B5.1; no
  labor-supply, claiming-timing, or savings response (#113 named-hard-part 4; M2
  "no behavioural response, no claiming change"). Behavioral modules plug in later
  behind their own gates.
- **No macro feedback.** Wages, prices, interest, the wage base are exogenous
  Trustees inputs; no general-equilibrium loop. The 2020–2022 shock is not modelled
  and its cells are report-only (§4.1).
- **No new spec estimation.** M6 composes the M1–M5 certified **specs** unchanged
  (same `CandidateSpec.sha256`); it re-runs their **estimates** on ≤`T*`-truncated
  panels (finding 3, §4.2) and fits nothing new except the alignment layer (a
  versioned adjustment, disclosed, not a certified generator). `gate_m6` certifies
  composition/projection, not spec re-estimation.
- **No trust-fund accounting (M7); no whole-panel rules scoring (M8).** M6 exposes
  the §6 levels and runs the per-year W2 seam (§5); it computes no actuarial balance
  and runs no simultaneous cross-year rules execution / W3 cross-validation.
- **No validated projection beyond the holdout window.** Only the post-demotion
  2015–2019 / 2016–2018 surface is gated; the 2100 path is report-only
  extrapolation, and — per finding 1 — even the gated claim is conditional on
  full-window spec selection (§4.9).
- **Deferred to the decision-5 successor gate:** the forward-projection variant
  (synthesized support, endogenous widowhood, re-drawn initial states — F1/F5); no
  claim of a certified projection *engine* until it passes.

## 8. Open design decisions (with adjudications)

The ten decisions were adjudicated in comment 4953722912; this section records the
outcome and the revision-1 refinements. Where the design referee's findings changed
a resolution, the change is noted.

1. **Earnings-moment cells — ACCEPTED.** Gate `change_moments`, `age_profile`,
   `mobility_matrix` (h 1–2), `zero_spells`; `autocorrelation` lag 5 report-only,
   machine reason `horizon_exceeds_holdout_span`, re-derived not grandfathered if
   `T*` moves.
2. **`T* = 2014` — ACCEPTED, shock demotion extended to all modules** with the
   dual-axis reference-year/event-year pin and machine reason
   `exogenous_shock_outside_model_class` (§4.1, finding 10).
3. **Flows family-A, stocks report-only with ≥3σ margin — ACCEPTED** (§4.6).
4. **Closed panel gated; entrants family B with named bridge requirements —
   ACCEPTED** (§4.8).
5. **Realized-`T*` seeding + M4 reproduction mode + exogenous widowhood —
   ACCEPTED, with two conditions:** (a) the forward-projection variant is a
   **registered successor gate**, not optional hardening; the 2100 path stays
   report-only until it passes; (b) `gate_m6` publishes a **report-only re-drawn-seed
   comparison** with a pre-named margin seeding the successor gate's flip criteria
   (§4.8). Finding 7 adds the support-basis split: mortality/marital are
   presence-conditioned forward-style cells, disability is reproduction-mode and
   certifies a temporal `gate_m4` extension (§4.4).
6. **One lagged edge — this revises adjudication 6, SUPERSEDED by the R2
   derivation.** Adjudication 6 accepted a lagged edge (couple formation reads
   `t-1` earnings). The code supersedes it: the 2c modifier's tercile is the
   person's **career-mean indexed positive-year earnings** — a person-constant
   capacity measure (`data/couple_earnings.py indexed_earnings_supply`;
   `couple_formation_sim_v2.py:261`), not an annual level, so there is no
   within-period annual cycle and a `t-1` (or `t`) annual level would violate R2.
   The R2-faithful composition feeds the predetermined **permanent-earnings
   tercile** at step 3, and the modifier is a marginal-preserving reweight
   (`apply_first_marriage_modifier` enforces `sum_t m·phi_cert = 1`, raising on
   violation — `couple_formation_sim_v2.py:520-525`), so composing it at step 3
   moves no (band, sex) first-marriage marginal. Adopted in place of the lagged
   edge, per adjudication 6's own instruction to derive the design from code
   rather than pick a side a priori.
7. **K = 20 on `5200 + k` verbatim — ACCEPTED**; degenerate ensemble width under
   compounding is a floors-ceremony finding, not a silent redesign.
8. **Single marital core — INJECTION, forced by R1+R2, with the R3 re-certification
   note.** P2 passed (the embedded core is the certified `CANDIDATE_16`), but the
   step order forces the core to run at step 3 (fertility reads married state), and
   avoiding candidate-9's internal re-simulation double-run requires injecting
   step-3 state and bypassing its internal `ft.simulate` — R1 surgery, adopted with
   the targeted re-certification note (§2.6). This resolves the finding-6
   contradiction by applying R1's line (bypassing a certified internal simulation =
   surgery + note), the more conservative reading.
9. **Alignment layer report-only; gate the un-aligned drift — ACCEPTED,** plus the
   per-run **maximum alignment displacement** published as a reported (not gated)
   magnitude (§4.8).
10. **Certify only the post-demotion holdout surface; 2100 = report-only
    extrapolation — ACCEPTED,** with the finding-1 certified-claims language: even
    the gated claim is conditional on full-window spec selection (§4.2, §4.9).

## 9. Floors-ceremony plan (§9 deliverables)

The ceremony that seeds from this revised §4 must produce, before locking:

- **the floor artifact** `runs/m6_holdout_floors_v1.json` — presence-conditioned
  (§4.4), start-wave-weighted (§4.7, F6), correlation-respecting half-splits, floor
  seeds `0–99`, on the post-demotion surface (§4.1 shock, §4.4 attrition).
- **the re-certification margin check (§2.6)** — candidate-9's household-composition
  output moments under injected-vs-internal marital state agree within a pre-named
  gate_m4-style margin; a floors-ceremony deliverable, not assumed here.
- **the OC-before-lock check (finding 14)** — compute the family-A operating
  characteristic on the **surviving** gated surface before locking, with a named
  weak-power threshold below which the ceremony **pauses for surface redesign**
  rather than locking a near-vacuous gate (the W1 lesson in both directions: no
  cell near-tautological, and no cell unfailable from wide small-N tolerances).
- **the household-ID + weight carriage rule (finding 13)** — the cross-period rule
  for household identity across composition changes (divorce → which IDs; new-entrant
  and child household assignment) and the per-period household weight the
  household-disjoint floor split and the household-level report-only cells depend on,
  consistent with the F6 weight convention (§4.7).
- **the claiming-vintage freeze (finding 11)** — a ≤`T*` claiming vintage for the
  report-only benefit levels, or the named 2023-Supplement vintage leak.

## 10. Design-parameters summary (design proposal — not a test-bound ledger)

This block summarizes the proposed parameters and the artifact/code fields they
derive from. Unlike a ratified amendment's consistency ledger, it is **not** bound
to committed artifacts by a test — this is a docs-only design PR that builds no
floor and writes no test. The numbers become test-bound only when
`runs/m6_holdout_floors_v1.json` and the `gates.gate_m6` block are authored in the
lock ceremony.

```json m6-design-parameters
{
  "design_id": "2026-07-12-m6-projection-engine",
  "revision": 1,
  "referee_round": "PR #170 comment 4953818376 (MAJOR REVISION)",
  "adjudication": "issue #42 comment 4953722912",
  "status": "design_draft",
  "gates_yaml_untouched_by_this_document": true,
  "fit_holdout": {
    "boundary_T_star": 2014,
    "prohibited_reestimation_includes_gate1_earnings_qrf": true,
    "spec_vs_fit": "same CandidateSpec.sha256 (params/impl), different truncated-panel fitted artifact; certified full-window artifacts neither read nor written",
    "spec_selection_on_full_sample_disclosed": true,
    "biennial_boundary_rule": "exclude any flow event whose dating needs a post-T* (2015+) interview from the fit",
    "person_overlap_intended": true,
    "disjointness": "temporal, not person"
  },
  "shock_window": {
    "machine_reason": "exogenous_shock_outside_model_class",
    "earnings_reference_year_axis": {"report_only": [2020, 2022], "note_2021_unobserved_odd_year": true, "gated": [2016, 2018]},
    "flows_event_year_axis": {"report_only": [2020, 2021], "gated": "2015-2019"}
  },
  "support_basis": {
    "disability": "reproduction mode (realized support); certifies a temporal gate_m4 extension, not M6 forward disability",
    "mortality_marital": "forward-style, presence-at-start-of-interval, conditioning-not-leakage",
    "attrition_friction": "F7; mortality worst; attrition_confounded_truth cells demoted report-only pre-lock"
  },
  "composition_R1_R2_R3": {
    "R2_marital_readers": ["fertility_step4_paternal_births", "household_composition_step8"],
    "R2_marital_blind": ["earnings_qrf", "disability", "claiming"],
    "gate2c_tercile_is_person_constant_career_capacity": true,
    "decision6_lagged_edge": "SUPERSEDED; feed permanent-earnings tercile at step 3",
    "injection_forced": true,
    "injection_class_under_R1": "surgery (bypasses candidate-9 internal ft.simulate)",
    "recertification": "targeted distributional margin check on candidate-9 composition outputs (P2 identical core; 2c marginal-preserving); not a gate-2b re-ceremony"
  },
  "scoring": {
    "estimator": "mean over K=20 draws, numpy.random.default_rng(5200 + k), scored once |ln(rbar/rate_a)|",
    "tolerance": "round(floor mean + k*sd, 3) capped ln(1.5)",
    "mixed_k": {"flow": 3, "stock": 4},
    "gate_seeds": [0, 1, 2, 3, 4],
    "floor_seeds": "0-99",
    "conjunction": "4 of 5 seeds",
    "weight": "realized-T* start-wave weight held fixed across the window (F6), on projection, truth, and floor",
    "floor": "half_vs_half real-vs-real, presence-conditioned, correlation-respecting split, runs/m6_holdout_floors_v1.json (NOT built here)"
  },
  "families": {
    "A_certifiable": "PSID-projected vs PSID-realized dynamics-drift flows (post-demotion, floor-priced)",
    "B_report_only_or_margin": "external alignment audit + alignment displacement + entrant cells (named bridge)",
    "C_binary_optional": "drift-direction fingerprints"
  },
  "registry_frictions": {
    "F1": "exogenous widowhood hazard vs mortality-induced widowhood",
    "F2": "household_composition embeds + re-simulates its own marital core (base_simulator.py:72-77)",
    "F3": "frozen core_seed=5200 has no period axis; engine-level injected generators (plumbing)",
    "F4": "gate-2c earnings axis is person-constant career capacity, not a contemporaneous cycle",
    "F5": "disability_hazard_sim preserves observed support (reproduction, not forward)",
    "F6": "weight semantics over the projection window (start-wave vs per-year calibrated)",
    "F7": "PSID attrition on the truth side (mortality worst)"
  },
  "ceremony_deliverables": ["floor_artifact", "recertification_margin_check", "OC_before_lock_weak_power_pause", "household_id_weight_rule", "claiming_vintage_freeze"],
  "non_goals": ["behavioral_response", "macro_feedback", "trust_fund_accounting_M7", "rules_on_whole_panel_M8", "new_spec_estimation", "validated_projection_beyond_holdout", "forward_engine_certification_deferred_to_successor_gate"],
  "process_addendum_bindings": {"pass_run_verification": "#42 comment 4948637741", "ladder_search_disclosure": "#42 comment 4948637741 + finding 1 spec-selection"}
}
```
