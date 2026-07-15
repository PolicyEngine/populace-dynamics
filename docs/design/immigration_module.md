# Immigration module — new-entrant cohorts through the scheduled-entry seam

- **Design id**: `2026-07-15-immigration-module`
- **Roadmap**: [#113](https://github.com/PolicyEngine/populace-dynamics/issues/113),
  M6 immigration entry cohorts and the versioned Trustees alignment layer.
- **Status**: DESIGN DRAFT (revision 1; adversarial-referee adjudication pending).
  No immigration surface is certified by this document.
- **Engine baseline**: `75d30dd57d71b91ee0929246b2f3cbb92263b350`.
  File:line pins refer to that tree unless a different source is named.
- **Information boundaries**: the inherited M6 temporal gate boundary remains
  `T* = 2014`. A current 2026 Trustees assumption may drive a separately
  versioned forward scenario, but is not estimation evidence and cannot enter or
  re-anchor the inherited M6 scored path.
- **Non-change declaration**: this is a docs-only design. It edits no code, gate,
  floor, run artifact, certified model, threshold, or test.

## Revision log (finding → section)

- Dormant-generator finding: the engine already owns an entrant seam, but there
  is no immigration schedule generator → §3.1, §4.
- Net-is-not-entry finding: the 2026 Trustees component table distinguishes
  1.340 million gross new-person inflows from 0.130 million total net change in
  2026 → §2.4, §4.3, §4.10.
- No-history finding: a row inserted through the seam is absent from the cached
  marital, disability, household, and earnings support objects → §3.4, §4.7–4.9.
- Certificate-boundary finding: §2.8.3a, §2.8.2g, and amendment 3h prohibit
  repairing an unsupported universe by fabricating state or bypassing a live-
  roster guard → §3.3–3.5, §5.6.
- Donor-support finding: MINT's sparse annual cohorts and the public surveys'
  different universes require floors before any accuracy threshold → §2.1–2.3,
  §5.3.

## 0. Executive disposition

The immigration v1 object is a **pre-projection schedule builder**, not a ninth
`PeriodModules` step. It produces deterministic, pre-ID-assigned entry frames and
places only those frames at metadata key
`m6_scheduled_entries_by_year`. The existing engine then activates each frame at
the top of its period.

The proposed design makes four provisional choices, all reviewable by the
referee:

1. Size literal entrant cohorts from **gross new-person inflows**, not from net
   migration. In the 2026 Trustees table, adjustment of status is a
   reclassification and is not a new person.
2. Use recent-arrival ACS donor units for the joint entry state. Estimation
   creates a versioned donor artifact; the runtime schedule is deterministic
   conditional on that artifact, a binding manifest, and a schedule seed.
3. Copy donor information only through entry. Post-entry outcomes come from
   named populace-dynamics laws; no donor's future is cloned.
4. Keep emigration outside the entry builder. Therefore entry-only v1 is
   report-only for population stocks and cannot claim Trustees net alignment.
   A separately designed exit law is mandatory before that claim can be made.

The fourth choice is intentionally costly but honest. Substituting total net
change for actual entrants would create a reduced-form residual cohort whose age,
family, earnings, and nativity composition has no literal interpretation. In
2026, it would also replace 1.340 million gross arrivals with 0.130 million
residual persons. That alternative remains an explicit referee decision, not a
silent implementation shortcut.

## 1. Scope and non-goals

### 1.1 Roadmap authority and the scope it does not settle

Roadmap issue #113 is open and explicitly places immigration in M6:

> “M6 | **Projection engine**: year-by-year forward simulation to ~2100 —
> demographics (M3) + earnings forward + DI (M4) + immigration entry cohorts +
> an explicit, versioned alignment layer to Trustees intermediate assumptions”

It also defines the target panel:

> “The proposal here is that **the canonical object is a person-period panel**:
> populace persons × years (~2026–2100), with future births and immigrant
> entrants as synthetic persons entering the panel, per-year calibrated weights,
> and the full variable surface as column groups.”

Those statements authorize entry cohorts, synthetic person rows, calibrated
weights, and Trustees alignment. They do **not** decide gross versus net flows,
emigration, legal status, donor construction, entry timing, family-unit
materialization, or the state histories required by downstream modules. This
document proposes that missing scope.

### 1.2 In scope for v1

V1 specifies:

- a vintage-pinned annual cohort-control manifest;
- an estimated, versioned recent-arrival donor artifact;
- a deterministic builder for entry frames and module-native entry-state
  packets;
- activation exclusively through `SCHEDULED_ENTRIES_KEY`;
- invariant checks for time coordinates, IDs, ordinals, weights, roster
  presence, and provenance;
- a proposed additive `gate_imm` for entrant characteristics; and
- report-only reconciliation to SSA and Census population projections.

The design covers people whose latest entry places them in the United States for
the modeled spell. It does not assert that a survey's latest year-of-entry answer
identifies a first migration, a legal admission, or continuous U.S. residence.

### 1.3 Non-goals

V1 does not:

- model legal-status transitions, visa classes, removals, adjustment of status,
  naturalization, or status-specific program eligibility;
- infer undocumented status from ACS, CPS, or SIPP;
- treat foreign earnings as Social Security covered earnings;
- model return migration as a distinct state;
- implement emigration or an overseas-beneficiary population;
- modify mortality to disguise emigration;
- claim that SSA's Social Security-area population and the Census resident
  population are the same universe;
- clone a donor's post-entry life;
- certify long-run entrant earnings, DI, marriage, fertility, household,
  claiming, benefit, or population-stock outcomes;
- perform M7 trust-fund accounting or M8 rules execution; or
- change the inherited M6 gate, frozen floors, model registries, RNG registry, or
  module order.

### 1.4 Emigration boundary

Emigration is **out of the v1 entry builder** because it has a different risk
universe and event law: it removes a roster-present person, can end an employment
spell, may preserve accrued U.S. benefit rights, and depends on time since entry.
The current entrant seam only adds prebuilt rows. Reusing mortality, negative
weights, or a fabricated negative entrant cohort would confound distinct events.

This is a component boundary, not a declaration that emigration is unimportant.
MINT, DYNASIM, PENSIM, and PENSIM2 all provide evidence that literal entrant
modeling and population alignment require exits. Until a separate exit design is
ratified, v1 may certify only entry-time characteristics conditional on a gross
inflow control. Every stock, dependency ratio, benefit total, and net-migration
reconciliation remains report-only.

## 2. Evidence and adjudication

### 2.1 What predecessor models actually do

| Model/source | Cohort count and exits | Entrant state | Transportable lesson |
|---|---|---|---|
| DYNASIM4 | The 2024 overview says immigration adds people, immigration and emigration are separate, and immigration is aligned to OACT targets (Cosic, Johnson, and Smith, *[Urban's Dynamic Simulation of Income Model 4](https://www.urban.org/sites/default/files/2024-09/Urban%E2%80%99s%20Dynamic%20Simulation%20of%20Income%20Model%204.pdf)*, pp. 1–2). | The public 2024 overview does not document a detailed donor algorithm. | Cite DYNASIM4 only for separate flows and OACT alignment; do not attribute undocumented donor mechanics to it. |
| DYNASIM3 detail | Table 1 of Favreault, Smith, and Johnson, *[The Dynamic Simulation of Income Model (DYNASIM): An Overview](https://www.urban.org/sites/default/files/publication/67366/2000391-The-Dynamic-Simulation-of-Income-Model-DYNASIM-%20An-Overview.pdf)* (2015), p. 7, uses OACT/Dowhan-Duleep targets by sex, age at entry, and source region, and a separate SSA-data emigration hazard using entry age and origin. | The same table says observed post-1980 immigrants' life histories are donors. | Donor histories plus distinct exit hazards are a useful precedent, but this report describes DYNASIM3 and cannot establish DYNASIM4's detailed implementation. |
| MINT6/MINT8 | MINT6 derived gross flows from OACT net targets and an emigration hazard; MINT8 uses projected gross legal and other-than-legal entries and models emigration separately (Smith et al., *[Modeling Income in the Near Term Version 6](https://www.urban.org/sites/default/files/publication/24986/412479-Modeling-Income-in-the-Near-Term-Version-.PDF)*, ch. II §VI, pp. II-24–II-28; Smith and Favreault, *[MINT8 and 2014: Primer](https://www.urban.org/sites/default/files/publication/100965/modeling_income_in_the_near_term_8_and_2014_primer.pdf)*, pp. 15–16, note 21 p. 29, Table 3 pp. 39–40). | MINT6 uses post-1990 SIPP immigrants to initialize sex, immigration age, source region, marital history/status, financial assets, and employment at arrival. It then runs ordinary post-entry modules. | Copy a coherent entry packet, never a donor's future. An initializer and the later transition laws are separate estimands. |
| PENSIM | *[PENSIM Overview](https://www.retirementplanblog.com/wp-content/uploads/sites/304/2006/10/overview.pdf)* (Holmer, Janney, and Cohen, 2006), §2.1.6 p. 8, combines SSASIM/Trustees net immigration with native- and foreign-born emigration assumptions to derive gross flows; Appendix B §§B.1.2–B.1.4 pp. 100–101 schedules entry and exit from a life synthesized at birth. | A person's whole pre-entry life exists before the immigration event. | Whole-life synthesis avoids missing histories, but is not portable to a roster that materializes a person at entry. The needed analogue is an explicit entry-state packet. |
| PENSIM2 | O'Donoghue, Redway, and Lennon, *[Simulating migration in the Pensim2 dynamic microsimulation model](https://www.microsimulation.pub/articles/00039)* (2010), §5.1 and Figure 3, disaggregates ONS net controls into gross immigration and emigration. | §5.2 samples immigrant **families** from the 2003 Labour Force Survey and calibrates person totals; Table 2 contrasts cloning and synthetic approaches. | Preserve joint family state and distinguish the control unit from the donor/simulation unit. Net-only migration can bias population structure (§3 and §4). |

MINT is the closest architectural analogue. Earlier MINT versions cloned a
donor's later trajectory, but MINT6 “modifies this approach markedly”: donor
information stops at arrival and the entrant then passes through ordinary
modules (MINT6, pp. II-24–II-26). MINT6 also reports roughly 300 synthetic entrant
records per year and noisy single-age results (pp. II-25–II-28, Tables 2-15 and
2-16). That is direct evidence for support floors and against an arbitrarily fine
matching grid.

The SSA research trilogy reaches a compatible conclusion. Duleep and Dowhan,
“[Incorporating Immigrant Flows into Microsimulation Models](https://www.ssa.gov/policy/docs/ssb/v68n1/v68n1p67.html),”
*Social Security Bulletin* 68(1), 2008, §§“Projecting the Flow of Immigrants” and
“Giving the New Immigrants Earnings Profiles,” identifies flow projection and
earnings imputation as different tasks. It recommends immigrant donors grouped
by sex, age at migration, and source region, and warns that a parametric approach
may suppress within-cell variation. The companion “[Adding Immigrants to
Microsimulation Models](https://www.ssa.gov/policy/docs/ssb/v68n1/v68n1p51.html),”
pp. 51–66, shows why initial relative earnings and time since entry matter and why
emigration cannot be ignored.

### 2.2 Public microdata: roles and limitations

| Source | Exact public support | V1 role | Binding limitation |
|---|---|---|---|
| ACS PUMS | Census, *[2010–2014 ACS 5-year PUMS Data Dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2010-2014.pdf)* (Jan. 14, 2016): `SERIALNO`, `PWGTP`, `AGEP`, `CIT`, disability items, `MAR`, `SCHL`, `SEX`, `WAGP`, dual `YOEP05`/`YOEP12`, `NATIVITY`, `PINCP`, and place of birth; the *[Accuracy of the Data](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2010_2014AccuracyPUMS.pdf)*, §3 pp. 5–7 and §4 p. 8, defines the pooled sample and dual-variable rule. | Primary joint-state donor and gate truth, weighted by `PWGTP` with replicate-weight uncertainty. | A recent-arrival respondent is a resident survivor/stayer observed at interview, not a gross arrival. `YOEP` is the latest entry, and `WAGP` covers the prior 12 months, possibly including pre-entry months. The 2016 publication date versus `T*=2014` is an open admissibility decision (§6.2, decision O6). |
| CPS ASEC | Census, *[2014 Annual Social and Economic Supplement Technical Documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar14R.pdf)*: demographics p. 65; six disability items pp. 68–69; `PENATVTY`, grouped `PEINUSYR`, `PRCITSHP`, `MARSUPWT` p. 69; wage/salary and earnings pp. 77, 83. | Report-only marginal and earnings triangulation. | Smaller civilian-noninstitutional sample; grouped entry years; survey-date demographics versus prior-calendar-year income; 2014 Traditional and Redesign files must never be silently combined. |
| SIPP | Census, *[2014 SIPP Metadata, all sections v2](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2014/w1/2014SIPP_Metadata_AllSections_v2.pdf)*: `WPFINWGT` p. 9; marital status p. 23; age p. 30; nativity/citizenship pp. 35–37; grouped `TYRENTRY` and entry-status item `TIMSTAT` pp. 38–39; education p. 42; sex p. 43; disability p. 1368; monthly earnings p. 2766. | Report-only joint-state and initializer plausibility check. | Wave 1 covers the 2013 reference year and a civilian-noninstitutional universe. Later waves miss newly arrived immigrant-only households. `TIMSTAT` is neither a legal-history panel nor authority to model status and is excluded from v1 state. |

The ACS donor predicate is named, not implied:

```text
foreign_born == true
and 0 <= survey_year - latest_year_of_entry <= recent_arrival_max_duration
```

`survey_year` is decoded from `SERIALNO`; the correct `YOEP05`/`YOEP12`
classification is selected by collection year. The initial proposal is a 0–4
year window, but the exact duration and fallback hierarchy remain decision O3.
The extractor must bind allocation-flag treatment, top/bottom-code treatment,
group-quarters policy, dual classifications, replicate weights, and the use of
`ADJINC` before any artifact can be certified.

### 2.3 Donor-based versus model-based assignment

**Provisional adjudication: donor-based joint assignment, with model-based
calibration and no cloned future.** This preserves observed covariance among age,
sex, education, marital/family state, disability indicators, employment, and
earnings. A purely parametric set of independent draws is rejected for v1 because
passing marginal cells would not establish a coherent person or family packet.

“Donor-based” does not mean raw row copying:

- the donor pool, matching grid, fallback ladder, unit construction, weight
  trimming, allocation-flag policy, and calibration loss are fitted/versioned;
- survey concepts are mapped to named entrant concepts with missing/unsupported
  states rejected rather than guessed;
- a SIPP history donor, if used, is a second named imputation with its own joint-
  fidelity diagnostics;
- future outcomes are never present in the donor artifact; and
- runtime realization is deterministic conditional on the frozen artifact,
  controls, and seed.

A later parametric alternative may compete only on the same joint holdout
surface and floors. Lower marginal error alone is insufficient.

### 2.4 Gross inflow, net change, and status adjustment

The current control candidate is the *2026 OASDI Trustees Report*, §V.A.3 and
[Table V.A2, “Immigration Assumptions, Calendar Years 1940–2100”](https://www.ssa.gov/oact/TR/2026/lr5a2.html).
The assumptions were set in February 2026. Table V.A2 carries, separately, LPR
new-arrival inflow, LPR/citizen outflow, adjustment to LPR status, LPR net change,
temporary-or-unlawfully-present inflow, outflow, the offsetting status adjustment,
that stock's net change, and total net change.

For intermediate 2026, in thousands:

| Component | Persons (thousands) | Entry-seam meaning |
|---|---:|---|
| LPR new-arrival inflow | 600 | New-person inflow |
| Temporary/unlawfully present inflow | 740 | New-person inflow |
| **Gross new-person inflow** | **1,340** | Candidate entry-cohort total |
| LPR/citizen outflow | 263 | Exit, never an entrant |
| Temporary/unlawfully present outflow | 947 | Exit, never an entrant |
| Adjustment to LPR status | 450 in each stock, opposite signs | Internal reclassification; not a new person |
| **Total net change** | **130** | Reconciliation target, not a literal cohort |

The 75-year intermediate average total net change is 1.138 million, but Table
V.A2 is annual and materially non-flat; the annual component values, not the
average or ultimate constant, are the binding shape. Components may differ by
rounding.

The table's universe is the Social Security area, not the Census resident
population. The report's glossary includes residents of U.S. territories,
certain people abroad, and other groups absent from ACS PUMS. Therefore even
gross inflow is not directly a resident-population control. A named universe
bridge is required before production use (§6.1). Until then, both the gross entry
schedule and net reconciliation are report-only.

## 3. Binding repository and domain law

### 3.1 The only entrant activation seam

The engine already defines
[`SCHEDULED_ENTRIES_KEY = "m6_scheduled_entries_by_year"`](../../src/populace_dynamics/engine/loop.py#L27)
at `src/populace_dynamics/engine/loop.py:27`. V1 feeds that seam and creates no
parallel entry hook.

The baseline contract is exact:

- `PeriodModules` contains the eight ordered adapters plus an initial-frame hook,
  but no entrant adapter (`engine/loop.py:110-122`).
- Initialization runs once on the original starting slice before the schedule is
  read (`engine/loop.py:180-192`). Scheduled entrants do not pass through it.
- Each schedule key must satisfy `start_year < entry_year <= end_year`; its value
  must be a validated DataFrame with a single coordinate
  `year == entry_year - 1` (`engine/loop.py:192-211`).
- Every scheduled `person_id` is pre-supplied, globally unique across the initial
  slice and all cohorts (`engine/loop.py:213-221`).
- The dynamic synthetic-ID allocator starts above the maximum initial or
  scheduled ID (`engine/loop.py:222-228`).
- Stable person RNG ordinals are preassigned over the complete initial-plus-
  scheduled ID universe before period 1 (`engine/loop.py:234-237`).
- At entry year, the frame is concatenated and stably sorted at the very top of
  the period (`engine/loop.py:239-257`), before mortality and the remaining
  ordered modules (`engine/loop.py:258-321`).
- Newborns and any other dynamically created rows receive ordinals only after the
  projected slice validates (`engine/loop.py:323-331`).

The seam is not wholly dormant. `M6RealizedPopulation.projection_metadata`
already publishes later PSID openers through the same key
(`harness/m6_population.py:49-73`), and the realized-population builder schedules
2017/2019 openers (`harness/m6_population.py:313-335`). The missing object is an
**immigration/new-entrant generator and its state packet**, not a generic engine
entry mechanism.

### 3.2 Consequences for timing, IDs, and RNG

An entry frame is a pre-period state. Mortality sees it first
(`engine/steps.py:113-134`); aging then adds one to `age` and writes the target
year (`engine/steps.py:137-166`). The existing PSID opener builder deliberately
stores the prior year coordinate while retaining the observed collection-wave
age (`harness/m6_population.py:313-320`). That convention does not by itself
settle whether an annual gross immigrant inflow should receive a full-year,
half-year, or no domestic mortality exposure in its entry year.

V1 therefore requires a named `entry_timing_basis` and a pre/post-aging age
identity check. Decision O2 must settle the convention before implementation;
the builder may not hide an age shift inside donor extraction. Whatever is
chosen, the schedule frame must use the seam's `year = entry_year - 1` contract.

IDs are assigned outside the loop in deterministic order. They must be finite
integers, greater than every starting-population ID, collision-free across all
years, and invariant to row order. These conditions preserve every original
person's sorted-ID ordinal. The loop then places the newborn allocator above all
scheduled IDs, so entrant IDs cannot collide with later births. No existing M6
module stream is consumed to construct the schedule.

### 3.3 Amendment 3h: live-roster materialization

Amendment 3h is not merged at the baseline commit. Its public source is the
[3h forensics/adjudication](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4984997277)
and draft [PR #216](https://github.com/PolicyEngine/populace-dynamics/pull/216).
This sibling design adopts its domain law as a dependency while accurately
describing its pending merge status:

> Scheduled open-addition events may materialize a related person only when the
> required parent/person is present in the live post-mortality roster. The
> frame-independent risk schedule may remain the scoring universe; absent-parent
> events are dropped after the draw and reconciled report-only so RNG addresses
> do not shift.

Entrants are roster-present when their own rows are activated. Their later
marriages, births, household links, and exits must still obey live-roster
materialization. A donor packet may not create a spouse, child, or other relation
whose person row is absent, and it may not weaken the existing birth-parent guard
at `engine/steps.py:409-414`.

### 3.4 Why an entry row is not an initialized person

The current certified assembly consumes fitted objects and native-panel builders
(`engine/assembly.py:81-99`). Several adapters construct and cache a whole-window
projection once per draw:

- marital support is built once and cached (`engine/assembly.py:271-295`), and
  its certified builder deliberately ignores the live frame
  (`engine/panel_builders.py:150-165`);
- fertility draws over the cached marital IDs (`engine/assembly.py:297-324`);
- disability simulates once over realized PSID support
  (`engine/assembly.py:326-377`);
- household composition builds and caches its native panel
  (`engine/assembly.py:385-424`); and
- earnings initialization runs only on the original initial slice
  (`engine/assembly.py:239-250`; `engine/loop.py:180-185`).

Worse, `_merge_period_columns` drops existing state columns before left-merging
the cached update (`engine/assembly.py:174-192`). An immigrant absent from that
update receives missing state even if the scheduled row carried a seed value.
The later PSID openers work because their realized support is known to the
whole-window builders in advance. An unknown synthetic immigrant has no such
support.

Therefore the immigration generator must produce both an entry row and an
explicit, module-native **EntrantStateBundle**. Merely adding more columns to the
entry frame is rejected.

### 3.5 The §2.8.3a / §2.8.2g / 3h domain-law family

The sibling M6 design's three laws bind this module:

- **Earnings §2.8.3a**: the certified forward generator's domain is the
  intersection with realized 2014 earnings state. Missing/new rows are false in
  `earnings_domain` (`engine/earnings_domain.py:68-100`); the initializer writes
  zero/missing outside that domain and validates exact fitted membership
  (`engine/earnings_domain.py:156-210`). An immigrant may not be assigned a fake
  2014 anchor, `u_w`, or PSID history to enter that domain.
- **Marital §2.8.2g**: preserve the invariant guard and repair an unsupported
  universe at builder/domain entry. The immigrant analogue is to construct a
  named entrant risk-set seed; it is not to weaken the core or pretend the PSID
  certificate covers the new population.
- **Fertility amendment 3h**: distinguish a frame-independent scoring schedule
  from live-roster materialization. No related row may be materialized against an
  absent person merely because that person exists in a donor or cached support
  artifact.

No existing certificate transfers across these bridges. Reuse of unchanged core
code may be plumbing at implementation time, but applying it to an entrant
population is a new domain claim and remains report-only until its expressly
named successor gate passes.

## 4. Proposed design

### 4.1 Components and one-way data flow

The design separates source estimation, scenario binding, schedule realization,
and period activation:

```text
ACS <=T* records ──fit──> ImmigrationDonorArtifact ─┐
                                                    ├─> ImmigrationScheduleBuilder
SSA annual components ──bind──> BindingManifest ───┘              │
                                                                  ├─> entry frames
SIPP/CPS ──diagnostics only───────────────────────────────────────┤
                                                                  ├─> EntrantStateBundle
Census scenarios ──report-only corridors─────────────────────────┘

entry frames ──metadata[SCHEDULED_ENTRIES_KEY]──> existing engine loop
EntrantStateBundle ──entrant-only adapters/builders──> downstream state
```

The conceptual products are:

- **`ImmigrationBindingManifest`**: immutable source/vintage/schema/hash records,
  annual flow controls, universe labels, and scenario identity;
- **`ImmigrationDonorArtifact`**: a fitted pool of recent-arrival person or family
  units, matching cells, fallback hierarchy, concept mappings, and provenance;
- **`EntrantStateBundle`**: the entry frame plus module-native initial state and
  exposure objects for the same people;
- **`ImmigrationSchedule`**: `entries_by_year`, the state bundle, deterministic
  ID ledger, and reconciliation tables; and
- **`ImmigrationAudit`**: cohort totals, donor support, calibration residuals,
  universe caveats, dropped/unsupported units, hashes, and certification labels.

`ImmigrationSchedule.entries_by_year` is the only product sent to
`SCHEDULED_ENTRIES_KEY`. The state bundle is consumed by explicit entrant-side
builders/adapters assembled before projection. There is no entrant event draw in
the annual engine loop.

### 4.2 Estimation versus deterministic realization

The campaign's estimation/determinism split is binding:

**Estimated and versioned**

- recent-arrival definition and survey concept map;
- donor unit, matching variables, and fallback order;
- sampling/calibration loss and weight trimming;
- any ACS→SIPP joint-history imputation;
- any entrant-specific earnings or disability initializer; and
- every evaluation floor and eventual gate threshold.

**Deterministic conditional on frozen inputs**

- parsing the annual external-control table;
- selecting an artifact by exact ID/hash;
- schedule realization under an explicit schedule seed;
- synthetic IDs and arrival-unit IDs;
- conversion to the seam's prior-year frame coordinate;
- cohort weight calibration to the bound control; and
- serialization and audit calculations.

The schedule is built **once per scenario before the K projection draws** and is
reused across those draws. It does not consume any of the eight existing M6
period-module streams. Sampling uncertainty is studied by separately named
schedule seeds/artifacts; it is not accidentally mixed into the engine's
process-error draws.

### 4.3 Annual cohort sizing

For Trustees calendar year `y`, define the unbridged new-person control

```text
G_ssa[y] = 1,000 * (
    V_A2_intermediate[y].lpr_new_arrival_inflow
    + V_A2_intermediate[y].temporary_or_unlawfully_present_inflow
)
```

Adjustment of status is excluded because the same person moves between the two
Trustees stocks. Outflows are excluded because they are exit events. The
intermediate total-net column is retained as `N_ssa[y]` for reconciliation only.

`G_ssa` is not yet a usable resident-population cohort. Production construction
requires an adjudicated, vintage-pinned bridge
`G_resident[y] = bridge(G_ssa[y])` between the Social Security-area and ACS
resident universes. Until decision O1 supplies that bridge, the builder must
either hard-stop or produce an explicitly named `ssa_area_proxy` schedule whose
entire projection is report-only. It may not silently set the bridge to identity
and label the result resident-population aligned.

The schedule uses a manageable synthetic sample and positive calibration
weights; it does not create 1.34 million physical rows in 2026. For each year:

- the number of donor units is set by support/precision requirements established
  in the floor ceremony, not by the external population count;
- unit sampling preserves all members selected together;
- person weights are finite and positive;
- the sum of person weights equals `G_resident[y]` within a pinned numerical
  tolerance;
- age/sex and other composition margins come from the frozen donor artifact,
  never from Table V.A2, which has no public age/sex detail; and
- no unannounced ultimate-value or nearest-year fallback is allowed. A missing
  year is a hard failure.

Low-cost and high-cost Trustees series are separately named report-only
scenarios. The 75-year average is a disclosure, never a replacement for annual
values.

### 4.4 Donor and simulation unit

PENSIM2's family sampling exposes a real unit mismatch: the external control is
people, while the state to preserve may be a family. V1's provisional unit is a
**co-resident co-arrival unit**, not a claimed historical travel party:

1. Start with a PUMS household (`SERIALNO`).
2. Select foreign-born recent-arrival people with the same latest-year-of-entry
   classification.
3. Preserve spouse/partner and parent/minor-child links only when both endpoints
   are in that selected set.
4. Assign a new `arrival_unit_id`; never expose `SERIALNO` as a synthetic person
   or household identifier.

A spouse, child, or parent who is native-born, entered in a different year, or is
absent from the PUMS household is not cloned. The entrant may retain an observed
marital/parental state with a named `relation_outside_entry_roster` marker, but
no related synthetic row or relationship ID is invented. This is required by
amendment 3h.

Sampling whole units while calibrating person totals requires an explicit rule
for within-unit weights. The provisional invariant is one common simulation
weight for every member of an arrival unit, with a unit-selection/calibration
algorithm that reproduces person margins. That is not yet binding; decision O4
may instead choose person donors with report-only relationship structure. The
floor ceremony must price whichever unit is selected.

### 4.5 Characteristic assignment

The fitted donor artifact preserves, at minimum:

- age at the named pre-period/entry timing coordinate;
- sex;
- education and enrollment;
- source-region grouping derived from place of birth;
- race and Hispanic-origin fields only where a downstream/reporting contract
  already has a defined concept;
- marital status/history fields available at the survey date;
- co-resident relationship structure and unit size;
- ACS disability-question indicators, without relabeling them DI status;
- employment, weeks/hours where available, wage/salary earnings, self-employment
  earnings, and zero-earnings status; and
- survey year, latest year of entry, observed duration, allocation flags, donor
  weight, and all concept-map provenance.

The default matching ladder begins with sex × broad age-at-entry × source region
and then uses education and family state when support permits, consistent with
the MINT/Duleep-Dowhan precedent. The precise cells are selected and frozen
before candidate scoring. Fallbacks coarsen in a published order; they never
cross a prohibited concept boundary merely to fill a cohort. Every fallback
count appears in the audit.

Long-run composition is held at the donor artifact's calibrated distribution
unless a separately sourced, gate-reviewed composition trajectory is bound.
Trustees aggregate totals cannot be used to manufacture one. Constant
composition through 2100 is therefore a visible extrapolation limitation, not an
empirical forecast claim.

### 4.6 Schedule seed, IDs, and reproducibility

The schedule PRNG address is conceptually

```text
(schedule_seed, binding_manifest_hash, donor_artifact_hash,
 entry_year, unit_slot, member_slot, purpose_tag)
```

It is independent of `ProjectionRNGRegistry`. Unit selection, tie-breaking, and
any stochastic calibration use named `purpose_tag` values so adding one draw does
not shift another purpose's addresses.

After the full schedule is realized, IDs are allocated in stable order
`(entry_year, unit_slot, member_slot)` beginning above the maximum initial
`person_id`. The builder stores the allocation ledger and verifies global
uniqueness before calling the engine. Arrival-unit and household IDs use their
own namespaces and may not alias person IDs.

Changing a source vintage, donor artifact, recent-arrival definition, or schedule
seed creates a new schedule identity. The design does not promise entrant or
newborn byte identity across such scenarios. It does promise that, within one
schedule, pre-supplying entrant IDs above the initial maximum leaves every
original person's sorted-ID ordinal unchanged and lets the engine place its
dynamic allocator above every scheduled entrant.

### 4.7 Entry-frame schema and hard invariants

Every scheduled entry frame contains these conceptual groups:

| Group | Required fields |
|---|---|
| Engine | `person_id`, `year`, `age`, `sex`, `weight` |
| Entry identity | `synthetic_entry = true`, `entry_kind = "immigration"`, `entry_year`, `arrival_unit_id`, `foreign_born = true` |
| Timing | `entry_timing_basis`, `age_timing_basis`, `years_since_entry = 0` |
| Provenance | binding-manifest ID/hash, donor-artifact ID/hash, schedule ID/seed, donor-cell and fallback code |
| Static/donor state | education, source region, marital/family seed, employment/earnings seed, disability-concept seed, household seed |
| Domain markers | `earnings_domain = false` and named entrant-domain flags for every composite adapter |

No `legal_status` field is inferred. A source-stock component from Table V.A2 may
be retained only at aggregate manifest level; it is not assigned to a person.

Before engine invocation the builder hard-checks:

- mapping keys are integer entry years in the projection range;
- each frame has exactly `year = entry_year - 1`;
- pre/post-aging age identities match decision O2;
- person IDs are finite integers, globally unique, and greater than the initial
  maximum;
- all relation endpoints are either scheduled no later than the relation's
  materialization year or explicitly marked outside-roster;
- weights are finite, positive, and reconcile at person and unit level;
- every required downstream state is supplied by the state bundle, not merely a
  frame column that a later merge will erase;
- no donor future outcome or post-cutoff gate input is present; and
- every source, parser, derived artifact, and schedule payload matches its bound
  hash.

Any failure aborts before period 1. There is no nearest-year, nearest-vintage,
native-donor, legal-status, or fabricated-history fallback.

### 4.8 `EntrantStateBundle`

The state bundle is atomic by scheduled person ID and contains:

| Native support object | Purpose |
|---|---|
| `entry_frames_by_year` | The exact frames sent through the seam. |
| `entrant_marital_seed` | Entry state/history, risk-set start year, relation IDs only for roster-present people, and entrant-domain marker. |
| `entrant_household_seed` | Arrival-unit household links, household state, and entry-year exposure mask. |
| `entrant_disability_seed` | Survey-concept indicators plus the explicitly estimated mapping to any entrant disability state. |
| `entrant_earnings_seed` | Entry employment/earnings state, zero pre-entry U.S.-covered history, any entrant-specific lag state, and normalization provenance. |
| `entrant_relationship_roster` | Parent/spouse/child endpoints and the first year each endpoint can materialize. |
| `entrant_audit` | Donor support, fallback, imputation, weight, universe, and hash records. |

The packet must be coherent. State surfaces come from the same donor unit unless
a separately fitted joint imputation is named and validated. Independent
hot-decks for education, marriage, disability, and earnings are rejected because
they can construct combinations absent from the observed support while still
passing marginal checks.

The packet includes only state through entry and the minimum pre-entry history
needed to define a downstream covariate. It never contains realized post-entry
outcomes. Foreign earnings may inform education/occupation matching if a future
source supports it, but U.S. Social Security covered earnings before modeled
entry are zero. Return migration is not inferred from a latest-entry response;
handling prior U.S. coverage is a successor design.

### 4.9 Downstream module-state contract

| Existing step | Entrant analogue | Certification boundary |
|---|---|---|
| Mortality | An activated entrant mechanically receives the existing age/sex mortality draw before aging. Decision O2 must define first-year exposure. | Mortality drift is already report-only; applying it to entrants is not newly certified. Emigration may not be encoded as excess mortality. |
| Aging | The existing deterministic `advance_age` runs unchanged after the prior-year coordinate check. | Plumbing only; the entry-age convention, not the function, is the new law. |
| Marital core | An entrant-side panel builder supplies one admissible entry seed and risk-set start. A production open-panel adapter may combine markets only in a report-only run. | Candidate-16's PSID certificate does not transfer to immigrants or to cross-domain matching. Closed-panel scored outputs must remain unchanged. |
| Fertility | Co-arriving children are entrant rows. Children conceived/born after entry use the normal fertility path and amendment-3h live-roster materialization. | No absent donor relative may be materialized. Entrant fertility remains report-only. |
| Disability | ACS disability questions and SIPP work-limit/benefit concepts feed a separately estimated entrant initializer/forward law; they are never relabeled as the realized PSID M4 status. | M4 reproduction support and certificate stay unchanged. Entrant disability is a successor-gate surface. |
| Earnings | Entrants remain `earnings_domain = false`. A separate immigrant-entrant generator owns entry earnings and all later entrant earnings until a future handoff law is designed. It uses zero pre-entry U.S.-covered earnings and a pinned wage normalization. | No fake 2014 earnings, `u_w`, `gen_earn_w2`, or `gen_earn_w4`; the §2.8.3a generator is untouched and its certificate does not transfer. |
| Claiming | The schedule may be reused as plumbing, but insured status and AIME/PIA must use only simulated U.S.-covered earnings. | Entrant claims and benefits are report-only until earnings/history and eligibility concepts are certified. |
| Household composition | An entrant-side native panel starts exposure at entry, carries arrival-unit links, and makes no link to an absent person. | Candidate-9's PSID certificate does not transfer. Open-market effects on existing people are report-only. |

The scored M6 closed-panel run and an open-population production run are distinct
products. Entrant interactions may affect existing people's marriages,
households, and births in the open run, but those effects cannot be allowed to
retroactively change the frozen closed-panel gate or its certificate.

### 4.10 Emigration and net reconciliation

The v1 audit publishes, by year:

```text
gross_entry_control
scheduled_entry_weight
trustees_gross_outflow_required
trustees_status_adjustment_reclassification
trustees_total_net_target
entry_only_minus_net_gap
universe_bridge_status
```

With no exit law, `entry_only_minus_net_gap` is expected and disqualifies any net
alignment claim. A chart may show what the Trustees total-net path would imply,
but no population stock produced by entry-only v1 is “aligned” to that path.

A successor emigration design must decide at least the exit timing/order, risk
population, age/sex/source/duration hazard, family versus individual exit,
return-entry identity, employment termination, household reconciliation,
accrued-coverage and overseas-benefit state, death-versus-exit observability, and
RNG addressing. Table V.A2 supplies aggregate outflow controls, not those hazards.

**Rejected for production:** feeding total net change through the entry seam and
calling the rows immigrants. If the referee authorizes a reduced-form
`net_entry_proxy` experiment, its metadata, report, and output columns must say
that exact name; it cannot enter `gate_imm`, family-unit evaluation, entrant
characteristic claims, or M7 accounting.

### 4.11 Fail-closed audit behavior

Every build writes a machine-readable audit before projection. The audit includes
source and derived hashes, retrieval/publication/observation dates, parser and
schema versions, schedule seed, ID ranges, per-year physical rows and weighted
totals, donor-cell effective sample sizes, fallback counts, calibration residuals,
top-code/allocation shares, unit-size distribution, state-bundle completeness,
and all report-only/certified labels.

The builder refuses to run when a binding is missing or mutable, a raw or derived
hash differs, a required year is absent, a unit crosses an unsupported concept,
an outside-roster relation is assigned an ID, an entry is accidentally admitted
to the certified earnings domain, or the universe bridge is absent for a run
claiming resident alignment. Network access and runtime redownload are prohibited
on a scored or production run.

## 5. Evaluation and proposed `gate_imm`

### 5.1 Gate identity and estimand

`gate_imm` is a proposed **additive successor gate**. This document does not add
it to `gates.yaml`, choose tolerances, build a floor, or lock an artifact.

The candidate estimand is narrow:

> Conditional on the named recent-arrival resident-stock proxy and its survey
> universe, does the frozen donor/schedule procedure reproduce held-out ACS
> entry-state marginals and named joint distributions at their empirical noise
> floor?

This is not a gate on external cohort counts: exact agreement with a forced
control is tautological. It is not truth for gross arrivals, people who left or
died before interview, legal status, or post-entry trajectories.

### 5.2 Temporal split and leakage fence

The primary proposed split uses collection-year information inside the
2010–2014 ACS 5-year PUMS:

- fit donor construction and all matching/calibration choices on collection
  years 2010–2013, recovered from `SERIALNO`;
- freeze the candidate artifact and surface;
- generate a synthetic 2014 recent-arrival resident cross-section without
  reading 2014 person records; and
- score against collection year 2014 using the same recent-arrival predicate,
  concept map, universe, and weight treatment.

Households/arrival units are indivisible. The truth and candidate normalize to
the same total before characteristic scoring; the total itself is not a gate
cell. No 2014 characteristic, marginal, top-code treatment selected after seeing
the holdout, CPS/SIPP statistic, current Trustees assumption, or Census projection
may affect the fit.

The 2010–2014 pooled file was published in January 2016. Decision O6 must ratify
the observation-date rule (all person observations are `<=T*`) or reject the file
under a publication-date rule. Rejection pauses the gate; it does not silently
move the boundary. A fallback using individually pinned one-year files published
by 2014 requires a new design amendment and floor.

Later ACS windows, including 2015–2019 and 2020–2024, are temporal-drift stress
tests only under this gate. Refitting on them creates a new production artifact
to which no exact-artifact certificate automatically transfers.

### 5.3 Floors before thresholds

The ceremony order is mandatory:

1. Freeze the recent-arrival predicate, donor/simulation unit, concept map,
   matching ladder, gate cells, metrics, and weighting rules without candidate
   holdout results.
2. Construct a correlation-respecting real-vs-real floor from deterministic,
   household-disjoint splits or survey replicate-weight pseudo-replicates of the
   2014 truth. Never split members of one donor/household unit.
3. Publish, for every proposed cell, raw person count, raw unit count, weighted
   effective sample size
   `n_eff = (sum(w))^2 / sum(w^2)`, denominator, allocation share, top-code share,
   and the distribution of the chosen distance under truth-vs-truth comparison.
4. Prune, pool, or demote every cell that fails predeclared support or whose floor
   is unstable. Record each demotion; do not widen a threshold to retain it.
5. Price thresholds from the surviving empirical floors using a predeclared
   transform. No numerical tolerance is assumed by this design.
6. Run an operating-characteristic experiment on the surviving conjunction
   using predeclared degraded pseudo-candidates. If power is weak or the gate is
   near-tautological, pause and redesign before any lock.
7. Only then score the registered candidate and, if it passes, bind the exact
   artifacts, floor, registry, code commit, and hashes.

This order is the M6 floors-before-thresholds law. Sparse single-age MINT6 cohorts
and the ACS weighted/unit structure make it substantive, not ceremonial.

### 5.4 Candidate gate surface

The floor ceremony may consider these predeclared families:

| Family | Candidate observables | Candidate metric |
|---|---|---|
| Demographic | broad age-at-entry × sex shares; arrival-duration band; broad source region | total-variation or weighted absolute-share distance |
| Education | attainment band overall and by broad age/sex | total-variation distance |
| Family state | marital-state shares; co-arrival unit size; spouse/child co-arrival indicators | total-variation distance, unit-weighted where applicable |
| Employment | employed/unemployed/NILF; zero earnings; work-intensity bands | absolute-share or total-variation distance |
| Positive earnings | `WAGP`/earnings normalized under the pinned economic index, p10/p50/p90 and log spread | floor-scaled log-quantile distance |
| Disability proxy | the six ACS question concepts individually and “any” | absolute-share distance; never named DI status |
| Named joints | age × sex × education; sex × marital state × duration; education × employment/earnings band | pooled-cell total-variation distance |

The final registry contains only cells that clear the floors. Fine source-region,
single-age, detailed education, high-order family, and earnings-tail cells are
presumptively report-only until their support proves otherwise.

### 5.5 Report-only evaluation

The following are useful but are not gate truth:

- CPS ASEC 2014 foreign-born tables and microdata marginals for age/sex, marital
  status, education, total money income, and earnings;
- SIPP 2014 Wave 1 joint plausibility for marital, employment/earnings,
  disability, and nativity/entry concepts;
- ACS 2015–2019 and 2020–2024 drift from the frozen donor composition;
- 2026 Trustees low-cost/intermediate/high-cost gross inflows, outflows, and
  total-net paths;
- the 2023 Census National Population Projections main/high/low/zero immigration
  corridors; and
- foreign-born population stocks, dependency ratios, AIME/PIA, DI, claiming,
  household, and benefit outputs.

The Census “zero immigration” scenario is especially easy to misstate: its
method sets gross foreign-born immigration to zero while leaving emigration and
net native migration in place, so net international migration may be negative.
Agreement with any Census corridor is a cross-model comparison, not validation.

### 5.6 What a PASS would and would not certify

A `gate_imm` PASS could certify only:

- the exact donor artifact, concept map, schedule procedure, seed protocol, and
  surviving registry named in the lock;
- reproduction of the named held-out ACS recent-arrival resident-stock
  cross-sections within floor-priced thresholds; and
- deterministic conversion of that artifact into seam-valid entry frames and a
  complete entrant-state packet.

It would not certify gross-flow truth, the Social Security-area→resident bridge,
emigration, legal status, current-vintage refits, composition through 2100,
entry-year mortality exposure, donor representativeness for people who left or
died before interview, any post-entry transition, any interaction with the
closed population, or any Social Security eligibility/benefit result.

It also would not modify or extend the M6 certificate. Entrants remain family-B
open additions under `m6_reporting.py:71-73` and its explicit immigrant
person-row bridge at `m6_reporting.py:104-118` until a later ratified gate says
otherwise.
