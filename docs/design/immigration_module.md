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
