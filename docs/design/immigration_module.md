# Immigration module — new-entrant cohorts through the scheduled-entry seam

- **Design id**: `2026-07-15-immigration-module`
- **Roadmap**: [#113](https://github.com/PolicyEngine/populace-dynamics/issues/113),
  M6 immigration entry cohorts and the versioned Trustees alignment layer.
- **Status**: DESIGN DRAFT (revision 3; adversarial-referee adjudication pending).
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

- Revision 3 replaces invalid collection-year slicing of five-year ACS PUMS with
  separately weighted annual 1-year files; freezes year-namespaced units, annual
  replicate designs and cross-year normalization; adds component-aware timing,
  conditional controls, a projection-origin/first-cohort decision, and explicit
  module-native initializer blockers → §2.2, §3.2, §4.2–4.9, §5, §6,
  O2/O3/O5/O6/O11–O15.
- Revision 2 incorporates the pre-PR adversarial review: it distinguishes the
  ACS survivor/stayer stock proxy from state at arrival; treats prior U.S.
  covered earnings as censored for possible return entrants; adds fertility,
  claiming, flow-to-stock, and stock-to-arrival blockers; composes immigrant IDs
  with the existing 2017/2019 schedule; guards caller-supplied allocators; and
  requires entrant RNG isolation before any byte-identity claim → §2.2–2.4,
  §3.1–3.4, §4.1–4.11, §5, §6, O2/O5/O8–O15.
- Dormant-generator finding: the engine already owns an entrant seam, but there
  is no immigration schedule generator → §3.1, §4.
- Net-is-not-entry finding: the 2026 Trustees component table distinguishes
  1.340 million positive stock-accounting inflows from 0.130 million total net
  change in 2026 → §2.4, §4.3, §4.10.
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

1. Use the positive inflow components of the 2026 Trustees stock accounting, not
   net migration, as the candidate size control. In that table, adjustment of
   status is a reclassification and not a new person. The temporary/unlawfully
   present inflow counts only people who remain to year-end, so it is not a count
   of every border arrival.
2. Use recent-arrival ACS donor units for a joint **resident survivor/stayer
   stock proxy**. A separately estimated stock-to-arrival bridge must backcast or
   otherwise map that proxy before it can be called entry-time state. The runtime
   schedule is deterministic conditional on the frozen artifacts, binding
   manifest, and schedule seed.
3. Copy donor information only through entry. Post-entry outcomes come from
   named populace-dynamics laws; no donor's future is cloned.
4. Keep emigration outside the entry builder. Therefore entry-only v1 is
   report-only for population stocks and cannot claim Trustees net alignment.
   A separately designed exit law is mandatory before that claim can be made.

The fourth choice is intentionally costly but honest. Substituting total net
change for a positive entrant control would create a reduced-form residual cohort whose age,
family, earnings, and nativity composition has no literal interpretation. In
2026, it would also replace a 1.340 million positive stock-accounting inflow proxy
with 0.130 million residual persons. The 1.340 million itself is not all physical
arrivals during the year. That alternative remains an explicit referee decision,
not a silent implementation shortcut.

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

The design covers people whose **reported year of entry** places them in the
recent-arrival proxy for the modeled spell. ACS interviewers request the most
recent entry from repeat entrants, but mail/self responses can be first or most
recent; the public item therefore does not establish a first migration, legal
admission, or continuous U.S. residence.

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
ratified, v1 may certify only the named recent-arrival stock-proxy characteristics
conditional on a positive inflow control. Every stock, dependency ratio, benefit total, and net-migration
reconciliation remains report-only.

## 2. Evidence and adjudication

### 2.1 What predecessor models actually do

| Model/source | Cohort count and exits | Entrant state | Transportable lesson |
|---|---|---|---|
| DYNASIM4 | The 2024 overview says immigration adds people, lists immigration and emigration separately, and says immigration is aligned to OACT targets (Cosic, Johnson, and Smith, *[Urban's Dynamic Simulation of Income Model 4](https://www.urban.org/sites/default/files/2024-09/Urban%E2%80%99s%20Dynamic%20Simulation%20of%20Income%20Model%204.pdf)*, pp. 1–2). | The public 2024 overview does not document a detailed donor algorithm. | Cite DYNASIM4 only for the listed flows and OACT alignment; do not attribute undocumented separation or donor mechanics to it. |
| DYNASIM3 detail | Table 1 of Favreault, Smith, and Johnson, *[The Dynamic Simulation of Income Model (DYNASIM): An Overview](https://www.urban.org/sites/default/files/publication/67366/2000391-The-Dynamic-Simulation-of-Income-Model-DYNASIM-%20An-Overview.pdf)* (2015), p. 7, uses OACT/Dowhan-Duleep targets by sex, age at entry, and source region, and a separate SSA-data emigration hazard using entry age and origin. | The same table says observed post-1980 immigrants' life histories are donors. | Donor histories plus distinct exit hazards are a useful precedent, but this report describes DYNASIM3 and cannot establish DYNASIM4's detailed implementation. |
| MINT6/MINT8 | MINT6 derived gross flows from OACT net targets and an emigration hazard; MINT8 uses projected gross legal and other-than-legal entries and models emigration separately (Smith et al., *[Modeling Income in the Near Term Version 6](https://www.urban.org/sites/default/files/publication/24986/412479-Modeling-Income-in-the-Near-Term-Version-.PDF)*, ch. II §VI, pp. II-24–II-28; Smith and Favreault, *[Modeling Income in the Near Term 8 and 2014: Primer](https://www.urban.org/sites/default/files/publication/100965/modeling_income_in_the_near_term_8_and_2014_primer.pdf)*, pp. 15–16, note 21 p. 29, Table 3 pp. 39–40). | MINT6 uses post-1990 SIPP immigrants to initialize sex, immigration age, source region, marital history/status, financial assets, and employment at arrival. It then runs ordinary post-entry modules. | Copy a coherent entry packet, never a donor's future. An initializer and the later transition laws are separate estimands. |
| PENSIM | Holmer, Janney, and Cohen, *[PENSIM Overview](https://www.retirementplanblog.com/wp-content/uploads/sites/304/2006/10/overview.pdf)* (Policy Simulation Group for the U.S. Department of Labor, Employee Benefits Security Administration, Sept. 2006), §2.1.6 p. 8, combines SSASIM/Trustees net immigration with native- and foreign-born emigration assumptions to derive gross flows; Appendix B §B.1 p. 99 begins a synthetic life at age zero, §§B.1.2–B.1.3 p. 100 schedule immigration/emigration, and §§B.7–B.8 pp. 106–107 execute them. | A person's whole pre-entry life exists before the immigration event. | Whole-life synthesis avoids missing histories, but is not portable to a roster that materializes a person at entry. The needed analogue is an explicit entry-state packet. |
| PENSIM2 | O'Donoghue, Redway, and Lennon, *[Simulating migration in the Pensim2 dynamic microsimulation model](https://www.microsimulation.pub/articles/00039)* (2010), §5.1 and Figure 3, disaggregates ONS net controls into gross immigration and emigration. | §3 discusses synthetic generation versus cloning and Table 2 inventories model components; §5.2 samples immigrant **families** from the 2003 Labour Force Survey and calibrates person totals. | Preserve joint family state and distinguish the control unit from the donor/simulation unit. Net-only migration can bias population structure (§3 and §4). |

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
| ACS PUMS | Census annual 1-year person PUMS files for [2010](https://www2.census.gov/programs-surveys/acs/data/pums/2010/1-Year/csv_pus.zip), [2011](https://www2.census.gov/programs-surveys/acs/data/pums/2011/1-Year/csv_pus.zip), [2012](https://www2.census.gov/programs-surveys/acs/data/pums/2012/1-Year/csv_pus.zip), [2013](https://www2.census.gov/programs-surveys/acs/data/pums/2013/1-Year/csv_pus.zip), and [2014](https://www2.census.gov/programs-surveys/acs/data/pums/2014/1-Year/csv_pus.zip); annual dictionaries [2010](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict10.pdf), [2011](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict11.pdf), [2012](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict12.pdf), [2013](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.pdf), and [2014](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict14.pdf); and corresponding annual Accuracy statements ([2010](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2010AccuracyPUMS.pdf), [2011](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2011AccuracyPUMS.pdf), [2012](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2012AccuracyPUMS.pdf), [2013](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2013AccuracyPUMS.pdf), [2014](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2014AccuracyPUMS.pdf)). The *[2010–2014 PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2010-2014.pdf)* is crosswalk evidence only. The *[2014 ACS Subject Definitions](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2014_ACSSubjectDefinitions.pdf)*, “Year of Entry,” pp. 128–129, documents response ambiguity. | Primary joint-stock donor and gate truth only for the named stock-proxy surface. Fit 2010–2013 with each annual file's `PWGTP` and replicate weights; hold the separately weighted 2014 file out in full. Bind annual `YOEP`, `POBP`, `AGEP`, `SEX`, `RELP`, education, disability, employment and earnings concepts. | A recent-arrival respondent is a resident survivor/stayer observed at interview, not a gross arrival. Interviewers request the most recent entry, but unclarified/mail answers can be first or most recent. `WAGP` covers the prior 12 months, possibly including pre-entry months. `RELP` identifies relationship to the reference person, not arbitrary relationship pointers. The gate never uses 5-year pooled weights. Current 2010/2011 bytes are corrected Mar. 4, 2013 re-releases ([erratum 87](https://www.census.gov/programs-surveys/acs/technical-documentation/errata/087.html)); the current 2013 person archive was reposted Feb. 12, 2015 alongside a housing-only `MV` correction ([erratum 97](https://www.census.gov/programs-surveys/acs/technical-documentation/errata/097.html)). The [2013 same-sex married-couple user note](https://www.census.gov/programs-surveys/acs/technical-documentation/user-notes/2013-03.html) is a marital/family concept break. Exact bytes/correction status and observation versus release date remain bound/O6 decisions. |
| CPS ASEC | Census, *[2014 Traditional ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar14.pdf)* or *[2014 Redesigned ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar14R.pdf)*: the cited Redesigned layout has demographics p. 65; six disability items pp. 68–69; `PENATVTY`, grouped `PEINUSYR`, `PRCITSHP`, `MARSUPWT` p. 69; wage/salary and earnings pp. 77, 83. | Report-only marginal and earnings triangulation. | Smaller civilian noninstitutional universe plus Armed Forces members living in civilian housing; grouped entry years; survey-date demographics versus prior-calendar-year income. Traditional and Redesign files must never be silently combined. |
| SIPP | Census, *[2014 SIPP Metadata, all sections v2](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2014/w1/2014SIPP_Metadata_AllSections_v2.pdf)*: `WPFINWGT` p. 9; marital status p. 23; age p. 30; nativity/citizenship pp. 35–37; grouped `TYRENTRY` and entry-status item `TIMSTAT` pp. 38–39; education p. 42; sex p. 43; disability p. 1368; monthly earnings p. 2766. | Report-only joint-state and initializer plausibility check. | Wave 1 covers the 2013 reference year and a civilian-noninstitutional universe. Later waves do not represent newly arrived immigrant-only households, although new co-residents of original sample people can enter. `TIMSTAT` is neither a legal-history panel nor authority to model status and is excluded from v1 state. |

The ACS donor predicate is named, not implied:

```text
foreign_born == true
and 0 <= survey_year - reported_year_of_entry <= recent_arrival_max_duration
```

`survey_year` comes only from the bound annual-file manifest; `SERIALNO` is a
within-file household/GQ identifier and contains no year. Annual `YOEP` supplies
the reported year. The training unit key is therefore
`(survey_year, SERIALNO)`. The initial proposal is a 0–4 year window, but the
exact duration, annual-pooling rule, and fallback hierarchy remain decision O3.
The extractor must bind allocation-flag treatment, top/bottom-code treatment,
group-quarters policy, per-year schema crosswalks, replicate weights, and the use
of `ADJINC` before any artifact can be certified.

This predicate defines a resident **stock proxy**, not entry-time truth. Education,
marriage, disability, employment, earnings, survival, and residence may all
change between reported entry and interview. Directly copying that state onto an
entry-year row is only a named report-only initializer. A literal arrival-state
claim requires the separately bound stock-to-arrival bridge in §6.2/O12.

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
- an explicit duration/backcast and survivor/stayer selection law maps the ACS
  stock proxy to arrival state, or the result stays labeled `stock_proxy`;
- a SIPP history donor, if used, is a second named imputation with its own joint-
  fidelity diagnostics;
- future outcomes are never present in the donor artifact; and
- runtime realization is deterministic conditional on the frozen artifact,
  controls, and seed.

A later parametric alternative may compete only on the same joint holdout
surface and floors. Lower marginal error alone is insufficient.

### 2.4 Gross inflow, net change, and status adjustment

The current control candidate is *The 2026 Annual Report of the Board of Trustees
of the Federal Old-Age and Survivors Insurance and Federal Disability Insurance
Trust Funds*, §V.A.3, released June 9, 2026, and the [Supplemental Single-Year
Table V.A2, “Immigration Assumptions, Calendar Years 1940–2100”](https://www.ssa.gov/oact/TR/2026/lr5a2.html).
The assumptions were set in February 2026. Table V.A2 carries, separately, LPR
new-arrival inflow, LPR/citizen outflow, adjustment to LPR status, LPR net change,
temporary-or-unlawfully-present inflow, outflow, the offsetting status adjustment,
that stock's net change, and total net change.

For intermediate 2026, in thousands:

| Component | Persons (thousands) | Entry-seam meaning |
|---|---:|---|
| LPR new-arrival inflow | 600 | SSA stock-accounting new-person inflow |
| Temporary/unlawfully present inflow | 740 | SSA stock-accounting new-person inflow, conditioned on remaining to year-end |
| **Positive SSA inflow components** | **1,340** | Candidate entry-control proxy, not every physical arrival |
| LPR/citizen outflow | 263 | Exit, never an entrant |
| Temporary/unlawfully present outflow | 947 | Exit, never an entrant |
| Adjustment to LPR status | 450 in each stock, opposite signs | Internal reclassification; not a new person |
| **Total net change** | **130** | Reconciliation target, not a literal cohort |

The 75-year intermediate average total net change is 1.138 million, but Table
V.A2 is annual and materially non-flat; the annual component values, not the
average or ultimate constant, are the binding shape. Components may differ by
rounding.

For the temporary/unlawfully present stock, “inflow” counts people who enter the
Social Security-area population and remain through the end of the year. Thus the
positive sum is neither a border-admissions series nor an unselected gross-arrival
flow.

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
- If metadata does not already supply one, the dynamic synthetic-ID allocator is
  created above the maximum initial or scheduled ID (`engine/loop.py:222-228`).
  Because the code uses `setdefault`, a caller-supplied allocator is only
  type-checked at `engine/loop.py:225-233`; v1 must separately range-check it.
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

The immigration builder must **compose with**, not replace, that existing mapping.
For each shared year it schema-reconciles and concatenates the unchanged PSID
opener frame with the immigrant frame, then validates one prior-year coordinate
and global ID uniqueness. It allocates immigrant IDs above the maximum of the
initial IDs **and every pre-existing scheduled ID**. Because all new IDs are above
that union, existing-person and existing-opener sorted ordinals remain unchanged.

### 3.2 Consequences for timing, IDs, and RNG

An entry frame is a pre-period state. Mortality sees it first
(`engine/steps.py:113-134`); aging then adds one to `age` and writes the target
year (`engine/steps.py:137-166`). The existing PSID opener builder deliberately
stores the prior year coordinate while retaining the observed collection-wave
age (`harness/m6_population.py:313-320`). That convention does not by itself
settle whether an annual gross immigrant inflow should receive a full-year,
half-year, or no domestic mortality exposure in its entry year.

V1 therefore requires a named `entry_timing_basis` and a pre/post-aging age
identity check. The unchanged adapters implement only a full mortality draw at
the scheduled age followed by unconditional `age += 1`; they cannot express a
fractional exposure, defer aging, or output a target-year age-zero immigrant
without an entrant-aware wrapper. Decision O2 must choose that feasible existing
convention or authorize a separately scoped wrapper inside the mortality/aging
slots. The builder may not hide an age shift or age `-1` inside donor extraction,
and every choice retains the seam's `year = entry_year - 1` contract.

That choice must respect the source component's exposure basis. Table V.A2's
temporary/unlawfully-present inflow is already conditioned on remaining through
year end; applying the unchanged engine's full source-year mortality to the raw
control would apply a second survival filter. O2 therefore blocks production
until a component-aware aggregate timing/exposure bridge is bound. It may alter
the cohort control/exposure convention, but it may not assign a Trustees stock or
legal-status label to individual donor rows.

IDs are assigned outside the loop in deterministic order. They must be finite
integers, greater than every starting-population **or pre-existing scheduled** ID,
collision-free across all years, and invariant to row order. These conditions
preserve every original and PSID-opener sorted-ID ordinal. Projection metadata
must omit a pre-supplied synthetic allocator, or assert its mutable `next_id` is
strictly above the combined maximum before period 1; otherwise the loop does not
protect births from collision. No existing M6 module stream is consumed to
construct the schedule.

### 3.3 Amendment 3h / M6 §2.8.2h: live-roster materialization

Amendment 3h is absent from the pinned engine baseline. It subsequently merged
to master as
[PR #216](https://github.com/PolicyEngine/populace-dynamics/pull/216), commit
`0e27be2d857719b30e33b556580b4a360808b5e0`, after this branch point. Its public
forensic source is the
[3h forensics/adjudication](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4984997277)
and the merged M6 §2.8.2h text. This sibling design adopts that law as a
dependency while retaining the stated baseline for every code pin:

> A scheduled maternal birth may materialize a child only when the mother is
> present in the live post-mortality roster. The frame-independent risk schedule
> may remain the scoring universe; absent-mother events are dropped after the draw
> and reconciled report-only so RNG addresses do not shift.

Entrants are roster-present when their own rows are activated, and their later
maternal births obey this exact 3h law. V1 separately proposes a broader
**immigration relationship-closure invariant**: a donor packet may not materialize
a linked spouse, child, or other person whose row is absent. That new invariant
is analogous to, but not certified by, 3h; becoming `married` in the marital core
does not itself materialize a spouse person. Neither law may weaken the existing
birth-parent guard at `engine/steps.py:409-414`.

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

When a cached update has rows for the current year, `_merge_period_columns` drops
existing state columns before left-merging it (`engine/assembly.py:174-192`). An
immigrant absent from that nonempty update receives missing state even if the
scheduled row carried a seed value; only a wholly empty year returns the frame
unchanged.
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
- **Fertility amendment 3h**: distinguish a frame-independent maternal-birth
  scoring schedule from live-roster materialization; a child may materialize
  only against a live mother. The broader relationship-closure rule in §3.3 is a
  new immigration invariant, not part of the 3h certificate.

No existing certificate transfers across these bridges. Reuse of unchanged core
code may be plumbing at implementation time, but applying it to an entrant
population is a new domain claim and remains report-only until its expressly
named successor gate passes.

## 4. Proposed design

### 4.1 Components and one-way data flow

The design separates source estimation, scenario binding, schedule realization,
and period activation:

```text
ACS <=T* ─> DonorArtifact ─> ArrivalStateBridge ─┐
SSA components ─────────────> BindingManifest ───┼─> ScheduleBuilder
SIPP/CPS ─────────────> diagnostic/bridge evidence┘       ├─> entry frames
Census scenarios ─────────────> report-only corridor audit └─> StateBundle

entry frames ──metadata[SCHEDULED_ENTRIES_KEY]──> existing engine loop
StateBundle ──entrant-only adapters/builders──> downstream state
```

The conceptual products are:

- **`ImmigrationBindingManifest`**: immutable source/vintage/schema/hash records,
  annual flow controls, universe labels, and scenario identity;
- **`ImmigrationDonorArtifact`**: a fitted pool of recent-arrival person or family
  units, matching cells, fallback hierarchy, concept mappings, and provenance;
- **`ArrivalStateBridge`**: an estimated duration/backcast and selection law, or
  an explicit `stock_proxy` identity label that prohibits an arrival-state claim;
- **`EntrantStateBundle`**: the entry frame plus module-native initial state and
  exposure objects for the same people;
- **`ImmigrationSchedule`**: `entries_by_year`, the state bundle, deterministic
  ID ledger, and reconciliation tables; and
- **`ImmigrationAudit`**: cohort totals, donor support, calibration residuals,
  universe caveats, dropped/unsupported units, hashes, and certification labels.

The builder merges `ImmigrationSchedule.entries_by_year` with the pre-existing
mapping and sends that single combined mapping to `SCHEDULED_ENTRIES_KEY`; it
never overwrites the PSID opener frames. The state bundle is consumed by explicit entrant-side
builders/adapters assembled before projection. There is no entrant event draw in
the annual engine loop.

### 4.2 Estimation versus deterministic realization

The campaign's estimation/determinism split is binding:

**Estimated and versioned**

- recent-arrival definition and survey concept map;
- donor unit, matching variables, and fallback order;
- sampling/calibration loss and weight trimming;
- any ACS→SIPP joint-history imputation;
- the stock-to-arrival duration/selection bridge, return-entry treatment, and
  fertility/parity and prior-coverage mappings;
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

For Trustees calendar year `y`, define the unbridged positive-inflow control

```text
G_ssa_stock[y] = 1,000 * (
    V_A2_selected_path[y].lpr_new_arrival_inflow
    + V_A2_selected_path[y].temporary_or_unlawfully_present_inflow
)
```

Adjustment of status is excluded because the same person moves between the two
Trustees stocks. Outflows are excluded because they are exit events. The
intermediate total-net column is retained as `N_ssa[y]` for reconciliation only.

`G_ssa_stock` is not yet a usable resident-population cohort. It includes the
temporary/unlawfully present end-of-year-stayer condition. Production construction
requires an adjudicated, vintage-pinned bridge
`G_resident_entry_proxy[y] = bridge(G_ssa_stock[y])` between the Social
Security-area stock accounting and ACS resident universes. The calibration target
is conditional and both source series remain in the audit:

```text
G_schedule_target[y] =
    G_resident_entry_proxy[y]  for a resident-labeled run with O11 bound
    G_ssa_stock[y]             only for an explicitly report-only ssa_area_proxy
```

Until O11 supplies the universe bridge, the builder must either hard-stop or use
the second branch. It may not silently set the bridge to identity and label the
result resident-population aligned.

The schedule interval is also conditional on O15 and must be continuous from
`start_year + 1` through 2100. A report-only open run retaining M6's 2014 initial
slice uses Table V.A2 historical/estimated rows for 2015–2025 (preserving the
source's row-class and footnote flags) and intermediate-assumption rows for
2026–2100. A separately bound 2025 initial slice needs only the latter. A 2026
initial slice cannot accept a 2026 schedule key: that cohort must be demonstrably
included in baseline stock or the run has an omission/double-count. No 2015–2025
gap or silently dropped first cohort is permitted.

The schedule uses a manageable synthetic sample and positive calibration
weights; it does not create 1.34 million physical rows in 2026. For each year:

- the number of donor units is frozen from training-only support/power analysis
  before the truth floor, not by the external population count or candidate;
- unit sampling preserves all members selected together;
- person weights are finite and positive;
- the sum of person weights equals the selected `G_schedule_target[y]` branch
  within a pinned numerical tolerance;
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

1. Start with an annual PUMS household (`survey_year, SERIALNO`).
2. Select foreign-born recent-arrival people with the same reported-year-of-entry
   classification.
3. Preserve only links unambiguously resolvable from the historical PUMS `RELP`
   relationship-to-reference-person field (for example, the reference person's
   spouse and children) when both endpoints are selected. Arbitrary spouse/parent
   pointers are unavailable and remain unresolved.
4. Assign a new `arrival_unit_id`; never expose either element of the source
   tuple as a synthetic person or household identifier.

A spouse, child, or parent who is native-born, entered in a different year, or is
absent from the PUMS household is not cloned. The entrant may retain an observed
marital/parental state with a named `relation_outside_entry_roster` marker, but
no related synthetic row or relationship ID is invented. This is the new
immigration relationship-closure invariant in §3.3, not a 3h-certified claim.

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
- reference-person relationship structure resolvable from `RELP`, unresolved-
  relation markers, and unit size;
- ACS disability-question indicators, without relabeling them DI status;
- employment, weeks/hours where available, wage/salary earnings, self-employment
  earnings, and zero-earnings status; and
- survey year, reported year of entry, observed duration, allocation flags, donor
  weight, and all concept-map provenance.

These are **interview-date stock characteristics**. They become entry-time
characteristics only after the `ArrivalStateBridge`; an identity bridge must be
named `stock_proxy` and is report-only.

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
`(entry_year, unit_slot, member_slot)` beginning above the maximum `person_id`
across the initial population **and the full pre-existing schedule**. The builder
stores the allocation ledger, merges same-year frames without changing their
existing rows, and verifies global uniqueness before calling the engine.
Arrival-unit and household IDs use their own namespaces and may not alias person
IDs.

The metadata adapter omits `synthetic_id_allocator` so the loop creates it from
the combined universe. If another caller requires a supplied allocator, the
adapter asserts `allocator.next_id > max(combined_person_ids)` before period 1;
the loop's type check alone is insufficient.

Changing a source vintage, donor artifact, recent-arrival definition, or schedule
seed creates a new schedule identity. The design does not promise entrant or
newborn byte identity across such scenarios. It does promise that, within one
schedule, pre-supplying entrant IDs above the initial maximum leaves every
original person's sorted-ID ordinal unchanged and lets the engine place its
dynamic allocator above every scheduled entrant only under the preceding
metadata guard.

Downstream RNG needs a separate isolation law. C16, fertility, and candidate-9
consume shared ordered generators, so merely adding entrants to their panels can
shift existing-person draws even though person ordinals are stable. A composite
adapter must run the baseline closed support in its original order with its
original generators, run entrant-only transitions under named disjoint
draw×period×person namespaces, and return one authoritative partitioned marital
result to the existing fertility/household reader slots. Any cross-domain
marriage or household reconciliation that can change baseline people is a
distinct report-only open-market product; it makes no byte-identity claim.

### 4.7 Entry-frame schema and hard invariants

Every scheduled entry frame contains these conceptual groups:

| Group | Required fields |
|---|---|
| Engine | `person_id`, `year`, `age`, `sex`, `weight` |
| Entry identity | `synthetic_entry = true`, `entry_kind = "immigration"`, `entry_year`, `arrival_unit_id`, `foreign_born = true` |
| Timing | `entry_timing_basis`, `age_timing_basis`, `reported_year_of_entry`, `entry_year`; `years_since_entry` is derived, not stored as mutable state |
| Provenance | binding-manifest ID/hash, donor-artifact and arrival-state-bridge ID/hash, schedule ID/seed, donor-cell and fallback code |
| Static/donor state | education, source region, marital/family seed, employment/earnings seed, disability-concept seed, household seed |
| Domain markers | `earnings_domain = false` and named entrant-domain flags for every composite adapter |

No `legal_status` field is inferred. A source-stock component from Table V.A2 may
be retained only at aggregate manifest level; it is not assigned to a person.

Before engine invocation the builder hard-checks:

- mapping keys are integer entry years in the projection range;
- each frame has exactly `year = entry_year - 1`;
- pre/post-aging age identities match decision O2;
- person IDs are finite integers, globally unique, and greater than the initial
  and every pre-existing scheduled maximum;
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

Every entrant adapter derives `years_since_entry = context.year - entry_year` at
read time. `advance_age` does not update that field, so a mutable carried counter
is prohibited.

### 4.8 `EntrantStateBundle`

The state bundle is atomic by scheduled person ID and contains:

| Native support object | Purpose |
|---|---|
| `entry_frames_by_year` | The exact frames sent through the seam. |
| `entrant_marital_seed` | Entry state/history, risk-set start year, relation IDs only for roster-present people, and entrant-domain marker. |
| `entrant_household_seed` | Arrival-unit household links, household state, and entry-year exposure mask. |
| `entrant_disability_seed` | Survey-concept indicators plus the explicitly estimated mapping to any entrant disability state. |
| `entrant_fertility_seed` | Prior parity/birth-history state, its source/uncertainty, and an exposure-start rule; absent a bound bridge the entrant is excluded from fertility risk. |
| `entrant_earnings_seed` | Current-spell entry employment/earnings state, censored/unknown prior U.S.-covered history, any entrant-specific lag state, and normalization provenance. |
| `entrant_claiming_seed` | `claimed`, `claim_age`, `claim_year`, prior-coverage/insured-status provenance, and an eligibility-domain marker. |
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
source supports it. Because reported year of entry does not identify first entry,
prior U.S. Social Security covered earnings are **unknown/censored**, not zero.
A separately named first-entry-only scenario may set them to zero only with an
adjudicated identification rule; otherwise insured-status and benefit outputs are
suppressed. Return migration and prior coverage are decision O12.

### 4.9 Downstream module-state contract

| Existing step | Entrant analogue | Certification boundary |
|---|---|---|
| Mortality | An activated entrant mechanically receives the existing age/sex mortality draw before aging. Decision O2 must define first-year exposure. | Mortality drift is already report-only; applying it to entrants is not newly certified. Emigration may not be encoded as excess mortality. |
| Aging | The existing deterministic `advance_age` runs unchanged after the prior-year coordinate check. | Plumbing only; the entry-age convention, not the function, is the new law. |
| Marital core | An entrant-side panel builder supplies one admissible entry seed and risk-set start. A production open-panel adapter may combine markets only in a report-only run. | Candidate-16's PSID certificate does not transfer to immigrants or to cross-domain matching. Closed-panel scored outputs must remain unchanged. |
| Fertility | Co-arriving children are entrant rows but do not reveal all prior births. The current kernel initializes parity to zero (`engine/marital.py:313`), so entrants are excluded from its fertility risk IDs until a bound parity/history seed and entrant-aware kernel exist. Later maternal births then use amendment-3h live-roster materialization. | No parity-zero default and no absent mother. Entrant fertility is blocked/report-only pending O13. |
| Disability | ACS disability questions and SIPP work-limit/benefit concepts feed a separately estimated entrant initializer/forward law; they are never relabeled as the realized PSID M4 status. | M4 reproduction support and certificate stay unchanged. Entrant disability is a successor-gate surface. |
| Earnings | Entrants remain `earnings_domain = false`. A separate immigrant-entrant generator owns current-spell entry earnings and all later entrant earnings until a future handoff law is designed. Prior U.S.-covered history remains censored unless O12 supplies it. | No fake 2014 earnings, `u_w`, `gen_earn_w2`, or `gen_earn_w4`; the §2.8.3a generator is untouched and its certificate does not transfer. |
| Claiming | Entrants are excluded by an eligibility-domain adapter unless the state packet resolves prior/current covered quarters and insured status. This is necessary because current `apply_claiming` draws for every person age 50+ without testing insurance, AIME, or PIA (`engine/steps.py:320-378`). | No unconditional claiming draw; entrant claims and benefits are blocked/report-only until O5/O12 and an eligibility gate. |
| Household composition | An entrant-side native panel starts exposure at entry, carries arrival-unit links, and makes no link to an absent person. | Candidate-9's PSID certificate does not transfer. Open-market effects on existing people are report-only. |

The scored M6 closed-panel run and an open-population production run are distinct
products. Entrant interactions may affect existing people's marriages,
households, and births in the open run, but those effects cannot be allowed to
retroactively change the frozen closed-panel gate or its certificate.

### 4.10 Emigration and net reconciliation

The v1 audit publishes, by year:

```text
ssa_stock_accounting_inflow_control
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

The builder refuses to run when a binding required by the selected run is missing
or mutable, a raw or derived hash differs, a required year is absent, a unit
crosses an unsupported concept, an outside-roster relation is assigned an ID, an
entry is accidentally admitted to the certified earnings domain, or the universe
bridge is absent for a run claiming resident alignment. Network access and runtime
redownload are prohibited on a scored or production run.

## 5. Evaluation and proposed `gate_imm`

### 5.1 Gate identity and estimand

`gate_imm` is a proposed **additive successor gate**. This document does not add
it to `gates.yaml`, choose tolerances, build a floor, or lock an artifact.

The candidate estimand is narrow:

> Conditional on the named recent-arrival resident-stock proxy and its survey
> universe, does the frozen donor/schedule procedure reproduce held-out ACS
> recent-arrival stock-proxy marginals and named joint distributions at their empirical noise
> floor?

This is not a gate on literal arrival-time state or external cohort counts: exact agreement with a forced
control is tautological. It is not truth for gross arrivals, people who left or
died before interview, legal status, or post-entry trajectories.

### 5.2 Temporal split and leakage fence

The primary proposed split uses five separately released annual ACS 1-year PUMS
files:

- fit donor construction and all matching/calibration choices on the 2010–2013
  annual files using each year's own person and replicate weights, with equal-
  year-mass versus population-mass pooling and cross-year normalization frozen
  from training only;
- freeze the candidate artifact and surface;
- generate a synthetic 2014 recent-arrival resident cross-section without
  reading any 2014 person record, weight, or marginal; and
- score against the separately weighted annual 2014 file using the same recent-
  arrival predicate, concept map, universe, and annual-weight treatment.

Households/arrival units keyed by `(survey_year, SERIALNO)` are indivisible. The
truth and candidate normalize to
the same total before characteristic scoring; the total itself is not a gate
cell. No 2014 characteristic, marginal, top-code treatment selected after seeing
the holdout, CPS/SIPP statistic, current Trustees assumption, or Census projection
may affect the fit. The 2010–2014 5-year PUMS is prohibited from both fitting and
truth: its five-year weights represent/rerake to the pooled period, so slicing it
by survey year would neither recover annual truth nor preserve the leakage fence.
The 2013 same-sex married-couple edit change is frozen in the per-year concept
map. Marital/family cells must harmonize to a definition stable across all five
annual files or be demoted before the truth floor; a dictionary crosswalk alone is
not evidence of longitudinal concept invariance.

The annual files are observations at or before `T*`, but at least the 2014 file
was released after its observation year. Decision O6 must ratify an observation-
date rule analogous to the M6 NAWI publication-lag reasoning or reject every
post-boundary release. Rejection pauses the gate; it does not silently move the
boundary or fall back to the pooled file.

Later ACS windows, including 2015–2019 and 2020–2024, are temporal-drift stress
tests only under this gate. Refitting on them creates a new production artifact
to which no exact-artifact certificate automatically transfers.

### 5.3 Floors before thresholds

The ceremony order is mandatory:

1. Using training-only power analysis, freeze the recent-arrival predicate,
   donor/simulation unit, physical annual sample size, weight caps, calibration
   constraints, fallback limits, schedule-seed grid and aggregation/conjunction
   rule, concept map, matching ladder, gate cells, metrics, and weighting rules
   without candidate holdout results.
2. Construct a correlation-respecting real-vs-real floor from deterministic,
   household-disjoint splits or the annual 2014 replicate-weight design. Never
   split members of one `(survey_year, SERIALNO)` unit. Training uncertainty
   treats each year's replicate set as a separate block; replicate columns are
   never concatenated across years as one common design.
3. On training-era pseudo-holdouts only, run the complete donor-selection,
   calibration, fallback, and schedule pipeline across the registered seed grid.
   Freeze how candidate-pipeline variability combines with the truth-side floor;
   no favorable single schedule seed can define a PASS.
4. Publish, for every proposed cell, raw person count, raw unit count, person-
   weight Kish ESS, cluster/unit-weight ESS, replicate-design variance,
   denominator, allocation share, top-code share, and both truth-side and
   pipeline distance distributions. The binding support measure is the most
   conservative applicable unit/design quantity, not person ESS alone.
5. Prune, pool, or demote every cell that fails predeclared support or whose floor
   is unstable. Record each demotion; do not widen a threshold to retain it.
6. Price thresholds from the surviving empirical floors using a predeclared
   transform. No numerical tolerance is assumed by this design.
7. Run an operating-characteristic experiment on the surviving conjunction
   using predeclared degraded pseudo-candidates. If power is weak or the gate is
   near-tautological, pause and redesign before any lock.
8. Only then score the registered candidate and, if it passes, bind the exact
   artifacts, floor, registry, code commit, and hashes.

This order is the M6 floors-before-thresholds law. Sparse single-age MINT6 cohorts
and the ACS weighted/unit structure make it substantive, not ceremonial.

### 5.4 Candidate gate surface

The floor ceremony may consider these predeclared families:

| Family | Candidate observables | Candidate metric |
|---|---|---|
| Demographic | broad derived reported-entry-age proxy × sex shares; reported-entry-duration proxy; broad source region | total-variation or weighted absolute-share distance |
| Education | attainment band overall and by broad age/sex | total-variation distance |
| Family state | marital-state shares; constructed co-resident same-reported-entry unit size; spouse/child same-reported-entry co-residence indicators | total-variation distance, unit-weighted where applicable |
| Employment | employed/unemployed/NILF; zero earnings; work-intensity bands | absolute-share or total-variation distance |
| Positive earnings | `WAGP`/earnings normalized under the pinned economic index, p10/p50/p90 and log spread | floor-scaled log-quantile distance |
| Disability proxy | the six ACS question concepts individually and “any” | absolute-share distance; never named DI status |
| Named joints | derived reported-entry-age proxy × sex × education; sex × marital state × reported-entry-duration proxy; education × employment/earnings band | pooled-cell total-variation distance |

The final registry contains only cells that clear the floors. Fine source-region,
single-age, detailed education, high-order family, and earnings-tail cells are
presumptively report-only until their support proves otherwise.

`reported_entry_duration_proxy = survey_year - reported_year_of_entry`; a
derived age proxy subtracts that duration from interview age under a frozen age/
birthday convention. Neither is literal duration or age at arrival, and neither
can certify O12's arrival-state bridge. The constructed unit is a co-resident
same-reported-entry unit, never a claimed historical travel party.

### 5.5 Report-only evaluation

The following are useful but are not gate truth:

- CPS ASEC 2014 foreign-born tables and microdata marginals for age/sex, marital
  status, education, total money income, and earnings;
- SIPP 2014 Wave 1 joint plausibility for marital, employment/earnings,
  disability, and nativity/entry concepts;
- ACS 2015–2019 and 2020–2024 drift from the frozen donor composition;
- 2026 Trustees low-cost/intermediate/high-cost positive stock-accounting inflows,
  outflows, and total-net paths;
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
- deterministic conversion of that artifact into seam-valid entry frames and
  **structural/schema completeness** of the packet fields exercised by the gate.
  It would not substantively validate latent histories or module-native mappings.

It would not certify gross-flow truth, the Social Security-area→resident bridge,
emigration, legal status, current-vintage refits, composition through 2100,
entry-year mortality exposure, donor representativeness for people who left or
died before interview, any post-entry transition, any interaction with the
closed population, or any Social Security eligibility/benefit result.

It also would not modify or extend the M6 certificate. Entrants remain family-B
open additions under `harness/m6_reporting.py:71-73` and its explicit immigrant
person-row bridge at `harness/m6_reporting.py:104-118` until a later ratified gate says
otherwise.

## 6. External bindings

### 6.1 Manifest contract

Every external or derived input selected for a run is represented by a field-
level manifest entry, following the sibling §2.8.10 pattern. Each entry must
contain:

- stable binding ID and semantic version;
- source agency, exact report/file title, table/section, scenario and columns;
- observation universe, reference-period convention, units, and covered years;
- information date, publication/release date, retrieval timestamp, canonical URL,
  and raw SHA-256;
- parser name/version, ordered source-field map, unit conversion, rounding rule,
  missing-year behavior, and universe transformation;
- derived artifact schema, row count, minimum/maximum year, SHA-256, and immutable
  artifact location;
- consuming component and whether use is gated, alignment-only, report-only, or
  blocking; and
- explicit alternatives rejected or still open.

An implementation factory must take no unbound source arguments. It resolves
repository-root-relative immutable artifacts, checks every raw and derived hash,
checks the declared time range and schema independently, and returns the ordered
binding objects. No scored or production run may fetch a source over the network.

This design records the sources and required fields, but it does not create raw
or derived artifacts. Consequently no source hash is invented here: `raw_sha256`
and `derived_sha256` are **required lock-ceremony fields** and remain blocking
until an acquisition PR records the actual bytes.

### 6.2 Binding ledger

| Binding ID | Exact source and vintage | Fields/transformation | Consumer and status |
|---|---|---|---|
| `ssa_tr2026_v_a2_components` | Social Security Administration, *[The 2026 Annual Report of the Board of Trustees of the Federal Old-Age and Survivors Insurance and Federal Disability Insurance Trust Funds](https://www.ssa.gov/OACT/TR/2026/tr2026.pdf)*, §V.A.3, and Supplemental Single-Year Table V.A2, “[Immigration Assumptions, Calendar Years 1940–2100](https://www.ssa.gov/oact/TR/2026/lr5a2.html),” assumptions set Feb. 2026, report released June 9, 2026. | Annual LPR inflow/outflow/status-adjustment/net; temporary-or-unlawfully-present inflow/outflow/status-adjustment/net; total net; thousands→persons; retain row class, estimate footnotes and rounding. Use historical/estimated 2015–2025 plus intermediate 2026–2100 for the 2014 M6 origin, or intermediate 2026–2100 for a bound 2025 origin. The positive sum is a stock-accounting inflow proxy; the temporary/unlawfully-present component counts year-end stayers. | Positive-inflow control candidate and net reconciliation. Report-only until universe/timing bridges and exit design exist; prohibited as gate-estimation data. |
| `ssa_tr2026_v_a2_sensitivity` | Same report/table/vintage, low-cost and high-cost alternatives. | Same component schema; never substitute 75-year average or ultimate value for annual rows. | Report-only scenario sensitivity. |
| `ssa_area_to_census_resident_bridge` | **UNBOUND.** Must reconcile the 2026 report's Social Security-area definition (glossary pp. 247–248) to the ACS/Census resident universe with an exact source and vintage. | Annual inclusion/exclusion or factor by population category; preserve an auditable raw-SSA series beside the bridged series. Identity is not an admissible silent default. | **BLOCKING** for a schedule labeled resident-population aligned. A raw `ssa_area_proxy` may run report-only. |
| `projection_origin_population` | Existing realized 2014 M6 initial slice for the recommended report-only integration path; a 2025/2026 resident baseline source and vintage are **UNBOUND** alternatives. | Bind `start_year`, population universe/artifact/hash, first schedule key, first-cohort disposition and continuous control interval. | **BLOCKING** for an origin other than the existing 2014 report-only open run; decision O15. |
| `entry_timing_exposure_bridge` | **UNBOUND.** Must document each Table V.A2 inflow component's event/survival timing and the selected population-origin convention with exact report text and vintage. | Translate controls to opening/mid/end-period exposure without applying source-year survival twice; bind scheduled and target-year age meaning. No person-level status assignment. | **BLOCKING** for a production entry schedule; decision O2. |
| `acs_pums_annual_2010_2014_recent_arrivals` | Census Bureau annual 1-year person PUMS `csv_pus.zip` files for survey years 2010–2014, with exact URLs, annual dictionaries and annual Accuracy statements in §2.2; 2014 Subject Definitions “Year of Entry,” pp. 128–129. Current 2010/2011 bytes are corrected Mar. 4, 2013 re-releases (erratum 87); 2012 is Dec. 17, 2013; current 2013 bytes are the Feb. 12, 2015 repost associated with housing-only erratum 97; 2014 is Oct. 27, 2015. | Bind each file/hash/release/correction status separately. Set survey year only from its manifest; namespace units as `(survey_year, SERIALNO)` and validate `SERIALNO` within file. Bind annual `PWGTP`/replicate weights, `ADJINC`, `AGEP`, `SEX`, `YOEP`, `POBP`, `NATIVITY`, `RELP`, education, disability, employment, income/earnings and allocation flags. Fit 2010–2013; seal annual 2014. The 5-year PUMS and weights are prohibited. | Primary resident survivor/stayer stock donor and proposed `gate_imm` truth, conditional on O6; never literal arrival-state truth. |
| `acs_stock_to_arrival_state_bridge` | **UNBOUND.** No cited ACS cross-section identifies state at the arrival instant or everyone who subsequently left/died. | Must pin a duration/backcast law, survivor/stayer selection adjustment, reported-entry ambiguity treatment, repeat-entry identification and source vintage. An identity mapping is labeled `stock_proxy` only. | **BLOCKING** for literal entry-time characteristics; decision O12. |
| `entrant_marital_household_history_initializer` | **UNBOUND.** Candidate evidence is the annual ACS stock donor plus an exact O8 SIPP file or another named history source; source, universe and vintage are not selected. | Map entry marital history/status, spouse/relations outside roster, household seed, exposure start and later entrant-only law jointly. | **BLOCKING** for entrant marital/household domains and every mixed market; decisions O4/O5/O14. |
| `entrant_disability_state_bridge` | **UNBOUND.** Candidate ACS disability questions and SIPP work-limit/benefit concepts have no selected module-native mapping source/vintage. | Estimate entry state and an entrant-specific forward law without relabeling survey concepts as M4 status. | **BLOCKING** for entrant disability outputs; decision O5 and a successor gate. |
| `entrant_earnings_initializer_forward_law` | **UNBOUND.** Candidate annual ACS current earnings and exact O8 SIPP spell/history evidence; source, universe and vintage are not selected. | Estimate partial-year current-spell state, lags/persistence and entrant-only forward earnings; `m6_projected_wage_index` normalizes dollars but supplies no behavior. | **BLOCKING** for entrant earnings and downstream benefit outputs; decisions O5/O12 and a successor gate. |
| `cps_asec_2014_foreign_born` | Census Bureau, *[2014 Traditional ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar14.pdf)*, *[2014 Redesigned ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar14R.pdf)*, and [2014 ASEC data page](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.2014.html). Calendar-year 2013 income, 2014 interview characteristics. | `A-AGE`, `A-MARITL`, `A-SEX`, `A-HGA`, six disability items, `PENATVTY`, grouped `PEINUSYR`, `PRCITSHP`, `MARSUPWT`, `WSAL-VAL`, `PEARNVAL`, `PTOTVAL`; civilian noninstitutional population plus Armed Forces members in civilian housing. Bind Traditional **or** Redesign bytes and correction/repost status. | Report-only marginal triangulation. Exact file choice is O7; the two designs may not be combined silently. |
| `sipp_2014_wave1_entry_state` | Census Bureau, *[2014 SIPP Metadata All Sections v2](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2014/w1/2014SIPP_Metadata_AllSections_v2.pdf)*, Sept. 12, 2017; [Wave 1 raw directory](https://www2.census.gov/programs-surveys/sipp/data/datasets/2014/w1/); *[SIPP 2014 Panel Source and Accuracy Statement, Wave 1](https://www2.census.gov/programs-surveys/sipp/tech-documentation/source-accuracy-statements/2014/sipp-2014-source-and-accuracy-statement.pdf)*. Interviews in 2014, reference year 2013. The directory is discovery evidence, not an immutable file binding; exact public-use filename/version and correction status are **UNBOUND** pending O8. | Candidate concepts: `WPFINWGT`, marital state/history, age, `EBORNUS`, `ECITIZEN`, `ENATCIT`, grouped `TYRENTRY`, education, sex, disability and monthly earnings. `TIMSTAT` is excluded from legal-status state. | Report-only joint-state/initializer diagnostics after exact bytes are bound. Later waves do not represent new immigrant-only households, though new co-residents of original sample people can enter. |
| `entrant_fertility_history_bridge` | **UNBOUND.** Candidate evidence must name exact SIPP fertility-history or CPS fertility-supplement files, variables, universe, observation years and release vintage. | Map prior parity/birth history and exposure start jointly with entrant family state; never default parity to zero. | **BLOCKING** for entrant fertility risk; decision O13. |
| `prior_us_covered_earnings_bridge` | **UNBOUND.** ACS reported year of entry and public survey earnings do not establish first entry or prior U.S. Social Security covered earnings. | Identify repeat-entry/first-entry status, covered quarters and prior indexed earnings with a source, universe and vintage; otherwise retain censored/unknown. | **BLOCKING** for entrant insured status, claiming, AIME/PIA and benefits; decision O12. |
| `census_np2023_nim_corridors` | Census Bureau, *[Methodology, Assumptions, and Inputs for the 2023 National Population Projections](https://www2.census.gov/programs-surveys/popproj/technical-documentation/methodology/methodstatement23.pdf)* (Nov. 2023), migration pp. 8–14; Table 1 in the Main Series and each Alternative Scenario, “Projected Population and Components of Change for the United States, [Main Series/High Immigration Scenario/Low Immigration Scenario/Zero Immigration Scenario]: 2022–2100”: [main](https://www2.census.gov/programs-surveys/popproj/tables/2023/2023-summary-tables/np2023-t1.xlsx), [high](https://www2.census.gov/programs-surveys/popproj/tables/2023/2023-summary-tables/np2023-t1-h.xlsx), [low](https://www2.census.gov/programs-surveys/popproj/tables/2023/2023-summary-tables/np2023-t1-l.xlsx), and [zero](https://www2.census.gov/programs-surveys/popproj/tables/2023/2023-summary-tables/np2023-t1-z.xlsx) workbooks. | Annual net international migration in thousands for main/high/low/zero scenarios; July 1 prior year–June 30 current year. Preserve scenario definitions: alternatives change gross foreign-born immigration, not every migration component. | Report-only cross-model corridors; never gate truth or a positive entrant control. |
| `m6_projected_wage_index` | Existing sibling design §2.7.6.3/§2.8.10: realized SSA NAWI through 2014 and `I_proj` beyond, estimated only from `<=T*`; see `m6_projection_engine.md:666-708,1723-1756`. | Annual-file `ADJINC` first expresses income in that survey year's dollars. Before cross-year training/holdout comparison bind `earnings_2014 = earnings_y_after_ADJINC * I_proj[2014] / I_proj[y]`; projection-year entrant earnings reverse the ratio from the 2014 base. Never use realized post-2014 NAWI on a scored path. | Reused by a future entrant-earnings initializer. No new external fetch and no certificate transfer to entrant earnings. |
| `emigration_duration_hazard` | **UNBOUND.** Table V.A2 supplies aggregate outflow counts only. The Duleep-Dowhan 2008 hazards and legacy model methods are research evidence, not a current operational binding. | Must identify age, sex, source grouping, time since entry, family/individual unit, re-entry treatment, universe and vintage. | **BLOCKING** for explicit exits and any Trustees net-alignment claim; outside entry-builder v1. |

### 6.3 Binding-specific guards

The SSA parser must assert scenario labels, all nine component columns, continuous
annual coverage from `start_year + 1` through 2100, units, and the internal
identities

```text
lpr_net = lpr_inflow - lpr_outflow + status_adjustment
temporary_net = temporary_inflow - temporary_outflow - status_adjustment
total_net = lpr_net + temporary_net
```

within the source's thousand-person rounding. It must not treat status adjustment
as two events or as a new arrival.

The ACS extractor must resolve five separate annual manifests, set and validate
`survey_year` from the manifest/path alone, namespace household/GQ linkage by
`(survey_year, SERIALNO)`, map that annual vintage's `YOEP` and `POBP`, use
only that file's person and replicate weights, and publish allocation/top-code/
group-quarters counts. It must hard-fail on a 5-year pooled input. It must use
`reported_year_of_entry`: the intended interviewer concept is most recent entry,
while unclarified/self responses may mean first or most recent entry.
`first_entry_year` is prohibited without another source.
The concept-map manifest must also name the 2013 same-sex married-couple edit
break and its common-definition harmonization or explicit report-only demotion.

The Census parser must preserve its July-to-June event year and resident-
population universe. A calendar-year bridge to Trustees may be displayed only as
a named transformation with both originals retained. The “zero” scenario may not
be rewritten to zero net international migration.

## 7. Open decisions for the referee

Nothing in this list is silently resolved by the provisional recommendation.

### O1. Positive SSA inflow, exit scope, and net proxy — hardest

Choose among:

- **recommended**: use the two positive SSA stock-accounting inflow components as
  an entry-control proxy through the seam, keep entry-only v1 report-only, and
  require a separate emigration design before any net-alignment claim;
- widen this design to include an audited exit law before implementing entries;
  or
- authorize a named reduced-form `net_entry_proxy` experiment, accepting that it
  is not a literal immigrant cohort and cannot enter entrant/family/benefit gates.

The temporary/unlawfully-present component counts people remaining through year
end, so none of these options may relabel the 1.340 million 2026 sum as every
physical border arrival. The population-universe bridge is a separate O11.

### O2. Entry-year timing, age, and mortality exposure

The seam inserts the prior-year-coordinate row before mortality and aging.
Choose either (a) the feasible unchanged-engine convention—after an explicit
source-to-opening-exposure transformation, full mortality at the scheduled age
followed by `age += 1`—or (b) an entrant-aware wrapper in the mortality/aging
slots for fractional or deferred exposure. The temporary/unlawfully-present
control already excludes people who do not remain through year end; applying
unadjusted full source-year mortality would double-filter survival and is
prohibited. Bind the component-specific source timing, the aggregate translation,
and what age means on each side without assigning source-stock/legal status to
persons. The builder may not use age `-1` or a hidden donor-age shift.

### O3. Recent-arrival window and matching ladder

Ratify the proposed 0–4-year **reported-entry** window or a different duration;
decide whether duration-zero/one donors receive priority; and freeze age/source/
education/family matching cells, annual pooling/replicate-block treatment,
same-sex-marriage concept harmonization, and fallback order. Repeat-entry
identification and prior U.S. coverage belong to O12.

### O4. Person versus co-arrival-family units and weights — hardest

Choose person donors, co-resident co-arrival units, or a mixed rule. If units are
chosen, bind relationship closure, common versus person-specific simulation
weights, calibration to person totals, partial families, group quarters, and
whether later exits occur by person or unit. This decision sets the correlation
unit for floors.

### O5. Atomic state/history packet and post-entry laws — hardest

Decide which state comes from one ACS donor, which history may come from a
jointly matched SIPP donor, and which requires a new model. In particular:

- marital history and spouse-outside-roster state;
- the ACS/SIPP disability-concept bridge;
- entry employment, partial-year earnings, lags and persistent earnings state;
- the entire entrant earnings law, since §2.8.3a membership cannot expand;
- fertility-history and claiming/insured-status exclusion masks;
- cross-domain marriage/household interactions with existing people; and
- the certification boundary for ordinary cores applied to entrant inputs.

The default is no independent marginal hot-decks and no certificate transfer.
Subparts may be ratified independently, but an entrant must remain out of every
unresolved module domain; a structurally complete packet is not substantive
validation.

### O6. Observation date versus publication date at `T*`

The proposed annual ACS files contain 2010–2014 observations, but the 2014 file
vintage is Oct. 27, 2015; corrected 2010/2011 bytes are Mar. 4, 2013; 2012 is
Dec. 17, 2013; and the current 2013 archive is the Feb. 12, 2015 repost whose
erratum affected only a housing-file variable. Ratify an observation-date rule
analogous to the M6 NAWI publication-lag reasoning, or reject every file whose
byte vintage crosses the boundary and redesign an earlier annual holdout. The
five-year pooled file is not an alternative under either rule. The same issue
affects later-published SIPP metadata, which is report-only here.

### O7. CPS ASEC diagnostic vintage

Choose Traditional or Redesign 2014 ASEC public-use data and bind its correction
history. This affects report-only triangulation, not candidate fitting or gate
truth.

### O8. Exact SIPP Wave 1 file and correction vintage

Choose and hash one exact 2014 SIPP Wave 1 public-use filename/version from the
mutable Census directory, record its correction/repost history, and bind the
metadata version to those bytes. Directory identity alone is not a binding.

### O9. Current-vintage production refits

Decide whether a later ACS donor refit (for example 2015–2019 or 2020–2024)
requires a new holdout/lock ceremony or may inherit a procedure-level certificate.
The conservative default is that the exact-artifact certificate does not
transfer and the refit remains report-only.

### O10. Physical cohort size and calibration constraints

Freeze annual physical unit counts, weight caps, calibration margins and maximum
fallback share through training-only power analysis **before** truth floors are
constructed. The floor may prune an infeasible surface, but it may not
candidate-adaptively choose the sample design. No arbitrary “one row per N
people” constant is adopted here.

### O11. Social Security-area to Census-resident universe bridge

Select or commission a vintage-pinned transformation from each relevant Social
Security-area population category to the ACS/Census resident universe. Choose
the time basis and treatment of territories and covered people abroad. Identity
is not presumed; without this bridge only an `ssa_area_proxy` schedule is
permitted and every population output remains report-only.

### O12. Stock-to-arrival state, repeat entry, and prior U.S. coverage

Choose a vintage-pinned law that maps the ACS interview-date survivor/stayer
stock to arrival-time state, including duration change, selection from pre-
interview death/emigration, and ambiguity between first and most recent entry.
Also choose whether first/return entry can be identified and how prior U.S.
covered quarters and earnings are sourced. Until then the identity initializer
is labeled `stock_proxy`, prior coverage is censored, and claiming/benefit
outputs are suppressed.

### O13. Fertility history and entrant exposure

Bind an exact parity/birth-history source and concept bridge, decide exposure
start, and authorize an entrant-aware fertility kernel. Until then no parity-zero
default is allowed and entrants remain outside fertility risk IDs; later maternal
births still obey the exact 3h live-mother materialization guard.

### O14. RNG-isolated composite and cross-domain markets

Choose the composite adapter that preserves the original closed support, order,
generators and outputs while giving entrant transitions disjoint named RNG
addresses. Decide whether entrant/existing-person marriage and household markets
remain a separate report-only product. No byte-identity claim is admissible from
stable person ordinals alone.

### O15. Projection origin and the first entrant cohort

Choose one complete origin contract:

- **recommended for the existing M6 integration path**: retain the realized 2014
  initial slice and bind the 2026 Table V.A2 historical/estimated 2015–2025 rows
  plus intermediate 2026–2100 rows in a separate report-only open run;
- bind an independently constructed 2025 initial resident slice, then schedule
  2026–2100 through the seam; or
- use a 2026 initial slice only if its construction proves the 2026 entrant cohort
  is already included, then begin seam scheduling in 2027.

The seam requires `entry_year > start_year`. The decision must name the initial-
population artifact/vintage, first schedule key, control range, and whether the
first cohort is baseline stock or a seam flow. It may neither omit 2015–2025 under
a 2014 start nor schedule an entry-year-equal-to-start-year frame.

## 8. What this design does not change

This document and its eventual entrant-side implementation must leave these
surfaces unchanged unless a later, separately adjudicated design explicitly
authorizes surgery:

- `gates.yaml`, every `gate_m6` cell/threshold, the v1/v2/v3 M6 floors and their
  hashes, `tests/tier_counts.json`, and all existing run artifacts;
- M6's `T* = 2014`, temporal holdout, shock partition, weight convention, and
  closed-panel scoring support;
- the certified candidate-16 marital core, candidate-9 household composition
  object, M4 disability reproduction object, gate-2c earnings modifier, and
  gate-1/backward or M6/forward certified earnings specifications;
- the existing age/sex mortality fit and frozen `<=2014` claiming schedule;
- the §2.8.3a earnings-domain predicate and its realized-2014 state maps;
- the §2.8.2g marital risk-set guard and seed-at-domain-entry law;
- amendment 3h's schedule-versus-live-roster materialization distinction and
  absent-parent guard;
- `SCHEDULED_ENTRIES_KEY`, its frame/year/ID contract, and every existing
  2017/2019 PSID opener row; the immigration builder merges that mapping and never
  overwrites it;
- the eight-member `PeriodModules` order, existing module RNG streams, original-
  person ordinals, synthetic-ID allocator semantics, and period trace;
- M6's current statement that immigrant/open-panel additions are report-only;
- M7 trust-fund accounting, M8 rules execution, and any PolicyEngine-US legal
  eligibility rule; and
- every source or artifact outside the immigration binding manifest.

An entrant adapter may call unchanged core code, but it must do so under an
entrant-domain label and separate report/gate surface. It may not alter the
closed-panel input or score and call the resulting difference “immigration.”
The metadata adapter also must omit a caller allocator or enforce the combined-
maximum guard in §4.6. Any mixed entrant/existing-person market remains the
report-only O14 product and makes no closed-run byte-identity claim.

## 9. Candidate-blind implementation and certification order

A later implementation should proceed in this order:

1. Referee resolves O1–O15 for the chosen implementation slice and ratifies the
   external-binding schema.
2. Acquisition PR commits/hash-binds exact source bytes and parsers; the zero-
   argument binding factory passes independently of any candidate.
3. Donor/bridge PR constructs only training artifacts, the separately identified
   arrival-state/fertility/prior-coverage bridges, and synthetic fixtures;
   holdout outcomes remain sealed.
4. Floors PR freezes the physical sample design and seed-grid aggregation,
   creates truth-vs-truth and pipeline-variability floors, prunes unsupported
   cells, and runs the operating-characteristic pause check.
5. Schedule/state PR implements the deterministic builder, seam adapter,
   structurally complete state packet, failure guards, and report-only open-
   population run without changing the M6 closed-panel score.
6. A registered `gate_imm` candidate is scored once against the locked surface.
7. If PASS is verified and ratified, the lock names exact source, derived, floor,
   code and schedule-protocol hashes.
8. Emigration and current-vintage refit designs run their own ceremonies before
   any net-population or benefit certification claim.

No stage reads a later stage's candidate outcomes to redesign an earlier frozen
surface.

A literal resident-entry schedule pauses while O11 and O12 remain unbound;
entrant fertility and claiming/benefit outputs pause while O13 and the prior-
coverage portion of O12 remain unbound. A `stock_proxy` diagnostic schedule may
exercise seam structure only in a report-only run with those module domains
excluded.

## 10. Referee citation ledger

- **DYNASIM4**: Cosic, Johnson, and Smith, *Urban's Dynamic Simulation of Income
  Model 4* (Urban Institute, September 2024), pp. 1–2. Detailed donor mechanics
  are not public in that overview.
- **DYNASIM3 detail, not DYNASIM4**: Favreault, Smith, and Johnson, *The Dynamic
  Simulation of Income Model (DYNASIM): An Overview* (Urban Institute, September
  2015), Table 1, report p. 7.
- **MINT**: Smith et al., *Modeling Income in the Near Term Version 6* (Urban
  Institute, December 2010), ch. II §VI, pp. II-24–II-28, Tables 2-14–2-16; Smith
  and Favreault, *Modeling Income in the Near Term 8 and 2014: Primer* (Urban
  Institute, April 2019), pp. 15–16, note 21 p. 29, Table 3 pp. 39–40.
- **PENSIM/PENSIM2**: Holmer, Janney, and Cohen, *PENSIM Overview* (Policy
  Simulation Group for the U.S. Department of Labor, Employee Benefits Security
  Administration, September 2006), §2.1.6 p. 8, Appendix B §B.1 p. 99,
  §§B.1.2–B.1.3 p. 100, and §§B.7–B.8 pp. 106–107;
  O'Donoghue, Redway, and Lennon, “Simulating migration in the Pensim2 dynamic
  microsimulation model,” *International Journal of Microsimulation* 3(2), 2010,
  Table 2, §§5.1–5.2.
- **SSA methods/current controls**: Duleep and Dowhan, “Adding Immigrants to
  Microsimulation Models” and “Incorporating Immigrant Flows into
  Microsimulation Models,” *Social Security Bulletin* 68(1), 2008; *The 2026
  Annual Report of the Board of Trustees of the Federal Old-Age and Survivors
  Insurance and Federal Disability Insurance Trust Funds*, §V.A.3, and
  Supplemental Single-Year Table V.A2.
- **Microdata**: Census annual 1-year ACS PUMS person files and annual Accuracy
  statements for 2010–2014, with the 2010–2014 dictionary as schema/crosswalk
  evidence and 2014 ACS Subject Definitions “Year of Entry,” pp. 128–129; Census
  2014 Traditional and Redesigned ASEC technical documentation; Census 2014 SIPP
  Wave 1 metadata and source/accuracy statement. The five-year ACS file is
  excluded from the gate; ACS errata 87/97 and the 2013 same-sex married-couple
  user note are manifest inputs; the exact SIPP raw file remains unbound at O8.
  Exact variables and pages are pinned in §2.2 and §6.2.
- **Projection corridors**: Census, *Methodology, Assumptions, and Inputs for the
  2023 National Population Projections* (November 2023), migration pp. 8–14, and
  Table 1 in the Main Series and each Alternative Scenario.

## 11. Design parameters and amendment history

```json immigration-design-parameters
{
  "design_id": "2026-07-15-immigration-module",
  "revision": 2,
  "status": "design_draft_referee_pending",
  "engine_baseline": "75d30dd57d71b91ee0929246b2f3cbb92263b350",
  "roadmap_issue": 113,
  "docs_only": true,
  "certifies_now": [],
  "information_boundaries": {
    "inherited_m6_T_star": 2014,
    "trustees_2026_role": "versioned forward assumption and report-only alignment; never gate-estimation evidence",
    "realized_post_T_star_nawi_on_scored_path": "prohibited"
  },
  "entry_seam": {
    "metadata_key": "m6_scheduled_entries_by_year",
    "definition_pin": "src/populace_dynamics/engine/loop.py:27",
    "contract_pin": "src/populace_dynamics/engine/loop.py:192-257",
    "frame_year": "entry_year - 1",
    "activation": "top of period before mortality",
    "new_period_module": false,
    "ids_preassigned": true,
    "synthetic_allocator_start": "engine default is max(initial and all scheduled person_id) + 1 when caller metadata omits the allocator",
    "existing_schedule_merge": "preserve and concatenate every PSID opener frame; allocate immigrants above max(initial plus all pre-existing scheduled ids)",
    "caller_allocator_guard": "omit caller allocator or assert next_id exceeds the combined maximum"
  },
  "provisional_adjudications": {
    "cohort_control": "positive SSA stock-accounting inflow proxy after an explicit Social-Security-area-to-Census-resident bridge",
    "status_adjustment": "aggregate reclassification; not an entrant and not assigned to persons",
    "emigration": "outside entry-builder v1; mandatory successor before net-alignment claim",
    "assignment": "recent-arrival ACS joint resident-stock donor units plus model-based calibration; literal arrival state requires a separate bound bridge; no cloned future",
    "prior_us_covered_earnings": "unknown/censored absent a first-or-return-entry and coverage bridge; never default zero",
    "entrant_fertility": "excluded from fertility risk absent a bound parity/history bridge and entrant-aware kernel",
    "entrant_claiming": "excluded absent insured-status and prior-coverage evidence",
    "rng_isolation": "required composite partition before any closed-person byte-identity claim",
    "runtime": "schedule built once per scenario and reused across K engine draws",
    "legal_status_dynamics": "out of scope",
    "current_entry_only_outputs": "report_only"
  },
  "external_bindings": [
    {
      "id": "ssa_tr2026_v_a2_intermediate",
      "source": "The 2026 Annual Report of the Board of Trustees of the Federal Old-Age and Survivors Insurance and Federal Disability Insurance Trust Funds, section V.A.3, and Supplemental Single-Year Table V.A2",
      "vintage": "assumptions set February 2026; report released June 9, 2026",
      "role": "positive stock-accounting inflow control candidate and net reconciliation",
      "status": "report_only_until_universe_bridge_and_exit_law"
    },
    {
      "id": "ssa_tr2026_v_a2_sensitivity",
      "source": "2026 Trustees Supplemental Single-Year Table V.A2 low-cost/high-cost",
      "vintage": "2026",
      "role": "scenario sensitivity",
      "status": "report_only"
    },
    {
      "id": "ssa_area_to_census_resident_bridge",
      "source": "unbound",
      "vintage": "unbound",
      "role": "population-universe bridge",
      "status": "blocking_for_resident_alignment"
    },
    {
      "id": "acs_pums_2010_2014_recent_arrivals",
      "source": "Census 2010-2014 ACS 5-year PUMS person file, dictionary, accuracy statement, and 2014 Subject Definitions",
      "vintage": "observations 2010-2014; published January 2016",
      "role": "resident survivor/stayer stock donor; fit 2010-2013 and hold out 2014",
      "status": "proposed_gate_binding_pending_O6_not_arrival_truth"
    },
    {
      "id": "acs_stock_to_arrival_state_bridge",
      "source": "unbound",
      "vintage": "unbound",
      "role": "map interview-date stock proxy to literal arrival state, including duration, selection, reported-entry ambiguity, and repeat entry",
      "status": "blocking_for_literal_entry_state_pending_O12"
    },
    {
      "id": "cps_asec_2014_foreign_born",
      "source": "Census 2014 ASEC public-use file and technical documentation",
      "vintage": "2014",
      "role": "marginal triangulation",
      "status": "report_only_pending_traditional_or_redesign_choice"
    },
    {
      "id": "sipp_2014_wave1_entry_state",
      "source": "Census 2014 SIPP Wave 1 metadata v2, source/accuracy statement, and mutable raw-file directory; exact public-use file unbound",
      "vintage": "2013 reference year; 2014 interviews",
      "role": "joint-state and initializer diagnostics",
      "status": "report_only_pending_exact_file_and_correction_vintage_O8"
    },
    {
      "id": "entrant_fertility_history_bridge",
      "source": "unbound exact SIPP fertility-history or CPS fertility-supplement file",
      "vintage": "unbound",
      "role": "parity, prior-birth history, and entrant fertility exposure start",
      "status": "blocking_for_entrant_fertility_pending_O13"
    },
    {
      "id": "prior_us_covered_earnings_bridge",
      "source": "unbound",
      "vintage": "unbound",
      "role": "first-versus-repeat entry, covered quarters, and prior indexed U.S. earnings",
      "status": "blocking_for_claiming_aime_pia_and_benefits_pending_O12"
    },
    {
      "id": "census_np2023_nim_corridors",
      "source": "Census 2023 National Population Projections, Alternative Scenarios Table 1",
      "vintage": "November 2023",
      "role": "main/high/low/zero net-international-migration corridors",
      "status": "report_only"
    },
    {
      "id": "m6_projected_wage_index",
      "source": "existing M6 <=2014 NAWI-derived I_proj binding",
      "vintage": "realized through 2014; projected thereafter",
      "role": "entrant earnings normalization",
      "status": "reuse_without_certificate_transfer"
    },
    {
      "id": "emigration_duration_hazard",
      "source": "unbound",
      "vintage": "unbound",
      "role": "allocate aggregate outflow controls to roster-present people",
      "status": "blocking_for_explicit_exit_and_net_alignment"
    }
  ],
  "gate_imm": {
    "exists_now": false,
    "estimand": "held-out ACS recent-arrival resident-stock characteristic reproduction",
    "fit_collection_years": [2010, 2011, 2012, 2013],
    "holdout_collection_years": [2014],
    "count_alignment_gated": false,
    "physical_sample_design_frozen_from_training_power_analysis": true,
    "schedule_seed_grid_and_aggregation_frozen_before_truth_floor": true,
    "floors_before_thresholds": true,
    "operating_characteristic_before_lock": true,
    "packet_certification": "structural_schema_only",
    "downstream_life_course_certified": false
  },
  "hardest_open_decisions": [
    "O1: positive SSA stock-accounting inflow proxy plus separate exits versus an expanded entry-and-exit design or named net-entry proxy",
    "O5/O12/O13: atomic downstream packet, stock-to-arrival and repeat-coverage bridges, and fertility/claiming exclusions",
    "O4/O14: person versus co-arrival-family units, relationship and weight closure, and RNG-isolated cross-domain markets"
  ],
  "certified_surfaces_untouched": [
    "gate_m6 registry, thresholds, floors, hashes, and closed-panel support",
    "candidate-16 marital core",
    "candidate-9 household composition object",
    "M4 disability reproduction object",
    "certified earnings specifications and section 2.8.3a domain",
    "section 2.8.2g marital domain law",
    "amendment 3h live-roster materialization law",
    "PeriodModules order and ProjectionRNGRegistry",
    "SCHEDULED_ENTRIES_KEY contract and every existing PSID opener row",
    "original closed-support generator order, RNG consumption, and outputs",
    "M7 trust-fund accounting and M8 rules execution"
  ],
  "amendment_history": [
    {
      "revision": 1,
      "date": "2026-07-15",
      "kind": "initial_docs_only_design",
      "changes": [
        "bind immigration activation to the existing scheduled-entry seam",
        "separate gross entry controls from outflows, status adjustments, and net reconciliation",
        "select donor-based entry state provisionally while prohibiting cloned futures",
        "define the EntrantStateBundle and no-certificate-transfer boundary",
        "propose floors-first gate_imm and enumerate external bindings and open referee decisions"
      ]
    },
    {
      "revision": 2,
      "date": "2026-07-15",
      "kind": "pre_pr_adversarial_hardening",
      "changes": [
        "distinguish SSA positive stock-accounting inflows from all physical arrivals and separate the population-universe bridge",
        "label ACS recent arrivals as a survivor/stayer stock proxy and make stock-to-arrival state an unbound bridge",
        "treat possible prior U.S. covered earnings as censored and block unsupported claiming and benefit outputs",
        "block entrant fertility until parity history and an entrant-aware kernel are bound",
        "merge rather than overwrite existing PSID scheduled entries and guard caller-supplied synthetic allocators",
        "require physical-sample and schedule-seed protocols before floors and limit packet PASS to structural/schema completeness",
        "require RNG-isolated composite adapters before any original-person byte-identity claim",
        "expand referee decisions through O14 and pin unresolved external bindings"
      ]
    }
  ]
}
```
