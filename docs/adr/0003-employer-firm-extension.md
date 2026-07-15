# ADR 0003: Employer-firm extension — C1 spell schema and C2 canonical firm-size banding

**Status:** Accepted — C1 and C2 frozen 2026-07-15. From this
point the contracts change only by joint PR between workstreams A
and B ([populace-dynamics#192](https://github.com/PolicyEngine/populace-dynamics/issues/192)).

**Sign-off:** @vahid-ahmadi (Workstream B, author) ·
@daphnehanse11 (Workstream A) — recorded by the merge of the freeze
PR, whose approvals from both workstream owners constitute the
joint sign-off. First scheduled amendment (pre-registered above):
the C1 ``hours_band``/monthly-hours column once phase-1 establishes
SIPP's supportable hours granularity.

## Context

The employer-firm plan (`docs/plans/employer-firm-plan.html`) splits
the extension into workstream A (person side: SIPP spells, CPS hosts,
imputation) and workstream B (firm side: external targets, banding,
calibration, register), meeting at three interface contracts. C1 (the
spell schema) and C2 (canonical firm-size banding and its semantics)
freeze in week 1. This ADR records both, plus the target/gate
partition rule, folding in the four contract-affecting findings from
the week-1 review on issue #192.

## Decision

### C2 — canonical firm-size banding

1. **Semantics (review finding F5).** The canonical firm-size
   variable means **administrative enterprise size**: total
   employment of the legal enterprise across all locations, as SUSB
   counts it. Survey labels are noisy measures of that quantity —
   CPS ASEC firm size (worker-reported, all locations, previous
   calendar year's longest job — under either the raw Census `NOEMP`
   or the IPUMS `FIRMSIZE` coding; see C2.5) is the primary training
   label; SIPP 2014+ `EJB1_EMPSIZE` (establishment size) is a proxy
   chain. SUSB is therefore the correct E1 reference.
2. **Bands are headcount bands.** Five canonical bands with edges at
   10 / 50 / 100 / 500: `LT10`, `B10_49`, `B50_99`, `B100_499`,
   `B500_PLUS` (`src/populace_dynamics/firms/banding.py`). The 50
   edge is mandatory: ACA and several state mandates key on 50, and
   both QWI (20-49 / 50-249) and the detailed SUSB classes (40-49 /
   50-74) support it. FTE-denominated thresholds (the ACA cut is 50
   full-time equivalents at 30 hours/week, not headcount) are
   resolved by a person-side hours join — out of C2 scope.
3. **Mappings are total but explicitly ambiguous where the source is
   coarse.** Every raw code from every source maps to exactly one
   `BandSpan` (a contiguous run of canonical bands with an `exact`
   flag). Source bands that straddle a canonical edge — BDS "20 to
   99", QWI "0-19", most SIPP bands — return multi-band spans rather
   than being silently rounded. Every ASEC firm-size vintage nests
   exactly (10-49 / 50-99), as do the SUSB detail classes.
4. **The 50 cut is directly observed in every supported ASEC vintage
   (corrected finding).** Earlier drafts held that the 2019+ ASEC
   label ("25 to 99") could not resolve the 50 edge. Real-data
   validation (issue #192) showed that relabeling is phantom — the
   2019-2025 Census/IPUMS dictionaries mislabel codes 2/3 (NOEMP) /
   2/5 (FIRMSIZE), but weighted code shares are continuous across the
   supposed 2018/2019 break and match SUSB only under the 10-49 /
   50-99 reading. Workstream A's loader (#204) and `banding.py` both
   use that reading. So the 50 edge is observed directly in 2011-2025
   and the two identification routes (within-band imputation; pooling
   across the "break") are **no longer needed**. The supporting
   replication (weighted code shares by year vs. SUSB) is committed
   as `runs/noemp_band_evidence_v1.json` with its build script and
   pinning tests (#211) — the reported-anchor convention, since it
   is derived evidence rather than a source extract — for the C3
   record.
5. **The person-side coding is explicit, not inferred (seam with
   #194).** The raw Census ASEC person file carries `NOEMP`
   (codes 1-6) and the IPUMS-CPS harmonised extract carries
   `FIRMSIZE` (a different, wider code set); the *same integer means
   different bands* — e.g. in the same 2011-2018 vintage code 4 is
   100-499 under `NOEMP` but 10-49 under IPUMS `FIRMSIZE`.
   `banding.cps_firmsize_to_canonical` therefore takes a **required**
   `coding` keyword (`"ipums_firmsize"` or `"census_noemp"`) with **no
   default** — three codes are valid under both codings with different
   bands, so a default would let a caller holding the wrong coding
   mis-band silently. `noemp_to_canonical` is the raw entry point, and
   both refuse vintage-impossible (and pre-2011) codes rather than
   silently borrowing another vintage's interval. Workstream A's
   loader (#194) emits `NOEMP`; whichever side owns the final
   `CanonicalBand` assignment must call the `census_noemp` route (or
   emit `CanonicalBand` directly), never feed `NOEMP` integers to the
   `ipums_firmsize` route.

### C1 — job-spell schema

One tidy table, written by workstream A, read by workstream B:

| column | type | notes |
|---|---|---|
| `person_id` | str (opaque key) | host CPS person key — an opaque string, **not** a numeric id (the ASEC `PERIDNUM` is 22 digits: int64 overflows and float64 rounds distinct persons together, per the #194 review). Workstream A may instead expose the host frame's internal integer row key; either way the column is a join key, never arithmetic |
| `spell_id` | int | unique within person |
| `start_period` | period | first period of the spell |
| `end_period` | period | last period; open spells use a sentinel |
| `industry` | str | NAICS major (sector) group |
| `firm_size_band` | enum | canonical band per C2 (`CanonicalBand`) |
| `class_of_worker` | enum | private / federal / state-local government / self-employed / unpaid family |
| `earnings_share` | float | share of the person's period earnings from this job |
| `primary_job` | bool | phase 0 is primary-job-only |

- **`class_of_worker` is load-bearing for the calibration universe**
  (issue #192 review, point 1): SUSB excludes government
  establishments, Public Administration (NAICS 92), crop/animal
  production and non-employers; QWI in-scope jobs are non-federal
  (and firm size is undefined for public-sector employers — the
  committed J2J extract drops NAICS 92 for the same reason).
  Government, self-employed, and unpaid-family spells are excluded
  from the SUSB/QWI calibration universe; self-employed spells have
  no defined `firm_size_band`.
- **Geography joins from the person table.** QWI/J2J targets are
  state-level; C1 deliberately carries no geography column. The
  state of a spell is the host person's state at `start_period`,
  joined on `person_id` — the join key lives on the person table,
  not the spell table.
- Multi-job holding is resolved primary-job-only in phase 0;
  `earnings_share` retains the information needed to revisit this.
- **Hours granularity is deferred, and the deferral is explicit**
  (registered downstream consumer: SNAP ABAWD work-requirement
  compliance, issue #192 — the 80-hours-per-month test of
  7 CFR 273.24 and the 3-in-36 countable-month clock need
  month-resolved hours, not spell start/end plus annual earnings).
  C1 as frozen carries no hours column, so it **cannot yet serve
  monthly-hours consumers**; a `hours_band` (or monthly hours)
  column is the first scheduled C1 amendment, to be added by joint
  PR once workstream A's phase-1 spell imputation establishes what
  hours granularity SIPP can support. Consumers must not proxy
  monthly compliance from annual quantities in the meantime.

### Conditioning DAG — the firm-size x tenure bridge

No current representative source observes firm size and tenure
jointly (issue #192 review, point 2): ASEC `FIRMSIZE` refers to the
preceding calendar year's longest job while the biennial tenure
supplement refers to the current job and asks no firm-size question;
SIPP 2014+ has tenure but only establishment size; NLSY has both but
for two non-representative cohorts. The joint dependence in the
phase-0 QRF therefore comes from a named bridge, not an implicit one:

1. **Primary bridge: the pre-redesign SIPP 2008 panel (2008-2013)**,
   which asked worker-reported firm size at all locations alongside
   job spells and tenure — the last representative panel observing
   the joint distribution. Its (dated) joint structure is the
   bridge, aged forward.
2. **Proxy chain:** SIPP 2014+ establishment size x tenure, mapped
   through the establishment-to-enterprise noise model implied by
   the C2 semantics.
3. **Pre-registered caveat:** the ASEC reference-period mismatch
   (`FIRMSIZE` = last calendar year's longest job; tenure supplement
   = current job) is carried into the C3 gate notes as a known
   label-misalignment term.

### Target/gate partition rule

Cells used in `microcalibrate` must be pre-registered as **disjoint**
from gate cells. The partition rule for the firm side: calibration
consumes the SUSB size x sector employment margins and the QWI
firm-size x sector flow margins committed under `data/external/`;
gates E1/E2/E7/E11 score on held-out dimensions of the same sources
(the sex/age demographic axes of QWI, the firm-age axis, and the
state axis) that calibration never touches. The exact cell lists lock
with C3 after the floor runs.

Three unit rules recorded now (issue #192 review, point 4; branch
review finding 3):

- **QWI/J2J cells count jobs, not persons** — each person-employer
  pair in a quarter is a separate job. Phase 0 is primary-job-only,
  so calibrating person-spells to QWI cells carries a wedge on the
  order of the multiple-jobholding rate (~5%, time-varying). A
  job-count -> person-count adjustment is an explicit pre-registered
  C3 item, not a footnote.
- **QWI publishes mean earnings (`EarnS`), never medians**; E7 is
  stated on means.
- **J2J's employer universe is broader than SUSB/QWI's.** The
  committed J2J extract uses ownership `oslp` (state/local government
  plus private), while the QWI extract is private-only (`op`) and
  SUSB excludes government entirely. Dropping NAICS 92 removes Public
  Administration but not state/local employment embedded in other
  sectors (notably 61 Educational Services and 62 Health Care). Any
  E11 cell definition must either restate J2J on a private-comparable
  basis or carry this scope difference as a pre-registered caveat;
  the choice locks with C3.

## Consequences

- `src/populace_dynamics/firms/banding.py` is the single canonical
  banding implementation; workstream A trains to it. Changing a band
  edge after the freeze requires a joint PR touching this ADR.
- The committed target extracts (`data/external/susb_us_*`,
  `bds_us_*`, `qwi_us_*`, `j2j_us_*`; provenance in
  `data/external/employer_firm_target_sources.md`) are external
  references, analogous to the NCHS/Census/ONS files — never scored
  model output. Raw microdata is never committed.
- No change to `gates.yaml`. Employer gates E1-E12 lock as a new
  block (C3) after noise-floor runs and a referee round, via the
  standard amendment process; no one-shot candidate runs before C3
  locks.
