# ADR 0004: Employer-firm linkage QC requirements for C3

**Status:** Proposed — input to the C3 referee round; this document
locks no C3 threshold. C1 and C2 are treated as frozen and immutable
for this work under the workstream directive and the freeze record in
[populace-dynamics#215](https://github.com/PolicyEngine/populace-dynamics/pull/215).
This ADR does not amend their schema, semantics, or readers.

## Context

The employer-firm plan on
[issue #192](https://github.com/PolicyEngine/populace-dynamics/issues/192)
requires noise floors before thresholds, a referee round before the C3
gate block locks, and no one-shot candidate run before that lock. The
same ordering must govern the worker-to-employer or worker-to-firm-type
assignment itself. A downstream E-cell cannot certify a model if the
links used to construct the cell have unknown quality.

Here, **link** includes any accepted worker-to-employer, employer-
attachment, or worker-to-firm-type assignment consumed by an E-cell.
That includes an assigned C2 `CanonicalBand` or industry type; it does
not turn a statistical imputation into an identified firm ID. A **link
unit** is the unit on which a decision is accepted or withheld. Paired
and run-level moments may require a stricter derived unit, as specified
below.

The frozen contract seam remains the C1 spell fields
`person_id`, `spell_id`, `start_period`, `end_period`, `industry`,
`firm_size_band`, `class_of_worker`, `earnings_share`, and
`primary_job`. The implemented C2 seam is the banding functions in
`src/populace_dynamics/firms/banding.py`. Linkage-QC decisions,
adjudication labels, match scores, and linkage weights therefore live
in versioned sidecars keyed to C1 rows; they are not new C1 columns.
`spell_type` is also not a C1 field. Where needed below, transition
type is derived from adjacent spell rows plus the versioned person-month
observation/nonemployment frame; C1 spells alone do not identify exits,
entries, or censoring.

### Evidence behind the precision-first rule

LIFE-M scales carefully reviewed hand links with supervised learning.
Its published workflow used independent double review, three new
independent reviewers for disagreements, a held-out test half, and
ten-fold cross-validation within the training half. It selected
thresholds to maximize recall subject to a project-specific
**"97% precision rate"**
and evaluated that training-selected cutoff out of sample. This ADR
adopts that ordering and separation, not LIFE-M's numeric 97% choice.
See Bailey et al. (2023), sections IV.C-IV.D and Figure 5, and the
[LIFE-M linking description](https://life-m.org/linking/).

The final abstract of Bailey, Cole, Henderson, and Massey (2020)
reports that trained reviewers rejected **"15 to 37 percent"** of
links from widely used automated methods and that the combined
problems studied attenuated an intergenerational-income-elasticity
estimate by **"up to 29 percent."** Sections III.B, IV, VI.C, and VII
show why these are not merely lost-sample problems: false links can be
systematic, estimates usually moved toward zero in their case study,
and removing false links brought estimates across algorithms together.
The authors consequently recommend putting more weight on precision
than on increasing the match count. Those numeric findings describe
their historical-data exercises, not an employer-firm floor to import.

Bailey, Cole, and Massey (online 2019; print 2020), section III.B and
Table 4, separately show how application-specific inverse-propensity
weights can improve balance between a linked sample and its reference
population on included observables. They also state the load-bearing
limits: common support and the paper's unconfoundedness/properly
specified propensity-score assumption are required, and balance on
omitted or unobserved characteristics is not guaranteed. Weighting
therefore complements the precision floor; it does not repair false
positive links.

## Decision

### 1. Precision first, before every C3 threshold

1. **Every link-producing component used by a gated E-cell must have
   an independent audit artifact.** For categorical assignments, the
   artifact reports the audit-design- and survey-weighted matrix of
   adjudicated true class by assigned class, including no-counterpart,
   abstain/unlinked, and indeterminate outcomes. It also reports
   accepted-assignment precision,
   end-to-end and stage-specific recall, false-negative and abstention
   rates, and all numerators and denominators, overall and for every
   pre-registered gate-relevant stratum. If it reports a pairwise
   false-positive rate, it must define the candidate-pair universe;
   `1 - precision` is the false-discovery rate, not that pairwise rate.
   Point estimates and one-sided confidence bounds are both required.
2. **The precision floor is pre-registered before E-cell thresholds.**
   C3 must name the assignment, eligible universe, link unit, floor
   `P_floor`, confidence level, required strata, pooling rule, and
   failure disposition before it names any threshold for an E-cell
   that consumes that assignment. Recall is always published. Whether
   recall also gates is an explicit C3 referee decision, not an
   after-the-fact response to results.
3. **Passing uses a confidence bound, not the observed proportion.**
   The lower one-sided confidence bound for precision must be at least
   `P_floor` overall and at every stratum or derived-unit level that
   C3 designates operative. Oversampled strata are combined only with
   their recorded sample inclusion weights.
4. **No threshold shopping follows a linkage failure.** A failed or
   unevaluable applicable floor makes the linked E-cell invalid. It is
   not a model miss that can be cured by widening the E-cell tolerance,
   dropping an inconvenient stratum, or selecting a different weighted
   result. A registered candidate with any required invalid cell cannot
   pass the employer block.
5. **The audit precedes the one-shot run.** Linkage-QC results and the
   immutable adjudication manifest must be on the C3 record before a
   candidate may consume the matcher. A materially changed matcher,
   cutoff, candidate-generation rule, source vintage, or target
   vocabulary requires a new versioned audit or the pre-registered
   transport test; it never inherits a pass silently.

### 2. Hand-adjudication sample

The C3 block must register the following sample design before labels
are opened.

1. **Frame and two audit arms.** Define the complete eligible universe,
   candidate-generation rules, accepted-link rule, true no-counterpart
   cases, and abstentions. Draw (a) an accepted-assignment arm, which
   identifies false positives and precision, and (b) an eligible-universe
   truth-search arm containing accepted, rejected, and no-link cases.
   The second arm uses an independent exhaustive search or external
   reference that can find a true counterpart omitted by candidate
   generation; reviewing only the matcher's candidate set is not an
   end-to-end recall study. Report candidate-generation recall separately
   from selector/cutoff recall. Because a unit may enter both arms, C3
   registers the dual-frame overlap and combined-inclusion estimator so
   it is neither omitted nor counted twice.
2. **Stratification follows the moments the matcher feeds.** For the
   accepted arm, stratify at minimum by the five assigned C2
   `CanonicalBand` values (plus withheld or unresolved assignments),
   assigned NAICS major industry, and the derived spell/transition class
   relevant to the battery: stay, job-to-job, exit, or entry, crossed
   with `primary_job` where that changes the estimand. Rare E11 origin-
   destination size pairs and E12 firm types are deliberately
   oversampled. Because rejected and no-link units lack an assigned or
   known-true class before review, the universe arm uses a separate
   pre-link stratification frame observed for every eligible unit, then
   reports recall by adjudicated true band, industry, and transition
   domain. C3 publishes any pooling of sparse strata before adjudication
   and retains every arm-specific inclusion probability.
3. **Target size is powered at the floor.** C3 registers `P_floor`, a
   substantively meaningful design precision `P_design > P_floor`,
   one-sided size `alpha`, power `1 - beta`, and its multiplicity rule.
   For independent, equal-probability accepted assignments within a
   simple-random operative stratum, the target is the smallest number
   `n` for which an integer critical count `c` exists such that

   `Pr[X >= c | X ~ Binomial(n, P_floor)] <= alpha`

   and

   `Pr[X >= c | X ~ Binomial(n, P_design)] >= 1 - beta`.

   Repeated spells, pairs, and runs are not independent Bernoulli draws.
   A worker-, firm-, or time-clustered, unequal-probability, finite, or
   dual-frame design instead uses design-based analytic or simulation
   power with the registered clustering unit, inclusion weights, finite-
   population correction where material, and anticipated design effect.
   The target includes unusable-record, indeterminate, and nonresponse
   inflation. If multiple strata must clear, C3 distinguishes simultaneous
   confidence coverage from an intersection-union pass rule and computes
   the **joint** probability that every required stratum passes at
   `P_design`; powering each stratum separately at `1 - beta` is not
   enough. A round number without the applicable calculation is not a
   target-size justification. The eligible-universe truth-search arm has
   its own registered target, based on expected true-counterpart
   prevalence and a desired recall-bound width or, if recall gates, an
   analogous floor-and-power calculation. It must contain enough
   independently searched true links to make false-negative uncertainty
   informative.
4. **Blind, independent coding.** Two trained coders independently see
   the same source evidence and candidate set, but not the matcher's
   score, cutoff, accepted choice, downstream outcome, or the other
   coder's decision. They code `link`, `no link`, or `insufficient
   evidence` under a frozen manual. A third coder or standing panel
   adjudicates disagreements without majority labels being disclosed
   first. Before labels open, C3 registers whether an indeterminate,
   unusable-evidence, or nonresponse case counts conservatively as
   incorrect, enters partial-identification bounds, or makes the floor
   unevaluable; hard cases may not be dropped from denominators after
   their labels are known. The artifact reports each rate, its effect on
   precision and recall denominators, agreement, disagreement, coder/
   manual versions, and final dispositions. Coders first calibrate on
   separate, vetted cases excluded from both audit arms; blinded
   audit/gold repeats measure
   continuing coder accuracy as well as agreement. The registered
   precision test either propagates estimated reference-label error or
   publishes the sensitivity bound needed to clear the floor. The label
   is **hand-adjudicated reference truth**, not a claim of infallible
   ground truth.
5. **Provenance is committed and immutable.** Commit a privacy-safe
   manifest containing the frame query and vintage, stratum definitions,
   random seed, selected-row hashes or access-controlled immutable IDs,
   inclusion probabilities, evidence and coding-manual versions, coder
   assignment protocol, adjudication rule, counts, and artifact hashes.
   Raw restricted records and direct identifiers remain outside git.
   Any replacement or exclusion is logged; the sample is never silently
   refreshed.
6. **No train-test leakage.** Neither adjudication arm, its final labels,
   nor disagreement dispositions may train, tune, select features for,
   set a cutoff for, or otherwise adapt the matcher it scores. A leaked
   sample is retired from evaluation and replaced under a new manifest.

### 3. Linkage-bias reweighting on observables

1. **Name the target population and identification assumption.** Each
   linked analysis declares its actual analysis unit (assignment, pair,
   transition, run, or firm/type cluster), the eligible reference
   population it is intended to represent, and the pre-link observables
   `X` available for both linked and unlinked units.
   Candidate observables include source/vintage, age, sex, state,
   `class_of_worker`, `primary_job`, industry information known before
   the assignment, earnings/tenure measures, missingness, and transition
   opportunity. For each E-cell outcome `Y`, the registered identification
   claim is usable-link inclusion independent of `Y` conditional on `X`,
   plus positivity on the target support. It is an assumption to defend,
   not a result of a balance test. Post-link outcomes cannot be used to
   manufacture balance.
2. **Estimate and publish actual inclusion propensities.** Fit and
   version `s_i = Pr(L_i = 1 | X_i)`, where `L_i` denotes inclusion in
   the usable linked subsample of the complete eligible universe at the
   E-cell's analysis unit. A person-level score does not automatically
   weight a pair, run, or cluster; C3 models that unit's inclusion or
   pre-registers and justifies a joint construction from component
   scores. Publish the model specification, training
   population, out-of-sample diagnostics, propensity distributions for
   linked and reference units, overlap/common-support checks, covariate
   balance before and after weighting, weight distribution, and effective
   sample size. Trimming, stabilization, normalization, and capping rules
   are pre-registered.
3. **Use a weight derived for the sampling construction.** For actual
   inclusion propensity `s_i`, full-population inverse-probability
   weighting uses `1 / s_i`, subject to the registered base design.
   Bailey, Cole, and Massey instead append a linked-sample copy to a
   reference-population copy and fit
   `r_i = Pr(D_i = linked copy | X_i)` in that stack. For this density-
   ratio construction only, their normalized inverse-odds weight is
   `[(1 - r_i) / r_i] [q / (1 - q)]`, where `q` is the linked-copy
   share of the stack. `s_i` and `r_i` are not interchangeable. The
   linkage adjustment multiplies the pre-existing survey/design/
   opportunity weight; it does not replace that base weight. C3 derives
   and registers the applicable form before seeing the E-cell.
4. **Publish weighted and unweighted with valid uncertainty.** Every
   E-cell consuming links publishes both, labels the registered operative
   version
   (`unweighted` or `linkage_ipw`), and explains its estimand. The
   `unweighted` label means base-weighted without the linkage adjustment,
   not equal-record weighting that discards a source survey design. The
   adjudication sample-inclusion weight and the linkage-propensity weight
   are distinct and must not be conflated. Intervals and gate statistics
   account for estimated propensities, base survey/design weights, any
   audit weights entering the estimator, trimming or stabilization, and
   repeated-person, firm/type, and time clustering. A point estimate and
   effective sample size alone are insufficient.
5. **Treat overlap failure as scope failure.** Extreme weights, absent
   common support, or material residual imbalance are reported, not
   hidden by ad hoc trimming. If the registered weighting diagnostic
   fails, the weighted cell is invalid. The unweighted diagnostic remains
   visible but cannot be substituted as the gate after results are known.
   Support-based trimming changes the target population, which must be
   renamed and reported rather than presented as the original estimand.
   These weights mitigate selection on included observables only; they
   do not correct a wrong link, establish balance on unobservables, or
   turn a firm type into an observed firm identity.

### 4. Battery wiring and invalidation

An overall floor failure invalidates every linked cell using that
matcher version. A required-stratum failure invalidates every cell whose
estimand includes that stratum, unless C3 pre-registers a genuinely
disjoint matcher and estimand. A passing marginal link floor does not by
itself certify a pair or a run: C3 must either audit the derived unit
directly or register and justify a conservative composition rule.

| cell | linkage unit and required QC | weighting and failure disposition |
|---|---|---|
| **E4 — retention pairs** | Audit endpoint assignments and, if C3 makes it operative, the derived same-employer/same-attribute decision. The audit strata include age, industry, C2 band, transition month, and `primary_job` status used by the cell. | Publish pair-opportunity estimates unweighted and with linkage-IPW. If any C3-designated endpoint or pair-level floor fails, all affected E4 retention cells are invalid. |
| **E5 — attachment runs** | If C3 designates a run-level floor, audit the full multi-window run label, including false continuation and false break errors. A per-month pass alone cannot certify a run because error compounds with length. | Weight the eligible run opportunity, not each observed linked month as if independent. If any C3-designated endpoint or run-level floor fails, the affected E5 run-length cells are invalid. |
| **E9 — earnings-change coherence** | Derive stay, job-to-job, exit, and entry from adjacent C1 spells plus the versioned person-month observation/nonemployment frame. Audit any C3-designated transition floor; for job-to-job cells, audit both origin and destination firm-size/industry assignments. | Define propensity and composite weight on the eligible transition opportunity, then publish both versions within class. A failed C3-designated origin, destination, or transition-class floor invalidates the corresponding E9 cells; the referee cannot replace them post hoc with a different definition. |
| **E11 — firm-size flow ladder** | The unit is an origin-destination job-to-job pair. Audit the joint ordered C2-band assignment. An unresolved `BandSpan` is not a correct categorical assignment merely because it contains the eventual band. | Model inclusion and weight at the ordered-pair opportunity. An overall or joint-pair floor failure invalidates E11; a required origin/destination stratum failure invalidates every E11 cell containing it. |
| **E12 — variance and coworker structure** | Phase 2 must audit worker-to-firm-type assignment and any generated same-firm or coworker co-assignment at the exact unit the E12 estimand uses. Type agreement alone cannot validate a claim about an identified firm. | Model inclusion at the worker pair, co-assignment, or cluster unit used by the decomposition; a worker-only propensity is insufficient without a justified composition. Publish both versions. If truth, floor, or support fails, every E12 cell using it is invalid and phase 2 is a no-go. |

E3 and E8 do not ordinarily require a worker-to-firm link, and E10
re-runs the existing locked PSID earnings gates without a new noise
floor. They are not blanket exemptions: if a final C3 implementation
constructs any of them from accepted employer or firm-type assignments,
the precision-first law applies. Linkage failure never weakens E10 or
changes an existing PSID threshold.

### 5. Phase scope and real seams

#### Phase 1 — spell hazards, no two-sided register

Phase 1 has within-panel employer attachment and firm attributes on
spells, not an observed two-sided worker-firm roster. The SIPP reader's
`EJB{n}_JOBID` is a within-panel attachment key. Its spell collapse
currently carries raw `empsize_code`; `sipp_empsize_to_canonical` in
`banding.py` preserves source-band ambiguity through `BandSpan`, while
the frozen C1 seam ultimately carries one `CanonicalBand`. Therefore:

1. QC scores the **final accepted assignment consumed by the E-cell**,
   after any ambiguity resolution, not the raw SIPP code or a claim that
   an ambiguous span is exact.
2. An exact SIPP interval-to-band map establishes only numeric interval
   nesting. SIPP measures establishment size, so it does not by itself
   validate administrative enterprise size.
3. Sidecars join to C1 with `person_id` and `spell_id`. They do not add
   `job_id`, `firm_id`, match score, adjudication status, or weights to
   frozen C1.
4. E4/E5 score retention and attachment, E9 scores transition-conditioned
   earnings changes, and E11 scores firm-size flows only after their
   applicable link and reweighting requirements above are evaluable.

The pre-C3 floor draft on
[PR #212](https://github.com/PolicyEngine/populace-dynamics/pull/212)
does not satisfy or conflict with this ADR: it estimates sampling noise
in E3/E4/E5/E8/E9 after the linkage inputs are defined. Its half-splits
cannot reveal a common linkage bias. The seam artifact on
[PR #214](https://github.com/PolicyEngine/populace-dynamics/pull/214)
likewise remains a separate prerequisite: it compares SIPP and J2J rate
levels, but does not estimate precision or recall for a worker-to-firm-
type assignment.

#### Phase 2 — BLM firm types, still not observed firms

Phase 2 proposes a BLM-style register of firm **types** (industry x
size x state), not identified enterprises or a public worker-firm
roster. E12 is especially sensitive to false assignments because a bad
worker-to-firm or coworker link moves covariance between the within- and
between-firm components. Bailey, Cole, Henderson, and Massey (2020),
section VI.C and Figure 7, show that false links often attenuated their
intergenerational-income-elasticity estimates and that removing them
reconciled estimates. Their broader result also warns that systematic
error can make the bias algorithm-dependent rather than always
attenuating.

For that reason, a phase-2 E12 story must identify an admissible
hand-adjudication frame for the actual assignment unit and clear its
precision floor. If public margins cannot support that truth, C3 records
the limitation as a phase-2 no-go; calibration fit to aggregates is not
a substitute. The register may support firm-type policy claims only at
the level it identifies. It may not relabel type agreement as firm-
identity or coworker validation.

### 6. Items for the C3 referee round — deliberately unresolved

The referee round must decide and pre-register the following. This ADR
does not resolve them:

1. The numeric precision floor or floors; `P_design`, `alpha`, power,
   confidence interval, multiplicity correction, and whether recall has
   an operative floor.
2. The phase-1 and phase-2 adjudication frames, permissible evidence,
   exact truth labels, privacy-safe manifest, and whether an E12 truth
   source exists at all.
3. Beyond E11's required ordered pair and the actual co-assignment unit
   used by E12, which endpoint, pair, transition, and run levels gate;
   how a passing endpoint result composes, if at all; and which sparse
   strata may be pooled before the powered sample target is calculated.
4. The target reference population, propensity model and observables,
   overlap and balance tolerances, weight stabilization/trimming rule,
   and the pre-registered operative weighting for E4, E5, E9, E11, and
   E12.
5. The exact calibration/gate cell partition, including held-out axes;
   the job-count-to-person-count adjustment for phase 0; the private-
   comparable versus caveat treatment of J2J's broader employer
   universe; and the ASEC firm-size/tenure reference-period mismatch.
6. PR #212's measurement choices: E3 quantile gaps versus weighted-ECDF
   gaps, and E9 stay IQR versus a broader distributional distance. The
   integer and wave-constant heaping degeneracies remain on the record.
7. PR #214's cross-wave SIPP job-ID consistency check and the final
   ruling on J2J rate levels, SIPP persistence, and seam-aware hazard
   estimation.
8. The E12 estimand and entity: firm type versus generated pseudo-firm,
   the minimum evidence for within/between variance and coworker
   correlation, and the phase-2 identification/go-no-go standard.
9. Long-window tenure evidence from PSID/NLSY and any transport test for
   applying one adjudication result across source vintages or populations.

The settled C1 fields, C2 semantics and five bands, explicit
`NOEMP`/`FIRMSIZE` coding, class-of-worker universe, and person-table
geography join are outside this list. Reopening them requires the joint
C1/C2 amendment process, not the C3 referee round.

## Consequences

- C3 gains a linkage-quality input gate before its model-fit gates.
  E4, E5, E9, E11, and E12 cannot certify a candidate from unaudited or
  floor-failing assignments.
- Every link-consuming cell exposes the observable-selection question
  by publishing linkage-IPW and unweighted estimates together, with one
  operative version chosen before the candidate result is seen.
- The required evidence is carried in sidecars and reported artifacts;
  C1, C2, `gates.yaml`, readers, and banding code are unchanged.
- Exact floor values, E-cell thresholds, and operative weighting choices
  remain decisions for the C3 referee round.

## References

- Bailey, Martha, Peter Z. Lin, A. R. Shariq Mohammed, Paul Mohnen,
  Jared Murray, Mengying Zhang, and Alexa Prettyman. 2023.
  ["The Creation of LIFE-M: The Longitudinal, Intergenerational Family
  Electronic Micro-Database Project."](https://doi.org/10.1080/01615440.2023.2239699)
  *Historical Methods* 56 (3): 138-159.
- Bailey, Martha J., Connor Cole, Morgan Henderson, and Catherine Massey.
  2020. ["How Well Do Automated Linking Methods Perform? Lessons from US
  Historical Data."](https://doi.org/10.1257/jel.20191526)
  *Journal of Economic Literature* 58 (4): 997-1044. The 29-percent
  figure above follows the
  [final AEA abstract](https://www.aeaweb.org/articles?id=10.1257/jel.20191526),
  while the
  [earlier deposited author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC8294155/)
  reports 20 percent in its abstract, introduction, and section VI.C.
- Bailey, Martha, Connor Cole, and Catherine Massey. 2019 online / 2020
  print. ["Simple Strategies for Improving Inference with Linked Data: A
  Case Study of the 1850-1930 IPUMS Linked Representative Historical
  Samples."](https://doi.org/10.1080/01615440.2019.1630343)
  *Historical Methods* 53 (2): 80-93.
