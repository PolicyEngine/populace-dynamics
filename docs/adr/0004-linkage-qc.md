# ADR 0004: Employer-firm linkage QC requirements for C3

**Status:** Proposed — input to the C3 referee round; this document
locks no C3 threshold. C1 and C2 are frozen by the joint sign-off
recorded in
[populace-dynamics#215](https://github.com/PolicyEngine/populace-dynamics/pull/215).
This ADR treats both contracts as immutable inputs and does not amend
their schema, semantics, or readers.

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

The implemented seams remain the frozen C1 spell fields
`person_id`, `spell_id`, `start_period`, `end_period`, `industry`,
`firm_size_band`, `class_of_worker`, `earnings_share`, and
`primary_job`, and the C2 banding functions in
`src/populace_dynamics/firms/banding.py`. Linkage-QC decisions,
adjudication labels, match scores, and linkage weights therefore live
in versioned sidecars keyed to C1 rows; they are not new C1 columns.
`spell_type` is also not a C1 field. Where needed below, transition
type is derived from adjacent spell rows.

### Evidence behind the precision-first rule

LIFE-M scales carefully reviewed hand links with supervised learning.
Its published workflow used independent double review, additional
review of disagreements, a held-out test half, and ten-fold
cross-validation within the training half. It selected thresholds to
maximize recall subject to a project-specific **"97% precision rate"**
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
limits: common support and a correctly specified selection model are
required, and balance on omitted or unobserved characteristics is not
guaranteed. Weighting therefore complements the precision floor; it
does not repair false positive links.

## Decision

### 1. Precision first, before every C3 threshold

1. **Every link-producing component used by a gated E-cell must have
   an independent audit artifact.** The artifact reports the full
   weighted confusion matrix, precision, recall, false-positive rate,
   false-negative rate, abstention/unlinked rate, and denominators,
   overall and for every pre-registered gate-relevant stratum. Point
   estimates and one-sided confidence bounds are both required.
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
   candidate-generation rules, accepted-link rule, and abstentions.
   Draw (a) an accepted-assignment arm, which identifies false positives
   and precision, and (b) an eligible-universe arm containing accepted,
   rejected, and no-link cases, whose complete candidate sets are
   reviewed to identify false negatives and recall. Reviewing accepted
   assignments alone is not a recall study.
2. **Stratification follows the moments the matcher feeds.** At minimum,
   stratify by the five C2 `CanonicalBand` values (plus missing,
   withheld, or unresolved assignments), NAICS major industry, and the
   derived spell/transition class relevant to the battery: stay,
   job-to-job, exit, or entry, crossed with `primary_job` where that
   changes the estimand. Rare E11 origin-destination size pairs and E12
   firm types are deliberately oversampled. C3 must publish any pooling
   of sparse strata before adjudication and retain each inclusion
   probability.
3. **Target size is powered at the floor.** C3 registers `P_floor`, a
   substantively meaningful design precision `P_design > P_floor`,
   one-sided size `alpha`, power `1 - beta`, and its multiplicity rule.
   For each operative pooling level, the target is the smallest number
   `n` of adjudicated accepted assignments for which an integer critical
   count `c` exists such that

   `Pr[X >= c | X ~ Binomial(n, P_floor)] <= alpha`

   and

   `Pr[X >= c | X ~ Binomial(n, P_design)] >= 1 - beta`.

   The registered target also includes anticipated abstention,
   unusable-record, and nonresponse inflation. If multiple strata must
   each clear the floor, the power calculation uses the pre-registered
   family-wise error allocation. A round number without this calculation
   is not a target-size justification.
4. **Blind, independent coding.** Two trained coders independently see
   the same source evidence and candidate set, but not the matcher's
   score, cutoff, accepted choice, downstream outcome, or the other
   coder's decision. They code `link`, `no link`, or `insufficient
   evidence` under a frozen manual. A third coder or standing panel
   adjudicates disagreements without majority labels being disclosed
   first. The artifact reports agreement, disagreement, insufficient-
   evidence rates, coder/manual versions, and final dispositions. The
   label is **hand-adjudicated reference truth**, not a claim of
   infallible ground truth.
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

1. **Name the target population.** Each linked analysis declares the
   eligible reference population it is intended to represent and the
   pre-link observables `X` available for both linked and unlinked units.
   Candidate observables include source/vintage, age, sex, state,
   `class_of_worker`, `primary_job`, industry information known before
   the assignment, earnings/tenure measures, missingness, and transition
   opportunity. Post-link outcomes cannot be used to manufacture
   balance.
2. **Estimate and publish link propensities.** Fit and version
   `p_i = Pr(L_i = 1 | X_i)`, where `L_i` denotes inclusion in the
   usable linked subsample. Publish the model specification, training
   population, out-of-sample diagnostics, propensity distributions for
   linked and reference units, overlap/common-support checks, covariate
   balance before and after weighting, weight distribution, and effective
   sample size. Trimming, stabilization, normalization, and capping rules
   are pre-registered.
3. **Use Bailey-Cole-Massey-style weights.** The default is a normalized
   inverse-link-propensity weight appropriate to the declared target.
   If the linked and reference samples are stacked as in Bailey, Cole,
   and Massey, the normalized inverse-odds form is
   `[(1 - p_i) / p_i] [q / (1 - q)]`, where `q` is the linked share.
   A different sampling construction may require `1 / p_i`; C3 must
   derive and register the form rather than choose it after seeing the
   E-cell.
4. **Publish weighted and unweighted together.** Every E-cell consuming
   links publishes both, labels the registered operative version
   (`unweighted` or `linkage_ipw`), and explains its estimand. The
   adjudication sample-inclusion weight and the linkage-propensity weight
   are distinct and must not be conflated.
5. **Treat overlap failure as scope failure.** Extreme weights, absent
   common support, or material residual imbalance are reported, not
   hidden by ad hoc trimming. If the registered weighting diagnostic
   fails, the weighted cell is invalid. The unweighted diagnostic remains
   visible but cannot be substituted as the gate after results are known.
   These weights mitigate selection on included observables only; they do
   not correct a wrong link, establish balance on unobservables, or turn a
   firm type into an observed firm identity.

## Consequences

- C3 gains a linkage-quality input gate before its model-fit gates.
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
  not the earlier author manuscript.
- Bailey, Martha, Connor Cole, and Catherine Massey. 2019 online / 2020
  print. ["Simple Strategies for Improving Inference with Linked Data: A
  Case Study of the 1850-1930 IPUMS Linked Representative Historical
  Samples."](https://doi.org/10.1080/01615440.2019.1630343)
  *Historical Methods* 53 (2): 80-93.
