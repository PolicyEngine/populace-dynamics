# UKHLS transition-table provenance

`ukhls_employment_state_transitions.csv` and
`ukhls_income_decile_transitions.csv` are aggregated,
disclosure-controlled derivatives of the Understanding Society
(UKHLS) main survey, Waves 1-15 (UKDS SN 6614, End User Licence),
601,795 person-wave observations.

Salvaged unchanged from archived
[policyengine-uk-data#346](https://github.com/PolicyEngine/policyengine-uk-data/pull/346)
(branch `feat/panel-persistent-ids-345`), per the UK-extension
decision thread
[populace#148](https://github.com/PolicyEngine/populace/issues/148).
The estimator that produced them is re-hosted at
`src/populace_dynamics/data/ukhls.py`; regenerating requires staging
the UKDS `.dta` download at `~/PolicyEngine/ukhls-data` (or
`POPULACE_DYNAMICS_UKHLS_DIR`) and running
`save_transition_tables()`.

## Construction

- Consecutive waves paired on `pidp` (cross-wave person key).
- Employment: four-state label (IN_WORK / UNEMPLOYED / RETIRED /
  INACTIVE) collapsed from `jbstat`; transitions grouped by 5-year
  age band and sex.
- Income deciles: within-wave decile rank of `fimngrs_dv` (gross
  monthly personal income), so the estimator is scale-invariant
  across years.
- Disclosure control: cells with fewer than 10 observations dropped
  (ONS/UKDS safeguarded convention); probabilities re-normalised
  within surviving `(age_band, sex, from)` groups. Raw microdata is
  never committed.

## External spot checks (from the source PR review)

- MALE 25-29 IN_WORK -> IN_WORK = 95.6% (matches ONS LFS 2-quarter
  flow rate).
- MALE 25-29 UNEMPLOYED -> IN_WORK = 36.5% (ONS published ~35-40%).
- Average probability of remaining in the same income decile
  year-on-year = 39.9% (consistent with IFS Living Standards
  mobility estimates).

These checks are pinned by `tests/test_ukhls.py`.
