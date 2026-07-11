# ADR 0002: UK extension — UKHLS as the second label panel

**Status:** Proposed (pending decision on
[populace#148](https://github.com/PolicyEngine/populace/issues/148))

## Context

Archived [policyengine-uk-data#346](https://github.com/PolicyEngine/policyengine-uk-data/pull/346)
built a full UK dynamic-panel pipeline (demographic ageing, household
transitions, UKHLS-estimated labour-market and income-mobility
matrices, a one-call `advance_year` composer). It was closed unmerged
during the uk-data archive cleanup with the note that UK panel work
needs a Populace-owned product/architecture decision, now tracked as
[populace#148](https://github.com/PolicyEngine/populace/issues/148).
The reviewer (second pass) also identified correctness blockers in the
transition-application machinery: mortality removed only person rows,
orphaning benunit/household entities with live weights, and
immigration cloned from unweighted donors.

This repository already has the architecture that PR lacked: a
population-view scoring harness, label-verified panel readers, and a
locked pre-registration evaluation contract. The question is what, if
anything, to bring across.

## Decision

1. **The UK becomes a candidate second country under this
   repository's harness**, with UKHLS (UKDS SN 6614) playing the role
   PSID plays for the US: the label panel from which transition
   structure is estimated and against which candidates are scored.
2. **Salvage the data machinery.** The UKHLS reader and the
   wave-pairing, cell-suppressed transition estimator
   (`src/populace_dynamics/data/ukhls.py`), the two committed
   aggregate tables (`data/external/ukhls_*_transitions.csv`, with a
   provenance note), and the ONS mortality/fertility loaders with
   their pinned public workbooks
   (`src/populace_dynamics/uk/ons_rates.py`,
   `data/external/ons_*.xlsx`). These are inputs and references —
   analogous to the NCHS/Census external references — not scored
   model output.
3. **Port the transition-application machinery with the review
   blockers fixed, as unscored model machinery.** The dynamics layer
   lives in `src/populace_dynamics/uk/` on a light three-table
   `UKPanelDataset` contract (no policyengine-uk dependency):
   `demographic_ageing`, `household_transitions`, and the
   `advance_year` composer. Both blockers from #346's second-pass
   review are fixed: every person-removal path (mortality,
   emigration) prunes benunits/households left with no surviving
   members via a shared `prune_orphaned_entities` helper, and
   immigration donors are drawn proportionally to household weight.
   This code carries **no gate certification**: any scored UK use
   goes through candidate registration (issue #42) and a one-shot
   pre-registered run under UK gates, like every US candidate.
4. **No change to `gates.yaml`.** The existing contract is US/PSID
   scoped and locked. UK evaluation gates, if the program proceeds,
   require their own pre-registration: UK-specific noise floors,
   moment battery scoping, and a referee round, via the standard
   amendment process.

## Consequences

- UKHLS raw microdata stages at `~/PolicyEngine/ukhls-data`
  (override: `POPULACE_DYNAMICS_UKHLS_DIR`), mirroring the PSID
  pattern; it is never committed, and UKHLS-dependent tests skip
  off-machine.
- Only aggregated, cell-suppressed (n >= 10) tables are committed,
  matching the ONS/UKDS safeguarded-microdata convention.
- The archived #346 branch (`feat/panel-persistent-ids-345` on
  policyengine-uk-data) remains the reference for future salvage;
  nothing else from it is imported until populace#148 resolves and UK
  gates exist.
