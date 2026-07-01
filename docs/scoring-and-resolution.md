# Scoring and resolution

## From fidelity to scoring

Most microsimulation validation asks: does the model match published
aggregates? That is fidelity, and it is necessary — but it is a weak
credential, because a model can be tuned to reproduce the tables it
was built from. This project holds itself to a stronger standard:
**a claim is validated when it improves prediction of something that
subsequently resolves.** The model's credential is its scorecard.

Nothing here is graded on a curve and nothing is quietly deleted. A
confident, wrong output scores worse than a hedged, correct one, and
the scorecard is public either way.

## Five scoring surfaces

### 1. Annually resolving components

The near-term outputs in tier 2 of
[domains-of-validity.md](domains-of-validity.md) resolve against
administrative publications on a calendar:

| Quantity | Resolves against | Cycle |
|---|---|---|
| Beneficiary counts by type, age, sex | SSA Annual Statistical Supplement | Annual |
| Average and aggregate benefits by type | SSA Annual Statistical Supplement | Annual |
| Covered workers and taxable payroll | Trustees Report | Annual |
| Cost-of-living adjustment | SSA COLA announcement | Annual (October) |
| DI incidence and awards | SSA disability statistics | Annual |
| Claiming-age distribution | SSA program statistics | Annual |

Each published forecast cell carries a resolution rule naming the
exact table and vintage that settles it. When the number lands, the
cell resolves and the score goes on the record.

### 2. Forecasting the forecasters

Official projections revise constantly. Predicting the *next
revision* — where the Trustees or CBO will move the depletion date,
the 75-year balance, or key assumption values — resolves in months
rather than decades, and it is decision-relevant: planners and
analysts act on the official number, so anticipating its movement is
useful even to those who treat it as authoritative.

### 3. Retrodiction with leakage control

Long-horizon dynamics cannot wait decades for a grade, but the past
already resolved. The protocol: build the panel from data vintages
available at time T, project forward, and score against realized
outcomes at T+k. The `populace` data registry pins source vintages,
which is what makes "what could the model have known on date X" an
enforceable constraint rather than an honor-system claim. Retrodictive
scores are necessary but not sufficient — calibration under the
historical regime does not guarantee calibration under a new one —
so they complement, and never substitute for, the live annual cells
in surface 1.

### 4. Statutory resolution

Much of a benefit calculation is fixed by law, not forecast. Where an
output turns on rules, the rules engine computes it exactly, and an
enacted policy settles the corresponding conditional cells
immediately. This cleanly separates the deterministic slice (scored
by computation) from the genuinely uncertain slice (scored by
resolution), so uncertainty budgets attach only to the parts that
are actually uncertain.

### 5. Held-out panel moments

The population layer itself is scored the way `populace` already
scores cross-sections: held-out evaluation against moments the model
was not fit to — earnings-mobility matrices, autocorrelation
structure and higher-order moments of earnings changes
[@guvenen2015earningsrisk], cohort age-earnings profiles, and
family-transition rates, evaluated on held-out panel records.

## Calibration as trust weight

The point of the scorecard is not self-congratulation; it is to give
consumers a number to weight the model by. An analyst — or an AI
agent composing this model with other tools — should put weight on an
output proportional to its demonstrated reliability in that domain.
Every API and MCP response therefore ships with the calibration
history of its output class: hit rates, interval coverage, and the
resolution record behind them. Where the model has no track record
yet, it says so.

## The contribution rule

Scoring is also the governance mechanism, inherited from `populace`:
**a contribution merges if and only if it improves the population's
score on held-out facts.** A better mortality module, a sharper
claiming model, a new earnings architecture — from this team or
anyone else's — earns its place by moving the scorecard, not by
seniority or affiliation. This is what makes the project open in a
sense stronger than its license: the standard of evidence is the
same for every contributor, and the evidence is public.

## Publication discipline

- Every forecast cell stores its question, assumptions, uncertainty,
  and resolution rule as data, not prose.
- Scores are reported per question and per model configuration, with
  proper scoring rules for probabilistic cells.
- Misses are published with the same prominence as hits, with a
  post-mortem note where the miss is instructive.
- Superseded methods keep their historical scorecards; improvement is
  demonstrated against the record, not by replacing it.

A demo that cannot be wrong proves nothing. The project would rather
publish a cell that resolves against it, in the open, than a polished
projection that never has to face a number.
