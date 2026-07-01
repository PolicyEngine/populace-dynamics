# Team and expertise

## Overview

The current planning lead can set the technical and product direction,
but that is not, by itself, a sufficient team for full implementation. A
credible plan distinguishes between:

- the current project lead
- the external review capacity the project needs
- the implementation expertise required to execute the roadmap

This chapter describes the kinds of roles and expertise the project
needs, not headcounts or compensation.

## Current project lead

### Max Ghenis

Max Ghenis leads PolicyEngine and brings the core infrastructure
experience behind the Enhanced CPS and PolicyEngine's tax-benefit
models. His role in this project is architectural leadership, product
direction, and integration with the broader PolicyEngine ecosystem.

## Needed review capacity

The public proposal should not imply named advisors, reviewers, or
staff are committed unless those commitments are explicit. Before full
implementation, the project should recruit review capacity covering:

- Social Security policy and benefit-rule expertise
- retirement economics and wealth measurement
- dynamic microsimulation practice
- statistical imputation, calibration, and validation
- public-interest product and user research

## Required expertise

The project should cover at least the following kinds of expertise
during implementation:

### Technical lead or research engineer

Implementation leadership is needed to own the longitudinal `populace`
pipeline, modeling infrastructure, and reproducibility workflow. This
capacity should not be treated as optional.

### Research economist or quantitative social scientist

This expertise is needed for validation design, policy interpretation,
benchmark replication, and reform analysis. The project needs people
whose job is to ask whether the results are economically credible, not
just whether the code runs.

### Data engineering and data science

The project requires substantial work on harmonization, ingestion,
versioning, and reproducibility across the surveys and administrative
sources that `populace` integrates and calibrates against. This is a
real workload, not a background task.

### Research assistance

Research assistance will likely be needed for:

- documentation of policy rules
- benchmark assembly
- literature synthesis
- validation-table construction

## Review and governance

The project should use structured outside review rather than relying
only on internal confidence. That review should include:

- Social Security policy experts
- microsimulation practitioners
- survey and wealth experts
- users from outside the immediate project team

The goal is to catch overclaiming early and document disagreements
openly.

## Operating model

The recommended operating model is:

- the project lead sets scope, standards, and go/no-go decisions
- external reviewers inform validation standards and policy relevance
- the implementation team owns delivery
- reviewers challenge the validation record at each major gate

That is a healthier structure than informal part-time execution by a
small core team.

## Why this matters

This project is easy to underscope because it looks like an extension of
existing PolicyEngine work. It is not just that. Dynamic microsimulation
adds persistent complexity in data construction, transition modeling,
validation, and product boundaries. The team section should therefore
signal seriousness about staffing from the outset.
