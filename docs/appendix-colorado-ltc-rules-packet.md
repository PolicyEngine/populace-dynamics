# Colorado LTC rules packet

This appendix translates the proposal's Colorado pilot idea into a
source-of-truth rules packet. The point is not to prove that one state
pilot solves long-term care modeling. The point is to show that a
serious first pilot can be scoped, sourced, and validated using
official materials rather than informal policy summaries.

Colorado is a good pilot state for three reasons.

- It has a visible and evolving LTSS ecosystem spanning nursing
  facilities, HCBS waivers, Community First Choice, PACE, and
  disability-related pathways
  [@colorado2026ltssprograms; @colorado2026pace; @colorado2026ebd].
- The state publishes member-facing and operational guidance that is
  detailed enough to support a real first-pass rules layer
  [@colorado2026trusts; @colorado2025incomeTrustMemo; @colorado2026cma].
- The design challenge is large enough to be meaningful but still narrow
  enough to fit into an adjacent pilot rather than a full national LTSS
  build.

## Source hierarchy

The pilot should treat sources in descending order of authority.

### 1. Federal floor

These sources define the national legal and parameter environment that
Colorado must operate within.

- Medicaid eligibility policy pages for spousal impoverishment, trust
  treatment, and transfers of assets
  [@medicaid2025eligibility; @medicaid2023spousalextension].
- Annual CMS bulletins on SSI, spousal impoverishment, and home-equity
  standards [@medicaid2026spousal].
- Federal estate-recovery guidance [@medicaid2025estate].
- Federal HCBS authority guidance [@medicaid2025hcbs].
- Federal PACE program guidance [@cms2025pace].

These are not enough to decide an individual case. They are the floor
under the state-specific implementation.

### 2. Colorado program pages

These sources identify the live program pathways a household can
actually enter.

- LTSS program hub page [@colorado2026ltssprograms]
- Elderly, Blind, and Disabled waiver page [@colorado2026ebd]
- PACE page [@colorado2026pace]
- Case Management Agency directory [@colorado2026cma]
- Buy-In for Working Adults with Disabilities page
  [@colorado2026buyin]
- Trust Policy and Recoveries page [@colorado2026trusts]
- Health First Colorado member handbook for member-facing policy
  implications, including estate recovery context
  [@colorado2024memberhandbook]

These pages are especially valuable for product design because they show
how the state itself explains pathway differences to applicants and
families.

### 3. Colorado operational memos and rules

This is where the pilot moves from a navigation prototype to a real
rules layer.

- HCPF OM 24-044 on income trusts and eligibility-site
  responsibilities [@colorado2025incomeTrustMemo]
- Department program rules and regulations, especially the sections of
  10 CCR 2505-10 governing long-term care medical assistance,
  post-eligibility treatment of income, transfers of assets, and estate
  recovery [@colorado2026programrules]

Operationally, this is the minimum layer needed to turn a general
"Medicaid LTSS" description into coded rules.

## Minimum rule blocks for a credible pilot

The first pilot does not need to solve every edge case in long-term care
law. It does need to solve the blocks that determine whether outputs are
useful.

### Intake and pathway selection

The engine should first identify which broad program path is being
tested.

- Nursing facility or institutional long-term care
- HCBS / EBD waiver
- PACE
- Community First Choice-adjacent home-care path
- Working disabled buy-in path

This sounds trivial, but it is not. Different pathways imply different
income handling, service packages, assessment requirements, and
post-eligibility payment rules
[@colorado2026ltssprograms; @colorado2026ebd; @colorado2026pace; @colorado2026buyin].

### Financial eligibility

The pilot should encode:

- income-cap logic tied to SSI-based standards
- countable versus exempt resources
- separate handling of home equity and other major exempt assets
- pathway-specific resource tests where they differ

Colorado's EBD waiver page is enough to establish the first-pass member
rules for the waiver pathway: income below three times the current SSI
limit, countable resources below $2,000 for a single person and $3,000
for a couple, plus nursing-facility-comparable level of care
[@colorado2026ebd].

The working-disabled buy-in path should be modeled separately. It uses a
different income frame and premium schedule and is not just a variant of
the same institutional/HCBS test [@colorado2026buyin].

### Spousal impoverishment

Any serious state LTSS pilot has to encode spousal impoverishment rather
than treating "married" as a footnote.

At minimum, the pilot should include:

- community spouse resource allowance logic
- minimum and maximum monthly maintenance needs allowance logic
- housing-allowance components
- treatment of spousal income transfers in post-eligibility payment
  calculations

The parameter values change over time and should come from the annual
CMS standards bulletins, not hard-coded prose
[@medicaid2026spousal].

### Income trusts

Colorado is a good pilot precisely because income-trust logic is real,
visible, and codifiable.

HCPF OM 24-044 states that for long-term care medical assistance
eligibility, an individual under the 300 percent institutional special
income category must establish an income trust if gross income exceeds
300 percent of the current individual SSI benefit level. The memo also
states that the Department must review each income trust and describes
eligibility-site responsibilities, monthly funding expectations, and
closure procedures [@colorado2025incomeTrustMemo].

This implies that the pilot should answer more than "trust required:
yes/no." It should be able to return:

- whether a trust is required for the selected pathway
- whether the household appears within the plausible state rate window
- what monthly funding or distribution logic applies
- what maintenance and closure obligations exist

### Post-eligibility treatment of income and patient liability

This is one of the easiest areas to omit and one of the most important
for user value.

The pilot should include:

- personal-needs-allowance handling
- spousal income payments where applicable
- patient liability or member contribution logic
- pathway-specific treatment for institutional versus HCBS cases

Colorado's trust memo is explicit that, for long-term care institution
clients, the allowed deductions from monthly trust distribution include
personal-needs allowance, spousal income payments, and approved PETI
payments, with the remainder paid toward the cost of care up to the
medical assistance reimbursement rate [@colorado2025incomeTrustMemo].

### Transfers, look-back, and penalties

A credible pilot also has to recognize when the right answer is
"probably ineligible for now because of a transfer penalty."

The engine should therefore be able to flag:

- five-year look-back relevance
- transfers for less than fair market value
- probable penalty period questions requiring additional fact gathering

Federal Medicaid eligibility policy makes clear that LTSS applicants can
be denied coverage when assets were transferred for less than fair
market value during the five-year period preceding application
[@medicaid2025eligibility].

### Functional eligibility and local entry points

The pilot should not pretend financial rules alone determine access.

The state-facing operational layer should therefore identify:

- whether nursing-facility-comparable level of care is required
- whether the relevant entry point is a Case Management Agency
- whether pathway-specific assessment or care-management steps are
  triggered

Colorado's EBD waiver page and CMA directory are enough to show that
financial screening and case-management routing are inseparable in real
operations [@colorado2026ebd; @colorado2026cma].

### Estate recovery and trust recovery

Estate recovery is a required part of any honest user-facing output in
this space.

Federal law requires states to seek recovery for certain LTSS-related
services for enrollees age 55 or older, subject to survivor protections
and hardship provisions [@medicaid2025estate].

Colorado's Trust Policy and Recoveries page also makes clear that the
state separately reviews trust submissions and recovers certain trust or
annuity balances after termination or death
[@colorado2026trusts].

The pilot therefore should at least be able to say:

- whether estate recovery is potentially in scope
- whether trust balances may be subject to state recovery
- whether home preservation is being evaluated under incomplete
  information

## Parameter inventory for phase 1

The first coded version should include, at minimum, the following
parameter groups.

| Parameter group | Why it matters | Primary official sources |
| --- | --- | --- |
| SSI-based income cap and resource standards | Determines baseline LTSS financial screening | `medicaid2026spousal`, `colorado2026ebd` |
| EBD waiver member-facing thresholds | Fast initial waiver screening and scenario output | `colorado2026ebd` |
| Working-disabled buy-in thresholds and premium schedule | Alternative path for working disabled adults | `colorado2026buyin` |
| CSRA, MMMNA, housing allowance, home-equity limits | Spousal impoverishment and home treatment | `medicaid2026spousal`, `medicaid2025eligibility` |
| Income-trust rules and workflow | Cases above the income cap | `colorado2025incomeTrustMemo`, `colorado2026trusts` |
| Post-eligibility deductions and member contribution logic | Patient liability / affordability outputs | `colorado2025incomeTrustMemo`, `colorado2026programrules` |
| HCBS, PACE, and nursing-facility pathway definitions | Pathway routing and service context | `colorado2026ltssprograms`, `colorado2026pace`, `colorado2026ebd` |
| Case-management routing | Operational entry to real programs | `colorado2026cma` |
| Transfer and estate-recovery rules | Risk flags and downstream household implications | `medicaid2025eligibility`, `medicaid2025estate`, `colorado2026trusts` |

## Maintenance requirements

This pilot only stays credible if the update burden is acknowledged
upfront.

- Federal SSI and spousal impoverishment standards update annually
  [@medicaid2026spousal].
- State waiver pages, program pages, and buy-in schedules change over
  time [@colorado2026ebd; @colorado2026pace; @colorado2026buyin].
- Operational memos can change workflow or form requirements even when
  the basic legal structure is stable
  [@colorado2025incomeTrustMemo].
- Colorado's LTSS system is actively changing, including Community First
  Choice and related program transitions
  [@colorado2026ltssprograms].

That means the first pilot should be designed like a maintained rules
product, not like a one-off memo.

## What the pilot should return

If the pilot is worth funding, it should answer better questions than a
simple binary eligibility calculator.

At minimum, a useful household-facing or analyst-facing output should
return:

- likely pathway or pathways worth investigating
- eligible now, potentially eligible after spend-down or trust setup, or
  likely ineligible under current facts
- major missing facts needed to finish the determination
- likely spousal-protection implications
- likely patient-liability or member-contribution implications
- estate-recovery or trust-recovery flags

That output is ambitious for a state pilot, but it is still much more
tractable than a national dynamic LTSS microsimulation. That is exactly
why Colorado is a credible adjacent work package.
