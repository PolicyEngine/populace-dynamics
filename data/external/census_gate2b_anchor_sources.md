# Gate-2b external anchor sources (Census / CPS household composition)

Provenance sidecar for the three committed Census benchmark files that feed
the gate-2b external-anchor decomposition
(`scripts/gate2b_anchor_decomposition.py` -> `runs/gate2b_anchor_v1.json`).
These are the concept-decomposed ACS/CPS shape/ratio references required by
`runs/gate2b_floors_v1.json` `external_anchor.required_before_ratifying_flip`
(round-1 referee finding F). **Reported, never gated** -- the anchor names
the PSID-family-unit-vs-Census-household concept delta; it does not move any
floor value.

All three were fetched on **2026-07-10** by direct HTTPS GET of the Census
Bureau's published, keyless time-series files on `www2.census.gov` (or the
America Counts story page). The Census Data API (`api.census.gov`) now
requires an API key and `data.census.gov` is a JavaScript single-page app, so
neither serves table values to a keyless programmatic client; the static
`.xls` time-series files and the published story figures ARE keyless and
authoritative. `.xls` files were parsed with `xlrd`.

## 1. `census_household_size_2023.json` -- household size distribution

- **Table id:** HH-4 (Households by Size: 1960 to Present)
- **Landing page:** https://www.census.gov/data/tables/time-series/demo/families/households.html
- **File URL:** https://www2.census.gov/programs-surveys/demo/tables/families/time-series/households/hh4.xls
- **Survey / year:** CPS ASEC, 2023 row
- **Fetch method:** HTTPS GET + xlrd parse of the 2023 row
- **Fetched (UTC):** 2026-07-10T09:53:47+00:00
- **Source file sha256:** `9ac805cd902da55988327a37e73e3ff5d1abd9d7aee014002e26b3e05384d79d`
- **Extracted:** all households 131,434k; 1p 38,097k; 2p 45,959k; 3p 19,771k;
  4p 16,038k; 5p 7,192k; 6p 2,721k; 7+p 1,656k; avg 2.51. Household-level and
  person-level shares are derived in the JSON (`derived`).

## 2. `census_living_arrangements_2023.json` -- coresidence by age x sex

- **Table id:** AD-3 (Living Arrangements of Adults by Age Group), five
  age-band files (18-24, 25-34, 35-64, 65-74, 75+)
- **Landing page:** https://www.census.gov/data/tables/time-series/demo/families/adults.html
- **File URLs:** `.../adults/ad3-18-24.xls`, `ad3-25-34.xls`,
  `ad3-35-64.xls`, `ad3-65-74.xls`, `ad3-75andover.xls` under
  https://www2.census.gov/programs-surveys/demo/tables/families/time-series/
- **Survey / year:** CPS ASEC, 2023 row of each file
- **Fetch method:** HTTPS GET + xlrd parse of the 2023 row, per age band
- **Fetched (UTC):** 2026-07-10T09:53:47+00:00
- **Source file sha256:** see `provenance.file_sha256` in the JSON (one per
  age band).
- **Extracted:** mutually-exclusive percent shares (living_alone,
  living_with_spouse, child_of_householder, living_with_partner,
  living_with_other_relatives, living_with_nonrelatives), by sex.

## 3. `census_multigenerational_2020.json` -- multigenerational households

- **Table id:** B11017 (Multigenerational Households), concept
- **Story URL:** https://www.census.gov/library/stories/2023/06/several-generations-under-one-roof.html
- **ACS table landing:** https://data.census.gov/table/ACSDT1Y2023.B11017
- **Survey / year:** 2020 Census figures as published by the Census Bureau
- **Fetch method:** WebFetch of the America Counts story page (B11017 exact
  counts need a keyed API; the Census Bureau's own published rounded figures
  are used)
- **Fetched (UTC):** 2026-07-10T09:53:47+00:00
- **Extracted:** 6.0M multigenerational households (2020), 4.7% of all
  households, 7.2% of family households; 5.1M (2010); 8.4% of children under
  18 in a grandparent's home (2020). Definition: three or more generations
  under one roof (concept-matched to the gate-2b B11017 >= 3-distinct-
  generations rule after fixes B/C).

## Concept bridge (why these are ratios, not level targets)

The PSID gate measures **person-weighted family-unit** composition; the
Census tables measure **households**. The decomposition names, per family:
household-vs-person weighting (hh_size, multigen), family-unit-vs-household
fragmentation (all), the spouse-OR-partner inclusion of PSID codes 20/22
(coresident_spouse vs Census living_with_spouse), the child-of-householder
vs any-coresident-parent gap (coresident_parent), the age-floor delta at
15-24 (PSID includes 15-17) and the band-aggregation delta at 35-64 (one
Census band spans three PSID bands), and the grandparent-side vs
grandchild-side denominator (coresident_grandchild). Transition families
(parental_home_exit, spousal_loss, multigen_entry/exit) are flows with no
cross-sectional Census level and carry no ratio (flip-note 4).
