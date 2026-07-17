# Employer-firm target extract provenance

Provenance sidecar for the committed aggregate extracts that
feed the employer-firm extension's calibration targets and gate
references (E1/E2/E7/E11/E12; issue #192, ADR 0003, workstream B).
All are small, tidy derivatives of published public aggregate files —
never raw microdata. Every file was fetched on **2026-07-14** by
keyless HTTPS GET via `scripts/fetch_employer_firm_targets.py`, which
pins the source URLs and the sha256 of each raw download and rebuilds
the extracts deterministically. Loaders (label-verified, with pinned
totals) live at `src/populace_dynamics/firms/targets.py`.

Note on the Census data API: `api.census.gov` now requires an API key
(probed 2026-07-14: keyless queries 302-redirect to
`missing_key.html`), so the BDS/QWI extracts come from the published
static CSV files on `www2.census.gov` / `lehd.ces.census.gov`, which
are keyless and authoritative.

## 1. `susb_us_sector_size_2022.csv` — SUSB enterprise size x sector

- **Source URL:** https://www2.census.gov/programs-surveys/susb/tables/2022/us_state_naics_detailedsizes_2022.txt
- **Landing page:** https://www.census.gov/data/tables/2022/econ/susb/2022-susb-annual.html
- **Vintage:** SUSB 2022 (latest annual release; March-12 employment)
- **Raw sha256:** `f07eef547763a7a21cb453bf6f4119f768f5939dc499a65850a532ea74489ccf`
- **Transformation:** rows filtered to STATE=00 (United States) and
  sector-level NAICS (2-digit codes, published ranges such as 31-33,
  and the `--` all-sectors total); columns renamed to tidy names; the
  `NN:` prefix stripped from size-class labels. No values altered.
- **Extracted check values:** US total (ENTRSIZE 01): 6,395,635
  firms; 8,298,562 establishments; 135,748,407 employment (pinned in
  the loader).
- **Universe caveat:** SUSB excludes government establishments,
  NAICS 92, crop/animal production, and non-employers.

## 2. `bds_us_firm_size_1978_2022.csv` — BDS firm-size time series

- **Source URL:** https://www2.census.gov/programs-surveys/bds/tables/time-series/2022/bds2022_fz.csv
- **Landing page:** https://www.census.gov/data/datasets/time-series/econ/bds/bds-datasets.html
- **Vintage:** BDS 2022 release (1978-2022, economy-wide by firm
  size)
- **Raw sha256:** `61e0e624b00ef3876a503fdaf8c1550d9aaa14070c8962dbd76ec500560ae86b`
- **Transformation:** column subset only (establishment counts,
  entry/exit, job creation/destruction levels and DHS rates,
  reallocation, firm deaths); all 450 rows (45 years x 10 firm-size
  categories) kept verbatim. Rates are per 100 employees, as
  published. The BDS `fsize` categories (1-4 ... 10000+) do not all
  nest in the canonical bands; see
  `banding.bds_fsize_to_canonical` (the 20-99 category straddles
  the canonical 50 edge).

## 3. `qwi_us_firmsize_sector_2015on.csv` — QWI flows by firm size x sector

- **Source URL:** https://lehd.ces.census.gov/data/qwi/R2026Q1/us/qwi_us_sa_fs_gn_ns_op_u.csv.gz
  (release-stamped, not `latest_release`, so the sha256 pin stays
  valid after LEHD rotates to the next quarterly release)
- **Release:** R2026Q1, V4.14.0 (`version_qwi.txt`: QWI_FS US
  1993:1-2024:4, `qwipu_us_20260311_1503`); not seasonally adjusted,
  private ownership (`op`)
- **Raw sha256:** `76f147615d796eb0ba93401107762e1c67667a25c468b2ee4ca07e07d9eabd85`
- **Transformation:** pure row/column filter, no re-aggregation:
  kept the all-sex (`sex=0`), all-age (`agegrp=A00`) margin, detail
  firm-size codes 1-5, NAICS-sector industry level plus the `00`
  all-industry margin, 2015Q1 onward; kept 14 stock/flow/earnings
  measures with their LEHD status flags. Firm-size labels joined
  from https://lehd.ces.census.gov/data/schema/latest/label_firmsize.csv
  (sha256 `29dfd8fed594be600c6c554b4cb27bd590c45da549c30e32824cea4248dffe1f`).
- **Unit caveats (pre-registered, ADR 0003):** QWI counts **jobs**,
  not persons — each person-employer pair in a quarter is a separate
  job, so person-spell calibration needs the job-to-person
  adjustment (~ multiple-jobholding rate, ~5%). `EarnS` is a
  **mean** monthly earnings measure; QWI never publishes medians.
  2024Q4 separations carry status flag -1 (not computable until the
  next quarter is released). Firm size here is administrative
  national March employment; public-sector firms are out of scope.

## 4. `j2j_us_firmsize_sector_2015on.csv` — J2J flows by firm size x sector

- **Source URL:** https://lehd.ces.census.gov/data/j2j/R2026Q1/us/j2j/j2j_us_d_fs_gn_ns_oslp_u.csv.gz
  (release-stamped, not `latest_release`; see the QWI note above)
- **Release:** R2026Q1, V4.14.0 (`version_j2j.txt`: J2J US
  2000:2-2025:1, `j2jpu_us_20260312_1118`); no demographic detail
  (`d`), not seasonally adjusted, state/local/private ownership
  (`oslp`)
- **Raw sha256:** `abdd573d414d66f864828952501cee0a6ef6c8db88cb789da38af1a3c9d55c6f`
- **Transformation:** pure row/column filter, no re-aggregation:
  detail firm-size codes 1-5, sector industry level plus the `00`
  margin, 2015Q1 onward; 12 flow measures with status flags.
  **NAICS 92 (Public Administration) dropped**: firm size is
  undefined for public-sector employers (LEHD codes it "N") and the
  residual 92 x firm-size cells are single-digit noise with zero
  denominators; the `00` margin still includes 92's (negligible)
  firm-size-coded contribution as published. 2025Q1 firm-size cells
  are suppressed in this release (status flag 5) and load as NaN.
- **Unit caveat:** job counts, as for QWI. Origin-x-destination
  size-ladder flows are in `j2jod_us_firmsize_od_2015on.csv` (entry
  6 below).

## 5. `j2j_us_sexage_2015on.csv` — J2J flows by sex x age group

Fetched on **2026-07-17** (extracts 5 and 6 are the second wave,
flagged in PR #223's method findings; same fetch script).

- **Source URL:** https://lehd.ces.census.gov/data/j2j/R2026Q1/us/j2j/j2j_us_sa_f_gn_ns_oslp_u.csv.gz
  (release-stamped, not `latest_release`; see the QWI note above)
- **Release:** R2026Q1, V4.14.0 (`version_j2j.txt`: J2J US
  2000:2-2025:1, `j2jpu_us_20260312_1118`); sex x age worker detail
  (`sa`), no firm characteristics (`f`), not seasonally adjusted,
  state/local/private ownership (`oslp`)
- **Raw sha256:** `0e043fc8796bd3e11231ff6d174fdfebed926c9d40da4f069a3ad31eed55aba0`
- **Transformation:** pure row/column filter, no re-aggregation:
  the all-industry margin (`industry == "00"`) only, the full
  sex (0/1/2) x age (A00-A08) grid margins included, 2015Q1 onward;
  12 flow measures with status flags. The full NAICS-sector detail
  would breach the 1 MB extract cap, so it is not committed; the
  raw sector file stays available at the pinned URL. Sex and
  age-group labels joined from the LEHD schema
  (https://lehd.ces.census.gov/data/schema/latest/label_agegrp.csv,
  sha256 `eb478c6eda6c12a57609afaf89bbb42dd4d9fb2ee883f6dd0399fb717b27889b`).
- **Naming caveat:** the age x sex tabulation is LEHD's `sa`
  crossing; LEHD's `se` crossing is sex x *education* (the gate-E2
  registration's "se" shorthand refers to sex x age, i.e. `sa`).
- **Unit caveats:** job counts, not persons (see the QWI entry);
  ownership is `oslp` here versus `op` for the QWI extract, so
  levels are not directly comparable across the two. 2025Q1
  separation-side measures carry status flag -1 (not computable
  until the next quarter is released) and load as NaN.

## 6. `j2jod_us_firmsize_od_2015on.csv` — J2J flows by origin x destination firm size

- **Source:** LED Extraction Tool query API,
  https://ledextract.ces.census.gov (POST the pinned JSON request in
  `scripts/fetch_employer_firm_targets.py` to `/j2j/download`, then
  GET `/j2j/download.csv?<encoded query>` from the 303 redirect).
  The LEHD flat J2JOD files (`j2jod_us_d_fs_*`) publish only the
  one-sided firm-size margins — the full origin x destination cross
  is not in any flat file — and the Census data API
  (`api.census.gov`) still requires a key (probed 2026-07-17), so
  the LED Extraction Tool is the pinned keyless source.
- **Release:** R2026Q1, V4.14.0 (the tool's `/j2j/schema` reports
  V4.14.0; matches `version_j2jod.txt` for R2026Q1: J2JOD US
  2000:2-2025:1, `j2jodpu_us_20260312_1118`); national, all
  industries, not
  seasonally adjusted, ownership A00 (state/local government plus
  private — the tool's reported `ownercode`; equivalent to `oslp`)
- **Fetched-CSV sha256:** `adbd16e2c23ee3a87a22c5f6520eca37b09f8036131d4147c081c89d6a5a867f`
  (the query was repeated at fetch time and is byte-stable). The
  tool serves the *current* release only, so this pin breaks loudly
  when LEHD rotates to R2026Q2; re-pin deliberately and update this
  entry.
- **Transformation:** column subset and sort only, no
  re-aggregation: the full 6 x 6 firm-size grid (codes 0-5 on both
  origin and destination sides), 2015Q1-2025Q1 (ordinal quarters
  8060-8100), six flow measures (EE, AQHire, J2J = EE + AQHire, and
  their stable variants) with status flags.
- **Detail-window caveat (important for E11):** the full 5 x 5
  origin x destination detail is released only for
  **2015Q1-2016Q1**; from 2016Q2 on every national detail cell
  carries status flag 11 ("aggregate of cells not released because
  component cells do not meet publication standards") and loads as
  NaN. The tool aggregates the state-level OD tabulations to the
  national level, and a state coverage gap from 2016Q2 blocks the
  aggregate; the same suppression governs the J2J Explorer, so no
  keyless public source carries the later cross. The one-sided
  margins (code 0 on either axis) remain published through 2025Q1.
  E11's origin x destination shape reference is therefore the
  2015Q1-2016Q1 window; later quarters constrain the margins only.
- **Margin caveat:** the code-0 margins are the tool's aggregates of
  the firm-size-coded tabulation (status flag 10/12), so they sit
  slightly below the flat-file `d_fs` margins, which include
  public-sector flows (firm size "N"): e.g. 2015Q1 all-size EE is
  3,985,308 here versus 3,988,566 in `j2jod_us_d_fs_gn_n_oslp_u`.
  Detail cells that fail publication standards (status flag 11)
  load as NaN — common in the small-x-large corners.
- **Unit caveat:** job counts, as for QWI/J2J; firm size is
  administrative national March employment on both sides.
