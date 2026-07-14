# Employer-firm target extract provenance

Provenance sidecar for the four committed aggregate extracts that
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

- **Source URL:** https://lehd.ces.census.gov/data/qwi/latest_release/us/qwi_us_sa_fs_gn_ns_op_u.csv.gz
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

- **Source URL:** https://lehd.ces.census.gov/data/j2j/latest_release/us/j2j/j2j_us_d_fs_gn_ns_oslp_u.csv.gz
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
  size-ladder flows (E11's J2JOD reference) are a later, separate
  extract.
