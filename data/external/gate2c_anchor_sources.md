# Gate-2c external anchor sources (published assortative-mating / spousal-earnings-correlation benchmarks)

Provenance sidecar for the two committed published-benchmark files that feed
the gate-2c external-anchor decomposition
(`scripts/gate2c_anchor_decomposition.py` -> `runs/gate2c_anchor_v1.json`).
These are the published couple-earnings-correlation / assortative-mating
references `runs/gate2c_floors_v1.json`
`external_anchor.required_before_ratifying_flip` demands before the ratifying
flip, drawn from the artifact's own `external_anchor.candidate_sources` list
(the CPS ASEC married-couple earnings correlation, and the academic
assortative-mating series it names). **Reported, never gated** -- the anchor
bridges the gate-2c per-year earnings-rank sorting to the published series; it
does not move any floor value and calibrates nothing.

The gate-2c couple axis is a within-couple **Spearman rank correlation of the
per-year NAWI-indexed positive-year earnings axis** (the fix-C axis), measured
over the SELECTED both-positive-earner PSID couple universe (age-60 indexing
year in the NAWI window, >= 5 observed positive-earnings years). Its value is
**0.4928** (frozen floor `assortative_correlation_report_only.decomposition`).
The anchor's job is to bridge THAT number (not the round-1 career-sum proxy
rho 0.1194, an observation-mechanics artifact) to the published sorting
series, naming the concept deltas that separate a rank-capacity correlation on
a selected, truncated-support couple set from the published annual-earnings
and educational series.

## 1. `schwartz_2010_spousal_earnings_correlation.json` -- CPS ASEC spouses' earnings correlation

- **Source:** Schwartz, Christine R. 2010. "Earnings Inequality and the
  Changing Association between Spouses' Earnings." *American Journal of
  Sociology* 115(5): 1524-1557.
- **DOI:** 10.1086/651373
- **Open-access full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2908420/
- **Survey / years:** March CPS (CPS ASEC), 1968-2006 (earnings years
  1967-2005)
- **Values (Figure 2), Pearson correlation of spouses' annual earnings:**
  all married couples -.08 (1967-70) -> .12 (2003-5); **dual-earner couples
  .08 (1967-70) -> .23 (2003-5)**.
- **Fetch method:** WebFetch of the open-access PMC full text on 2026-07-10;
  the Figure-2 values and the verbatim sentence were transcribed. The PMC PDF
  endpoint returns an HTML interstitial to keyless programmatic clients (the
  same wall the gate-2b Census-API sources hit), so there is **no fixed-byte
  source-file sha256**; the pinned URL + DOI + verbatim quote are the record.
- This is the direct concept match for the gate-2c earnings axis (candidate
  source: "CPS ASEC married-couple earnings-rank contingency tables"). The
  concept-matched row is DUAL-EARNER (both positive), mirroring the gate-2c
  both-positive selection.

## 2. `greenwood_2014_assortative_mating.json` -- educational assortative-mating series (Kendall's tau, relative sum of diagonals)

- **Source:** Greenwood, Jeremy, Nezih Guner, Georgi Kocharkov, and Cezar
  Santos. 2014. "Marry Your Like: Assortative Mating and Income Inequality."
  *American Economic Review: Papers & Proceedings* 104(5): 348-353 (NBER
  Working Paper 19829).
- **File URL:** https://www.nber.org/system/files/working_papers/w19829/w19829.pdf
- **Survey / years:** U.S. Census / ACS, married couples, 1960-2005.
- **Values (Figure 1):** Kendall's tau (husband-wife education) rose over
  1960-2005 on a ~0.33-0.40 axis; delta_t, the relative sum of the
  education-contingency diagonals, on a ~1.6-2.0 axis. Both are **read from a
  figure**, recorded as approximate ranges, not transcribed from a value
  table. Married-couple income Gini 0.34 (1960) -> 0.43 (2005).
- **Fetch method:** HTTPS GET of the NBER working-paper PDF on 2026-07-10;
  `pdftotext` extraction.
- **Source file sha256:**
  `e134e53a9a17bcb0571bb58245eb61f8b707a5734694f65def6fe037873d37cb`
  (246,004 bytes).
- This is the academic assortative-mating series the candidate-source list
  names (adjacent to Schwartz & Mare 2005 / Eika-Mogstad-Zafar 2019). It
  anchors EDUCATION sorting (a proxy for earnings capacity), so the bridge
  carries an education-vs-earnings delta; the relative-sum-of-diagonals
  statistic is computed the same way on the gate-2c earnings contingency, for
  direction and order of magnitude only.

## What the decomposition does with these

`scripts/gate2c_anchor_decomposition.py` reads the FROZEN floor
`runs/gate2c_floors_v1.json` (the per-year rank correlation and the
own-tercile x spouse-tercile contingency -- never rewritten) and the two files
above, and reports, per facet, our value next to the published value with the
concept delta NAMED:

1. **within-couple rank correlation** -- our per-year Spearman 0.4928 vs
   Schwartz dual-earner Pearson .08->.23 (and all-couples -.08->.12). Deltas:
   rank vs Pearson; per-year earnings-capacity vs annual earnings; the
   selected both-positive universe; the pooled/older marriage decades. The
   career-sum proxy 0.1194 sits near Schwartz's all-couples .12 but for the
   WRONG reason (observation mechanics), which is exactly why the gated axis
   is the per-year measure.
2. **contingency diagonal concentration** -- our relative-sum-of-diagonals
   delta (computed from the frozen 3x3 contingency) vs Greenwood's education
   delta_t ~1.6-2.0. Deltas: 3x3 terciles vs 5x5 education levels; earnings
   capacity vs education.

No calibration: the report names concept deltas and validates DIRECTION
(positive, moderate sorting); it never moves a floor value or a tolerance.
