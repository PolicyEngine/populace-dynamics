# SSA DI Annual Statistical Report 2023 — extraction provenance

- Source: https://www.ssa.gov/policy/docs/statcomps/di_asr/2023/ (sect01c.html, sect03a.html, sect03f.html)
- Method: in-browser same-origin fetch + DOM table extraction via Control Chrome MCP (ssa.gov returns 403 to all programmatic fetch; browser session required — same recipe as ../ssa_supplement_2023_6b.txt, 2026-07-07)
- Extracted: 2026-07-10, verbatim cell text, no edits
- Files: tables.json (Table 19 prevalence by sex×age 1960–2023; Table 35 awards series 1960–2023; Table 36 awards by basis×age×sex 2023; Table 49 terminations number+rate 1960–2023; Table 50 terminations by reason 2023)
- Note: report corrigendum on Table 50 ("original version contained errors... Workers, Wi[dowers]") — this extraction is the CORRECTED live version
- sect01a.xlsx also archived (browser download; Tables 1–2)
- Purpose: M4 disability gate anchors (populace-dynamics #113 M4, #123 wanted_ssa_tables items 1/2/4 partial)
- Still wanted: Trustees V.C5/V.C6 (incidence/termination assumptions), Supplement 4.C2 (insured denominator), age-sex-adjusted incidence rate series
- Key anchor values: 2023 worker termination rate 107/1,000; FRA conversions 455,267 of 788,327 worker terminations (57.8%); deaths 230,502 (29.2%); medical-cessation family ~99,721 incl. SGA 63,699
