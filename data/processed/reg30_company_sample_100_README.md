# Reg 30 Event Extraction — 100-Company Sample

## Source and method

Sampled from `data/processed/industry_map.xlsx` (the 247-company matched universe used
throughout this project — CG scores, `numerical_indices.xlsx`, `06_index_calculation.ipynb`),
**not** from `data/raw/top_500_companies.xlsx`. That keeps Reg 30 events mergeable into the
existing firm-quarter panel without a separate crosswalk.

Selection is proportional stratified sampling by `Industry`:

1. Each of the 247 companies' `Industry` gives a target share of 100 = `industry_count * 100/247`.
2. Fractional shares are rounded down, then the largest remainders get the leftover slots
   (largest-remainder method) so the total lands on exactly 100.
3. Companies within each industry are chosen by random shuffle, seed=42 (reproducible).
4. 5 companies had no `Industry` value in `industry_map.xlsx` (GE T&D India, ICICI
   Securities, Piramal Enterprises, Suven Pharmaceuticals, Zomato/Eternal — all had
   corporate-action name/symbol changes that likely broke the yfinance lookup in
   `01_industries_map.ipynb`) — bucketed as `Unclassified` rather than dropped.

Result: **100 companies across 46 of the 247-universe's 70 industries.** Smaller industries
(1-2 companies) mostly round to a zero share at this sample size and aren't represented —
this is a size-representative sample of the universe, not a full industry census.

Full list with BSE Code / Industry: `data/processed/reg30_company_sample_100.csv`

## Pipeline state as of this writing

- **Announcement index**: complete for all 100 companies — `data/processed/reg30_index.csv`,
  29,549 Reg 30 rows total (`regulation\s*30\b` match against NEWSSUB/SUBCATNAME, case-
  insensitive — broader than BSE's own "Announcement under Regulation 30 (LODR)" prefix,
  which misses ~25% of genuine Reg 30 filings phrased differently).
- **Document cache**: 22 companies fully downloaded (6,513 PDFs/HTML, moved to
  `data/raw/archive/reg30_announcements/` to keep the working tree light); the remaining
  ~23,000 rows are indexed with `content_type='pdf_pending'` — metadata only, document
  fetched lazily only for rows actually selected for extraction.
- **Event extraction**: 50-document review batch complete (`reg30_sample_review.csv`,
  `reg30_events.csv`) using qwen2.5:14b — 46/50 succeeded, 79 events found. Full-corpus
  extraction uses qwen2.5:7b-instruct instead (see below).

## Why 7b, not 14b, for the full run

This machine has 16GB RAM. qwen2.5:14b (9GB) doesn't comfortably coexist with everything
else running — measured swap usage at 7.4GB/8GB during the 14b review batch, which is
disk-thrashing, not compute-bound slowness (~2 tokens/sec, versus the expected 15-25+
tok/s an M2 should give a 4-bit 14B model with proper Metal residency). qwen2.5:7b-instruct
(4.7GB) doesn't hit this ceiling. A same-document, same-prompt comparison: 14b took ~120s,
7b took 20.5s, and pulled the same two director-change events with correct fields. At 14b's
pace, the full relevant-row corpus (below) would take ~64 hours; at 7b's, ~11.

## Scope of the full extraction run

Of 29,549 total Reg 30 rows, only rows whose `subcategory`/`newssub` can plausibly contain
one of the three target event types are extracted — running an LLM over press releases,
investor-meet notices, and board-meeting intimations wastes compute for zero possible
signal. Filter: `subcategory` or `newssub` matches
`director|auditor|credit rating|kmp|resignation|appointment|cessation|chief executive|chief
financial|company secretary` (case-insensitive).

**1,909 of 29,549 rows (6.5%) match.** Breakdown of the largest categories:

| Subcategory | Rows |
|---|---|
| Change in Directorate | 785 |
| Credit Rating | 561 |
| Cessation | 133 |
| Appointment of Statutory Auditor/s | 115 |
| General | 107 |
| Resignation of Director | 90 |
| Resignation of Company Secretary / Compliance Officer | 47 |
| Resignation of Chief Financial Officer (CFO) | 28 |
| Resignation of Managing Director | 11 |
| Resignation of Statutory Auditors | 9 |
| Resignation of Chairman | 6 |
| Resignation of Chief Executive Officer (CEO) | 5 |
