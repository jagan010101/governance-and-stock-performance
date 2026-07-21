# Corporate Governance & Stock Performance

A study of whether corporate governance quality predicts forward stock performance for
247 large-cap Indian listed companies (FY23-FY24), built on a governance index derived
from BSE/NSE regulatory filings (XBRL) plus an LLM-scored narrative-quality layer, tested
against forward returns, risk-adjusted alpha, volatility, and ROE.

**Headline result: a clean null.** After correcting for the fact that the governance
index has six-then-five separate sub-dimensions tested against seven outcomes, none of
the sub-index → outcome relationships survive multiple-testing correction in the primary
specification. Two narrow exceptions exist in robustness cuts (not the headline spec) —
see [Findings](#findings) below. The full result set, including every robustness
variant, is in `regression_results_full.txt`.

## Contents

- [Data sources](#data-sources)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
- [Findings](#findings)
- [Limitations](#limitations)
- [Setup](#setup)
- [Project structure](#project-structure)

## Data sources

| Source | What | Where |
|---|---|---|
| BSE governance filings (XBRL) | Corporate Governance Report quarterly filings — board, audit, ownership, remuneration, disclosure metrics | `data/raw/governance_reports.zip` |
| Annual Reports (PDF) | 739 PDFs, 247 firms × up to 3 FYs — board structure, audit fees, RPT, pledge, contingent liabilities | `data/raw/annual_reports/` |
| BSE Reg 30 (SEBI LODR) announcements | Director/auditor/credit-rating-change disclosures, 100-firm stratified sample | `data/raw/archive/reg30_announcements/` |
| Prowess (CMIE) | Company-financials cross-reference for BSE↔Prowess name matching | `data/raw/prowess_raw_data.xlsx` |
| yfinance | Daily prices, volume, financial statements, shares outstanding | live-fetched, not stored raw |
| Fama-French 5 + Momentum factors | Monthly India factor returns | `data/processed/ff5mom_factors_monthly.csv` (static — no in-repo downloader; provenance undocumented, see Limitations) |

Universe: `data/raw/top_500_companies.xlsx` (500 companies, ranked by market cap
Jul-Dec 2025) → `data/raw/matched_companies_seed.xlsx` (247, manually curated — the exact
selection criteria for 500→247 are not recoverable from this repo) → matched to Prowess →
`data/processed/industry_map.xlsx` (final 247-firm regression universe).

## Pipeline

Numbered notebooks run in order; each depends only on the outputs of earlier ones (all
saved to `data/processed/`).

| Notebook | Purpose |
|---|---|
| `00_cleaning_prowess.ipynb` | Prowess data cleaning, BSE↔Prowess name matching |
| `01_industries_map.ipynb` | Industry/sector classification for the 247-firm universe |
| `02_download_scripts.ipynb` | Orchestrates the scrapers in `download_scripts/` |
| `03_targets.ipynb` | Forward returns, alpha, volatility, ROE construction |
| `04_controls.ipynb` | Beta, momentum, market cap, D/E ratio control variables |
| `05_alphas.ipynb` | CAPM/FF5 alpha estimation |
| `06_index_calculation.ipynb` | Builds the CG sub-indices (AINDEX, BINDEX, CINDEX, DINDEX, OINDEX, TRINDEX) from raw XBRL/NLP scores |
| `07_index_validation.ipynb` | Index construction sanity checks |
| `08_ff5_regression.ipynb` | FF5+Momentum factor regression exploration |
| `09_regression.ipynb` | **Main panel regression** — forward-looking `CG_t → outcome_{t+1}` design. Table A (original) + Table B (+AR) + Table C (+AR+Reg30) + Section 10 (corrected primary spec) + Section 11 (AR/Reg30-augmented robustness) + Section 12 (expanded-controls robustness) |
| `09_regression_v1_contemporaneous.ipynb` | Superseded contemporaneous-timing (`CG_t → outcome_t`) version, preserved for comparison |
| `10_event_study.ipynb` | CAR around CG filing dates, cross-sectional regression on CG scores |
| `11_portfolios.ipynb` | Long-short quintile portfolio sorts on CG scores |
| `12_summary.ipynb` | Pipeline's own original summary (Table A results only) |
| `13_findings_summary.ipynb` | **Diagnostic narrative** — all 14 findings from the diagnose-and-fix effort, one at a time, each with a live computation against real saved data |
| `14_index_extension.ipynb` | Builds `cg_scores_augmented_fy.csv` — extends AINDEX/BINDEX/CINDEX with 6 AR/Reg30 metrics, consumed by `09_regression.ipynb` Section 11 |

Supporting scripts:

- `download_scripts/` — scrapers and extractors: `bse_cg_downloader.py`,
  `nse_ar_downloader.py`, `bse_ar_fallback.py`, `reg30_scraper.py`,
  `reg30_fetch_relevant.py`, `reg30_repair_attachments.py`,
  `build_filing_dates_db.py`, `finalize_dataset.py`, and the three LLM-based
  extractors — `cg_nlp_scorer.py`, `ar_extractor.py`, `reg30_extractor.py` (all three
  require a local Ollama server, see [Setup](#setup)).
- `reg30_firm_fy_agg.py` / `reg30_firm_quarter_agg.py` — collapse Reg 30 event-level data
  to firm-FY/firm-quarter aggregates.
- `xbrl_narrative_inventory.py` — standalone audit of XBRL narrative-text tag coverage.
- `diagnostics.py` — the Phase-1 diagnostic script behind `13_findings_summary.ipynb`'s
  Findings 1-5.

## Methodology

- **Timing**: forward-looking, `CG_t → outcome_{t+1}` — every outcome is measured strictly
  after `CG_t` becomes public. Panel is `T=2` (FY23, FY24), determined empirically (not
  assumed) by checking real data coverage in both directions — see Finding 5.
- **CG score construction**: quarterly `Avg_Score` per sub-index, averaged to FY level,
  Van der Waerden normal-score transformed within each FY cross-section before use as a
  regressor.
- **Regression**: firm-clustered standard errors (HC3 fallback), industry + year fixed
  effects, outcomes winsorized 1%-99%.
- **Multiple-testing correction**: Romano-Wolf stepdown (cluster/firm bootstrap, B=2000),
  applied across the *full* hypothesis family actually tested in each exhibit — never
  narrowed after seeing results. This is the actual inferential standard throughout; raw
  p-values are reported alongside for readability but are not the basis for any claim of
  significance.
- **No synthetic data, no rescue imputation**: where a variable is majority-missing, it
  is reported as non-estimable rather than imputed into existence (see DINDEX below).

## Findings

Full narrative with live-computed numbers: `13_findings_summary.ipynb`. Full result
tables in classic regression-table format: `regression_results_full.txt`.

**Root-caused, not assumed**: two real design defects were found and fixed before
reaching the headline result —

1. **DINDEX (Disclosure) is dropped from the primary spec.** Its D2-D5 components fail
   LLM scoring on 61-67% of observations, and those failures are hard-coded to a score of
   0 (a governance *failure*) rather than treated as missing — collapsing DINDEX to 66
   unique values across 4,197 observations (1.57% distinct). Re-imputing was explicitly
   rejected as fabricating the very data in question; the index is reported as
   non-estimable instead. (DINDEX was also tested included, on request, purely for the
   record — see `panel_romano_wolf_dindex_included.csv` — it contributes no signal that
   changes the verdict.)
2. **26 singleton-industry firms are dropped from the primary spec.** They mechanically
   absorb 100% of their own cross-sectional variation once industry fixed effects are
   applied, contributing nothing to identification. A Sector-FE version (broader FE, no
   firms dropped) is reported as a robustness check, not the headline.

**Primary spec result**: 5 sub-indices × 7 outcomes = 35 tests, **0/35 survive** RW
correction (smallest RW p = 0.133).

**Two narrow exceptions, both explicitly secondary**, robustness/exploratory cuts:
- Sector-FE robustness: TRINDEX (Transparency/Remuneration) → lower volatility survives
  (2/35, RW p = 0.004 and 0.013).
- Event study: CINDEX (narrative quality) → ±1-day CAR, marginal (RW p = 0.080).

Every other cut — Table B (+AR variables), Table C (+AR+Reg30), the AR/Reg30-augmented
indices, expanded controls (liquidity/ROA/asset growth), portfolio sorts, and even the
superseded contemporaneous-timing design (retroactively RW-corrected, 66 hypotheses) —
comes back to the same null. This convergence across independently-built variable sets
and timing conventions is itself evidence the null is real, not an artifact of any one
design choice.

## Limitations

Stated plainly, not smoothed over:

1. **Survivorship / look-ahead in the universe definition.** The 247-firm universe
   requires being large-cap *as of Jul-Dec 2025* — after the FY23-FY24 study window ends
   — and Prowess-matched. If poor governance predicts financial distress (the standard
   finding in the literature), this conditioning removes the worst tail of both
   governance scores and outcomes jointly, which *attenuates* any true relationship
   toward zero. A null here is at least as consistent with survivorship-driven
   attenuation as with a genuine population-level absence of the effect. The exact
   500→247 selection criteria could not be recovered from anything in this repo.
2. **DINDEX exclusion** means the primary spec speaks to 5 of the original 6 governance
   dimensions.
3. **T=2, structurally fixed.** Verified (not assumed) unextendable in either direction —
   backward is blocked by the SEBI disclosure mandate's phase-in (CG-score coverage was
   0.8% of firms in FY20, rising to 100% only by FY23) and by the factor-return series
   only starting mid-2022; forward is blocked by the calendar itself (a full forward
   return window for FY25 hasn't elapsed) and by `ff5mom_factors_monthly.csv` having no
   recoverable source/construction methodology in this repo.
4. **No human-coded validation of the LLM-scored layer.** Cohen's κ against hand-coded
   labels was never computed for the NLP-scored sub-indices, despite being flagged as a
   prerequisite for trusting that layer.

## Setup

```bash
pip install -r requirements.txt
```

The three LLM-based extractors (`download_scripts/cg_nlp_scorer.py`,
`download_scripts/ar_extractor.py`, `download_scripts/reg30_extractor.py`) additionally
require a local [Ollama](https://ollama.com) server with `qwen2.5:7b-instruct` and
`qwen2.5:14b` pulled. They are not needed to re-run the regression notebooks — their
outputs are already saved under `data/processed/`.

Raw data (`data/raw/annual_reports/`, `data/raw/governance_reports.zip`,
`data/raw/archive/`) totals ~9.3GB and is not included in version control (see
`.gitignore`) — obtain separately or re-run the download scripts.

## Project structure

```
├── 00-14_*.ipynb              pipeline notebooks, run in numeric order
├── diagnostics.py             Phase-1 diagnostic script
├── reg30_firm_fy_agg.py       Reg 30 event → firm-FY aggregation
├── reg30_firm_quarter_agg.py  Reg 30 event → firm-quarter aggregation
├── xbrl_narrative_inventory.py
├── reg30_ar_metric_labels.csv metric-labelling scaffold for AR/Reg30 variables
├── regression_results_full.txt   every regression result, one file
├── requirements.txt
├── download_scripts/          scrapers + LLM extractors
└── data/
    ├── raw/                   source data (XBRL, PDFs, Reg30 announcements, Prowess)
    ├── processed/             every derived CSV/xlsx the pipeline produces
    └── logs/                  download provenance logs (audit trail)
```
