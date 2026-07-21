"""
Phase 1 Diagnostics — WHY does 09_regression.ipynb find nothing after
Romano-Wolf correction? Diagnostics only. No fixes here.

Four checks, run independently, each printing its own numbers:
  1. CG-score variance/concentration per sub-index (+ histograms to disk)
  2. Survivorship/look-ahead audit of the price panel's universe construction
  3. LLM (cg_nlp_scorer.py) measurement quality — failure/placeholder rates
  4. Power/df budget for the actual 09_regression panel

Usage
-----
    python diagnostics.py
"""
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lxml import etree

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
FIG_DIR = BASE / "data" / "processed" / "diagnostic_figures"
FIG_DIR.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# 1 — CG-SCORE VARIANCE
# ═══════════════════════════════════════════════════════════════════════════
def diagnostic_1():
    section("1 — CG-SCORE VARIANCE (data/processed/cg_scores.csv)")

    cg = pd.read_csv(PROC / "cg_scores.csv")
    g = cg.groupby("Category")["Avg_Score"].agg(["mean", "std", "min", "max", "nunique", "count"])
    g["unique_pct"] = (g["nunique"] / g["count"] * 100).round(2)

    print(g.round(4).to_string())

    print("\nConcentration check — % of obs in the single most common value, and top-2:")
    concentration_rows = []
    for cat, sub in cg.groupby("Category"):
        vc = sub["Avg_Score"].value_counts(normalize=True)
        top1 = vc.iloc[0] * 100 if len(vc) else 0
        top2 = vc.iloc[:2].sum() * 100 if len(vc) >= 2 else top1
        concentration_rows.append({"Category": cat, "top1_value": vc.index[0] if len(vc) else np.nan,
                                    "top1_pct": round(top1, 1), "top2_pct": round(top2, 1)})
        flag = ""
        if sub["Avg_Score"].std() < 0.02:
            flag += " [NEAR-ZERO SD]"
        if top2 > 80:
            flag += " [>80% MASS IN <=2 VALUES]"
        print(f"  {cat:10s}  top1={vc.index[0]:.4f} ({top1:5.1f}%)   top2={top2:5.1f}%{flag}")

        # histogram
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sub["Avg_Score"].hist(bins=40, ax=ax)
        ax.set_title(f"{cat} — Avg_Score distribution (N={len(sub)}, nunique={sub['Avg_Score'].nunique()})")
        ax.set_xlabel("Avg_Score"); ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"hist_{cat}.png", dpi=110)
        plt.close(fig)

    print(f"\nHistograms saved to {FIG_DIR}/hist_<CATEGORY>.png")

    # trace DINDEX's collapse to its NLP source components (D2-D5)
    print("\n--- Why DINDEX specifically? Raw D2-D5 component scores (data/raw/numerical_indices.xlsx) ---")
    ni = pd.read_excel(RAW / "numerical_indices.xlsx")
    for mid in ["D2", "D3", "D4", "D5", "A8"]:
        sub = ni[ni["ID"] == mid]
        vc = sub["Score"].astype(str).value_counts()
        placeholder_mask = sub["Score"].astype(str).isin(
            ["NLP scoring failed", "No policy URL found", "Auditor opinion undetermined"])
        n_placeholder = placeholder_mask.sum()
        print(f"  {mid}: {len(sub)} rows, {n_placeholder} ({100*n_placeholder/len(sub):.1f}%) are "
              f"failure placeholders, not real LLM scores")
        top_vals = vc.head(4).to_dict()
        print(f"       top values: {top_vals}")

    return g


# ═══════════════════════════════════════════════════════════════════════════
# 2 — SURVIVORSHIP / LOOK-AHEAD AUDIT
# ═══════════════════════════════════════════════════════════════════════════
def diagnostic_2():
    section("2 — SURVIVORSHIP / LOOK-AHEAD AUDIT")

    top500 = pd.read_excel(RAW / "top_500_companies.xlsx")
    mc_col = [c for c in top500.columns if "market cap" in c.lower()][0]
    print(f"Universe seed: {RAW/'top_500_companies.xlsx'} ({len(top500)} companies)")
    print(f"  Ranking column: {mc_col!r}")
    print("  *** This ranks companies by market cap AFTER the FY23-FY25 study window ends ***")
    print("  (FY25 ends March 2025; the ranking window is Jul-Dec 2025 — i.e. this universe was")
    print("   selected using information from 4-9 months in the FUTURE relative to the last FY")
    print("   being studied. A firm that was large during FY23 but has since been delisted,")
    print("   acquired, or fallen out of the top 500 by late 2025 is structurally absent from")
    print("   the sample -- this is look-ahead bias in the sample DEFINITION, upstream of any")
    print("   code in this repo.")

    seed = pd.read_excel(RAW / "matched_companies_seed.xlsx")
    matched = pd.read_excel(PROC / "matched_companies.xlsx")
    imap = pd.read_excel(PROC / "industry_map.xlsx")
    print(f"\nmatched_companies_seed.xlsx: {len(seed)} companies "
          f"(manually curated from the 500 -- {len(top500)-len(seed)} dropped, criteria not in this repo)")
    no_prowess = matched["Prowess Code"].isna().sum()
    print(f"matched_companies.xlsx: {len(matched)} companies, {no_prowess} ({100*no_prowess/len(matched):.1f}%) "
          f"have no Prowess match (financial-statement linkage failed)")
    print(f"industry_map.xlsx (final regression universe): {len(imap)} companies")

    print("\n--- yfinance ticker construction (identical pattern in notebooks 05, 08, 09) ---")
    print("  imap = pd.read_excel('industry_map.xlsx')  # CURRENT company list, no point-in-time snapshot")
    print("  tickers_ns = [f'{s}.NS' for s in imap['NSE Symbol']]")
    print("  px_raw = yf.download(tickers_ns, start='2019-07-01', end=<today>, auto_adjust=True)")
    print("  valid = px_raw.columns[px_raw.isna().mean() < 0.5].tolist()")
    print("\n  Re-running this exact call to measure the drop rate live:")

    imap2 = imap[["BSE Code", "NSE Symbol", "Industry"]].dropna(subset=["NSE Symbol"])
    tickers_ns = [f"{s}.NS" for s in imap2["NSE Symbol"]]
    try:
        import yfinance as yf
        px_raw = yf.download(tickers_ns, start="2019-07-01", end="2026-07-21",
                              auto_adjust=True, progress=False)["Close"]
        valid = px_raw.columns[px_raw.isna().mean() < 0.5].tolist()
        dropped = [t for t in tickers_ns if t not in valid]
        print(f"  Tickers requested : {len(tickers_ns)}")
        print(f"  Valid (kept)      : {len(valid)}")
        print(f"  Dropped (>50% NaN): {len(dropped)}  ({100*len(dropped)/len(tickers_ns):.1f}%)")
        if dropped:
            print(f"  Dropped tickers: {dropped}")
    except Exception as e:
        print(f"  [live yfinance check failed: {e}]")
        dropped = None

    print("\n  Note: this filter only catches firms with THIN data in the current yf.download() call.")
    print("  It CANNOT catch firms that delisted before ever being tickers in industry_map.xlsx in")
    print("  the first place -- those were already excluded at the Stage-1/Stage-2 universe-")
    print("  selection step above, not by this filter. The %-missing filter under-counts the true")
    print("  survivorship gap; it only measures its own, smaller, downstream slice of it.")


# ═══════════════════════════════════════════════════════════════════════════
# 3 — LLM MEASUREMENT QUALITY
# ═══════════════════════════════════════════════════════════════════════════
def diagnostic_3():
    section("3 — LLM (cg_nlp_scorer.py) MEASUREMENT QUALITY")

    ni = pd.read_excel(RAW / "numerical_indices.xlsx")
    print("Failure/placeholder rate by metric ID (from data/raw/numerical_indices.xlsx,")
    print("the data actually feeding cg_scores.csv):")
    placeholder_strs = ["NLP scoring failed", "No policy URL found", "Auditor opinion undetermined"]
    for mid in ["A8", "D2", "D3", "D4", "D5"]:
        sub = ni[ni["ID"] == mid]
        if not len(sub):
            continue
        counts = sub["Score"].astype(str).value_counts()
        n_fail = sub["Score"].astype(str).isin(placeholder_strs).sum()
        print(f"  {mid}: N={len(sub):5d}  failed/placeholder={n_fail:5d} ({100*n_fail/len(sub):5.1f}%)  "
              f"breakdown={ {k: v for k, v in counts.items() if k in placeholder_strs} }")

    print("\nNote on provenance: the failure-label strings actually observed in numerical_indices.xlsx")
    print("('No policy URL found', 'NLP scoring failed', 'Auditor opinion undetermined') do NOT match")
    print("cg_nlp_scorer.py's own labels in this repo ('URL fetch failed', 'LLM parse failed') --")
    print("meaning the scores currently in use were NOT produced by literally running this exact")
    print("script as it stands today; some earlier/different version produced them. I'm diagnosing")
    print("(a) the ACTUAL data's failure rate [above, solid ground] and (b) THIS repo's current")
    print("code's specific fragility points [below, code-level facts + a live spot-check].")

    print("\n--- Code-level fragility in cg_nlp_scorer.py (exact line references) ---")
    print("  Line 155: `if 'pdf' in ct: return None`  -- any PDF policy is silently skipped entirely,")
    print("            not sent to the LLM at all, regardless of content quality.")
    print("  Line 167: `text = ' '.join(words[:3000])`  -- fetched page text capped at 3000 words.")
    print("  Line 187: `{text[:4000]}` inside the prompt -- a SECOND, more severe cap: the already-")
    print("            3000-word text gets cut to its first 4000 characters before the LLM ever sees")
    print("            it. At ~5.5 chars/word (English prose), 4000 chars is ~730 words -- roughly")
    print("            75% of any policy longer than ~1 printed page is silently discarded pre-scoring.")
    print("  Line 204: `re.search(r'\\{[^}]+\\}', output)` -- a single-brace-depth regex. Any JSON with")
    print("            a NESTED object or array (e.g. `{\"criteria_met\": [1,2,3], ...}`, which the")
    print("            rubric's own requested output shape includes!) will not match past the first")
    print("            `}`, truncating/corrupting the match. This is a self-inflicted parse failure:")
    print("            the rubric ASKS for a list field, and the regex can't handle it.")

    print("\n--- Live spot-check: fetching a small sample of real policy URLs from cached CG filings ---")
    _live_url_spotcheck()


def _live_url_spotcheck(n_sample=15):
    """Extract real policy URLs from a handful of cached CG XML filings and test
    cg_nlp_scorer.py's actual fetch/parse behavior on them today."""
    import zipfile
    zpath = RAW / "governance_reports.zip"
    if not zpath.exists():
        print("  governance_reports.zip not found -- skipping live spot-check.")
        return

    sys.path.insert(0, str(BASE / "download_scripts"))
    from cg_nlp_scorer import extract_urls_from_xml, fetch_page_text

    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if n.endswith("_CG.xml")]
    rng = np.random.default_rng(7)
    sample_names = rng.choice(names, size=min(60, len(names)), replace=False)

    found_urls = []
    for n in sample_names:
        with z.open(n) as f:
            data = f.read()
        tmp = FIG_DIR / "_tmp.xml"
        tmp.write_bytes(data)
        info = extract_urls_from_xml(str(tmp))
        if info and info.get("urls"):
            for metric_id, url in info["urls"].items():
                found_urls.append((n, metric_id, url))
        if len(found_urls) >= n_sample:
            break
    tmp.unlink(missing_ok=True)

    print(f"  Sampled {len(sample_names)} CG.xml filings, found {len(found_urls)} policy URLs to test "
          f"(capped at {n_sample})")
    n_ok, n_fail, n_pdf, n_over3000w, n_over4000c = 0, 0, 0, 0, 0
    for _, metric_id, url in found_urls[:n_sample]:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, verify=False)
            ct = resp.headers.get("Content-Type", "")
            if "pdf" in ct:
                n_pdf += 1
                print(f"    [PDF-SKIP] {url[:70]}")
                continue
            text = fetch_page_text(url)
            if not text:
                n_fail += 1
                print(f"    [FETCH-FAIL] {url[:70]}")
                continue
            n_ok += 1
            n_words = len(text.split())
            if n_words >= 3000:
                n_over3000w += 1
            if len(text) > 4000:
                n_over4000c += 1
        except Exception:
            n_fail += 1
            print(f"    [FETCH-FAIL] {url[:70]}")
        time.sleep(0.3)

    tested = n_ok + n_fail + n_pdf
    print(f"\n  Of {tested} URLs tested live today:")
    print(f"    fetch failed          : {n_fail}")
    print(f"    PDF (silently skipped): {n_pdf}")
    print(f"    fetched OK            : {n_ok}")
    if n_ok:
        print(f"    of those fetched OK, hit the 3000-word fetch cap : {n_over3000w}/{n_ok}")
        print(f"    of those fetched OK, exceed the 4000-char prompt cap: {n_over4000c}/{n_ok}")


# ═══════════════════════════════════════════════════════════════════════════
# 4 — POWER / DF BUDGET
# ═══════════════════════════════════════════════════════════════════════════
def diagnostic_4():
    section("4 — POWER / DEGREES-OF-FREEDOM BUDGET (09_regression.ipynb's actual panel)")

    panel = pd.read_csv(PROC / "panel_regression_ready.csv")
    panel["BSE Code"] = pd.to_numeric(panel["BSE Code"], errors="coerce")

    n_firms = panel["BSE Code"].nunique()
    n_obs = len(panel)
    n_industries = panel["Industry"].nunique()
    fe_industry = n_industries - 1  # drop_first=True
    fe_year = panel["FY"].nunique() - 1
    n_controls = 4  # Beta_Market, Momentum, Log_MarketCap, DE_Ratio
    n_cg_joint = 6  # AINDEX..TRINDEX, for the joint spec

    print(f"Final panel: N={n_obs} firm-FY rows, {n_firms} unique firms, {panel['FY'].nunique()} FYs")
    print(f"Industries represented: {n_industries} -> {fe_industry} industry FE dummies (drop_first)")
    print(f"Year FE dummies: {fe_year}")

    for spec_name, k_x in [("Individual sub-index (1 CG var)", 1 + n_controls),
                            ("Joint (6 CG vars)", n_cg_joint + n_controls)]:
        k_total = 1 + k_x + fe_industry + fe_year  # +1 for intercept
        df_resid = n_obs - k_total
        print(f"\n  Spec: {spec_name}")
        print(f"    Regressors (incl. intercept + FE): {k_total}")
        print(f"    Residual df: {n_obs} - {k_total} = {df_resid}")
        print(f"    FE share of used df: {(fe_industry+fe_year)/k_total*100:.1f}%")
        if df_resid < 50:
            print("    *** LOW RESIDUAL DF -- FE absorb a large share of the sample ***")

    # per-industry cell sizes -- small industries + industry FE is a classic
    # "FE absorbs almost all the variation" trap
    print("\nFirms per industry (the FE dummy structure) -- smallest 10:")
    counts = panel.groupby("Industry")["BSE Code"].nunique().sort_values()
    print(counts.head(10).to_string())
    n_singleton = (counts == 1).sum()
    print(f"\nIndustries with only 1 firm in the panel: {n_singleton} / {n_industries}")
    print("(a singleton-firm industry dummy perfectly absorbs that firm's entire cross-sectional")
    print(" variation -- CG_t for that firm can contribute nothing to the industry-FE model)")


if __name__ == "__main__":
    diagnostic_1()
    diagnostic_2()
    diagnostic_3()
    diagnostic_4()
    print("\n" + "=" * 78)
    print("  DIAGNOSTICS COMPLETE — see written verdict in the conversation, not here.")
    print("=" * 78)
