"""
Annual Report Quantitative Feature Extractor
===============================================
Extracts NEW quantitative governance constructs from the 739 Annual Report
PDFs under data/raw/annual_reports/ — NOT re-scoring the constructs the
existing CG index already covers (A8 auditor opinion, D2-D5 policy quality).
That re-scoring is a separate, explicitly-opt-in robustness exhibit (see
ar_robustness_rescoring.py), never mixed into these features.

Constructs (all sourced from the Corporate Governance Report chapter and the
financial-statement notes, both inside the same AR PDF):
  board_size                    — total directors on the board
  board_independence_ratio      — independent directors / board_size
  board_meeting_count           — board meetings held during the year
  board_avg_attendance_pct      — average attendance %, ONLY if the AR states
                                   this as an explicit summary figure (never
                                   computed by the LLM from a per-director
                                   table — that's a different, unreliable task)
  promoter_pledge_pct           — % of promoter shareholding pledged
  rpt_aggregate_inr_crore       — aggregate related-party-transaction value
  audit_fees_inr_crore          — statutory audit fee paid
  contingent_liabilities_inr_crore — aggregate contingent liabilities

Two-stage approach (a whole AR is 50-300+ pages; qwen2.5's context window is
4096 tokens, nowhere near enough to feed one in full):
  1. LOCATE: search every page's text for section-anchor keywords, take a
     bounded window of pages around each hit cluster.
  2. EXTRACT: one combined Ollama call per document with all located
     excerpts, format=json, explicit instruction to output null rather than
     fabricate when a field isn't clearly stated (checked: LLM must not
     compute board_avg_attendance_pct itself from a per-director table).

Monetary fields ask for {value, unit} separately (Indian ARs mix Crore/Lakh/
Thousand/actual-rupee reporting depending on company size) and are
normalized to INR Crore in post-processing — a raw unit-less number would be
incomparable across firms of different scale.

Found-flags: derived as notna(value) post-hoc, not requested as a separate
LLM-reported field — a null already unambiguously means "not found," so a
redundant self-reported confidence score would only add a second,
potentially-inconsistent signal without new information.

_provenance: 'ok' (extraction ran), 'no_anchor' (couldn't locate the CG
report chapter at all — PDF likely scanned/non-standard), 'llm_failed'.

Checkpointed every 20 documents; resumable exactly like reg30_extractor.py.

Usage
-----
    python download_scripts/ar_extractor.py --limit 5      # test batch
    python download_scripts/ar_extractor.py --limit 0      # full 739-document run
"""
import argparse
import json
import re
from pathlib import Path

import fitz
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
AR_DIR = BASE_DIR / "data" / "raw" / "annual_reports"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "ar_features_firm_fy.csv"
MANIFEST_CSV = BASE_DIR / "data" / "logs" / "ar_extractor_manifest.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"
CHECKPOINT_EVERY = 20

# ── Stage 1: section location ────────────────────────────────────────────────
# A bare case-insensitive match on "Corporate Governance Report" fires on the
# table of contents, cross-references, and running headers throughout the
# document, not just the chapter's actual opening page — validated on 4 ARs
# of very different layout (360 ONE WAM, ABB India, Hero MotoCorp, Maharashtra
# Scooters): the phrase (or its near-universal Schedule-V-mandated first
# subsection, "Company's philosophy on corporate governance") appearing
# within the first 80 characters of a page's extracted text reliably means
# "this page IS the chapter opening," while a mid-page match means "this is
# a mention/reference," almost without exception across all 4 formats tested.
CG_ANCHOR_RE = re.compile(
    r"report\s+on\s+corporate\s+governance|corporate\s+governance\s+report|philosophy\s+on\s+corporate\s+governance",
    re.I,
)
CG_ANCHOR_MAX_POS = 80  # match must start within this many chars of page start
CG_WINDOW_PAGES = 14  # CG Report chapter is compact (~10-20pp) in Indian ARs

FINANCIAL_PATTERNS = {
    "rpt": re.compile(r"related party transaction", re.I),
    "audit_fees": re.compile(r"payment to auditor|auditor.{0,15}remuneration|remuneration to auditor", re.I),
    "contingent": re.compile(r"contingent liabilit", re.I),
    "pledge": re.compile(r"pledg(e|ed|ing)", re.I),
}
FIN_WINDOW_BEFORE, FIN_WINDOW_AFTER = 0, 1  # each hit + the following page
MAX_HITS_PER_PATTERN = 4  # cap runaway matches (e.g. "pledge" boilerplate repeated many times)
MAX_EXCERPT_CHARS = 9000  # keep total prompt comfortably under the 4096-token ctx window

SYSTEM_PROMPT = """You are extracting quantitative governance data from excerpts of an Indian listed company's Annual Report. The excerpts below were located by keyword search across a much longer document, so they may include irrelevant surrounding text — only extract a field if the excerpt clearly and specifically states it. If a field is not clearly stated in the excerpts, output null for it. NEVER compute, estimate, or infer a value that is not explicitly written — e.g. do not calculate an average attendance percentage yourself from a per-director attendance table; only use board_avg_attendance_pct if the report states an average/overall attendance figure directly.

Extract these fields:
- board_size (integer): total number of directors currently on the Board
- board_independent_count (integer): number of those directors who are Independent Directors
- board_meeting_count (integer): number of Board meetings held during the year. This is very often stated as a parenthetical in an attendance table's column header, e.g. "No. of Board Meetings attended during the year (Total 4 Meetings)" — if you see a phrase like "(Total N Meetings)" in or near the board attendance table, N is the answer; you do not need a separate standalone sentence.
- board_avg_attendance_pct (number 0-100): ONLY if an overall/average attendance percentage is explicitly stated as a summary figure
- promoter_pledge_pct (number 0-100): percentage of promoter shareholding that is pledged/encumbered. If explicitly stated as "Nil" or "0%", output 0, not null.
- rpt_aggregate: {"value": number, "unit": one of "crore","lakh","thousand","actual_rupees","unclear"} — the aggregate/total value of related party transactions for the year, if a single clear total is stated
- audit_fees: {"value": number, "unit": same enum} — the statutory audit fee paid to the auditor for the year. This number must come from a table row whose label is LITERALLY "Audit fee" or "Statutory audit fee" (or a very close variant) — a row that appears among OTHER itemized fee rows such as "Tax audit fee", "Limited review", "Certification", "Reimbursement of expenses" in the same small table under a heading like "Auditor's remuneration" or "Payment to auditor". Do NOT use a subtotal/total row that sums several fee types together, and do NOT use any number from an unrelated table on the same page (e.g. CSR spend, EPS, discontinued operations, KMP remuneration) — if you are not looking at that specific itemized fee table, output null.
- contingent_liabilities: {"value": number, "unit": same enum} — the aggregate contingent liabilities figure (a specific total number from a table/note, not the accounting-policy definition text)

If you are not highly confident a number comes from exactly the table/line-item described above, output null rather than guess — a wrong number is worse than a missing one.

Respond ONLY with a JSON object of exactly this shape:
{"board_size": ..., "board_independent_count": ..., "board_meeting_count": ..., "board_avg_attendance_pct": ..., "promoter_pledge_pct": ..., "rpt_aggregate": {"value": ..., "unit": ...}, "audit_fees": {"value": ..., "unit": ...}, "contingent_liabilities": {"value": ..., "unit": ...}}
Use null for any field/sub-value you cannot find, and null for a monetary "value" whose unit you truly cannot determine (do not guess unit)."""

UNIT_TO_CRORE = {"crore": 1.0, "lakh": 0.01, "thousand": 0.0001, "actual_rupees": 1e-7}


def get_page_texts(path):
    doc = fitz.open(str(path))
    texts = [doc[i].get_text() for i in range(doc.page_count)]
    doc.close()
    return texts


CG_SUB_BUDGET = 5000  # guaranteed share of MAX_EXCERPT_CHARS for the board/CG section


def _merge_ranges(chunks):
    chunks = sorted(chunks, key=lambda c: c[1])
    merged = []
    for label, start, end in chunks:
        if merged and start <= merged[-1][2]:
            merged[-1] = (merged[-1][0] + f"+{label}", merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((label, start, end))
    return merged


def _render_chunks(page_texts, chunks, budget):
    parts, total_len = [], 0
    for label, start, end in chunks:
        if total_len >= budget:
            break
        text = "\n".join(page_texts[start:end])
        if total_len + len(text) > budget:
            text = text[: max(0, budget - total_len)]
        parts.append(f"--- {label} (pages {start}-{end - 1}) ---\n{text}")
        total_len += len(text)
    return "\n\n".join(parts), total_len


def locate_excerpts(page_texts):
    """Returns (excerpt_text, found_cg_anchor: bool). The board/CG section gets
    a guaranteed sub-budget so it can never be crowded out by financial-note
    excerpts assembled in page order — board_size/independence/meeting_count
    all depend on it, and financial statements often appear earlier in the
    PDF's page order than the CG Report chapter, so a naive shared budget
    would silently drop the board data first (this was caught and fixed
    after the first real test run came back with ~0% board-field coverage)."""
    n = len(page_texts)

    def _is_chapter_open(t):
        m = CG_ANCHOR_RE.search(t)
        return m is not None and m.start() < CG_ANCHOR_MAX_POS

    cg_hit = next((i for i, t in enumerate(page_texts) if _is_chapter_open(t)), None)

    cg_text = ""
    remaining_budget = MAX_EXCERPT_CHARS
    if cg_hit is not None:
        end = min(n, cg_hit + CG_WINDOW_PAGES)
        cg_text, used = _render_chunks(page_texts, [("BOARD/CG REPORT SECTION", cg_hit, end)], CG_SUB_BUDGET)
        remaining_budget -= used

    # Each financial category gets its OWN slice of the remaining budget —
    # same crowding-out bug as the CG section, one level down: RPT hits are
    # often the most numerous (several pages), and page-order rendering let
    # RPT alone consume the whole remaining budget before audit_fees or
    # contingent_liabilities ever got a slot (caught via direct inspection:
    # ABB's audit-fee table is on a page I'd manually verified was clean, but
    # came back null because RPT chunks at earlier page numbers ate the
    # budget first). Splitting evenly per category guarantees every field
    # gets a shot regardless of how page-hit-heavy any one category is.
    per_category_budget = remaining_budget // len(FINANCIAL_PATTERNS)
    fin_parts = []
    for label, pattern in FINANCIAL_PATTERNS.items():
        hits = [i for i, t in enumerate(page_texts) if pattern.search(t)][:MAX_HITS_PER_PATTERN]
        cat_chunks = []
        for h in hits:
            start = max(0, h - FIN_WINDOW_BEFORE)
            end = min(n, h + FIN_WINDOW_AFTER + 1)
            cat_chunks.append((label.upper(), start, end))
        cat_text, _ = _render_chunks(page_texts, _merge_ranges(cat_chunks), per_category_budget)
        if cat_text:
            fin_parts.append(cat_text)

    return "\n\n".join(p for p in [cg_text] + fin_parts if p), cg_hit is not None


def call_ollama(excerpt_text):
    prompt = f"{SYSTEM_PROMPT}\n\n--- DOCUMENT EXCERPTS ---\n{excerpt_text}\n--- END ---"
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.1, "num_predict": 400},
        }, timeout=180)
        resp.raise_for_status()
        output = resp.json().get("response", "")
        return json.loads(output), None
    except json.JSONDecodeError as e:
        try:
            m = re.search(r"\{.*\}", output, re.DOTALL)
            if m:
                return json.loads(m.group()), None
        except Exception:
            pass
        return None, f"json_decode_error: {e}"
    except Exception as e:
        return None, str(e)


def normalize_money(field):
    if not isinstance(field, dict):
        return None
    val, unit = field.get("value"), field.get("unit")
    if val is None or unit not in UNIT_TO_CRORE:
        return None
    try:
        return float(val) * UNIT_TO_CRORE[unit]
    except (TypeError, ValueError):
        return None


def find_ar_pdfs():
    rows = []
    for company_dir in sorted(AR_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        for pdf in company_dir.glob("*_AnnualReport.pdf"):
            m = re.match(r"(.+)_(\d+)_(FY\d+)_AnnualReport\.pdf", pdf.name)
            if not m:
                continue
            company_name, bse_code, fy = m.groups()
            rows.append({"BSE Code": int(bse_code), "company_name": company_name, "FY": fy, "path": pdf})
    return pd.DataFrame(rows)


def run(limit=None):
    docs = find_ar_pdfs()
    print(f"Found {len(docs)} AR PDFs", flush=True)
    if limit and limit > 0:
        docs = docs.head(limit)

    done_ids = set()
    prior_rows = []
    if OUTPUT_CSV.exists():
        prior = pd.read_csv(OUTPUT_CSV)
        target_keys = set(zip(docs["BSE Code"], docs["FY"]))
        overlap = prior[prior.apply(lambda r: (r["BSE Code"], r["FY"]) in target_keys, axis=1)]
        if len(overlap):
            done_ids = set(zip(overlap["BSE Code"], overlap["FY"]))
            prior_rows = overlap.to_dict("records")
            print(f"Resuming: {len(done_ids)} of {len(docs)} already done", flush=True)
            docs = docs[~docs.apply(lambda r: (r["BSE Code"], r["FY"]) in done_ids, axis=1)]

    rows = list(prior_rows)

    def checkpoint():
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    for n_done, (_, doc_row) in enumerate(docs.iterrows(), start=1):
        provenance = "ok"
        parsed = None
        try:
            page_texts = get_page_texts(doc_row["path"])
            excerpt, found_anchor = locate_excerpts(page_texts)
        except Exception as e:
            excerpt, found_anchor = "", False
            provenance = f"pdf_error: {e}"

        if provenance == "ok" and not found_anchor and not excerpt.strip():
            provenance = "no_anchor"
        elif provenance == "ok":
            parsed, err = call_ollama(excerpt)
            if parsed is None:
                provenance = f"llm_failed: {err}"

        row = {
            "BSE Code": doc_row["BSE Code"], "company_name": doc_row["company_name"], "FY": doc_row["FY"],
            "_provenance": provenance,
        }
        if parsed:
            row["board_size"] = parsed.get("board_size")
            row["board_independent_count"] = parsed.get("board_independent_count")
            row["board_meeting_count"] = parsed.get("board_meeting_count")
            row["board_avg_attendance_pct"] = parsed.get("board_avg_attendance_pct")
            row["promoter_pledge_pct"] = parsed.get("promoter_pledge_pct")
            row["rpt_aggregate_inr_crore"] = normalize_money(parsed.get("rpt_aggregate"))
            row["audit_fees_inr_crore"] = normalize_money(parsed.get("audit_fees"))
            row["contingent_liabilities_inr_crore"] = normalize_money(parsed.get("contingent_liabilities"))
            bs, bic = row["board_size"], row["board_independent_count"]
            row["board_independence_ratio"] = (bic / bs) if (bs not in (None, 0) and bic is not None) else None
        rows.append(row)

        print(f"  [{n_done}/{len(docs)}] {doc_row['company_name'][:30]:30s} {doc_row['FY']}  "
              f"provenance={provenance}", flush=True)

        if n_done % CHECKPOINT_EVERY == 0:
            checkpoint()

    checkpoint()
    out = pd.DataFrame(rows)
    print(f"\nWrote {len(out)} rows -> {OUTPUT_CSV}")

    print("\nProvenance breakdown:")
    print(out["_provenance"].apply(lambda p: p.split(":")[0]).value_counts().to_string())

    print("\nPer-field non-null coverage:")
    fields = ["board_size", "board_independence_ratio", "board_meeting_count", "board_avg_attendance_pct",
              "promoter_pledge_pct", "rpt_aggregate_inr_crore", "audit_fees_inr_crore",
              "contingent_liabilities_inr_crore"]
    for f in fields:
        if f in out.columns:
            n = out[f].notna().sum()
            print(f"  {f}: {n}/{len(out)} ({100*n/len(out):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AR quantitative features.")
    parser.add_argument("--limit", type=int, default=5, help="0 = full 739-doc run")
    args = parser.parse_args()
    run(limit=args.limit)
