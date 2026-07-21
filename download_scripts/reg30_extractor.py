"""
Reg 30 Event Extractor
=======================
Reads data/processed/reg30_index.csv (produced by download_scripts/reg30_scraper.py),
pulls raw text out of each cached PDF/HTML attachment, and asks a local Ollama
model (qwen2.5:14b, format=json) to pull out three event classes:

  1. director_change      — {name, event, effective_date, reason, is_independent}
  2. auditor_change       — {old_auditor, new_auditor, effective_date, reason_stated}
  3. credit_rating_change — {agency, rating_old, rating_new, outlook_old,
                             outlook_new, effective_date, direction}

A single filing can report zero, one, or several events (a board-meeting
outcome often reports multiple director changes at once), so the LLM returns
lists per class and each list item becomes one row in reg30_events.csv.

_provenance values
-------------------
  ok          — text extracted, LLM call succeeded (whether or not it found events)
  ocr_used    — reserved: no OCR engine is available in this environment
                (no Homebrew/tesseract, no local vision LLM). PDFs with no
                text layer fall through to 'no_text' instead. Wired up so a
                future OCR backend only needs to set this value, not touch
                the CSV schema.
  llm_failed  — text was extracted but the Ollama call/JSON parse failed
  no_text     — no attachment, or the attachment has no extractable text
                (e.g. scanned image PDF with no text layer)

Usage
-----
    python download_scripts/reg30_extractor.py --limit 50    # review batch (default)
    python download_scripts/reg30_extractor.py --limit 0     # full corpus (0 = no limit)
"""
import argparse
import json
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_CSV = BASE_DIR / "data" / "processed" / "reg30_index.csv"
EVENTS_CSV = BASE_DIR / "data" / "processed" / "reg30_events.csv"
REVIEW_CSV = BASE_DIR / "data" / "processed" / "reg30_sample_review.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
# CPU-bound inference on this machine runs ~2 tokens/sec (confirmed via Ollama's
# server logs, no Metal/GPU acceleration in play) — 6000 chars + a 1500-token
# generation budget meant single documents were taking 2.5+ minutes, with some
# prompts approaching the model's 4096-token context window. Trimmed both:
# most Reg 30 disclosures front-load the material fact in the first ~3-4K
# chars (boilerplate letterhead/signature blocks are what's being cut), and a
# handful of events rarely needs more than a few hundred output tokens.
TEXT_CHARS_TO_LLM = 3500
PREVIEW_CHARS = 300
MIN_TEXT_LEN = 20  # below this, treat as no extractable text

DIRECTOR_FIELDS = ["name", "event", "effective_date", "reason", "is_independent"]
AUDITOR_FIELDS = ["old_auditor", "new_auditor", "effective_date", "reason_stated"]
RATING_FIELDS = ["agency", "rating_old", "rating_new", "outlook_old", "outlook_new",
                  "effective_date", "direction"]

EVENTS_COLUMNS = [
    "BSE Code", "company_name", "NEWSID", "news_dt", "event_class",
    "name", "event", "reason", "is_independent",
    "old_auditor", "new_auditor", "reason_stated",
    "agency", "rating_old", "rating_new", "outlook_old", "outlook_new", "direction",
    "effective_date", "_provenance",
]

SYSTEM_PROMPT = """You are extracting structured data from an Indian listed company's SEBI LODR Regulation 30 disclosure filed with BSE.

Identify every event of the following three types mentioned in the disclosure text below. A single disclosure may contain zero, one, or several events of the same or different types (e.g. a board meeting outcome may report multiple director changes at once). Only extract events that are actually stated in the text -- do not infer or fabricate names, dates, or ratings that are not present.

1. director_change -- a director or KMP being appointed, resigned, ceased holding office, or reappointed.
   Fields: name (string), event (one of: appointed, resigned, ceased, reappointed), effective_date (YYYY-MM-DD or null if not stated), reason (one of: personal, term_ended, regulatory_age, health, other_professional, unstated, other), is_independent (true, false, or null if not stated)

2. auditor_change -- a change of statutory auditor.
   Fields: old_auditor (string or null), new_auditor (string or null), effective_date (YYYY-MM-DD or null), reason_stated (the reason given in the text, verbatim or closely paraphrased, empty string if none stated)

3. credit_rating_change -- a credit rating action by a rating agency (ICRA, CRISIL, CARE, India Ratings, Brickwork, etc).
   Fields: agency (string), rating_old (string or null), rating_new (string or null), outlook_old (string or null), outlook_new (string or null), effective_date (YYYY-MM-DD or null), direction (one of: upgrade, downgrade, reaffirmed)

Respond ONLY with a JSON object of exactly this shape, nothing else:
{"director_changes": [...], "auditor_changes": [...], "credit_rating_changes": []}

Use an empty list for any class with no events. Every event object must include all fields listed for its class."""


def extract_text(local_path, content_type):
    """Returns (text, provenance). provenance is 'ok' or 'no_text' at this stage
    (LLM-stage failures are layered on by the caller)."""
    # pandas represents a missing local_path as float NaN, not "" — `not nan`
    # is False (NaN is truthy), so this must be checked explicitly rather
    # than relying on `not local_path` alone, or a NaN slips through as if it
    # were a real path and crashes the Path join below.
    if not isinstance(local_path, str) or not local_path or content_type == "none":
        return "", "no_text"

    full_path = BASE_DIR / "data" / local_path
    if not full_path.exists():
        return "", "no_text"

    try:
        if content_type == "pdf":
            doc = fitz.open(str(full_path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        elif content_type == "html":
            raw = full_path.read_text(encoding="utf-8", errors="ignore")
            text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
        else:
            return "", "no_text"
    except Exception:
        return "", "no_text"

    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return "", "no_text"
    return text, "ok"


def call_ollama(text, company_name, headline, subcategory, model=MODEL):
    prompt = f"""{SYSTEM_PROMPT}

Company: {company_name}
Filing headline: {headline}
Filing subcategory: {subcategory}

--- DISCLOSURE TEXT ---
{text[:TEXT_CHARS_TO_LLM]}
--- END ---"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 700},
        }, timeout=180)
        resp.raise_for_status()
        output = resp.json().get("response", "")
        parsed = json.loads(output)
        for key in ("director_changes", "auditor_changes", "credit_rating_changes"):
            parsed.setdefault(key, [])
        return parsed, None
    except json.JSONDecodeError as e:
        # fallback: try to locate the first {...} block
        try:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for key in ("director_changes", "auditor_changes", "credit_rating_changes"):
                    parsed.setdefault(key, [])
                return parsed, None
        except Exception:
            pass
        return None, f"json_decode_error: {e}"
    except Exception as e:
        return None, str(e)


def flatten_events(parsed, row, provenance):
    """Yield one events-CSV row per extracted event. If parsed is None or all
    lists are empty, yield a single row recording provenance with blank fields."""
    base = {
        "BSE Code": row["BSE Code"], "company_name": row["company_name"],
        "NEWSID": row["NEWSID"], "news_dt": row["news_dt"],
    }

    if parsed is None:
        yield {**base, "event_class": "", "_provenance": provenance}
        return

    n_events = 0
    for d in parsed.get("director_changes", []):
        n_events += 1
        r = {**base, "event_class": "director_change", "_provenance": provenance}
        for f in DIRECTOR_FIELDS:
            r[f] = d.get(f)
        yield r

    for a in parsed.get("auditor_changes", []):
        n_events += 1
        r = {**base, "event_class": "auditor_change", "_provenance": provenance}
        for f in AUDITOR_FIELDS:
            r[f] = a.get(f)
        if r.get("reason_stated"):
            r["reason_stated"] = str(r["reason_stated"])[:500]
        yield r

    for c in parsed.get("credit_rating_changes", []):
        n_events += 1
        r = {**base, "event_class": "credit_rating_change", "_provenance": provenance}
        for f in RATING_FIELDS:
            r[f] = c.get(f)
        yield r

    if n_events == 0:
        yield {**base, "event_class": "none", "_provenance": provenance}


CHECKPOINT_EVERY = 20


def run(limit=50, input_csv=None, model=MODEL, resume=True):
    df = pd.read_csv(input_csv or INDEX_CSV)
    if limit and limit > 0:
        df = df.head(limit)

    # Resume support: a long run (hours) with no checkpointing means any crash
    # loses everything done so far. If REVIEW_CSV already has rows for this
    # exact input's NEWSIDs (from a prior interrupted run), skip them and
    # keep their results rather than reprocessing from scratch.
    prior_review_rows, prior_event_rows = [], []
    done_ids = set()
    if resume and REVIEW_CSV.exists():
        prior_review = pd.read_csv(REVIEW_CSV)
        target_ids = set(df["NEWSID"])
        overlap = prior_review[prior_review["NEWSID"].isin(target_ids)]
        if len(overlap):
            done_ids = set(overlap["NEWSID"])
            prior_review_rows = overlap.to_dict("records")
            if EVENTS_CSV.exists():
                prior_events = pd.read_csv(EVENTS_CSV)
                prior_event_rows = prior_events[prior_events["NEWSID"].isin(done_ids)].to_dict("records")
            print(f"Resuming: {len(done_ids)} of {len(df)} already done in a prior run, skipping them", flush=True)
            df = df[~df["NEWSID"].isin(done_ids)]

    print(f"Processing {len(df)} announcements (limit={limit or 'all'}) with model={model}", flush=True)

    event_rows = list(prior_event_rows)
    review_rows = list(prior_review_rows)

    def checkpoint():
        pd.DataFrame(event_rows, columns=EVENTS_COLUMNS).to_csv(EVENTS_CSV, index=False)
        pd.DataFrame(review_rows).to_csv(REVIEW_CSV, index=False)

    for n_done, (i, row) in enumerate(df.iterrows(), start=1):
        text, provenance = extract_text(row.get("local_path"), row.get("content_type"))
        parsed = None
        llm_err = None

        if provenance == "ok":
            parsed, llm_err = call_ollama(
                text, row.get("company_name"), row.get("headline"), row.get("subcategory"), model=model
            )
            if parsed is None:
                provenance = "llm_failed"

        event_rows.extend(flatten_events(parsed, row, provenance))

        review_rows.append({
            "BSE Code": row["BSE Code"],
            "company_name": row["company_name"],
            "NEWSID": row["NEWSID"],
            "headline": row.get("headline"),
            "subcategory": row.get("subcategory"),
            "extracted_json": json.dumps(parsed) if parsed is not None else "",
            "raw_text_preview": text[:PREVIEW_CHARS] if text else "",
            "_provenance": provenance,
            "llm_error": llm_err or "",
        })

        print(f"  [{n_done}/{len(df)}] {row['company_name'][:30]:30s} "
              f"provenance={provenance}", flush=True)

        if n_done % CHECKPOINT_EVERY == 0:
            checkpoint()

    checkpoint()
    events_df = pd.DataFrame(event_rows, columns=EVENTS_COLUMNS)
    review_df = pd.DataFrame(review_rows)
    print(f"\nWrote {len(events_df)} event rows -> {EVENTS_CSV}")
    print(f"Wrote {len(review_df)} review rows -> {REVIEW_CSV}")

    print("\nProvenance breakdown (documents):")
    print(review_df["_provenance"].value_counts().to_string())
    print("\nEvent class breakdown (rows):")
    print(events_df["event_class"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Reg 30 events via Ollama.")
    parser.add_argument("--limit", type=int, default=50, help="0 = process full index")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Use an alternate index CSV instead of reg30_index.csv "
                             "(e.g. a pre-selected diverse sample).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Ollama model tag. Default qwen2.5:14b; qwen2.5:7b-instruct "
                             "is ~6x faster on memory-constrained machines (swap thrashing "
                             "at 14b when RAM is tight) with comparable spot-checked quality.")
    args = parser.parse_args()
    run(limit=args.limit, input_csv=args.input_csv, model=args.model)
