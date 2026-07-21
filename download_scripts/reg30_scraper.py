"""
BSE Regulation 30 Announcement Scraper
=======================================
Fetches all SEBI LODR Regulation 30 corporate announcements for every scrip
in data/raw/top_500_companies.xlsx, from START_DATE to today.

Unlike bse_cg_downloader.py, this does NOT need Playwright. Investigation
(see conversation) showed BSE's announcements page is backed by a plain JSON
API (api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData) and serves PDF
attachments from a predictable static path — neither is behind the Akamai
bot-detection that guards the XBRL download flow bse_cg_downloader.py
exists to bypass. Plain `requests` with a normal User-Agent + Referer works,
which is far faster and more reliable than driving a browser per company.
Politeness is enforced via REQUEST_DELAY between calls regardless.

Reg 30 filter: BSE files these disclosures under multiple NEWSSUB phrasings
("Announcement under Regulation 30 (LODR)-...", "Disclosure Under The
Regulation 30 Of The SEBI...", "Board Meeting Outcome for Disclosure Under
The Regulation 30..."). A strict "Announcement under Regulation 30 (LODR)"
prefix match misses ~25% of genuine Reg 30 rows (verified on scrip 500002:
251/456 broad match vs ~190/456 strict prefix). REG30_PATTERN below matches
"regulation 30" (case-insensitive, tolerant of spacing) anywhere in NEWSSUB
or SUBCATNAME instead.

Output
------
- data/raw/reg30_announcements/<bse_code>/<NEWSID>.pdf   — attachment, when present
- data/raw/reg30_announcements/<bse_code>/<NEWSID>.html  — inline disclosure text
  (BSE's "MORE" field), used when there is no PDF attachment
- data/processed/reg30_index.csv  — one row per Reg 30 announcement
- data/logs/reg30_download_log.csv — per-company completion log (resumable)

Usage
-----
    python download_scripts/reg30_scraper.py                  # all companies
    python download_scripts/reg30_scraper.py --limit 5         # first 5, for testing
    python download_scripts/reg30_scraper.py --batch 1 --batch-size 100
"""
import argparse
import csv
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / "data"
_RAW_DIR = _DATA_DIR / "raw"
_PROC_DIR = _DATA_DIR / "processed"
_LOG_DIR = _DATA_DIR / "logs"

COMPANIES_XLSX = _RAW_DIR / "top_500_companies.xlsx"
OUTPUT_DIR = _RAW_DIR / "reg30_announcements"
INDEX_CSV = _PROC_DIR / "reg30_index.csv"
LOG_FILE = _LOG_DIR / "reg30_download_log.csv"

START_DATE = "20220401"  # 2022-04-01, per spec — matches Q1FY23, the earliest quarter
                          # in quarterly_returns_from_filing.csv (the regression panel)
END_DATE = "20250331"    # 2025-03-31 — matches Q4FY25, the latest quarter in cg_scores.csv
                          # and quarterly_returns_from_filing.csv. Fixed rather than "today"
                          # so Reg30 stays aligned with the existing regression panel and
                          # every scrape (this run or a rerun next month) covers the same
                          # window instead of silently drifting forward each time it runs.
PAGE_SIZE = 50
REQUEST_DELAY = 0.35    # seconds between BSE API/attachment calls — politeness
# BSE's servers throw an intermittent SSLV3_ALERT_BAD_RECORD_MAC on a small but
# real fraction of requests (reproduced with Connection: close too, so it's not
# a keep-alive issue) — transient and clears up on retry, so retry harder than
# a normal "give up fast" policy rather than let it look like a dead link.
MAX_RETRIES = 5
RETRY_BACKOFF = 1.5
ATTACH_TIMEOUT = 20

ANN_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_BASES = [
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/",
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
]

REG30_PATTERN = re.compile(r"regulation\s*30\b", re.IGNORECASE)

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")

# Two distinct header sets, not one shared dict. Root-caused during the
# annual-report fallback work: sending Accept: application/json (meaningless
# for a PDF) on attachment requests causes BSE to intermittently fail them —
# not a session/keep-alive issue (Connection: close alone didn't fix it).
# Matching the exact minimal header set that's proven 100% reliable for
# attachment fetches (User-Agent + Referer only) is what actually works.
API_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bseindia.com/corporates/ann.html",
    "Accept": "application/json",
}
ATTACH_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bseindia.com/corporates/ann.html",
}

INDEX_FIELDS = [
    "BSE Code", "company_name", "NEWSID", "news_dt", "period_end_guess",
    "headline", "newssub", "category", "subcategory",
    "attachment_name", "attachment_url", "local_path", "content_type",
    "critical_news", "fetched_at",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(_LOG_DIR / "reg30_scraper.log")],
)
log = logging.getLogger(__name__)


def load_companies(limit=None, sample=None, companies_csv=None):
    if companies_csv:
        df = pd.read_csv(companies_csv)
        companies = list(zip(df["BSE Code"].astype(int), df["Company Name"].astype(str)))
        log.info(f"Loaded {len(companies)} companies from {companies_csv}")
        return companies

    df = pd.read_excel(COMPANIES_XLSX)
    companies = list(zip(df["Scrip Code"].astype(int), df["Company Name"].astype(str)))
    if sample:
        # top_500_companies.xlsx is sorted by market cap — taking the first N
        # would only sample mega-caps (unrepresentative: they file far more
        # disclosures than typical mid/small-caps). An evenly-spaced
        # systematic sample spans the whole size distribution instead.
        step = max(1, len(companies) // sample)
        companies = companies[::step][:sample]
    elif limit:
        companies = companies[:limit]
    log.info(f"Loaded {len(companies)} companies from {COMPANIES_XLSX.name}")
    return companies


def load_done_set():
    done = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "done":
                    done.add(int(row["bse_code"]))
    return done


def load_previously_failed_set():
    """Companies with a 'failed' (incomplete pagination) entry — these will
    be reprocessed on rerun, so their partial rows from the earlier attempt
    must be purged before re-appending to avoid duplicate NEWSID rows."""
    failed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "failed":
                    failed.add(int(row["bse_code"]))
    return failed


def purge_company_rows(scrip_code):
    """Remove any existing reg30_index.csv rows for scrip_code (used before
    re-appending a retried company's fresh, complete row set)."""
    if not INDEX_CSV.exists():
        return
    with open(INDEX_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(r["BSE Code"]) != scrip_code]
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)


def append_log(row: dict):
    write_header = not LOG_FILE.exists()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "bse_code", "company_name",
                                           "status", "n_total", "n_reg30", "note"])
        if write_header:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(), **row})


def _get_with_retry(url, headers, params=None, timeout=20):
    """Retry only on transient failures (timeouts, connection errors, 5xx).
    A 404 means the resource genuinely isn't at this URL — return it
    immediately so the caller can try a fallback base without wasting
    backoff time retrying a request that will never succeed."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200 or r.status_code == 404:
                return r
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    log.warning(f"  Giving up on {url} after {MAX_RETRIES} attempts: {last_err}")
    return None


def fetch_announcements(scrip_code, to_date):
    """Paginate through AnnSubCategoryGetData for one scrip. Returns
    (rows, complete) — rows unfiltered (Reg 30 filtering happens in the
    caller), complete=False if a page failed after all retries, so the
    caller can avoid marking a company 'done' on a silently truncated
    fetch (which would make the gap permanent — 'done' companies are
    skipped on rerun)."""
    all_rows = []
    pageno = 1
    while True:
        params = {
            "pageno": pageno, "strCat": -1, "strPrevDate": START_DATE,
            "strScrip": str(scrip_code), "strSearch": "P", "strToDate": to_date,
            "strType": "C", "subcategory": -1,
        }
        resp = _get_with_retry(ANN_API, API_HEADERS, params=params)
        time.sleep(REQUEST_DELAY)
        if resp is None or resp.status_code != 200:
            log.warning(f"  scrip {scrip_code}: page {pageno} fetch failed after retries")
            return all_rows, False
        try:
            data = resp.json()
        except ValueError:
            log.warning(f"  scrip {scrip_code}: page {pageno} non-JSON response")
            return all_rows, False
        rows = data.get("Table") or []
        if not rows:
            break
        all_rows.extend(rows)
        rowcnt = None
        t1 = data.get("Table1")
        if t1 and isinstance(t1, list) and t1:
            rowcnt = t1[0].get("ROWCNT")
        if rowcnt is not None and len(all_rows) >= rowcnt:
            break
        if len(rows) < PAGE_SIZE:
            break
        pageno += 1
        if pageno > 200:  # sanity guard against runaway pagination
            log.warning(f"  scrip {scrip_code}: aborting after 200 pages (unexpected)")
            break
    return all_rows, True


def download_attachment(attachment_name, dest_path):
    """Try AttachHis then AttachLive. Returns True on success."""
    for base in ATTACH_BASES:
        resp = _get_with_retry(base + attachment_name, ATTACH_HEADERS, timeout=ATTACH_TIMEOUT)
        time.sleep(REQUEST_DELAY)
        if resp is not None and resp.status_code == 200 and resp.content:
            dest_path.write_bytes(resp.content)
            return True
    return False


def process_company(scrip_code, company_name, to_date, metadata_only=False):
    comp_dir = OUTPUT_DIR / str(scrip_code)
    rows, complete = fetch_announcements(scrip_code, to_date)
    reg30_rows = [
        r for r in rows
        if REG30_PATTERN.search(r.get("NEWSSUB") or "") or REG30_PATTERN.search(r.get("SUBCATNAME") or "")
    ]

    index_rows = []
    for r in reg30_rows:
        newsid = r.get("NEWSID")
        attach_name = (r.get("ATTACHMENTNAME") or "").strip()
        local_path = ""
        content_type = "none"

        if metadata_only:
            # Pagination is cheap (seconds/company); attachment downloads are
            # the actual bottleneck (minutes/company). This mode builds the
            # full index — including attachment_url, so nothing is lost —
            # without fetching any PDFs, so the index can be built for many
            # companies fast, with document fetch deferred to whichever rows
            # are actually needed for extraction.
            content_type = "pdf_pending" if attach_name else "none"
        elif attach_name:
            ext = Path(attach_name).suffix or ".pdf"
            dest = comp_dir / f"{newsid}{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                local_path = str(dest.relative_to(_RAW_DIR.parent))
                content_type = "pdf"
            else:
                comp_dir.mkdir(parents=True, exist_ok=True)
                if download_attachment(attach_name, dest):
                    local_path = str(dest.relative_to(_RAW_DIR.parent))
                    content_type = "pdf"
                else:
                    log.warning(f"  {scrip_code}/{newsid}: attachment download failed ({attach_name})")
        else:
            more_html = (r.get("MORE") or "").strip()
            if len(more_html) > 20:
                comp_dir.mkdir(parents=True, exist_ok=True)
                dest = comp_dir / f"{newsid}.html"
                dest.write_text(more_html, encoding="utf-8")
                local_path = str(dest.relative_to(_RAW_DIR.parent))
                content_type = "html"

        attach_url = (ATTACH_BASES[0] + attach_name) if attach_name else ""
        index_rows.append({
            "BSE Code": scrip_code,
            "company_name": company_name,
            "NEWSID": newsid,
            "news_dt": r.get("NEWS_DT"),
            "period_end_guess": (r.get("NEWS_DT") or "")[:10],
            "headline": r.get("HEADLINE"),
            "newssub": r.get("NEWSSUB"),
            "category": r.get("CATEGORYNAME"),
            "subcategory": r.get("SUBCATNAME"),
            "attachment_name": attach_name,
            "attachment_url": attach_url,
            "local_path": local_path,
            "content_type": content_type,
            "critical_news": r.get("CRITICALNEWS"),
            "fetched_at": datetime.now().isoformat(),
        })

    return rows, reg30_rows, index_rows, complete


def append_index_rows(rows):
    write_header = not INDEX_CSV.exists()
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def run(limit=None, batch=0, batch_size=100, sample=None, companies_csv=None, metadata_only=False):
    companies = load_companies(limit=limit, sample=sample, companies_csv=companies_csv)
    if batch > 0:
        start_idx = (batch - 1) * batch_size
        end_idx = start_idx + batch_size
        companies = companies[start_idx:end_idx]
        log.info(f"Batch {batch}: companies {start_idx + 1}-{end_idx}")

    done_set = load_done_set()
    previously_failed = load_previously_failed_set()
    to_date = END_DATE

    stats = {"companies_done": 0, "companies_skipped": 0, "reg30_total": 0}

    for scrip_code, company_name in companies:
        if scrip_code in done_set:
            stats["companies_skipped"] += 1
            continue

        log.info(f"-> {scrip_code} {company_name}")
        try:
            if scrip_code in previously_failed:
                purge_company_rows(scrip_code)
            all_rows, reg30_rows, index_rows, complete = process_company(
                scrip_code, company_name, to_date, metadata_only=metadata_only)
            if index_rows:
                append_index_rows(index_rows)
            # A company whose pagination didn't complete must NOT be marked
            # "done" — done_set skips it on rerun, which would make a
            # transient failure a permanent silent gap in reg30_index.csv.
            status = "done" if complete else "failed"
            append_log({
                "bse_code": scrip_code, "company_name": company_name, "status": status,
                "n_total": len(all_rows), "n_reg30": len(reg30_rows),
                "note": "" if complete else "pagination incomplete — will retry on rerun",
            })
            if complete:
                stats["companies_done"] += 1
            stats["reg30_total"] += len(reg30_rows)
            log.info(f"   {len(all_rows)} announcements total, {len(reg30_rows)} Reg 30"
                     f"{'' if complete else ' (INCOMPLETE — will retry)'}")
        except Exception as e:
            log.error(f"   ERROR: {e}")
            append_log({
                "bse_code": scrip_code, "company_name": company_name, "status": "failed",
                "n_total": 0, "n_reg30": 0, "note": str(e),
            })

    log.info("=" * 60)
    log.info(f"DONE | companies processed: {stats['companies_done']} | "
             f"skipped (already done): {stats['companies_skipped']} | "
             f"Reg 30 rows found: {stats['reg30_total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape BSE Regulation 30 announcements.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N companies (testing).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evenly-spaced sample of N companies across the full "
                             "market-cap range (unlike --limit, not biased toward mega-caps).")
    parser.add_argument("--batch", type=int, default=0, help="1-indexed batch number.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--companies-csv", type=str, default=None,
                        help="CSV with 'BSE Code' and 'Company Name' columns — overrides "
                             "top_500_companies.xlsx as the company source entirely.")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Build the index (incl. attachment_url) without downloading any "
                             "PDF/HTML attachments. content_type is 'pdf_pending' for rows that "
                             "have an attachment to fetch later. Pagination-only — much faster.")
    args = parser.parse_args()
    run(limit=args.limit, batch=args.batch, batch_size=args.batch_size,
        sample=args.sample, companies_csv=args.companies_csv, metadata_only=args.metadata_only)
