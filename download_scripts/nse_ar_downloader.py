"""
Indian Annual Report Downloader
================================
Downloads FY23, FY24, FY25 Annual Reports for companies listed in
matched_companies.xlsx, sourcing PDFs from NSE India archives.

BSE's announcement API requires JavaScript-set session cookies and cannot
be accessed from scripts. NSE's annual-report API is accessible with a
simple session warmup and curl_cffi browser impersonation.

Source:
  - NSE India : https://www.nseindia.com

Usage:
  python3 nse_ar_downloader.py                        # downloads all companies
  python3 nse_ar_downloader.py --company INFOSYS      # single company by name/BSE code
  python3 nse_ar_downloader.py --workers 2 --delay 3  # custom concurrency/delay
"""

import io
import re
import csv
import json
import time
import logging
import argparse
import hashlib
import random
import threading
import zipfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from curl_cffi import requests as curl_requests
from tqdm import tqdm


SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR.parent / "data"               # DSL/data/
RAW_DIR       = DATA_DIR / "raw"
PROC_DIR      = DATA_DIR / "processed"
BASE_DIR      = RAW_DIR / "annual_reports"                # DSL/data/raw/annual_reports/
LOG_DIR       = DATA_DIR / "logs"                          # DSL/data/logs/
MANIFEST_FILE = LOG_DIR / "ar_manifest.json"                # DSL/data/logs/ar_manifest.json

# matched_companies.xlsx: columns BSE Name, BSE Code, Prowess Name, Prowess Code
COMPANIES_FILE = PROC_DIR / "matched_companies.xlsx"

NSE_AR_API = "https://www.nseindia.com/api/annual-reports"

# Maps our FY label → NSE toYr field value
NSE_FY_MAP = {"FY25": "2025", "FY24": "2024", "FY23": "2023"}

CHUNK_SIZE    = 1024 * 512   # 512 KB
MIN_PDF_BYTES = 50_000       # reject files < 50 KB (likely error page)


# ── Company loading ───────────────────────────────────────────────────────────

def load_companies() -> list[tuple[str, str, str]]:
    """
    Load (bse_code, company_name, nse_symbol) triples from matched_companies.xlsx.
    The 'NSE Symbol' column must be present — run build_nse_symbols.py to
    generate or refresh it.
    """
    df = pd.read_excel(COMPANIES_FILE,
                       usecols=["BSE Code", "BSE Name", "NSE Symbol"])
    df = df.dropna(subset=["BSE Code", "BSE Name"])
    df["BSE Code"]    = df["BSE Code"].astype(int).astype(str)
    df["NSE Symbol"]  = df["NSE Symbol"].fillna("")

    no_sym = df[df["NSE Symbol"] == ""]["BSE Name"].tolist()
    if no_sym:
        logging.warning(f"No NSE symbol for {len(no_sym)} companies: {no_sym}")

    return list(zip(df["BSE Code"], df["BSE Name"], df["NSE Symbol"]))


# ── Download result dataclass ─────────────────────────────────────────────────

@dataclass
class DownloadResult:
    bse_code:     str
    company_name: str
    fy:           str
    status:       str           # "success" | "not_found" | "error"
    filepath:     Optional[str] = None
    source:       Optional[str] = None   # "nse" | "cached"
    url:          Optional[str] = None
    size_bytes:   Optional[int] = None
    sha256:       Optional[str] = None
    error_msg:    Optional[str] = None
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── NSE scraper ───────────────────────────────────────────────────────────────

class NSEScraper:
    """
    Uses curl_cffi with Chrome impersonation to bypass NSE's bot protection.
    A single session is shared across all downloads; the homepage warmup
    sets the cookies NSE requires before the API can be called.

    NSE rate-limits sessions after extended use. A threading lock guards
    session refresh so only one thread re-warms at a time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.session = curl_requests.Session(impersonate="chrome110")
        self.session.headers.update({
            "Referer":         "https://www.nseindia.com/",
            "Accept":          "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._warmup()

    def _warmup(self):
        try:
            self.session.get("https://www.nseindia.com/", timeout=20)
            time.sleep(2)
            logging.info("NSE session warmed up.")
        except Exception as e:
            logging.warning(f"NSE warmup failed: {e}")

    def _refresh(self):
        """Re-warm the session under lock. Called when the session appears expired."""
        with self._lock:
            logging.warning("NSE session appears expired — re-warming...")
            self._warmup()

    def find_annual_report_url(self, nse_symbol: str, fy: str) -> Optional[str]:
        """Return direct PDF URL for the given symbol and FY, or None."""
        to_yr = NSE_FY_MAP.get(fy)
        if not to_yr:
            return None

        for attempt in range(2):
            try:
                resp = self.session.get(
                    NSE_AR_API,
                    params={"index": "equities", "symbol": nse_symbol},
                    timeout=20,
                )
                resp.raise_for_status()
                reports = resp.json().get("data", [])

                # Empty data on first attempt usually means session expired, not absent data.
                if not reports and attempt == 0:
                    self._refresh()
                    continue

                for r in reports:
                    if str(r.get("toYr", "")) == to_yr:
                        return r.get("fileName") or None
                return None  # reports present but no matching FY — genuinely not available
            except Exception as e:
                if attempt == 0:
                    logging.warning(f"NSE API error [{nse_symbol}/{fy}], retrying after refresh: {e}")
                    self._refresh()
                else:
                    logging.warning(f"NSE API error [{nse_symbol}/{fy}]: {e}")
        return None

    def download(self, url: str, dest: Path) -> int:
        """Stream-download a PDF (or ZIP containing the PDF). Returns file size in bytes."""
        for attempt in range(2):
            resp = self.session.get(url, timeout=120)
            if resp.status_code == 403 and attempt == 0:
                self._refresh()
                continue
            resp.raise_for_status()
            break

        content = resp.content

        # NSE sometimes serves a ZIP archive containing the annual report PDF.
        # Detect by magic bytes PK\x03\x04 and extract the AR_ member.
        if content[:4] == b"PK\x03\x04":
            content = _extract_ar_pdf_from_zip(content)

        dest.write_bytes(content)
        return len(content)


# ── File helpers ──────────────────────────────────────────────────────────────

def _extract_ar_pdf_from_zip(data: bytes) -> bytes:
    """Extract the annual report PDF from a ZIP archive returned by NSE.
    Picks the member whose name starts with 'AR_'; falls back to the first PDF.
    Raises ValueError if the ZIP is unreadable (e.g., truncated/corrupt server-side file)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            ar_name = next((n for n in names if n.upper().startswith("AR_") and n.lower().endswith(".pdf")), None)
            if ar_name is None:
                ar_name = next((n for n in names if n.lower().endswith(".pdf")), None)
            if ar_name is None:
                raise ValueError(f"No PDF found inside ZIP. Members: {names}")
            return z.read(ar_name)
    except zipfile.BadZipFile:
        raise ValueError("NSE returned a corrupt/truncated ZIP — likely a server-side issue")

def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def dest_path(company_name: str, bse_code: str, fy: str) -> Path:
    # Flat per-company folder; FY differentiates files within it.
    # e.g. annual_reports/HDFC_BANK_LIMITED_500180/HDFC_BANK_LIMITED_500180_FY25_AnnualReport.pdf
    folder = BASE_DIR / f"{sanitize(company_name)}_{bse_code}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{sanitize(company_name)}_{bse_code}_{fy}_AnnualReport.pdf"


# ── Per-task download ─────────────────────────────────────────────────────────

def download_one(
    bse_code:     str,
    company_name: str,
    nse_symbol:   str,
    fy:           str,
    nse:          NSEScraper,
    delay:        float = 2.0,
) -> DownloadResult:
    """Download one company/FY annual report from NSE. Returns DownloadResult."""

    dest = dest_path(company_name, bse_code, fy)

    if dest.exists() and dest.stat().st_size > MIN_PDF_BYTES:
        logging.info(f"[SKIP] {company_name} {fy} already exists.")
        return DownloadResult(
            bse_code=bse_code, company_name=company_name, fy=fy,
            status="success", filepath=str(dest), source="cached",
            size_bytes=dest.stat().st_size, sha256=sha256_of(dest),
        )

    if not nse_symbol:
        return DownloadResult(
            bse_code=bse_code, company_name=company_name, fy=fy,
            status="not_found", error_msg="No NSE symbol mapping",
        )

    time.sleep(delay + random.uniform(0, 0.5))

    try:
        url = nse.find_annual_report_url(nse_symbol, fy)
        if url:
            size = nse.download(url, dest)
            if size < MIN_PDF_BYTES:
                dest.unlink(missing_ok=True)
                raise ValueError(f"File too small ({size} bytes) — likely error page")
            logging.info(f"[OK] {company_name} {fy} ({size/1024/1024:.1f} MB)")
            return DownloadResult(
                bse_code=bse_code, company_name=company_name, fy=fy,
                status="success", filepath=str(dest), source="nse",
                url=url, size_bytes=size, sha256=sha256_of(dest),
            )
    except Exception as e:
        logging.warning(f"[FAIL] {company_name}/{fy}: {e}")
        dest.unlink(missing_ok=True)

    return DownloadResult(
        bse_code=bse_code, company_name=company_name, fy=fy,
        status="not_found", error_msg=f"Not found on NSE ({nse_symbol})",
    )


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

def manifest_key(bse_code: str, fy: str) -> str:
    return f"{bse_code}::{fy}"


# ── Reporting ─────────────────────────────────────────────────────────────────

def write_csv_report(results: list[DownloadResult]):
    """Append results to shared download_log.csv and failed_downloads.csv."""
    log_path    = LOG_DIR / "ar_download_log.csv"
    failed_path = LOG_DIR / "ar_failed_downloads.csv"

    def row(r: DownloadResult) -> dict:
        return {
            "timestamp":    r.timestamp,
            "scrip_code":   r.bse_code,
            "company_name": r.company_name,
            "quarter":      r.fy,
            "filename":     Path(r.filepath).name if r.filepath else "",
            "status":       r.status,
            "note":         r.error_msg or r.source or "",
        }

    fields = ["timestamp", "scrip_code", "company_name", "quarter", "filename", "status", "note"]

    for path, subset in [
        (log_path,    results),
        (failed_path, [r for r in results if r.status != "success"]),
    ]:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            for r in subset:
                w.writerow(row(r))

    logging.info(f"Results appended to {log_path} ({len(results)} rows)")


def print_summary(results: list[DownloadResult]):
    total    = len(results)
    success  = sum(1 for r in results if r.status == "success")
    cached   = sum(1 for r in results if r.source == "cached")
    fail     = sum(1 for r in results if r.status == "not_found")
    total_mb = sum(r.size_bytes or 0 for r in results) / (1024 ** 2)

    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Total tasks : {total}")
    print(f"  Success     : {success}  (cached: {cached})")
    print(f"  Not found   : {fail}")
    print(f"  Total size  : {total_mb:.1f} MB")
    print("=" * 60 + "\n")


# ── Main run ──────────────────────────────────────────────────────────────────

def run(
    companies:      list          = None,
    fiscal_years:   list          = ["FY25", "FY24", "FY23"],
    workers:        int           = 2,
    delay:          float         = 2.0,
    filter_company: Optional[str] = None,
):
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "ar_downloader.log", mode="a"),
            logging.StreamHandler(),
        ],
    )

    if companies is None:
        companies = load_companies()   # list of (bse_code, company_name, nse_symbol)

    if filter_company:
        f = filter_company.upper()
        companies = [c for c in companies if f in c[0] or f in c[1].upper() or f in c[2].upper()]
        if not companies:
            logging.error(f"No company matched filter '{filter_company}'")
            return

    manifest = load_manifest()

    tasks = []
    for bse_code, company_name, nse_symbol in companies:
        for fy in fiscal_years:
            key = manifest_key(bse_code, fy)
            if key in manifest and manifest[key]["status"] == "success":
                continue
            tasks.append((bse_code, company_name, nse_symbol, fy))

    logging.info(f"Tasks: {len(tasks)} | Companies: {len(companies)} | FYs: {fiscal_years}")

    if not tasks:
        logging.info("All requested annual reports already downloaded (per manifest) — nothing to do.")
        return

    # Single shared NSE session — NSE rate-limits per IP, keep workers low
    nse = NSEScraper()
    results: list[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, bse_code, company_name, nse_symbol, fy, nse, delay): (bse_code, fy)
            for bse_code, company_name, nse_symbol, fy in tasks
        }
        with tqdm(total=len(futures), desc="Downloading", unit="report") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                manifest[manifest_key(result.bse_code, result.fy)] = asdict(result)
                save_manifest(manifest)
                pbar.set_postfix({"last": f"{result.company_name[:18]} {result.fy} {result.status}"})
                pbar.update(1)

    print_summary(results)
    write_csv_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Indian Company Annual Reports via NSE")
    parser.add_argument("--company", type=str, help="Filter by BSE code or company name substring")
    parser.add_argument("--fy",      nargs="+", default=["FY25", "FY24", "FY23"],
                        choices=["FY25", "FY24", "FY23"], help="Fiscal years to download")
    parser.add_argument("--workers", type=int,   default=2,   help="Parallel threads (default 2)")
    parser.add_argument("--delay",   type=float, default=2.0, help="Base delay between requests (s)")
    args = parser.parse_args()

    run(
        fiscal_years=args.fy,
        workers=args.workers,
        delay=args.delay,
        filter_company=args.company,
    )