"""
BSE Annual Report Fallback
===========================
Fills gaps left by nse_ar_downloader.py using BSE's own filing archive instead
of NSE's. Every listed company must file its Annual Report with BSE under
SEBI LODR Regulation 34(1) — this filing is indexed by BSE Code, which never
changes, unlike NSE's ticker symbol (which does, on renames/demergers/spin-
offs). That's exactly why NSE lookups fail for some companies even with a
"correct" current symbol: e.g. GE T&D India Limited renamed to GE Vernova T&D
India Ltd and its NSE annual-reports symbol changed too, while its BSE Code
(522275) stayed the same the whole time. Confirmed by direct query: BSE has
clean Reg. 34(1) Annual Report filings for FY22-FY25 for this company, while
NSE's annual-reports API returns zero results under any symbol we tried.

Uses the same BSE JSON API + attachment-download pattern validated for
download_scripts/reg30_scraper.py earlier — no Playwright needed, no
Akamai-style bot detection on this path.

Scope: only the (bse_code, fy) pairs still marked non-success in
data/logs/ar_manifest.json after nse_ar_downloader.py has run. Shares that
same manifest and the same data/raw/annual_reports/ output layout — this is
a complement to nse_ar_downloader.py, not a replacement.

Usage
-----
    python download_scripts/bse_ar_fallback.py
"""
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nse_ar_downloader import (  # noqa: E402  (reuse, don't duplicate, shared conventions)
    dest_path, sha256_of, load_manifest, save_manifest, manifest_key,
    MIN_PDF_BYTES, LOG_DIR,
)

REQUEST_DELAY = 1.5
ATTACH_TIMEOUT = 30

ANN_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_BASES = [
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/",
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
]
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")

# Two distinct header sets rather than one shared dict: repeated isolated
# tests against the attachment endpoint (User-Agent + Referer only) were 100%
# reliable, while the same requests inside this script — sending an extra
# Accept: application/json (meaningless for a PDF) and Connection: close —
# failed intermittently to consistently. Not fully root-caused (BSE isn't
# globally blocking this client — the same file fetched standalone mid-failed-
# run succeeds every time), but matching the proven-good request shape for
# attachments is a concrete, evidence-based change, unlike another retry-count bump.
API_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bseindia.com/corporates/ann.html",
    "Accept": "application/json",
}
ATTACH_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bseindia.com/corporates/ann.html",
}

# BSE files the standalone Reg 34(1) copy under this exact subcategory name;
# combined AGM-notice-plus-AR filings also exist but bundle other content, so
# the standalone one is the cleaner source.
AR_SUBCAT_RE = re.compile(r"reg\.?\s*34\s*\(1\)\s*annual report", re.IGNORECASE)

START_DATE = "20220101"  # wide enough to catch FY23 (filed mid-2023) with margin


def fy_label_for_date(iso_dt: str) -> str:
    """A company files its FY-ending-March-Y annual report within calendar
    year Y (typically May-Sep). Late filings can slip into Jan-Mar of Y+1
    but still belong to FY-ending-March-Y, hence the month<4 adjustment.

    Only year/month are needed, so pull them with a regex instead of
    datetime.fromisoformat() — BSE's timestamps have inconsistent fractional-
    second precision (e.g. '.12' vs '.120000'), which fromisoformat rejects
    on some inputs."""
    m = re.match(r"(\d{4})-(\d{2})-\d{2}", iso_dt)
    if not m:
        raise ValueError(f"Unrecognized NEWS_DT format: {iso_dt!r}")
    year, month = int(m.group(1)), int(m.group(2))
    y = year if month >= 4 else year - 1
    return f"FY{y % 100}"


def get_with_retry(url, headers, params=None, timeout=20, retries=5):
    """BSE's servers throw an intermittent SSLV3_ALERT_BAD_RECORD_MAC on a
    small but real fraction of requests, unrelated to connection reuse
    (reproduced with Connection: close too) — it's transient and clears up
    on retry, so retry harder here than a normal "give up fast" policy."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (200, 404):
                return r
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    print(f"  giving up on {url}: {last_err}")
    return None


def fetch_ar_filings(bse_code):
    """Paginate BSE's announcement API for one scrip and return
    {fy_label: attachment_name} for every Reg 34(1) Annual Report filing found,
    keeping the most recent filing per FY if there are duplicates/refilings."""
    result = {}
    pageno = 1
    while True:
        params = {"pageno": pageno, "strCat": -1, "strPrevDate": START_DATE,
                   "strScrip": str(bse_code), "strSearch": "P",
                   "strToDate": datetime.now().strftime("%Y%m%d"),
                   "strType": "C", "subcategory": -1}
        resp = get_with_retry(ANN_API, API_HEADERS, params=params)
        time.sleep(REQUEST_DELAY)
        if resp is None or resp.status_code != 200:
            # A page fetch failing after all retries shouldn't silently
            # truncate this company's history to whatever pages came before
            # it — surface it so a rerun (which resumes from the manifest)
            # can pick up the FYs this page would have contained.
            print(f"   [WARN] page {pageno} fetch failed after retries — "
                  f"filings on this and later pages may be missed this run")
            break
        data = resp.json()
        rows = data.get("Table") or []
        if not rows:
            break
        for r in rows:
            if AR_SUBCAT_RE.search(r.get("SUBCATNAME") or ""):
                fy = fy_label_for_date(r["NEWS_DT"])
                attach = (r.get("ATTACHMENTNAME") or "").strip()
                if attach:
                    prev = result.get(fy)
                    if prev is None or r["NEWS_DT"] > prev[1]:
                        result[fy] = (attach, r["NEWS_DT"])
        rowcnt = (data.get("Table1") or [{}])[0].get("ROWCNT")
        if rowcnt is not None and pageno * 50 >= rowcnt:
            break
        if len(rows) < 50:
            break
        pageno += 1
    return {fy: attach for fy, (attach, _) in result.items()}


def download_attachment(attachment_name, dest):
    for base in ATTACH_BASES:
        resp = get_with_retry(base + attachment_name, ATTACH_HEADERS, timeout=ATTACH_TIMEOUT)
        time.sleep(REQUEST_DELAY)
        if resp is not None and resp.status_code == 200 and resp.content:
            dest.write_bytes(resp.content)
            return True
    return False


def run():
    manifest = load_manifest()
    missing = [(k, v) for k, v in manifest.items() if v.get("status") != "success"]
    print(f"{len(missing)} (bse_code, fy) pairs still missing after NSE — trying BSE fallback")

    # Group by company so we only paginate each scrip's announcement history once
    by_company = {}
    for key, v in missing:
        bse_code, fy = key.split("::")
        by_company.setdefault(bse_code, {"company_name": v["company_name"], "fys": []})
        by_company[bse_code]["fys"].append(fy)

    recovered, still_missing = 0, 0
    for bse_code, info in by_company.items():
        company_name = info["company_name"]
        print(f"-> {bse_code} {company_name} (need {info['fys']})")
        ar_filings = fetch_ar_filings(bse_code)
        print(f"   BSE Reg 34(1) filings found: {sorted(ar_filings.keys())}")

        for fy in info["fys"]:
            key = manifest_key(bse_code, fy)
            attach_name = ar_filings.get(fy)
            if not attach_name:
                print(f"   [MISS] {fy}: no matching BSE filing")
                still_missing += 1
                continue

            dest = dest_path(company_name, bse_code, fy)
            ok = download_attachment(attach_name, dest)
            if ok and dest.exists() and dest.stat().st_size >= MIN_PDF_BYTES:
                manifest[key] = {
                    "bse_code": bse_code, "company_name": company_name, "fy": fy,
                    "status": "success", "filepath": str(dest), "source": "bse",
                    "url": ATTACH_BASES[0] + attach_name,
                    "size_bytes": dest.stat().st_size, "sha256": sha256_of(dest),
                    "error_msg": None, "timestamp": datetime.utcnow().isoformat(),
                }
                save_manifest(manifest)
                print(f"   [OK] {fy} ({dest.stat().st_size/1024/1024:.1f} MB) <- BSE")
                recovered += 1
            else:
                dest.unlink(missing_ok=True)
                print(f"   [FAIL] {fy}: attachment download failed")
                still_missing += 1

    print("\n" + "=" * 60)
    print(f"  Recovered via BSE : {recovered}")
    print(f"  Still missing     : {still_missing}")
    print("=" * 60)


if __name__ == "__main__":
    run()
