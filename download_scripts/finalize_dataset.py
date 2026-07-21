"""
BSE CG Dataset Finalizer
========================
Runs three steps in order:

  Step 1 — Retry pending failures
      Reads failed_downloads.csv, cross-checks against download_log.csv,
      and retries every quarter that is still outstanding using a live
      Playwright session. Skips this step if nothing is pending.

  Step 2 — Delete invalid files
      Removes any file in governance_reports/ whose name does not contain
      "cg" (case-insensitive). Invalid files are BSE redirect/error pages
      (e.g. scrip_128.html) that slipped through when the wrong table column
      was clicked. Note: ICGIG files are valid — "cg" appears in "icgig".

  Step 3 — Package into ZIP
      Only runs if Step 1 left zero pending failures.
      Zips the entire governance_reports/ folder into
      governance_reports.zip in the data/ directory.

Usage:
    python download_scripts/finalize_dataset.py
"""

import csv
import time
import zipfile
from pathlib import Path

import pip

from download_scripts.bse_cg_downloader import (
    load_base_page, search_and_submit, extract_table_rows,
    download_xbrl, append_log, safe_dirname,
    OUTPUT_DIR, LOG_FILE, FAILED_FILE, log,
)
from playwright.sync_api import sync_playwright

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR   = _SCRIPT_DIR.parent / "data" / "raw"
ZIP_PATH    = _DATA_DIR / "governance_reports.zip"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_pending() -> list[tuple[int, str, str]]:
    """Return list of (scrip_code, company_name, quarter) not yet downloaded."""
    done_set = set()
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "downloaded":
                done_set.add(f"{row['scrip_code']}|{row['quarter']}")

    seen, targets = set(), []
    with open(FAILED_FILE, newline="") as f:
        for row in csv.DictReader(f):
            key = f"{row['scrip_code']}|{row['quarter']}"
            if key not in done_set and key not in seen:
                seen.add(key)
                targets.append((int(row["scrip_code"]), row["company_name"], row["quarter"]))
    return targets


# ── Step 1: Retry pending failures ───────────────────────────────────────────

def step1_retry(targets: list) -> int:
    """Retry all pending quarters. Returns number still failing after retry."""
    if not targets:
        log.info("Step 1: No pending failures — skipping retry.")
        return 0

    log.info(f"Step 1: Retrying {len(targets)} pending quarter(s)...")
    still_failed = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        page = context.new_page()

        log.info("  Loading BSE CG page...")
        load_base_page(page)
        time.sleep(5)   # let session settle before the first click
        log.info("  Page ready.")

        for scrip_code, company_name, target_quarter in targets:
            log.info(f"  → {scrip_code}  {company_name}  [{target_quarter}]")
            comp_dir = OUTPUT_DIR / f"{scrip_code}_{safe_dirname(company_name)}"

            try:
                search_and_submit(page, scrip_code, company_name)
                rows = extract_table_rows(page)

                matched = [r for r in rows if r["quarter"] == target_quarter]
                if not matched:
                    log.warning(f"    Quarter '{target_quarter}' not found in table")
                    still_failed += 1
                    load_base_page(page)
                    continue

                row    = matched[0]
                q_safe = row["quarter"].replace(" ", "_")

                # Skip if a file for this quarter already exists in the company folder
                existing = list(comp_dir.glob(f"*{q_safe}*")) if comp_dir.exists() else []
                if existing:
                    log.info(f"    Already exists — skipping: {[f.name for f in existing]}")
                    continue

                result = download_xbrl(page, row["xbrl_index"], comp_dir, scrip_code, q_safe)

                if result and result is not False:
                    log.info(f"    ✓ {target_quarter}  →  {result.name}")
                    append_log({"scrip_code": scrip_code, "company_name": company_name,
                                "quarter": target_quarter, "filename": result.name,
                                "status": "downloaded", "note": ""})
                else:
                    log.warning(f"    ✗ FAILED: {target_quarter}")
                    append_log({"scrip_code": scrip_code, "company_name": company_name,
                                "quarter": target_quarter, "filename": "",
                                "status": "failed", "note": ""})
                    still_failed += 1

                load_base_page(page)

            except Exception as e:
                log.warning(f"    Error: {e}")
                still_failed += 1
                load_base_page(page)

        browser.close()

    log.info(f"Step 1 done. Still failing: {still_failed}")
    return still_failed


# ── Step 2: Delete invalid files ──────────────────────────────────────────────

def step2_delete_invalid():
    """Delete files in governance_reports/ whose name has no 'cg' in it."""
    log.info("Step 2: Scanning for invalid files (no 'cg' in filename)...")
    invalid = [f for f in OUTPUT_DIR.rglob("*")
               if f.is_file() and "cg" not in f.name.lower()]

    if not invalid:
        log.info("  No invalid files found.")
        return

    log.info(f"  Found {len(invalid)} invalid file(s) — deleting...")
    for f in invalid:
        log.info(f"  Deleting: {f.relative_to(OUTPUT_DIR)}")
        f.unlink()

    # Remove any company folders that are now empty after deletion
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            log.info(f"  Removed empty folder: {folder.name}")

    log.info("Step 2 done.")


# ── Step 3: Zip the dataset ───────────────────────────────────────────────────

def step3_zip():
    """Zip governance_reports/ into governance_reports.zip."""
    log.info(f"Step 3: Creating ZIP at {ZIP_PATH} ...")
    all_files = list(OUTPUT_DIR.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]

    mode = "a" if ZIP_PATH.exists() else "w"
    with zipfile.ZipFile(ZIP_PATH, mode, zipfile.ZIP_DEFLATED) as zf:
        for f in files_only:
            arcname = f.relative_to(_DATA_DIR)   # path inside zip starts at governance_reports/
            zf.write(f, arcname)

    size_mb = ZIP_PATH.stat().st_size / 1_048_576
    log.info(f"Step 3 done. ZIP contains {len(files_only)} files — {size_mb:.1f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if ZIP_PATH.exists():
        log.info(f"{ZIP_PATH} already exists — dataset already finalized, skipping. "
                 "Delete it to re-run.")
        raise SystemExit(0)

    # Cross-check pending failures before doing anything
    pending = build_pending()
    log.info(f"Pending failures before retry: {len(pending)}")

    still_failing = step1_retry(pending)

    step2_delete_invalid()

    if still_failing == 0:
        step3_zip()
    else:
        log.warning(
            f"Step 3 skipped — {still_failing} quarter(s) still failing. "
            f"Re-run finalize_dataset.py to try again, or investigate manually."
        )