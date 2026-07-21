"""
Fetch attachments for the rows selected for full-corpus Reg 30 extraction
(subcategory/newssub matches one of the 3 target event classes) that are
still 'pdf_pending' from the metadata-only scrape. Writes directly into the
archive location (data/raw/archive/reg30_announcements/) since that's the
established home for downloaded Reg 30 documents going forward — avoids
recreating a separate "live" folder for what is a small, curated subset.

Usage
-----
    python download_scripts/reg30_fetch_relevant.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg30_scraper import download_attachment, _RAW_DIR  # noqa: E402  (reuse, don't duplicate)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_CSV = BASE_DIR / "data" / "processed" / "reg30_index.csv"
RELEVANT_CSV = BASE_DIR / "data" / "processed" / "reg30_relevant_rows.csv"
ARCHIVE_DIR = _RAW_DIR / "archive" / "reg30_announcements"


def run():
    df = pd.read_csv(INDEX_CSV)
    relevant_ids = set(pd.read_csv(RELEVANT_CSV)["NEWSID"])
    todo = df[
        (df["content_type"] == "pdf_pending")
        & df["attachment_name"].notna() & (df["attachment_name"] != "")
        & df["NEWSID"].isin(relevant_ids)
    ]
    print(f"{len(todo)} relevant rows to fetch (of {len(relevant_ids)} total relevant)", flush=True)

    fetched, failed = 0, 0
    for idx, row in todo.iterrows():
        scrip_code = row["BSE Code"]
        newsid = row["NEWSID"]
        attach_name = row["attachment_name"]
        ext = Path(attach_name).suffix or ".pdf"
        comp_dir = ARCHIVE_DIR / str(scrip_code)
        comp_dir.mkdir(parents=True, exist_ok=True)
        dest = comp_dir / f"{newsid}{ext}"

        if dest.exists() and dest.stat().st_size > 0:
            df.at[idx, "content_type"] = "pdf"
            df.at[idx, "local_path"] = str(dest.relative_to(_RAW_DIR.parent))
            fetched += 1
            continue

        if download_attachment(attach_name, dest):
            df.at[idx, "content_type"] = "pdf"
            df.at[idx, "local_path"] = str(dest.relative_to(_RAW_DIR.parent))
            fetched += 1
        else:
            failed += 1

        if fetched % 50 == 0 and fetched > 0:
            print(f"  fetched {fetched}/{len(todo)} so far ({failed} failed)...", flush=True)
            df.to_csv(INDEX_CSV, index=False)  # periodic checkpoint

    df.to_csv(INDEX_CSV, index=False)
    print(f"\nFetched {fetched}/{len(todo)}, failed {failed}", flush=True)


if __name__ == "__main__":
    run()
