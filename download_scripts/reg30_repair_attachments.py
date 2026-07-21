"""
Reg 30 Attachment Repair Pass
===============================
Retries attachment downloads for reg30_index.csv rows where content_type is
'none' but an attachment_name was recorded (i.e. BSE had the file, but the
download failed at scrape time). Updates content_type/local_path in place on
success — does not touch rows with no attachment to begin with.

Useful both as a one-off cleanup (e.g. after fixing a bug in the scraper's
request headers) and as a general "top up whatever didn't make it" pass for
a corpus this size, since some fraction of BSE attachment fetches fail for
reasons outside our control even with retries.

Usage
-----
    python download_scripts/reg30_repair_attachments.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg30_scraper import download_attachment, _RAW_DIR  # noqa: E402  (reuse, don't duplicate)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_CSV = BASE_DIR / "data" / "processed" / "reg30_index.csv"


def run():
    df = pd.read_csv(INDEX_CSV)
    needs_retry = df[(df["content_type"] == "none") & df["attachment_name"].notna() & (df["attachment_name"] != "")]
    print(f"{len(needs_retry)} rows need an attachment retry")

    recovered = 0
    for idx, row in needs_retry.iterrows():
        scrip_code = row["BSE Code"]
        newsid = row["NEWSID"]
        attach_name = row["attachment_name"]
        ext = Path(attach_name).suffix or ".pdf"
        comp_dir = _RAW_DIR / "reg30_announcements" / str(scrip_code)
        comp_dir.mkdir(parents=True, exist_ok=True)
        dest = comp_dir / f"{newsid}{ext}"

        if download_attachment(attach_name, dest):
            df.at[idx, "content_type"] = "pdf"
            df.at[idx, "local_path"] = str(dest.relative_to(_RAW_DIR.parent))
            recovered += 1
            if recovered % 25 == 0:
                print(f"  recovered {recovered}/{len(needs_retry)} so far...")
                df.to_csv(INDEX_CSV, index=False)  # checkpoint periodically

    df.to_csv(INDEX_CSV, index=False)
    print(f"\nRecovered {recovered}/{len(needs_retry)} attachments")
    print(f"Still missing: {len(needs_retry) - recovered}")


if __name__ == "__main__":
    run()
