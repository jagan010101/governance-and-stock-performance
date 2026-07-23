"""Snapshot daily adjusted close prices for the study universe into a versioned parquet.

Why this exists: the pipeline notebooks originally called yf.download(...) live on every
run with end=today. That makes results irreproducible three separate ways — (1) the price
history itself moves (Yahoo re-adjusts the whole series whenever a dividend/split lands),
(2) delisted/renamed tickers silently vanish from later fetches, and (3) transient Yahoo
failures silently shrink the sample through the <50%-missing validity filter (this
actually happened: JINDALSTEL and HINDCOPPER were dropped from the original run by a
transient download failure, not by any property of their data).

This script fetches once, retries failures single-threaded (the usual failure mode is
yfinance's local sqlite cache lock, which a serial retry clears), and writes:

    data/processed/prices_daily.parquet       wide Close-price panel, tickers + ^CRSLDX
    data/processed/prices_daily_meta.json     fetch provenance (date, range, failures)

The committed snapshot pins the study dataset. Notebooks read the parquet; none of them
need network access or yfinance. Re-running this script with --refresh consciously
re-pins the dataset to a new fetch — expect small drift in adjusted prices and possibly
newly-delisted tickers; downstream numbers will move accordingly.

Usage:
    python download_scripts/fetch_market_data.py            # refuses if snapshot exists
    python download_scripts/fetch_market_data.py --refresh  # re-pin to a fresh fetch
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / 'data' / 'processed'

START = '2019-07-01'
INDEX_TICKER = '^CRSLDX'   # Nifty 500
N_RETRIES = 3


def fetch(start, end):
    imap = pd.read_excel(PROC / 'industry_map.xlsx')[['NSE Symbol']].dropna()
    tickers = [f'{s}.NS' for s in imap['NSE Symbol']] + [INDEX_TICKER]
    print(f'Fetching {len(tickers)} tickers, {start} -> {end}')

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    px = px.reindex(columns=tickers)
    failed = [t for t in tickers if px[t].notna().sum() == 0]
    print(f'First pass: {len(tickers) - len(failed)} ok, {len(failed)} empty')

    for attempt in range(1, N_RETRIES + 1):
        if not failed:
            break
        time.sleep(2)
        retry = yf.download(failed, start=start, end=end, auto_adjust=True,
                            progress=False, threads=False)['Close']
        if isinstance(retry, pd.Series):
            retry = retry.to_frame(failed[0])
        for t in list(failed):
            if t in retry.columns and retry[t].notna().sum() > 0:
                px[t] = retry[t].reindex(px.index)
                failed.remove(t)
        print(f'Retry {attempt}: {len(failed)} still empty')

    if failed:
        print(f'Unavailable from Yahoo at fetch time (delisted/renamed): {sorted(failed)}')
    return px, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=None, help='exclusive end date; default = tomorrow')
    ap.add_argument('--refresh', action='store_true',
                    help='overwrite an existing snapshot (re-pins the study dataset)')
    args = ap.parse_args()

    out_px = PROC / 'prices_daily.parquet'
    out_meta = PROC / 'prices_daily_meta.json'
    if out_px.exists() and not args.refresh:
        sys.exit(f'{out_px} already exists — the snapshot pins the study dataset.\n'
                 'Pass --refresh only if you intend to re-pin it (downstream numbers will move).')

    end = args.end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    px, failed = fetch(args.start, end)

    px.to_parquet(out_px)
    meta = {
        'fetched': pd.Timestamp.now().isoformat(timespec='seconds'),
        'start': args.start, 'end_exclusive': end,
        'first_date': str(px.index.min().date()), 'last_date': str(px.index.max().date()),
        'n_tickers': int(px.shape[1]), 'n_rows': int(px.shape[0]),
        'failed': sorted(failed),
        'yfinance_version': yf.__version__,
    }
    out_meta.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'Saved {out_px.name} ({px.shape[0]} rows x {px.shape[1]} tickers, '
          f'{out_px.stat().st_size / 1e6:.1f} MB) + {out_meta.name}')


if __name__ == '__main__':
    main()
