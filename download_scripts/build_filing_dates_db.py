"""
build_filing_dates_db.py
────────────────────────
Ground truth = actual files on disk in governance_reports/.
Each physical file → exactly one row.

Steps:
  1. Walk governance_reports/ → all .xml / .html files that exist
  2. Parse BSE code + filing date from filename
  3. Join download_log on filename to get quarter label
     - Duplicates (same file, multiple log quarters): keep closest quarter to filing date
     - Missing (file on disk, not in log): infer quarter from filing date
  4. Enrich with company name (industry_map → matched_companies → log name)

Output: filing_dates_db.csv
  BSE_Code, Company, Sector, Q_FY, Quarter_Period, Quarter_End_Date,
  Filing_Date, Source_File
"""

import re
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import calendar

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
RAW_DIR  = DATA_DIR / 'raw'
PROC_DIR = DATA_DIR / 'processed'
GOV_DIR  = RAW_DIR / 'governance_reports'
LOG_CSV  = DATA_DIR / 'logs' / 'cg_download_log.csv'
OUT_CSV  = PROC_DIR / 'filing_dates_db.csv'

if OUT_CSV.exists():
    print(f"{OUT_CSV} already exists — skipping. Delete it to rebuild.")
    sys.exit(0)

# ── Company name lookup (priority: industry_map > matched_companies > log) ────
ind_map = pd.read_excel(PROC_DIR / 'industry_map.xlsx')[
    ['BSE Code', 'BSE Name', 'Sector']
]
ind_map['BSE Code'] = pd.to_numeric(ind_map['BSE Code'], errors='coerce')
ind_map = ind_map.dropna(subset=['BSE Code']).drop_duplicates('BSE Code')
code_to_name   = dict(zip(ind_map['BSE Code'].astype(int), ind_map['BSE Name']))
code_to_sector = dict(zip(ind_map['BSE Code'].astype(int), ind_map['Sector']))

mc = pd.read_excel(PROC_DIR / 'matched_companies.xlsx')[['BSE Code', 'BSE Name']]
mc['BSE Code'] = pd.to_numeric(mc['BSE Code'], errors='coerce')
mc = mc.dropna(subset=['BSE Code']).drop_duplicates('BSE Code')
for _, row in mc.iterrows():
    code = int(row['BSE Code'])
    if code not in code_to_name:
        code_to_name[code] = row['BSE Name']

# ── Quarter helpers ───────────────────────────────────────────────────────────
MONTH_TO_QUARTER = {
    'June':      ('Q1', +1),
    'September': ('Q2', +1),
    'December':  ('Q3', +1),
    'March':     ('Q4',  0),
}
QUARTER_END_DATE = {
    'June':      lambda y: date(y, 6, 30),
    'September': lambda y: date(y, 9, 30),
    'December':  lambda y: date(y, 12, 31),
    'March':     lambda y: date(y, 3, 31),
}

def quarter_label(month_name: str, year: int) -> str | None:
    if month_name not in MONTH_TO_QUARTER:
        return None
    q, offset = MONTH_TO_QUARTER[month_name]
    fy = (year + offset) % 100
    return f"{q}FY{fy:02d}"

def parse_quarter_str(qtr_raw: str):
    """Parse 'June 2024' → (q_fy, quarter_end_date). Returns (None, None) if invalid."""
    parts = str(qtr_raw).strip().split()
    if len(parts) != 2:
        return None, None
    month_name, yr_str = parts
    if month_name not in MONTH_TO_QUARTER:
        return None, None
    try:
        yr = int(yr_str)
        if not (2015 <= yr <= 2030):
            return None, None
        return quarter_label(month_name, yr), QUARTER_END_DATE[month_name](yr)
    except ValueError:
        return None, None

def infer_quarter_from_date(filing_date: date):
    """
    Infer Q_FY from filing date: find the quarter whose end date is within
    30–210 days before the filing (typical SEBI window is 30–120 days).
    Among candidates, pick the one with smallest absolute distance.
    """
    candidates = []
    for yr in range(filing_date.year - 1, filing_date.year + 2):
        for month_name, qe_fn in QUARTER_END_DATE.items():
            qe = qe_fn(yr)
            delay = (filing_date - qe).days
            if 10 <= delay <= 210:
                candidates.append((abs(delay - 90), month_name, yr, qe))
    if not candidates:
        return None, None
    candidates.sort()
    _, month_name, yr, qe = candidates[0]
    return quarter_label(month_name, yr), qe

# ── Timestamp parser ──────────────────────────────────────────────────────────
def parse_filing_datetime(ts: str) -> datetime | None:
    """
    Parse a variable-length BSE filing timestamp with no consistent zero-padding.
    Tries every plausible year match in the string (left-to-right) until a
    fully valid datetime can be constructed.
    """
    # Scan every position for a valid year (allows overlapping matches)
    year_positions = [
        (i, int(ts[i:i+4]))
        for i in range(len(ts) - 3)
        if ts[i:i+4].isdigit() and 2015 <= int(ts[i:i+4]) <= 2029
    ]
    for yi, year in year_positions:
        dm   = ts[:yi]
        rest = ts[yi + 4:]

        if not (2 <= len(dm) <= 4):   # dm must hold exactly D+M digits
            continue

        # ── Parse day and month ───────────────────────────────────────────────
        day = month = None
        if len(dm) == 2:
            day, month = int(dm[0]), int(dm[1])
        elif len(dm) == 3:
            c1d, c1m = int(dm[:2]), int(dm[2])
            c2d, c2m = int(dm[0]), int(dm[1:])
            if 1 <= c1d <= 31 and 1 <= c1m <= 12:
                day, month = c1d, c1m
            elif 1 <= c2d <= 31 and 1 <= c2m <= 12:
                day, month = c2d, c2m
        elif len(dm) == 4:
            day, month = int(dm[:2]), int(dm[2:])

        if day is None or not (1 <= day <= 31 and 1 <= month <= 12):
            continue

        # ── Parse time (H, M, S) — no zero-padding assumed ───────────────────
        h = mn = s = 0
        found = False
        for h_len in [2, 1]:
            if len(rest) < h_len:
                continue
            try:
                hh = int(rest[:h_len])
            except ValueError:
                continue
            if not 0 <= hh <= 23:
                continue
            rem = rest[h_len:]
            if not rem:
                h = hh; found = True; break
            for m_len in [2, 1]:
                if len(rem) < m_len:
                    continue
                try:
                    mm = int(rem[:m_len])
                except ValueError:
                    continue
                if not 0 <= mm <= 59:
                    continue
                srem = rem[m_len:]
                if not srem:
                    h, mn = hh, mm; found = True; break
                if len(srem) <= 2:
                    try:
                        ss = int(srem)
                    except ValueError:
                        continue
                    if 0 <= ss <= 59:
                        h, mn, s = hh, mm, ss; found = True; break
            if found:
                break

        try:
            return datetime(year, month, day, h, mn, s)
        except ValueError:
            continue   # try next year match

    return None

def parse_bse_from_filename(fname: str) -> int | None:
    stem  = Path(fname).stem
    parts = stem.split('_')
    for p in parts:
        if p.isdigit() and len(p) >= 5:
            return int(p)
    return None

def parse_ts_from_filename(fname: str) -> str | None:
    stem  = Path(fname).stem
    parts = stem.split('_')
    for p in parts:
        if re.search(r'(201[5-9]|202[0-9])', p) and len(p) >= 8:
            return p
    return None

# ── Step 1: Walk governance_reports → all actual files ────────────────────────
print('Scanning governance_reports/ for actual files...')
disk_files = []
for subdir in sorted(GOV_DIR.iterdir()):
    if not subdir.is_dir() or subdir.name.startswith('.'):
        continue
    for fpath in subdir.iterdir():
        if fpath.suffix in ('.xml', '.html') and not fpath.name.startswith('.'):
            disk_files.append(fpath.name)

print(f'  Files on disk : {len(disk_files):,}')
disk_set = set(disk_files)

# ── Step 2: Parse BSE code + filing date from each filename ───────────────────
records = []
unparsed = []
for fname in disk_files:
    bse  = parse_bse_from_filename(fname)
    ts   = parse_ts_from_filename(fname)
    dt   = parse_filing_datetime(ts) if ts else None
    if bse and dt:
        records.append({
            'filename':    fname,
            'BSE_Code':    bse,
            'Filing_Date': dt.date(),
        })
    else:
        unparsed.append(fname)

print(f'  Date parsed   : {len(records):,}')
print(f'  Unparseable   : {len(unparsed)} — {unparsed[:5]}')

df_disk = pd.DataFrame(records)

# ── Step 3: Load download_log — keep only entries whose file is on disk ────────
log = pd.read_csv(LOG_CSV)
log = log[log['status'] == 'downloaded'].copy()
log['filename'] = log['filename'].astype(str).str.strip()

# Drop log entries whose file does not exist on disk
log = log[log['filename'].isin(disk_set)].copy()
print(f'\nLog entries referencing real files: {len(log):,}')

# Parse quarter from log
log[['Q_FY', 'Quarter_End_Date']] = log['quarter'].apply(
    lambda q: pd.Series(parse_quarter_str(q))
)
log = log.dropna(subset=['Q_FY'])  # drop rows with invalid quarter string
log = log.rename(columns={'quarter': 'Quarter_Period'})
print(f'Log entries with valid quarter    : {len(log):,}')

# Build filename → company_name from log (last-resort name fallback)
log_name_map = (log[['scrip_code', 'company_name']]
                .assign(scrip_code=lambda x: pd.to_numeric(x['scrip_code'], errors='coerce'))
                .dropna(subset=['scrip_code'])
                .drop_duplicates('scrip_code')
                .set_index('scrip_code')['company_name'].to_dict())

# When same filename appears in log multiple times (different quarters),
# keep the entry whose quarter_end_date is closest to the filing date.
log_best = (
    df_disk[['filename', 'Filing_Date']]
    .merge(log[['filename', 'Q_FY', 'Quarter_Period', 'Quarter_End_Date']], on='filename', how='left')
)
log_best['Quarter_End_Date'] = pd.to_datetime(log_best['Quarter_End_Date'])
log_best['Filing_Date']      = pd.to_datetime(log_best['Filing_Date'])
log_best['_delay'] = (log_best['Filing_Date'] - log_best['Quarter_End_Date']).dt.days

# Keep only rows where delay is plausible (10–210 days)
log_best = log_best[log_best['_delay'].between(10, 210)]

# Among valid candidates per file, keep the one with delay closest to 90 days
log_best['_dist'] = (log_best['_delay'] - 90).abs()
log_best = log_best.sort_values('_dist').drop_duplicates('filename', keep='first')
log_best = log_best[['filename', 'Q_FY', 'Quarter_Period', 'Quarter_End_Date']].copy()

print(f'Files matched to log quarter      : {len(log_best):,}')

# ── Step 4: Merge back to disk records; infer quarter for unmatched files ──────
df = df_disk.merge(log_best, on='filename', how='left')

# Infer quarter for files that had no log match
no_match = df['Q_FY'].isna()
print(f'Files needing quarter inference   : {no_match.sum()}')

inferred = df.loc[no_match, 'Filing_Date'].apply(
    lambda d: pd.Series(infer_quarter_from_date(d))
)
inferred.columns = ['Q_FY_inf', 'Quarter_End_Date_inf']
df.loc[no_match, 'Q_FY']             = inferred['Q_FY_inf'].values
df.loc[no_match, 'Quarter_End_Date'] = pd.to_datetime(inferred['Quarter_End_Date_inf'].values)

# ── Step 5: Enrich ────────────────────────────────────────────────────────────
df['Company'] = df['BSE_Code'].map(code_to_name)
df['Sector']  = df['BSE_Code'].map(code_to_sector)

# Fill blanks from log name map
blank = df['Company'].isna()
df.loc[blank, 'Company'] = df.loc[blank, 'BSE_Code'].map(log_name_map)

df['Filing_Date']       = pd.to_datetime(df['Filing_Date'])
df['Quarter_End_Date']  = pd.to_datetime(df['Quarter_End_Date'])

# ── Final output ──────────────────────────────────────────────────────────────
cols = ['BSE_Code', 'Company', 'Sector',
        'Q_FY', 'Quarter_Period', 'Quarter_End_Date',
        'Filing_Date', 'filename']
df = df[cols].rename(columns={'filename': 'Source_File'})
df = df.sort_values(['BSE_Code', 'Q_FY']).reset_index(drop=True)

df.to_csv(OUT_CSV, index=False)

print(f'\n{"─"*60}')
print(f'Saved → {OUT_CSV}')
print(f'Rows (= files on disk)  : {len(df):,}')
print(f'Firms                   : {df["BSE_Code"].nunique():,}')
print(f'Blank company names     : {df["Company"].isna().sum()}')
print(f'Rows with no Q_FY       : {df["Q_FY"].isna().sum()}')
print(f'Quarters covered: {sorted(df["Q_FY"].dropna().unique())}')
print(f'\nSample (Exide 500086):')
print(df[df['BSE_Code'] == 500086].to_string(index=False))
