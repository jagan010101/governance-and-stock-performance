"""
Reg 30 Firm-Quarter Aggregation
=================================
Rolls the event-level data in data/processed/reg30_events.csv up to one row
per (BSE Code, Q_FY), producing the four panel variables:

  id_resig_ct        — count of independent-director resignations
                        (event_class='director_change', event='resigned', is_independent=True)
  id_unstated_flag    — 1 if any independent director LEFT (resigned or ceased) that
                        quarter with reason='unstated' (a governance transparency signal:
                        an independent director departing without a disclosed reason),
                        else 0
  auditor_change_flag — 1 if any auditor_change event that quarter, else 0
  credit_downgrade_ct — count of credit_rating_change events with direction='downgrade'

Quarter assignment uses news_dt (the BSE filing/announcement date), not any
event effective_date — effective_date is frequently null/unstated in the
extracted data, while news_dt is always present, and it's the date the market
actually learns of the event, matching how Q_FY is assigned to returns data
elsewhere in this project (quarterly_returns_from_filing.csv keys off Filing_Date).

Uses the identical to_indian_fiscal_quarter() logic from 06_index_calculation.ipynb.

Produces a complete grid: every (BSE Code, Q_FY) combination actually covered
by the Reg 30 scrape (Q1FY23-Q4FY25, the 100-company sample) gets a row, with
zeros where no qualifying event occurred — not just the firm-quarters that had
one. That's what makes this safely left-joinable into the existing regression
panel without conflating "no event" with "not merged".

All 1270/1270 target rows have an extraction result (1267 ok, 3 llm_failed —
the 3 failures are excluded via the _provenance == 'ok' filter below).

Usage
-----
    python reg30_firm_quarter_agg.py
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EVENTS_CSV = BASE_DIR / "data" / "processed" / "reg30_events.csv"
INDEX_CSV = BASE_DIR / "data" / "processed" / "reg30_index.csv"
COMPANY_SAMPLE_CSV = BASE_DIR / "data" / "processed" / "reg30_company_sample_100.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "reg30_firm_quarter.csv"

START_DATE = "2022-04-01"
END_DATE = "2025-03-31"


def to_indian_fiscal_quarter(date):
    """Identical to the function in 06_index_calculation.ipynb."""
    m, y = date.month, date.year
    if m <= 3:       # Jan-Mar -> Q4 of same FY (e.g. Mar 2024 -> Q4FY24)
        q, fy = 4, y
    elif m <= 6:     # Apr-Jun -> Q1 (e.g. Jun 2023 -> Q1FY24)
        q, fy = 1, y + 1
    elif m <= 9:     # Jul-Sep -> Q2
        q, fy = 2, y + 1
    else:            # Oct-Dec -> Q3
        q, fy = 3, y + 1
    return f"Q{q}FY{str(fy)[2:]}"


def full_quarter_grid(start, end):
    """All Q_FY labels from start through end, in chronological order.
    Walking month-by-month is simplest since fiscal-quarter boundaries
    don't align uniformly with calendar-quarter starts under the mapping."""
    seen = []
    for dt in pd.date_range(start, end, freq="MS"):
        q = to_indian_fiscal_quarter(dt)
        if q not in seen:
            seen.append(q)
    return seen


def run():
    events = pd.read_csv(EVENTS_CSV)
    events = events[events["_provenance"] == "ok"]
    events["news_dt"] = pd.to_datetime(events["news_dt"], format="mixed")
    events["Q_FY"] = events["news_dt"].apply(to_indian_fiscal_quarter)

    dc = events[events["event_class"] == "director_change"]
    id_resig = dc[(dc["event"] == "resigned") & (dc["is_independent"] == True)]  # noqa: E712
    id_resig_ct = id_resig.groupby(["BSE Code", "Q_FY"]).size().rename("id_resig_ct")

    id_left = dc[dc["event"].isin(["resigned", "ceased"]) & (dc["is_independent"] == True)]  # noqa: E712
    id_unstated = id_left[id_left["reason"] == "unstated"]
    id_unstated_flag = (
        id_unstated.groupby(["BSE Code", "Q_FY"]).size().gt(0).astype(int).rename("id_unstated_flag")
    )

    ac = events[events["event_class"] == "auditor_change"]
    auditor_change_flag = (
        ac.groupby(["BSE Code", "Q_FY"]).size().gt(0).astype(int).rename("auditor_change_flag")
    )

    cr = events[events["event_class"] == "credit_rating_change"]
    downgrades = cr[cr["direction"] == "downgrade"]
    credit_downgrade_ct = downgrades.groupby(["BSE Code", "Q_FY"]).size().rename("credit_downgrade_ct")

    # Full grid: every company in the 100-sample x every quarter in the
    # scraped window, so downstream merges get real zeros, not missing rows.
    companies = pd.read_csv(COMPANY_SAMPLE_CSV)["BSE Code"].unique()
    quarters = full_quarter_grid(START_DATE, END_DATE)
    grid = pd.MultiIndex.from_product([companies, quarters], names=["BSE Code", "Q_FY"]).to_frame(index=False)

    out = grid.merge(id_resig_ct, on=["BSE Code", "Q_FY"], how="left")
    out = out.merge(id_unstated_flag, on=["BSE Code", "Q_FY"], how="left")
    out = out.merge(auditor_change_flag, on=["BSE Code", "Q_FY"], how="left")
    out = out.merge(credit_downgrade_ct, on=["BSE Code", "Q_FY"], how="left")

    for col in ["id_resig_ct", "id_unstated_flag", "auditor_change_flag", "credit_downgrade_ct"]:
        out[col] = out[col].fillna(0).astype(int)

    out = out.sort_values(["BSE Code", "Q_FY"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(out)} firm-quarter rows ({len(companies)} companies x {len(quarters)} quarters) -> {OUTPUT_CSV}")
    print(f"\nQuarters covered: {quarters}")
    print("\nNon-zero cells per variable:")
    for col in ["id_resig_ct", "id_unstated_flag", "auditor_change_flag", "credit_downgrade_ct"]:
        print(f"  {col}: {(out[col] > 0).sum()} / {len(out)} firm-quarters")
    print(f"\nTotals: id_resig={out['id_resig_ct'].sum()}, "
          f"id_unstated_flag={out['id_unstated_flag'].sum()}, "
          f"auditor_change_flag={out['auditor_change_flag'].sum()}, "
          f"credit_downgrade={out['credit_downgrade_ct'].sum()}")


if __name__ == "__main__":
    run()
