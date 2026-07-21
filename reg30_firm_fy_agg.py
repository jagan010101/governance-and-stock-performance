"""
Reg 30 Firm-FY Aggregation (for 09_regression.ipynb integration)
====================================================================
Collapses data/processed/reg30_events.csv to one row per (BSE Code, FY),
producing the three regressor variables notebook 09's Table C needs:

  n_director_changes  — count of ALL director_change events that FY
                         (appointed + resigned + ceased + reappointed; the
                         variable name is "changes", not "departures", so
                         every event type counts)
  n_rating_changes    — count of credit_rating_change events whose direction
                         is a genuine upgrade or downgrade (reaffirmations,
                         first-time rating assignments, and withdrawals are
                         NOT "changes" in the rating level and are excluded —
                         see the direction crosswalk below)
  auditor_change_any  — binary 0/1 (not a count): auditor changes are too
                         sparse (~0.4/firm-FY) for a count to be meaningful
                         under fixed effects
  reg30_covered       — 1 for every row (all 100 sampled firms x all 3 FYs
                         get a row here) — carried through so a left-merge
                         in the notebook can distinguish "covered, zero
                         events" from "not in the Reg30 sample at all"

FY uses notebook 09's own convention (`'FY' + Q_FY.str[-2:]`), applied to
Q_FY computed the same way as reg30_firm_quarter.csv (to_indian_fiscal_quarter
on news_dt). Covers FY23/FY24/FY25 for completeness even though only
FY23/FY24 rows will find a match in 09's panel (FY25 CG_t rows don't exist
there — see INTEGRATION_NOTES.md).

Two known data-quality issues, resolved here (not dropped):

1. PHANTOM COMPANIES. reg30_events.csv carries 44 rows from 2 companies
   (Bharti Airtel 532454, Hindalco 500440) that were scraped during early
   pipeline testing before the 100-company stratified sample was finalized.
   Excluded so the base is exactly the 100-firm sample.

2. OUT-OF-SCHEMA ENUM VALUES. The LLM extractor's prompt constrained
   `event`/`reason`/`direction` to fixed enums, but on free-text disclosure
   language it doesn't always comply (measured: 3/2016 event, 169/1237
   director_change reason, 29/537 credit_rating_change direction rows off-
   schema — much of the "reason" drift specifically because "reason" is a
   departure concept the schema also asked for on appointment/reappointment
   events, where there usually isn't a "why" in the disclosure at all).
   Deterministic, documented crosswalks below map every observed value to
   a valid bucket — nothing is dropped, everything is auditable via the
   printed before/after tables.

Usage
-----
    python reg30_firm_fy_agg.py
"""
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EVENTS_CSV = BASE_DIR / "data" / "processed" / "reg30_events.csv"
COMPANY_SAMPLE_CSV = BASE_DIR / "data" / "processed" / "reg30_company_sample_100.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "reg30_firm_fy.csv"

FYS = ["FY23", "FY24", "FY25"]

# ── Crosswalk 1: `event` (director_change) ──────────────────────────────────
VALID_EVENT = {"appointed", "resigned", "ceased", "reappointed"}
EVENT_CROSSWALK = {
    "reappointment": "reappointed",   # noun-form typo for reappointed
    "continued": "reappointed",       # director continuing in office is
                                       # semantically closest to reappointed —
                                       # not a fresh appointment, not a departure
}


def crosswalk_event(raw):
    if raw in VALID_EVENT:
        return raw
    return EVENT_CROSSWALK.get(raw, "other")  # unreached given the 2 known values, but safe


# ── Crosswalk 2: `reason` (director_change) ──────────────────────────────────
VALID_REASON = {"personal", "term_ended", "regulatory_age", "health", "other_professional", "unstated", "other"}

# Ordered regex rules — first match wins. Applied only to departure events
# (resigned/ceased, post-event-crosswalk); appointment-side events go
# straight to 'unstated' below since "reason for appointment" isn't a
# concept the schema's categories (all departure-framed) can represent.
_REASON_RULES = [
    (re.compile(r"health|ill.?health|medical", re.I), "health"),
    (re.compile(r"statutory age|regulatory age|attained.{0,15}age|age of \d+|completion of \d+ years of age", re.I), "regulatory_age"),
    (re.compile(r"retir(e|ed|ement)|superannuat|completion of (tenure|term|second term)|expir(e|y)|end of.{0,20}term", re.I), "term_ended"),
    (re.compile(r"personal", re.I), "personal"),
    (re.compile(r"professional|other engagement|pre-?occupation|business commitment", re.I), "other_professional"),
]


def crosswalk_reason(raw, crosswalked_event):
    if raw in VALID_REASON:
        return raw
    if pd.isna(raw):
        return "unstated"
    if crosswalked_event in ("appointed", "reappointed"):
        return "unstated"
    for pattern, bucket in _REASON_RULES:
        if pattern.search(str(raw)):
            return bucket
    return "other"  # has real descriptive text, doesn't fit a named bucket


# ── Crosswalk 3: `direction` (credit_rating_change) ──────────────────────────
VALID_DIRECTION = {"upgrade", "downgrade", "reaffirmed"}
DIRECTION_CROSSWALK = {
    "affirmed": "reaffirmed",
    "re-affirmed": "reaffirmed",
    "reaffirmed/assigned": "reaffirmed",
    "reaffirmed and withdrawn": "reaffirmed",
    "reaffirm & outlook upgraded": "reaffirmed",  # rating LEVEL unchanged; only
                                                   # outlook moved — n_rating_changes
                                                   # tracks the rating, not outlook
    "assigned": "assigned",       # new bucket: first-time rating, no prior
                                   # rating to compare against — not a "change"
    "revised": "unclear",         # new bucket: direction not determinable from
                                   # the label alone; excluded from the count
                                   # rather than guessed
    "withdrawal": "withdrawn",    # new bucket: agency stopped rating the
                                   # instrument — not a rating-level change
}
# Buckets that do NOT count as a rating change for n_rating_changes:
NON_CHANGE_DIRECTIONS = {"reaffirmed", "assigned", "unclear", "withdrawn"}


def crosswalk_direction(raw):
    if raw in VALID_DIRECTION:
        return raw
    return DIRECTION_CROSSWALK.get(raw, "unclear")


def report_crosswalk(series, crosswalked, label):
    before = series.value_counts(dropna=False)
    changed = series.ne(crosswalked)
    print(f"\n{label} crosswalk: {changed.sum()} / {len(series)} rows remapped")
    if changed.sum():
        table = pd.DataFrame({"raw": series[changed], "crosswalked": crosswalked[changed]})
        summary = table.groupby(["raw", "crosswalked"]).size().reset_index(name="n").sort_values("n", ascending=False)
        print(summary.to_string(index=False))


def run():
    ev = pd.read_csv(EVENTS_CSV)
    ev = ev[ev["_provenance"] == "ok"]

    sample_ids = set(pd.read_csv(COMPANY_SAMPLE_CSV)["BSE Code"])
    phantom_mask = ~ev["BSE Code"].isin(sample_ids)
    print(f"Excluding {phantom_mask.sum()} rows from {ev.loc[phantom_mask, 'BSE Code'].nunique()} "
          f"phantom (non-sample) companies: {sorted(ev.loc[phantom_mask, 'BSE Code'].unique())}")
    ev = ev[~phantom_mask]

    ev["news_dt"] = pd.to_datetime(ev["news_dt"], format="mixed")
    ev["Q_FY"] = ev["news_dt"].apply(_to_indian_fiscal_quarter)
    ev["FY"] = "FY" + ev["Q_FY"].str[-2:]

    # --- director_change: event + reason crosswalks ---
    dc = ev[ev["event_class"] == "director_change"].copy()
    dc["event_cw"] = dc["event"].apply(crosswalk_event)
    report_crosswalk(dc["event"], dc["event_cw"], "director_change.event")
    dc["reason_cw"] = dc.apply(lambda r: crosswalk_reason(r["reason"], r["event_cw"]), axis=1)
    report_crosswalk(dc["reason"].fillna("<NA>"), dc["reason_cw"], "director_change.reason")

    n_director_changes = dc.groupby(["BSE Code", "FY"]).size().rename("n_director_changes")

    # --- credit_rating_change: direction crosswalk ---
    cr = ev[ev["event_class"] == "credit_rating_change"].copy()
    cr["direction_cw"] = cr["direction"].apply(crosswalk_direction)
    report_crosswalk(cr["direction"], cr["direction_cw"], "credit_rating_change.direction")

    real_changes = cr[~cr["direction_cw"].isin(NON_CHANGE_DIRECTIONS)]
    n_rating_changes = real_changes.groupby(["BSE Code", "FY"]).size().rename("n_rating_changes")

    # --- auditor_change: binary flag ---
    ac = ev[ev["event_class"] == "auditor_change"]
    auditor_change_any = (
        ac.groupby(["BSE Code", "FY"]).size().gt(0).astype(int).rename("auditor_change_any")
    )

    # --- full grid: 100 firms x FY23-FY25 ---
    companies = sorted(sample_ids)
    grid = pd.MultiIndex.from_product([companies, FYS], names=["BSE Code", "FY"]).to_frame(index=False)
    out = grid.merge(n_director_changes, on=["BSE Code", "FY"], how="left")
    out = out.merge(n_rating_changes, on=["BSE Code", "FY"], how="left")
    out = out.merge(auditor_change_any, on=["BSE Code", "FY"], how="left")
    for col in ["n_director_changes", "n_rating_changes", "auditor_change_any"]:
        out[col] = out[col].fillna(0).astype(int)
    out["reg30_covered"] = 1

    out = out.sort_values(["BSE Code", "FY"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # --- reconciliation ---
    print(f"\nWrote {len(out)} firm-FY rows ({len(companies)} firms x {len(FYS)} FYs) -> {OUTPUT_CSV}")
    print("\n--- Reconciliation ---")
    print(f"director_change events: {len(dc)} raw (after phantom exclusion) "
          f"-> n_director_changes sums to {out['n_director_changes'].sum()} (should match)")
    print(f"credit_rating_change events: {len(cr)} raw -> "
          f"{len(real_changes)} are genuine upgrade/downgrade -> "
          f"n_rating_changes sums to {out['n_rating_changes'].sum()} (should match {len(real_changes)})")
    print(f"  excluded as non-change: {cr['direction_cw'].isin(NON_CHANGE_DIRECTIONS).sum()} "
          f"({cr[cr['direction_cw'].isin(NON_CHANGE_DIRECTIONS)]['direction_cw'].value_counts().to_dict()})")
    print(f"auditor_change events: {len(ac)} raw -> "
          f"{out['auditor_change_any'].sum()} firm-FY cells flagged (some firm-FYs have >1 event)")

    print(f"\nFY breakdown (only FY23/FY24 will merge into 09's panel):")
    print(out.groupby("FY")[["n_director_changes", "n_rating_changes", "auditor_change_any"]].sum())


def _to_indian_fiscal_quarter(date):
    """Identical to 06_index_calculation.ipynb / reg30_firm_quarter_agg.py."""
    m, y = date.month, date.year
    if m <= 3:
        q, fy = 4, y
    elif m <= 6:
        q, fy = 1, y + 1
    elif m <= 9:
        q, fy = 2, y + 1
    else:
        q, fy = 3, y + 1
    return f"Q{q}FY{str(fy)[2:]}"


if __name__ == "__main__":
    run()
