"""
XBRL Narrative Field Inventory — Phase 1 (schema discovery)
=============================================================
Parses a stratified sample of BSE Corporate Governance ("CG") XBRL filings
(both classic XML instances and the newer inline-XBRL-in-HTML format BSE
switched to) and enumerates every `in-bse-cg:*` element whose cleaned text
content exceeds 50 characters, excluding fields already consumed by:

  - cg_nlp_scorer.py's URL_TAGS (5 policy-weblink tags: D2-D5, A8)

Note: 00_cleaning_prowess.ipynb (as referenced in the task) only cleans
Prowess financial data — it contains no XBRL parsing. The numeric CG metrics
(B1, A1, etc.) come pre-computed from data/raw/numerical_indices.xlsx; the
extraction code that produced it is not present in this repo. Since numeric/
boolean facts never reach the 50-char threshold, this has no practical effect
on the exclusion set, but is flagged here for the record.

Scope: only "*_CG.xml" and "*_CG.html" filings (the Corporate Governance
Report) for the 247 companies in data/processed/industry_map.xlsx.
"*_ICGIG.html" filings are EXCLUDED — inspection showed they use a distinct
SEBI taxonomy (in-capmkt, "Integrated Governance"), not in-bse-cg at all, so
they are a different filing type rather than a format-variant of the CG
report. Flagged for the user rather than silently included or dropped.

Output: data/processed/xbrl_narrative_inventory.csv
"""
import re
import random
import zipfile
import statistics
from collections import defaultdict
from pathlib import Path

import warnings

import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "data" / "raw" / "governance_reports.zip"
INDUSTRY_MAP_PATH = BASE_DIR / "data" / "processed" / "industry_map.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "xbrl_narrative_inventory.csv"

SAMPLE_SIZE = 30
RANDOM_SEED = 42
LEN_THRESHOLD = 50
SNIPPET_MAXLEN = 300
N_EXAMPLES = 3

# ── Tags already consumed by cg_nlp_scorer.py (URL_TAGS) — exclude from inventory ──
ALREADY_CONSUMED_TAGS = {
    "DisclosureWebLinkOfCompanyAtWhichDetailsOfEstablishmentOfVigilMechanismOrWhistleBlowerPolicyIsPlaced",
    "DisclosureWebLinkOfCompanyAtWhichPolicyOnDealingWithRelatedPartyTransactionsIsPlaced",
    "DisclosureWebLinkOfCompanyAtWhichDetailsOfFamiliarizationProgrammesImpartedToIndependentDirectorsIsPlaced",
    "DisclosureWebLinkOfCompanyAtWhichCodeOfConductOfBoardOfDirectorsAndSeniorManagementPersonnelIsPlaced",
    "DisclosureWebLinkOfCompanyAtSeparateAuditedFinancialStatementsOfEachSubsidiaryOfTheListedEntityIsPlaced",
}


def clean_text(raw):
    """Strip HTML-escaped markup that XBRL TextBlocks wrap narrative text in."""
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)


def is_cg_namespace(tag):
    """True if a (possibly namespaced) lxml tag belongs to the in-bse-cg taxonomy
    (excluding the -roles and -types helper namespaces, same filter cg_nlp_scorer.py uses)."""
    if not isinstance(tag, str) or not tag.startswith("{"):
        return False
    ns = tag.split("}")[0][1:]
    return "in-bse-cg" in ns and "roles" not in ns and "types" not in ns


def extract_long_text_facts(file_bytes, is_html):
    """Return list of (local_tag_name, cleaned_text) for every in-bse-cg element
    whose cleaned text exceeds LEN_THRESHOLD chars."""
    if is_html:
        # BSE's newer CG.html filings embed raw `in-bse-cg:Concept` elements directly
        # in HTML (not standard ix:nonNumeric wrapping). libxml2's HTML parser
        # lowercases tag names, which would break exact-name matching, so we parse
        # as XML-with-recovery instead — this preserves case and namespaces.
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.fromstring(file_bytes, parser=parser)
    else:
        parser = etree.XMLParser(huge_tree=True)
        root = etree.fromstring(file_bytes, parser=parser)

    if root is None:
        raise ValueError("empty parse tree")

    facts = []
    for el in root.iter():
        if not is_cg_namespace(el.tag):
            continue
        local = etree.QName(el).localname
        if local in ALREADY_CONSUMED_TAGS:
            continue
        text = clean_text(el.text)
        if len(text) > LEN_THRESHOLD:
            facts.append((local, text))
    return facts


def build_candidate_list():
    """List (zip_name, bse_code, industry, approx_year, fmt) for every *_CG.xml /
    *_CG.html filing belonging to a company in industry_map.xlsx."""
    industry_map = pd.read_excel(INDUSTRY_MAP_PATH)
    code_to_industry = dict(
        zip(industry_map["BSE Code"].astype(int).astype(str), industry_map["Industry"])
    )
    universe = set(code_to_industry)

    z = zipfile.ZipFile(ZIP_PATH)
    names = sorted(set(z.namelist()))

    candidates = []
    year_re = re.compile(r"20\d{2}")
    for n in names:
        if n.endswith("/"):
            continue
        if not (n.endswith("_CG.xml") or n.endswith("_CG.html")):
            continue
        parts = n.split("/")
        code = parts[1].split("_")[0]
        if code not in universe:
            continue
        base = parts[-1]
        year_candidates = [int(y) for y in year_re.findall(base) if 2015 <= int(y) <= 2027]
        approx_year = year_candidates[0] if year_candidates else None
        fmt = "xml" if n.endswith(".xml") else "html"
        candidates.append(
            {
                "zip_name": n,
                "bse_code": code,
                "industry": code_to_industry[code],
                "approx_year": approx_year,
                "fmt": fmt,
            }
        )
    return candidates, z


def _stratified_pick(pool, n, seed):
    """Spread n picks from `pool` across industries, filing years and companies."""
    rng = random.Random(seed)

    by_industry = defaultdict(list)
    for c in pool:
        by_industry[c["industry"]].append(c)

    # Rank industries by how many candidate filings they have, then interleave
    # the biggest sectors with a shuffled slice of the long tail for diversity.
    industries = sorted(by_industry, key=lambda k: -len(by_industry[k]))
    top = industries[:12]
    tail = industries[12:]
    rng.shuffle(tail)
    chosen_industries = top + tail[: max(0, 20 - len(top))]

    picked = []
    used_codes = set()
    idx = 0
    stall = 0
    while len(picked) < n and chosen_industries and stall < len(chosen_industries) * 2:
        industry = chosen_industries[idx % len(chosen_industries)]
        already = {p["zip_name"] for p in picked}
        candidates_here = [c for c in by_industry[industry] if c["zip_name"] not in already]
        if not candidates_here:
            chosen_industries.remove(industry)
            stall += 1
            if not chosen_industries:
                break
            continue
        stall = 0
        candidates_here.sort(key=lambda c: (c["bse_code"] in used_codes, c["approx_year"] or 0))
        # bias pick toward alternating early/late year within this industry's pool
        half = len(candidates_here) // 2
        choice_pool = candidates_here[:half] if len(picked) % 2 == 0 and half else candidates_here
        chosen = rng.choice(choice_pool)
        picked.append(chosen)
        used_codes.add(chosen["bse_code"])
        idx += 1

    return picked


def stratified_sample(candidates, n=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Spread the sample across industries, filing years, and both formats
    (xml / html) — split roughly evenly between formats so newer HTML-format
    filings (which carry additional/updated taxonomy tags) aren't crowded out
    by the more numerous older XML filings."""
    xml_pool = [c for c in candidates if c["fmt"] == "xml"]
    html_pool = [c for c in candidates if c["fmt"] == "html"]

    n_html = min(len(html_pool), n // 2)
    n_xml = n - n_html

    picked = _stratified_pick(xml_pool, n_xml, seed) + _stratified_pick(html_pool, n_html, seed + 1)
    return picked


def main():
    print(f"Building candidate list from {ZIP_PATH.name} ...")
    candidates, z = build_candidate_list()
    print(f"  {len(candidates)} CG.xml/CG.html filings across "
          f"{len({c['bse_code'] for c in candidates})} companies, "
          f"{len({c['industry'] for c in candidates})} industries")

    sample = stratified_sample(candidates)
    print(f"\nStratified sample: {len(sample)} files")
    print(f"  Industries covered : {len({s['industry'] for s in sample})}")
    print(f"  Years covered      : {sorted({s['approx_year'] for s in sample if s['approx_year']})}")
    print(f"  Formats            : xml={sum(1 for s in sample if s['fmt']=='xml')}, "
          f"html={sum(1 for s in sample if s['fmt']=='html')}")

    tag_stats = defaultdict(lambda: {
        "files_present": set(),
        "total_occurrences": 0,
        "lengths": [],
        "examples": [],
        "formats": set(),
        "industries": set(),
    })

    parse_failed = []
    for c in sample:
        try:
            file_bytes = z.read(c["zip_name"])
            facts = extract_long_text_facts(file_bytes, is_html=(c["fmt"] == "html"))
        except Exception as e:
            parse_failed.append((c["zip_name"], str(e)))
            continue

        seen_tags_this_file = set()
        for local_tag, text in facts:
            st = tag_stats[local_tag]
            seen_tags_this_file.add(local_tag)
            st["total_occurrences"] += 1
            st["lengths"].append(len(text))
            st["formats"].add(c["fmt"])
            st["industries"].add(c["industry"])
            if len(st["examples"]) < N_EXAMPLES:
                snippet = text[:SNIPPET_MAXLEN] + ("…" if len(text) > SNIPPET_MAXLEN else "")
                st["examples"].append(snippet)
        for t in seen_tags_this_file:
            tag_stats[t]["files_present"].add(c["zip_name"])

    if parse_failed:
        print(f"\nParse failures: {len(parse_failed)}")
        for name, err in parse_failed:
            print(f"  {name}: {err}")

    rows = []
    for tag, st in tag_stats.items():
        examples = st["examples"] + [""] * (N_EXAMPLES - len(st["examples"]))
        rows.append({
            "tag_name": tag,
            "files_present": len(st["files_present"]),
            "pct_files_present": round(100 * len(st["files_present"]) / len(sample), 1),
            "total_occurrences": st["total_occurrences"],
            "median_length_chars": round(statistics.median(st["lengths"]), 1),
            "source_formats": "+".join(sorted(st["formats"])),
            "n_industries_seen": len(st["industries"]),
            "example_1": examples[0],
            "example_2": examples[1],
            "example_3": examples[2],
        })

    out_df = pd.DataFrame(rows).sort_values(
        ["files_present", "total_occurrences"], ascending=False
    ).reset_index(drop=True)

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out_df)} distinct tags → {OUTPUT_PATH}")
    print(out_df[["tag_name", "files_present", "pct_files_present", "median_length_chars"]].to_string())


if __name__ == "__main__":
    main()
