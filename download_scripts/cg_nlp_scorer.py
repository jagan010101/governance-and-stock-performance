"""
CG NLP Scorer — Scores A8, D2-D5 using Ollama LLM
Step 1: Extract policy URLs from CG XML files
Step 2: Fetch policy text from URLs
Step 3: Score using Ollama

Usage:
  # Setup Ollama first (one-time):
  brew install ollama
  ollama serve                    # start server (keep running in separate terminal)
  ollama pull llama3.1:8b         # download model (~4.7GB, fits 16GB RAM)

  # Run scorer:
  python download_scripts/cg_nlp_scorer.py <root_folder> [--model llama3.1:8b] [--output nlp_scores.csv]
  
  # Merge with main scores:
  python download_scripts/cg_nlp_scorer.py <root_folder> --merge scores.xlsx --output scores_final.xlsx
"""
import os, sys, re, json, time, csv
from lxml import etree
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

# ── URL TAGS in CG XML mapped to metrics ──
URL_TAGS = {
    'D2': {
        'tag': 'DisclosureWebLinkOfCompanyAtWhichDetailsOfEstablishmentOfVigilMechanismOrWhistleBlowerPolicyIsPlaced',
        'name': 'Whistleblower / Vigil Mechanism Policy',
    },
    'D4': {
        'tag': 'DisclosureWebLinkOfCompanyAtWhichPolicyOnDealingWithRelatedPartyTransactionsIsPlaced',
        'name': 'Related Party Transactions Policy',
    },
    'D5': {
        'tag': 'DisclosureWebLinkOfCompanyAtWhichDetailsOfFamiliarizationProgrammesImpartedToIndependentDirectorsIsPlaced',
        'name': 'Board Evaluation / Familiarization Policy',
    },
    # D3 (insider trading) - usually under code of conduct or SEBI PIT regulations
    'D3': {
        'tag': 'DisclosureWebLinkOfCompanyAtWhichCodeOfConductOfBoardOfDirectorsAndSeniorManagementPersonnelIsPlaced',
        'name': 'Code of Conduct / Insider Trading Policy',
    },
    # A8 - Auditor's report is in Annual Report
    'A8': {
        'tag': 'DisclosureWebLinkOfCompanyAtSeparateAuditedFinancialStatementsOfEachSubsidiaryOfTheListedEntityIsPlaced',
        'name': "Auditor's Report / Annual Report",
    },
}

# ── SCORING RUBRICS ──
RUBRICS = {
    'D2': """Score the Whistleblower / Vigil Mechanism Policy on a scale of 0 to 1 based on these criteria. 
Award points for each criterion met (each worth 0.2):
1. Anonymity protection for complainants
2. Anti-retaliation clause protecting whistleblowers
3. Clear investigation process with timelines
4. Escalation matrix (who to report to, including direct access to Audit Committee chair)
5. Coverage scope (covers employees, directors, AND stakeholders/vendors)

Respond ONLY with a JSON object: {"score": <float 0-1>, "criteria_met": [<list of criteria numbers met>], "brief_reason": "<1 sentence>"}""",

    'D3': """Score the Insider Trading Policy (Code of Conduct for Prevention of Insider Trading) on a scale of 0 to 1 based on these criteria.
Award points for each criterion met (each worth 0.2):
1. Clear definition of "insider" and "unpublished price sensitive information (UPSI)"
2. Trading window restrictions with specific closure periods
3. Pre-clearance requirements for designated persons
4. Disclosure obligations (initial, continual, and transaction-based)
5. Penalties and consequences for violations clearly stated

Respond ONLY with a JSON object: {"score": <float 0-1>, "criteria_met": [<list of criteria numbers met>], "brief_reason": "<1 sentence>"}""",

    'D4': """Score the Related Party Transactions (RPT) Policy on a scale of 0 to 1 based on these criteria.
Award points for each criterion met (each worth 0.2):
1. Clear definition of related parties and material RPTs with thresholds
2. Prior Audit Committee approval requirement for all RPTs
3. Shareholder approval requirement for material RPTs
4. Arm's length pricing requirement with verification mechanism
5. Disclosure and review mechanism (periodic reporting to Board/Audit Committee)

Respond ONLY with a JSON object: {"score": <float 0-1>, "criteria_met": [<list of criteria numbers met>], "brief_reason": "<1 sentence>"}""",

    'D5': """Score the Board Evaluation Policy on a scale of 0 to 1 based on these criteria.
Award points for each criterion met (each worth 0.2):
1. Evaluation covers Board, committees, AND individual directors separately
2. Specific evaluation criteria defined (attendance, participation, expertise, independence)
3. Independent directors are evaluated on additional independence-specific criteria
4. Process includes self-assessment AND peer review components
5. Action plan / follow-up mechanism for evaluation outcomes

Respond ONLY with a JSON object: {"score": <float 0-1>, "criteria_met": [<list of criteria numbers met>], "brief_reason": "<1 sentence>"}""",

    'A8': """Score the Auditor's Report / Opinion on a scale of 0 to 1 based on these criteria:
- 1.0 = Unqualified opinion with no emphasis of matter
- 0.8 = Unqualified opinion with emphasis of matter paragraph(s)
- 0.6 = Qualified opinion (minor qualifications)
- 0.3 = Qualified opinion (significant qualifications)
- 0.0 = Adverse opinion or Disclaimer of opinion

If you cannot determine the auditor's opinion from the text, respond with score -1.

Respond ONLY with a JSON object: {"score": <float 0-1 or -1>, "opinion_type": "<unqualified/qualified/adverse/disclaimer/unknown>", "brief_reason": "<1 sentence>"}""",
}


# ── EXTRACT URLs FROM CG XML ──
def extract_urls_from_xml(filepath):
    """Extract policy URLs from a CG XML file."""
    try:
        root = etree.parse(filepath).getroot()
        ns = {}
        for p, u in root.nsmap.items():
            if 'in-bse-cg' in u and 'roles' not in u and 'types' not in u:
                ns = {'cg': u}; break
        if not ns: return None

        # Get company info
        def val(tag):
            els = root.findall(f'.//cg:{tag}', ns)
            return els[0].text if els else None

        info = {
            'company_name': val('NameOfTheCompany'),
            'scrip_code': val('ScripCode'),
            'symbol': val('Symbol'),
        }

        urls = {}
        for metric_id, cfg in URL_TAGS.items():
            els = root.findall(f'.//cg:{cfg["tag"]}', ns)
            if els and els[0].text:
                url = els[0].text.strip()
                # Clean up: some have multiple URLs separated by newlines
                url = url.split('\n')[0].strip()
                if url.startswith('http'):
                    urls[metric_id] = url

        return {**info, 'urls': urls}
    except:
        return None


# ── FETCH PAGE TEXT ──
def fetch_page_text(url, timeout=15):
    """Fetch and extract text from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
        resp.raise_for_status()

        ct = resp.headers.get('Content-Type', '')
        if 'pdf' in ct:
            return None  # Can't parse PDF inline — would need separate handling

        soup = BeautifulSoup(resp.text, 'html.parser')
        # Remove script/style
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        # Truncate to ~3000 words to fit context
        words = text.split()
        if len(words) > 3000:
            text = ' '.join(words[:3000])
        return text
    except Exception as e:
        return None


# ── OLLAMA SCORING ──
def score_with_ollama(text, metric_id, model=DEFAULT_MODEL):
    """Score policy text using Ollama."""
    if not text or len(text.strip()) < 50:
        return None

    rubric = RUBRICS.get(metric_id)
    if not rubric: return None

    prompt = f"""You are a corporate governance expert evaluating Indian listed companies' policies.

Below is the text extracted from a company's {URL_TAGS[metric_id]['name']}. 

--- POLICY TEXT ---
{text[:4000]}
--- END ---

{rubric}"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 200}
        }, timeout=120)
        resp.raise_for_status()
        output = resp.json().get('response', '')

        # Parse JSON from response
        # Try to find JSON in output
        match = re.search(r'\{[^}]+\}', output)
        if match:
            result = json.loads(match.group())
            return result
        return None
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


def check_ollama(model):
    """Check if Ollama is running and model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in resp.json().get('models', [])]
        if not any(model in m for m in models):
            print(f"Model '{model}' not found. Available: {models}")
            print(f"Run: ollama pull {model}")
            return False
        return True
    except:
        print("Ollama not running. Start it with: ollama serve")
        return False


# ── MAIN ──
def main():
    if len(sys.argv) < 2:
        print("Usage: python download_scripts/cg_nlp_scorer.py <root_folder> [--model llama3.1:8b] [--output nlp_scores.csv] [--merge scores.xlsx]")
        sys.exit(1)

    root = sys.argv[1]
    model = DEFAULT_MODEL
    output = 'nlp_scores.csv'
    merge_file = None

    for flag in ['--model', '--output', '--merge']:
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                val = sys.argv[i + 1]
                if flag == '--model': model = val
                elif flag == '--output': output = val
                elif flag == '--merge': merge_file = val

    if not check_ollama(model):
        sys.exit(1)

    print(f"Model: {model}")

    # Step 1: Collect all XML files and extract URLs
    print("Step 1: Extracting policy URLs from CG XML files...")
    file_data = []  # (filepath, filename, info_with_urls)
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.xml') and 'CG' in f.upper():
                fp = os.path.join(dirpath, f)
                info = extract_urls_from_xml(fp)
                if info and info.get('urls'):
                    file_data.append((fp, f, info))

    print(f"  Found {len(file_data)} XML files with policy URLs")

    # Deduplicate URLs per company (same company has same policies across quarters)
    company_urls = {}  # scrip_code -> {metric_id: url}
    company_info = {}  # scrip_code -> {name, symbol}
    for fp, fn, info in file_data:
        scrip = info.get('scrip_code', '')
        if scrip not in company_urls:
            company_urls[scrip] = {}
            company_info[scrip] = {'name': info.get('company_name'), 'symbol': info.get('symbol')}
        company_urls[scrip].update(info['urls'])

    print(f"  {len(company_urls)} unique companies with policy URLs")
    total_urls = sum(len(v) for v in company_urls.values())
    print(f"  {total_urls} total URLs to fetch and score")

    # Step 2 & 3: Fetch and score
    print("\nStep 2-3: Fetching policies and scoring with Ollama...")
    results = []  # (scrip, metric_id, score, reason)
    done = 0

    for scrip, urls in company_urls.items():
        cname = company_info[scrip]['name'] or scrip
        for metric_id, url in urls.items():
            done += 1
            print(f"  [{done}/{total_urls}] {cname} → {metric_id}...", end=' ', flush=True)

            text = fetch_page_text(url)
            if not text:
                print("fetch failed")
                results.append((scrip, cname, company_info[scrip].get('symbol'), metric_id, None, 'URL fetch failed'))
                continue

            result = score_with_ollama(text, metric_id, model)
            if result and 'score' in result:
                score = result['score']
                reason = result.get('brief_reason', '')
                print(f"score={score}")
                results.append((scrip, cname, company_info[scrip].get('symbol'), metric_id, score, reason))
            else:
                print("LLM parse failed")
                results.append((scrip, cname, company_info[scrip].get('symbol'), metric_id, None, 'LLM parse failed'))

            time.sleep(0.5)  # rate limit

    # Write CSV
    print(f"\nWriting {len(results)} scores to {output}")
    with open(output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scrip_code', 'company_name', 'symbol', 'ID', 'Score', 'Reason'])
        w.writerows(results)

    # Merge if requested
    if merge_file:
        import pandas as pd
        print(f"\nMerging into {merge_file}...")
        main_df = pd.read_excel(merge_file)
        nlp_df = pd.read_csv(output)

        # Build lookup: (scrip, metric_id) -> score
        nlp_lookup = {}
        for _, r in nlp_df.iterrows():
            key = (str(r['scrip_code']), r['ID'])
            if r['Score'] is not None and str(r['Score']) not in ('', 'None'):
                nlp_lookup[key] = r['Score']

        # Update main scores
        updated = 0
        for idx, r in main_df.iterrows():
            scrip = str(r.get('scrip_code', ''))
            # Also try extracting from filename
            if not scrip or scrip == 'nan':
                m = re.match(r'^(\d+)', str(r.get('file', '')))
                scrip = m.group(1) if m else ''
            key = (scrip, r['ID'])
            if key in nlp_lookup and (str(r['Score']) in ('NLP-based', '', 'None', 'nan') or pd.isna(r['Score'])):
                main_df.at[idx, 'Score'] = nlp_lookup[key]
                updated += 1

        out_merged = output.replace('.csv', '.xlsx') if output.endswith('.csv') else output
        if merge_file.endswith('.xlsx'):
            out_merged = merge_file.replace('.xlsx', '_with_nlp.xlsx')

        MAX_ROWS = 1_048_000
        if len(main_df) <= MAX_ROWS:
            main_df.to_excel(out_merged, index=False, sheet_name='CG_Scores')
        else:
            chunks = [main_df.iloc[i:i+MAX_ROWS] for i in range(0, len(main_df), MAX_ROWS)]
            for ci, chunk in enumerate(chunks):
                fname = out_merged.replace('.xlsx', f'_part{ci+1}.xlsx')
                chunk.to_excel(fname, index=False, sheet_name='CG_Scores')

        print(f"  Updated {updated} scores. Output: {out_merged}")

    print("Done.")


if __name__ == '__main__':
    main()
