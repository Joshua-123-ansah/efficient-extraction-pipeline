#!/usr/bin/env python3
"""
NLP Extraction Pipeline
- Recursively scan a root folder for PDFs
- Filter/process only papers containing any target authors
- Extract metadata using local NLP (spaCy + regex heuristics)
- Export Excel with columns: Title, Target Author, Found Author Name, Author's Position,
  Total Authors, Affiliation, Abstract, File Path, Paper Row
"""

import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import unicodedata
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional engines for PDF text
try:
    import pymupdf as fitz
except Exception:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

# Optional progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Excel export
try:
    import pandas as pd
except Exception:
    print("pandas is required for Excel export. Please install with: pip install pandas openpyxl")
    raise

# -------------------- CONFIG --------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline_paths import MY_PAPERS, NLP_RESULTS_XLSX, PIPELINE_RUN_STATS_JSON

# Root folder variable (can be changed):
ROOT_FOLDER: Path = Path(os.environ.get("ROOT_PAPERS_DIR", str(MY_PAPERS))).expanduser().resolve()

# Target authors variable (can be changed):
TARGET_AUTHORS: List[str] = [
    'Alaine Allen', 'Amon Milner', 'Angela Byars-Winston', 'Asia Fuller-Hamilton',
    'Ayesha Boyce', 'Beronda Montgomery', 'Bevlee Watford', 'Brain A. Burt',
    'Brian Nord', 'Brooke Coley', 'Bruk Berhane', 'Chanda Prescod-Weinstein',
    'Cherie Avent', 'Christina S. Morton', 'Christopher C. Jett', 'Christopher G. Wright',
    'Courtney Smith-Orr', 'Danny Bernard Martin', 'Darryl Dickerson', 'David A. Delaine',
    'DeLean Tolbert', 'Denise R. Simmons',
    'Devin Guillory', 'Grace A. Gonce', 'Stephanie Dinkins',
    'Tiffany Lethabo King', 'Eboni M. Zamani-Gallaher', 'Ebony O. McGee',
    'Erika Bullock', 'Felicia Moore Mensah', 'Fredericka Brown', 'Jakita Thomas',
    'James Holly, Jr.', 'Jeremi London', 'Joi-Lynn Mondisa', 'Jomo Mutegi',
    'Joy Buolamwini', 'Julius E. Davis', 'Kelly Cross', 'Kinnis Gosha',
    'LaVar J. Charleston', 'Leroy Long', 'Lola Eniola-Adefeso', 'Lorenzo Baber',
    'Maisie L. Gholson', 'Mark A. Melton', 'Monica Cox', 'Monica Lynn Miles',
    'Monique S. Ross', "Na'ilah Suad Nasir", 'Nichole Pinkard', 'Nicki Washington',
    'Nicole M. Joseph', 'Nicole Pitterson', 'Patrice Prince', 'Quincy Brown',
    'Racheida Lewis', 'Renetta Garrison Tull', 'Robert T. Palmer', 'Ruby Mendenhall',
    'Sharon Fries-Britt', 'Shaun Harper', 'Shaundra Daily', 'Sheena Erete',
    'Walter Lee', 'Trevion Henderson', 'Jessica Rush Leeker', 'Jerrod Henderson',
    'Karis Boyd-Sinkler', 'Jeremy A. Magruder Waisome', 'Christina Alston',
    'Kaitlyn Cage', 'Robert Downey', 'Carlotta A. Berry', 'Rickey Caldwell',
    'Clausell Mathis', 'Whitney Gaskins', 'Christopher Dancy', 'Christy Chatmon',
    'Tanya Ennis', 'Geraldine Cochran', 'Yolanda Rankin', 'John Palmore'
]

# Path shared with compare_pipelines.py for time/cost reporting
RUN_STATS_JSON = os.environ.get("RUN_STATS_JSON", str(PIPELINE_RUN_STATS_JSON))

# Output Excel file:
OUTPUT_XLSX: Path = Path(os.environ.get("OUTPUT_XLSX", str(NLP_RESULTS_XLSX))).resolve()
# Optional worker count for concurrency
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "4"))

# Heuristics / limits
SCAN_FIRST_N_PAGES_FOR_BYLINE = 3
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+\b', re.I)

# Non-capturing: these are embedded in larger patterns whose group(1) is
# meant to be the BODY text. As capturing groups they shifted every index,
# so abstract/keyword extraction returned the literal word "Abstract".
ABSTRACT_HDR = r'(?:Abstract|Summary|Résumé|Resumé|Zusammenfassung|Resumen|Резюме|摘要|概要|초록|ملخص)'
KEYWORDS_HDR = r'(?:Keywords?|Index Terms|Mots[- ]clés|Palabras clave|Schlüsselwörter|Ключевые слова|关键词)'

# Common affiliation keywords for heuristics
AFFILIATION_HINTS = (
    'university', 'department', 'institute', 'school', 'college', 'centre', 'center',
    'laboratory', 'lab', 'faculty'
)

# -------------------- Utilities --------------------

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip()

def strip_accents(s: str) -> str:
    try:
        return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    except Exception:
        return s

def normalize_name_for_compare(s: str) -> str:
    t = normalize_spaces(s).lower()
    t = strip_accents(t)
    return re.sub(r'[\s\.,;:\-]+', '', t)

def split_name(name: str) -> Tuple[List[str], str]:
    parts = [p for p in re.split(r'[\s\-]+', normalize_spaces(name)) if p]
    if not parts:
        return [], ''
    return parts[:-1], parts[-1]

def initials_from_given(given_tokens: List[str]) -> List[str]:
    return [t[0].lower() for t in given_tokens if t]

# -------------------- TARGET AUTHOR ABBREVIATION --------------------

def parse_name_for_abbrev(name: str) -> Tuple[str, List[str], str]:
    # Strip suffixes first: "James Holly, Jr." must not be read as
    # Last="James Holly", First="Jr." by the "Last, First" branch below.
    n = strip_name_suffix(name)
    if not n:
        return '', [], ''
    if ',' in n:
        last, rest = [x.strip() for x in n.split(',', 1)]
        tokens = [t for t in re.split(r'[\s\-]+', rest) if t]
        first = tokens[0] if tokens else ''
        middle = tokens[1:] if len(tokens) > 1 else []
        return first, middle, last
    tokens = [t for t in re.split(r'[\s\-]+', n) if t]
    if not tokens:
        return '', [], ''
    if len(tokens) >= 2:
        last = tokens[-1]
        first = tokens[0]
        middle = tokens[1:-1] if len(tokens) > 2 else []
        return first, middle, last
    elif len(tokens) == 1:
        return tokens[0], [], ''
    else:
        return '', [], ''

def abbreviate_target_name(name: str) -> str:
    first, middles, last = parse_name_for_abbrev(name)
    if not first or not last:
        return name
    if len(first) == 1 and first.endswith('.'):
        return name
    if not middles:
        return f"{first} {last}"
    mid_initials: List[str] = []
    for middle in middles:
        if middle and len(middle) > 1:
            mid_initials.append(middle[0].upper() + '.')
        else:
            mid_initials.append(middle)
    return f"{first} {' '.join(mid_initials)} {last}".strip()

def abbreviate_targets(targets: List[str]) -> List[str]:
    return [abbreviate_target_name(t) for t in (targets or [])]

# -------------------- Name comparison --------------------

def extract_first_last(name: str) -> Tuple[str, str]:
    n = strip_name_suffix(name)
    if not n:
        return '', ''
    if ',' in n:
        parts = [p.strip() for p in n.split(',', 1)]
        if len(parts) == 2:
            last_part = parts[0]
            first_part = parts[1]
            first_tokens = [t for t in re.split(r'[\s\-]+', first_part) if t]
            first = first_tokens[0] if first_tokens else ''
            last_tokens = [t for t in re.split(r'[\s\-]+', last_part) if t]
            last = last_tokens[-1] if last_tokens else ''
            return first, last
    tokens = [t for t in re.split(r'[\s\-]+', n) if t]
    if not tokens:
        return '', ''
    if len(tokens) == 1:
        return tokens[0], ''
    return tokens[0], tokens[-1]

# Generational/credential suffixes are not part of a person's name and must be
# stripped before first/last extraction, otherwise "James Holly, Jr." parses as
# first="Jr", last="Holly".
NAME_SUFFIX_RE = re.compile(
    r"[,\s]+(jr|sr|ii|iii|iv|v|phd|ph\.d|md|m\.d|dr|prof)\.?\s*$", re.I)


def strip_name_suffix(s: str) -> str:
    prev = None
    out = normalize_spaces(s)
    while prev != out:
        prev = out
        out = normalize_spaces(NAME_SUFFIX_RE.sub('', out))
    return out


def normalize_name_for_compare_robust(s: str) -> str:
    if not s:
        return ''
    t = normalize_spaces(s).lower()
    t = strip_accents(t)
    # Drop apostrophes and hyphens too: "Na'ilah" and "Nailah", "Boyd-Sinkler"
    # and "Boyd Sinkler" are the same person in a byline.
    t = re.sub(r"[^\w]", '', t)
    return t

def _is_initials(raw: str, norm: str) -> bool:
    """True if a given-name field is an initial form: "B", "B.", "B.A.", "B. A."."""
    if len(norm) == 1:
        return True
    compact = re.sub(r'\s+', '', (raw or '').lower())
    return bool(re.fullmatch(r'(?:[a-z]\.){1,3}', compact))

def robust_author_match(target: str, found: str) -> bool:
    """
    Match two renderings of the same author.

    Surnames must agree exactly. Given names agree if they are equal, or if
    either side is an initial that starts the other -- journals routinely print
    "B. Coley" for "Brooke Coley", and requiring full equality silently drops
    those papers.
    """
    if not target or not found:
        return False
    t_first, t_last = extract_first_last(target)
    f_first, f_last = extract_first_last(found)
    if not t_first or not t_last or not f_first or not f_last:
        return False
    t_first_norm = normalize_name_for_compare_robust(t_first)
    t_last_norm = normalize_name_for_compare_robust(t_last)
    f_first_norm = normalize_name_for_compare_robust(f_first)
    f_last_norm = normalize_name_for_compare_robust(f_last)
    if not t_last_norm or t_last_norm != f_last_norm:
        return False
    if t_first_norm == f_first_norm:
        return True
    # Initial vs full given name. "B." and "B.A." are both initial forms.
    if _is_initials(t_first, t_first_norm) or _is_initials(f_first, f_first_norm):
        return bool(t_first_norm) and bool(f_first_norm) and t_first_norm[0] == f_first_norm[0]
    return False

# -------------------- spaCy model --------------------

def load_spacy_model():
    try:
        import spacy  # noqa: F401
    except Exception:
        return None
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        # Try auto-download (requires internet)
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"], check=True)
            import spacy as sp
            return sp.load("en_core_web_sm")
        except Exception:
            return None

# -------------------- PDF text helpers --------------------

def pdf_text_all(pdf_path: Path) -> str:
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                return "\n".join(page.get_text("text") for page in doc)
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            return pdfminer_extract_text(str(pdf_path))
        except Exception:
            pass
    return ""

def pdf_text_first_n(pdf_path: Path, n_pages: int) -> str:
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                n = min(n_pages, len(doc))
                return "\n".join(doc[i].get_text("text") for i in range(n))
        except Exception:
            pass
    return pdf_text_all(pdf_path)

# Quick prefilter: check head text for any target author to avoid heavy processing on non-matches

def quick_author_hit(text_head: str, targets: List[str]) -> bool:
    if not targets:
        return True
    hay = (text_head or '').lower()
    if not hay:
        return False
    for t in targets:
        t_norm = normalize_spaces(t).lower()
        if not t_norm:
            continue
        if t_norm in hay:
            return True
        first, middles, last = parse_name_for_abbrev(t)
        if not first or not last:
            continue
        first_lower = first.lower()
        last_lower = last.lower()
        if re.search(rf'\b{re.escape(first_lower)}\s+{re.escape(last_lower)}(?:\d+)?\b', hay):
            return True
        if re.search(rf'\b{re.escape(first_lower[0])}\.?\s+{re.escape(last_lower)}(?:\d+)?\b', hay):
            return True
        if re.search(rf'\b{re.escape(last_lower)}\s*,\s*{re.escape(first_lower)}\b', hay):
            return True
        if re.search(rf'\b{re.escape(last_lower)}\s*,\s*{re.escape(first_lower[0])}\.?\b', hay):
            return True
    return False

def byline_block(text: str) -> str:
    lines = [normalize_spaces(ln) for ln in text.splitlines()]
    block: List[str] = []
    for ln in lines[:300]:
        if re.search(rf'\b{ABSTRACT_HDR}\b|\b{KEYWORDS_HDR}\b|\bIntroduction\b', ln, re.I):
            break
        if ln:
            block.append(ln)
    return " ".join(block)

def parse_authors_from_byline(byline: str) -> List[str]:
    parts = re.split(r',|\band\b', byline, flags=re.I)
    out: List[str] = []
    for chunk in parts:
        name = re.sub(r'[\*\d\u00B9\u00B2\u00B3\u2070-\u2079\u2020\u2021\u00A7\u00B6]+', '', chunk)
        name = normalize_spaces(name).strip(',:;')
        if 2 <= len(name.split()) <= 6 and re.search(r'\b[A-Z][a-z]', name):
            out.append(name)
    # dedupe preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

# -------------------- NLP extraction --------------------

def extract_title_from_head(text_head: str) -> str:
    if not text_head:
        return ''
    lines = [normalize_spaces(ln) for ln in text_head.splitlines() if normalize_spaces(ln)]
    for ln in lines[:40]:
        if len(ln) < 6:
            continue
        if re.search(rf'^({ABSTRACT_HDR}|{KEYWORDS_HDR}|Introduction)\b', ln, re.I):
            continue
        # Heuristic: title often in Title Case or ALL CAPS and not ending with a period
        if len(ln) <= 180 and not ln.endswith('.'):
            # avoid author-like lines containing 'and' many commas
            if not (ln.count(',') >= 2 and ' and ' in ln):
                return ln
    return lines[0] if lines else ''

def spacy_persons(text: str, nlp) -> List[str]:
    if not text or nlp is None:
        return []
    try:
        doc = nlp(text)
        names: List[str] = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                nm = normalize_spaces(ent.text)
                if 2 <= len(nm.split()) <= 6:
                    names.append(nm)
        # dedupe preserving order
        seen = set()
        uniq: List[str] = []
        for x in names:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    except Exception:
        return []

def spacy_orgs(text: str, nlp) -> List[str]:
    if not text or nlp is None:
        return []
    try:
        doc = nlp(text)
        orgs: List[str] = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                tx = normalize_spaces(ent.text)
                if len(tx) > 3:
                    orgs.append(tx)
        # dedupe preserving order
        seen = set()
        uniq: List[str] = []
        for x in orgs:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    except Exception:
        return []

def extract_keywords(text: str) -> str:
    if not text:
        return ''
    m = re.search(rf'\b{KEYWORDS_HDR}\b\s*[:\-]?\s*(.+)', text, re.I)
    if m:
        # stop at line break or period if extremely long
        kws = m.group(1)
        kws = kws.split('\n', 1)[0]
        return normalize_spaces(kws)[:2000]
    return ''

def extract_year(text: str) -> str:
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text[:5000]) if text else []
    return years[0] if years else ''

def extract_journal_meta(text: str) -> Tuple[str, str, str, str]:
    if not text:
        return '', '', '', ''
    head = text[:8000]
    # Journal name (heuristic: line containing 'Journal' or 'Proceedings of')
    journal = ''
    for ln in head.splitlines()[:80]:
        ln_norm = normalize_spaces(ln)
        if re.search(r'\b(Journal|Proceedings of|Transactions on|Conference on)\b', ln_norm, re.I):
            journal = ln_norm
            break
    # Volume/Issue
    volume = ''
    issue = ''
    mvi = re.search(r'Vol\.?\s*(\d+)(?:\s*[,;| ]\s*No\.?\s*(\d+))?', head, re.I)
    if mvi:
        volume = mvi.group(1) or ''
        issue = mvi.group(2) or ''
    else:
        mvi2 = re.search(r'Volume\s*(\d+)\s*(?:Issue|No\.)\s*(\d+)', head, re.I)
        if mvi2:
            volume = mvi2.group(1) or ''
            issue = mvi2.group(2) or ''
    # Pages
    pages = ''
    mp = re.search(r'pp\.?\s*(\d+\s*[-–]\s*\d+)', head, re.I)
    if mp:
        pages = normalize_spaces(mp.group(1))
    return journal, volume, issue, pages

def extract_affiliations(byline_text: str, head_text: str, nlp) -> List[str]:
    candidates: List[str] = []
    for src in [byline_text or '', head_text or '']:
        # explicit affiliation lines first
        for ln in src.split('\n'):
            ln_norm = normalize_spaces(ln)
            low = ln_norm.lower()
            if any(k in low for k in AFFILIATION_HINTS) and len(ln_norm) > 8:
                candidates.append(ln_norm)
        # NER-based orgs
        candidates.extend(spacy_orgs(src, nlp))
    # Clean & dedupe
    seen = set()
    cleaned: List[str] = []
    for a in candidates:
        c = re.sub(r'\s+', ' ', a).strip()
        c = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', c)
        if c and c not in seen and len(c) > 8:
            seen.add(c)
            cleaned.append(c)
    return cleaned[:10]

def extract_from_pdf_nlp(pdf_path: Path, nlp) -> Dict[str, Any]:
    text_head = pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
    text_all = pdf_text_all(pdf_path)
    md = {
        "format": "nlp",
        "authors": [],
        "affiliations": [],
        "emails": [],
        "title": "",
        "abstract": "",
        "keywords": "",
        "journal": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "year": "",
        "doi": ""
    }
    if not (text_head or text_all):
        return md

    by = byline_block(text_head or text_all)
    # Authors via byline regex + spaCy PERSONs near the top
    authors_regex = parse_authors_from_byline(by)
    authors_ner = spacy_persons(by or text_head, nlp)
    authors: List[str] = []
    for nm in [*authors_regex, *authors_ner]:
        if nm not in authors:
            authors.append(nm)
    md["authors"] = authors

    # Title
    md["title"] = extract_title_from_head(text_head or text_all)

    # Abstract: capture full text; end strictly at Keywords if present, else intro-like fallback
    abs_text_source = text_head or text_all or ''
    m_abs_kw = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)\b{KEYWORDS_HDR}\b\s*[:\n]', abs_text_source, re.I)
    if m_abs_kw:
        md["abstract"] = normalize_spaces(m_abs_kw.group(1))
    else:
        m_abs_intro = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)(^Introduction|\n\s*1\.\s*Introduction)', abs_text_source, re.I | re.M)
        if m_abs_intro:
            md["abstract"] = normalize_spaces(m_abs_intro.group(1))

    # Keywords
    md["keywords"] = extract_keywords(text_head or text_all)

    # Emails
    emails = EMAIL_RE.findall(text_head or '') or EMAIL_RE.findall(text_all or '')
    if emails and authors:
        md["emails"].append({"author": authors[0], "emails": emails[:5]})

    # DOI
    mdoi = DOI_RE.search(text_all or text_head or "")
    md["doi"] = mdoi.group(0).rstrip('.,);]') if mdoi else ""

    # Journal-like metadata
    journal, volume, issue, pages = extract_journal_meta(text_head or text_all)
    md["journal"] = journal
    md["volume"] = volume
    md["issue"] = issue
    md["pages"] = pages
    md["year"] = extract_year(text_head or text_all)

    # Affiliations
    affs = extract_affiliations(by, text_head, nlp)
    if authors and affs:
        for a in authors:
            md["affiliations"].append({"author": a, "affiliations": affs})

    return md

# -------------------- Recursion & Row helpers --------------------

def gather_pdfs(root: Path) -> List[Path]:
    paths: List[Path] = []
    if root.is_file() and root.suffix.lower() == '.pdf':
        return [root]
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith('.pdf'):
                paths.append(Path(r) / fn)
    # Deduplicate & sort for stable row indexing
    uniq: List[Path] = []
    seen = set()
    for p in sorted(paths):
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq

from functools import lru_cache

@lru_cache(maxsize=1000)
def get_file_row_cached(file_path: str) -> int:
    try:
        path = Path(file_path)
        directory = path.parent
        filename = path.name
        pdf_files = [f for f in directory.iterdir() if f.suffix.lower() == '.pdf']
        pdf_files.sort()
        for i, file in enumerate(pdf_files, 1):
            if file.name == filename:
                return i
        return 1
    except Exception:
        return 1

# Excel text sanitizer

def clean_for_excel(text: str) -> str:
    if text is None:
        return ''
    return re.sub(r'[\x00-\x1F\x7F]', '', str(text))

# -------------------- Main aggregation --------------------

def main():
    # CLI overrides: root folder and comma-separated authors
    root = ROOT_FOLDER
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1]).expanduser().resolve()
    targets = abbreviate_targets(TARGET_AUTHORS[:])
    if len(sys.argv) >= 3:
        targets = abbreviate_targets([a.strip() for a in sys.argv[2].split(',') if a.strip()])

    if not root.exists():
        print(f"❌ Root folder does not exist: {root}")
        sys.exit(1)

    # Load spaCy model (auto-download if missing)
    nlp = load_spacy_model()
    if nlp is None:
        print("ℹ️ spaCy model 'en_core_web_sm' not available; proceeding with regex-only heuristics.")

    pdfs = gather_pdfs(root)
    print(f"📄 Found {len(pdfs)} PDF(s) under {root}")

    # Pre-filter: only keep candidates that show a quick author hit in head text
    t0 = time.time()
    candidates: List[Path] = []
    for p in (tqdm(pdfs, desc='Scanning', unit='pdf') if tqdm else pdfs):
        head = pdf_text_first_n(p, SCAN_FIRST_N_PAGES_FOR_BYLINE)
        if quick_author_hit(head, targets):
            candidates.append(p)
    print(f"✅ Prefilter retained {len(candidates)} / {len(pdfs)} PDFs ({len(pdfs)-len(candidates)} skipped).")

    rows: List[Dict[str, Any]] = []

    def process_one(pdf: Path) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        md = extract_from_pdf_nlp(pdf, nlp)
        authors: List[str] = md.get('authors', []) or []
        if not authors:
            return out_rows
        # Attempt to match any target author
        found_pairs: List[Tuple[str, str]] = []  # (target, found)
        if targets:
            for t in targets:
                for f in authors:
                    if robust_author_match(t, f):
                        found_pairs.append((t, f))
        else:
            found_pairs = [(f, f) for f in authors]
        if not found_pairs:
            return out_rows
        # Affiliation mapping
        aff_map: Dict[str, List[str]] = {}
        for item in md.get('affiliations', []) or []:
            name = item.get('author', '')
            vals = item.get('affiliations', []) or []
            if name:
                aff_map.setdefault(name, [])
                for v in vals:
                    if v and v not in aff_map[name]:
                        aff_map[name].append(v)
        total_authors = len(authors)
        title = md.get('title', '')
        abstract = md.get('abstract', '')
        file_path_str = str(pdf.resolve())
        paper_row = get_file_row_cached(file_path_str)
        for target, found in found_pairs:
            try:
                position = authors.index(found) + 1 if found in authors else ''
            except Exception:
                position = ''
            affs = aff_map.get(found, [])
            aff_text = '; '.join(affs) if affs else 'Not specified'
            out_rows.append({
                'Title': clean_for_excel(title),
                'Target Author': clean_for_excel(target),
                'Found Author Name': clean_for_excel(found),
                "Author's Position": position,
                'Total Authors': total_authors,
                'Affiliation': clean_for_excel(aff_text),
                'Abstract': clean_for_excel(abstract),
                'File Path': file_path_str,
                'Paper Row': paper_row,
            })
        return out_rows

    # Parallel processing for speed
    processed = 0
    failed: List[Tuple[str, str]] = []
    if MAX_WORKERS > 1 and len(candidates) > 1:
        iterator = None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, p): p for p in candidates}
            iterator = tqdm(as_completed(futures), total=len(futures), desc='Processing', unit='pdf') if tqdm else as_completed(futures)
            for fut in iterator:
                try:
                    rs = fut.result()
                    rows.extend(rs)
                except Exception as e:
                    failed.append((futures[fut].name, f"{type(e).__name__}: {e}"))
                processed += 1
    else:
        iterable = tqdm(candidates, desc='Processing', unit='pdf') if tqdm else candidates
        for p in iterable:
            # Serial path had no guard: one unreadable PDF aborted the whole run.
            try:
                rows.extend(process_one(p))
            except Exception as e:
                failed.append((p.name, f"{type(e).__name__}: {e}"))
            processed += 1

    dt = time.time() - t0
    rate = (processed / dt) if dt > 0 else 0
    print(f"⏱️ Done in {dt:.1f}s ({rate:.2f} pdf/s). Matched rows: {len(rows)}")
    if failed:
        # Previously these were swallowed silently, so a run could quietly drop
        # papers and still look successful.
        print(f"⚠️  {len(failed)} PDF(s) failed to process:")
        for nm, err in failed[:10]:
            print(f"     {nm}: {err}")
        if len(failed) > 10:
            print(f"     ... and {len(failed)-10} more")

    if not rows:
        print("ℹ️ No matching papers/authors found; nothing to write.")
        return

    df = pd.DataFrame(rows, columns=[
        'Title', 'Target Author', 'Found Author Name', "Author's Position",
        'Total Authors', 'Affiliation', 'Abstract', 'File Path', 'Paper Row'
    ])
    try:
        df.to_excel(OUTPUT_XLSX, index=False)
        print(f"✅ Wrote {len(df)} rows to {OUTPUT_XLSX}")
    except Exception as e:
        print(f"❌ Failed to write Excel: {e}")

    # Record runtime so compare_pipelines.py can show a complete Time column.
    # Without this, NLP's time was always NaN in the comparison table.
    try:
        import json as _json
        stats_path = Path(RUN_STATS_JSON)
        stats = {}
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as f:
                stats = _json.load(f)
        stats["nlp"] = {"runtime_seconds": round(dt, 3), "cost_usd": 0.0}
        with stats_path.open("w", encoding="utf-8") as f:
            _json.dump(stats, f, indent=2)
        print(f"⏱️  Recorded NLP stats in {stats_path}")
    except Exception as e:
        print(f"⚠️ Failed to write NLP stats: {e}")

if __name__ == "__main__":
    main()


