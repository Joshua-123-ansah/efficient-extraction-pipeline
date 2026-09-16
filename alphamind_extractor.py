#!/usr/bin/env python3
"""
AlphaMind Extractor (Alphamind_Extractor) — hybrid GROBID + LLM extraction pipeline
- Combines GROBID and LLM extraction methods
- Uses GROBID for fields where GROBID F1 >= 0.85:
  - Title (F1: 0.960)
  - Target Author (F1: 0.960)
  - Found Author Name (F1: 0.960)
  - Author's Position (F1: 0.950)
  - Total Authors (F1: 0.917)
  - Abstract (F1: 0.905)
- Uses LLM for fields where GROBID F1 < 0.85:
  - Affiliation (F1: 0.038)
- Export Excel with columns: Title, Target Author, Found Author Name, Author's Position,
  Total Authors, Affiliation, Abstract, File Path, Paper Row

This pipeline reduces LLM costs by using GROBID for most fields and LLM only for Affiliation.
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import unicodedata
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

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

# Excel export — optional. Affiliation extraction (and the dashboard LLM path)
# must still import this module when pandas/numpy are mismatched in the env.
try:
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        import pandas as pd
except Exception:
    pd = None

# GROBID imports
import requests
from lxml import etree

# -------------------- CONFIG --------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline_paths import ALPHAMIND_EXTRACTOR_RESULTS_XLSX, MY_PAPERS

# GROBID
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT = 60

# GROBID endpoint. Measured on GROBID 0.8.3-SNAPSHOT: processHeaderDocument does
# NOT honour includeRawAffiliations (no <note type="raw_affiliation"> in its TEI),
# and the affiliation parser depends on that note. It is also only ~1.6x faster
# here (1.09s vs 1.79s), not the 15x the docs suggest. Stay on fulltext.
GROBID_SERVICE = os.environ.get("GROBID_SERVICE", "processFulltextDocument")

# consolidateHeader=1 makes GROBID call out to CrossRef/biblio-glutton for every
# paper (extra network round trip). Default 1; set CONSOLIDATE_HEADER=0 or
# --no-consolidate-header for PDF-only headers.
CONSOLIDATE_HEADER = os.environ.get("CONSOLIDATE_HEADER", "1")

# -------------------- LLM provider (ASU CreateAI) --------------------
# CreateAI is OpenAI-wire-compatible, so the official `openai` SDK works once
# base_url and the bearer token are pointed at ASU's gateway.
#   Production base: https://api-main.aiml.asu.edu
#   Beta base:       https://api-main-beta.aiml.asu.edu
# CREATEAI_BASE_URL must be the OpenAI-compatible root (the path the SDK appends
# /chat/completions to). Confirm the exact suffix on
# https://docs.aiml.asu.edu/openai-compatible and override if it differs.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "createai").strip().lower()
CREATEAI_BASE_URL = os.environ.get("CREATEAI_BASE_URL", "https://api-main.aiml.asu.edu/v1").rstrip('/')
CREATEAI_QUERY_URL = os.environ.get("CREATEAI_QUERY_URL", "https://api-main.aiml.asu.edu/query")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "60"))


def get_llm_api_key() -> str:
    """
    Bearer token for the configured provider.

    CreateAI service tokens are JWTs and live in ASU_CREATE_AI_KEY;
    CREATEAI_API_KEY is accepted as an alias. Falls back to OPENAI_API_KEY so
    the pipeline still runs against api.openai.com with LLM_PROVIDER=openai.
    """
    if LLM_PROVIDER == "createai":
        for var in ("CREATEAI_API_KEY", "ASU_CREATE_AI_KEY"):
            key = os.environ.get(var, "").strip()
            if key:
                return key
    return os.environ.get("OPENAI_API_KEY", "").strip()

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

# Output Excel file:
OUTPUT_XLSX: Path = Path(os.environ.get("OUTPUT_XLSX", str(ALPHAMIND_EXTRACTOR_RESULTS_XLSX))).resolve()

# Concurrency: be conservative for API rate-limits
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "2"))

# Heuristics / limits
# Two pages, not three -- must match LLM_extraction_pipeline.py or the cost
# comparison is measuring two different input windows. Used here only by the
# tier-2 escalation (tier 1 is the much smaller byline window). Measured on 422
# PDFs: the title -> authors -> abstract region closes by page 2 for all of
# them, and by page 1 for 417. Override with HEAD_PAGES.
SCAN_FIRST_N_PAGES_FOR_BYLINE = int(os.environ.get("HEAD_PAGES", "2"))
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+\b', re.I)

# Non-capturing: these are embedded in larger patterns whose group(1) is
# meant to be the BODY text. As capturing groups they shifted every index,
# so abstract/keyword extraction returned the literal word "Abstract".
ABSTRACT_HDR = r'(?:Abstract|Summary|Résumé|Resumé|Zusammenfassung|Resumen|Резюме|摘要|概要|초록|ملخص)'
KEYWORDS_HDR = r'(?:Keywords?|Index Terms|Mots[- ]clés|Palabras clave|Schlüsselwörter|Ключевые слова|关键词)'

# -------------------- Confidence routing --------------------
# Skip the LLM entirely on papers where GROBID's affiliation clears a quality
# bar. Before the raw_affiliation fix this was not viable (GROBID's affiliations
# were doubled, so the gate was right only ~80% of the time). After the fix,
# measured on the gold corpus: 94.6% precision (35/37), ~80% of papers skipped.
# Set ROUTE_ON_GROBID_CONFIDENCE=0 to always call the LLM instead.
ROUTE_ON_GROBID_CONFIDENCE = os.environ.get("ROUTE_ON_GROBID_CONFIDENCE", "1") not in ("0", "false", "False")
GATE_MIN_CHARS = int(os.environ.get("GATE_MIN_CHARS", "12"))
GATE_MAX_CHARS = int(os.environ.get("GATE_MAX_CHARS", "300"))
GATE_MAX_SEGMENTS = int(os.environ.get("GATE_MAX_SEGMENTS", "1"))

# Pricing per 1M tokens. Derived from a real CreateAI response:
# 801 input tokens billed at $0.004005 => $5.00 / 1M input, not OpenAI's $0.15.
# Confirm the output rate from metadata.usage_metric.output_token_cost.
OPENAI_INPUT_PRICE_PER_M = float(os.environ.get("OPENAI_INPUT_PRICE_PER_M", "5.00"))
OPENAI_OUTPUT_PRICE_PER_M = float(os.environ.get("OPENAI_OUTPUT_PRICE_PER_M", "15.00"))

# Path shared with compare_pipelines.py for time/cost reporting
RUN_STATS_JSON = os.environ.get("RUN_STATS_JSON", "pipeline_run_stats.json")

# Global usage accumulator (guarded by a lock because of ThreadPoolExecutor)
_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost_usd": 0.0,
}
_USAGE_LOCK = threading.Lock()

# Legacy aliases (tests / older scripts)
_JOSHCOLEY_USAGE = _USAGE
_JOSHCOLEY_USAGE_LOCK = _USAGE_LOCK

# How often confidence routing let us skip the LLM. Reported at the end of a run
# so the skip rate is visible rather than assumed.
_ROUTING_COUNTS = {"grobid_only": 0, "llm_called": 0}


def _update_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    """Accumulate token usage and cost for this run (hybrid pipeline)."""
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return
    call_cost = (
        (prompt_tokens / 1_000_000.0) * OPENAI_INPUT_PRICE_PER_M
        + (completion_tokens / 1_000_000.0) * OPENAI_OUTPUT_PRICE_PER_M
    )
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += prompt_tokens
        _USAGE["completion_tokens"] += completion_tokens
        _USAGE["cost_usd"] += call_cost


# Legacy alias (tests / older scripts)
_update_joshcoley_usage = _update_usage

# -------------------- Utilities --------------------

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip()

# ---- Affiliation near-duplicate collapsing -------------------------------
# GROBID emits each affiliation twice: once as raw <affiliation> text and once
# as the structured <orgName> reassembly. The two differ only in punctuation
# and word order, so an exact-string dedupe lets both survive and they get
# joined into "Purdue University Purdue University". That doubling drove the
# measured affiliation F1 down to 0.038. Compare on a punctuation-free token
# set instead of the raw string.

def aff_key(s: str) -> str:
    """Punctuation-insensitive comparison key for an affiliation string."""
    s = re.sub(r'[^\w\s]', ' ', (s or '').lower())
    return re.sub(r'\s+', ' ', s).strip()


def aff_is_duplicate(candidate: str, kept: List[str], overlap: float = 0.8) -> bool:
    """True if `candidate` says the same thing as something already in `kept`."""
    ck = aff_key(candidate)
    if not ck:
        return True
    ctoks = set(ck.split())
    for existing in kept:
        ek = aff_key(existing)
        if not ek:
            continue
        if ck in ek or ek in ck:
            return True
        etoks = set(ek.split())
        denom = min(len(ctoks), len(etoks))
        if denom and len(ctoks & etoks) / denom >= overlap:
            return True
    return False

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


# Cap on the small window. Sized from the gold corpus: the byline block plus
# affiliation footnotes averages ~1.4k chars, so 2500 leaves ample headroom.
BYLINE_WINDOW_CHARS = int(os.environ.get("BYLINE_WINDOW_CHARS", "2500"))


def pdf_text_byline_window(pdf_path: Path) -> str:
    """
    Page 1 up to the Abstract heading, plus the page-1 tail where journals
    often print affiliation footnotes.

    Everything after the Abstract is body text that cannot contain an
    affiliation, so sending it is pure cost.
    """
    if fitz is None:
        return pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return ""
            page1 = doc[0].get_text("text")
    except Exception:
        return pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)

    m = re.search(rf'\b{ABSTRACT_HDR}\b', page1, re.I)
    head = page1[:m.start()] if m else page1[:BYLINE_WINDOW_CHARS]
    head = head[:BYLINE_WINDOW_CHARS]
    tail = page1[-900:] if len(page1) > len(head) + 900 else ""
    return (head + ("\n" + tail if tail else "")).strip()


def _has_affiliation(result: Optional[Dict[str, Any]]) -> bool:
    """True if an LLM response actually carries at least one affiliation."""
    if not result:
        return False
    for item in result.get("affiliations") or []:
        try:
            if any(normalize_spaces(x) for x in (item.get("affiliations") or [])):
                return True
        except Exception:
            continue
    return False

# -------------------- Prefilter --------------------

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
        # full name
        if t_norm in hay:
            return True
        # Parse target into components
        first, middles, last = parse_name_for_abbrev(t)
        if not first or not last:
            continue
        first_lower = first.lower()
        last_lower = last.lower()
        
        # Pattern 1: First + Last (e.g., "Monica Miles")
        if re.search(rf'\b{re.escape(first_lower)}\s+{re.escape(last_lower)}(?:\d+)?\b', hay):
            return True
        
        # Pattern 2: Initial + Last (e.g., "M. Miles" or "M Miles")
        if re.search(rf'\b{re.escape(first_lower[0])}\.?\s+{re.escape(last_lower)}(?:\d+)?\b', hay):
            return True
        
        # Pattern 3: Last, First (e.g., "Miles, Monica")
        if re.search(rf'\b{re.escape(last_lower)}\s*,\s*{re.escape(first_lower)}\b', hay):
            return True
        
        # Pattern 4: Last, Initial (e.g., "Miles, M.")
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

# -------------------- GROBID extraction functions --------------------

def is_grobid_alive() -> bool:
    try:
        with requests.Session() as s:
            r = s.get(f"{GROBID_URL}/api/isalive", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def extract_with_grobid(pdf_path: Path) -> Optional[str]:
    session = requests.Session()
    try:
        with open(pdf_path, 'rb') as f:
            files = {'input': f}
            data = {
                'consolidateHeader': os.environ.get("CONSOLIDATE_HEADER", CONSOLIDATE_HEADER),
                'includeRawAffiliations': '1',
            }
            if GROBID_SERVICE == 'processFulltextDocument':
                # Only meaningful for the fulltext service.
                data.update({
                    'consolidateCitations': '0',
                    'includeRawCitations': '0',
                    'teiCoordinates': '0',
                })
            resp = session.post(f"{GROBID_URL}/api/{GROBID_SERVICE}", files=files, data=data, timeout=GROBID_TIMEOUT)
        resp.raise_for_status()
        return resp.content.decode('utf-8', errors='ignore')
    except Exception:
        return None
    finally:
        try:
            session.close()
        except Exception:
            pass

def parse_bibtex(content: str) -> Dict[str, Any]:
    md = {
        "format": "bibtex",
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
    def rex(field):
        m = re.search(rf'{field}\s*=\s*{{([^}}]+)}}', content, flags=re.I)
        return m.group(1).strip() if m else ""
    md["title"] = rex('title')
    md["abstract"] = rex('abstract')
    md["keywords"] = rex('keywords')
    md["journal"] = rex('journal')
    md["volume"] = rex('volume')
    md["issue"]  = rex('number')
    md["pages"]  = rex('pages')
    md["year"]   = rex('year')
    md["doi"]    = rex('doi')
    a = rex('author')
    if a:
        md["authors"] = [x.strip() for x in a.split(' and ') if x.strip()]
    note_full = rex('note')
    if note_full and md["authors"]:
        mails = EMAIL_RE.findall(note_full)
        if mails:
            md["emails"].append({"author": md["authors"][0], "emails": mails})
        if any(k in note_full.lower() for k in ['university','department','institute','school','college','centre','center']):
            md["affiliations"].append({"author": md["authors"][0], "affiliations": [note_full]})
    email_field = rex('email')
    if email_field and md["authors"]:
        emails = [e.strip() for e in re.split(r'[;, ]+', email_field) if EMAIL_RE.fullmatch(e.strip())]
        if emails:
            md["emails"].append({"author": md["authors"][0], "emails": emails})
    return md

def parse_tei(content: str) -> Dict[str, Any]:
    if content.startswith('<?xml'):
        content = content.split('\n', 1)[1] if '\n' in content else content
    xml = etree.fromstring(content.encode('utf-8'))
    tei = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def itxt(node): return normalize_spaces(''.join(node.itertext())) if node is not None else ""

    org_by_id = {}
    for org in xml.xpath('.//tei:listOrg/tei:org', namespaces=tei):
        org_id = org.get('{http://www.w3.org/XML/1998/namespace}id')
        if org_id: org_by_id[f"#{org_id}"] = itxt(org)

    md = {
        "format": "tei",
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

    # Title
    title = xml.find('.//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:title', namespaces=tei)
    if title is None or not title.text:
        title = xml.find('.//tei:titleStmt/tei:title', namespaces=tei)
    if title is not None and title.text:
        md["title"] = itxt(title)

    def author_name_from(node) -> str:
        pn = node.find('.//tei:persName', namespaces=tei)
        if pn is None:
            return ""
        fn = [itxt(x) for x in pn.findall('.//tei:forename', namespaces=tei) if itxt(x)]
        sname = pn.findtext('.//tei:surname', default='', namespaces=tei)
        return ' '.join([*fn, sname]).strip()

    analytic_authors = xml.xpath('.//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author[.//tei:persName]', namespaces=tei)
    title_authors    = xml.xpath('.//tei:titleStmt/tei:author[.//tei:persName]', namespaces=tei)

    author_nodes_ordered = []
    seen_nodes = set()
    for node in analytic_authors + title_authors:
        if id(node) not in seen_nodes:
            seen_nodes.add(id(node))
            author_nodes_ordered.append(node)

    authors: List[str] = []
    author_nodes_by_name: Dict[str, List[Any]] = {}
    for a in author_nodes_ordered:
        nm = author_name_from(a)
        if nm:
            # GROBID lists the same people under BOTH <analytic> and <titleStmt>,
            # often rendered differently ("Monica F. Cox" vs "Monica F Cox",
            # "Jiabin Zhu" vs "J. Zhu"). An exact-string check lets the variant
            # through, inflating Total Authors and shifting Author's Position.
            # Match on identity instead, and keep the first (fuller) rendering.
            existing = next((x for x in authors if robust_author_match(x, nm)), None)
            if existing is None:
                authors.append(nm)
                author_nodes_by_name.setdefault(nm, []).append(a)
            else:
                author_nodes_by_name.setdefault(existing, []).append(a)
    md["authors"] = authors

    # Affiliations mapping (Note: GROBID affiliations are extracted but replaced with LLM results)
    def aff_text_single(aff) -> str:
        """
        Text for one <affiliation>, without the raw/structured double-count.

        With includeRawAffiliations=1, GROBID nests a
        <note type="raw_affiliation"> inside <affiliation> alongside the
        parsed <orgName> elements. itertext() concatenates both, producing
        "Purdue University Purdue University". Read them separately and keep
        the raw form, which preserves the punctuation the gold standard uses.
        """
        raw_nodes = aff.findall('.//tei:note[@type="raw_affiliation"]', namespaces=tei)
        raw_txt = normalize_spaces(' '.join(''.join(n.itertext()) for n in raw_nodes))
        clone = copy.deepcopy(aff)
        for n in clone.findall('.//tei:note[@type="raw_affiliation"]', namespaces=tei):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)
        struct_txt = normalize_spaces(''.join(clone.itertext()))
        # Strip the affiliation marker GROBID keeps on raw notes ("a", "1", "*")
        chosen = raw_txt or struct_txt
        chosen = re.sub(r'^[a-z]?[\d\*†‡§¶]*[\s,]*', '', chosen)
        # GROBID leaves dangling separators where it dropped a sub-element
        chosen = re.sub(r'\s*[,;]\s*$', '', chosen)
        return re.sub(r'\s+,', ',', chosen).strip()

    def itxt_aff(node) -> List[str]:
        out: List[str] = []
        for aff in node.findall('.//tei:affiliation', namespaces=tei):
            t = aff_text_single(aff)
            if t:
                out.append(t)
            for ref in aff.xpath('.//@ref', namespaces=tei):
                if ref in org_by_id and org_by_id[ref]:
                    out.append(org_by_id[ref])
        cleaned: List[str] = []
        for a in out:
            c = re.sub(r'\s+', ' ', a).strip()
            c = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', c)
            if c and len(c) > 8 and not aff_is_duplicate(c, cleaned):
                cleaned.append(c)
        return cleaned

    persons = xml.xpath('.//tei:profileDesc/tei:particDesc/tei:listPerson/tei:person', namespaces=tei)

    def person_name_from(node) -> str:
        pn = node.find('.//tei:persName', namespaces=tei)
        if pn is None:
            return ""
        fn = [itxt(x) for x in pn.findall('.//tei:forename', namespaces=tei) if itxt(x)]
        sname = pn.findtext('.//tei:surname', default='', namespaces=tei)
        return ' '.join([*fn, sname]).strip()

    person_by_name: Dict[str, Any] = {}
    for p in persons:
        nmp = person_name_from(p)
        if nmp:
            person_by_name[normalize_spaces(nmp).lower()] = p

    authors_any = xml.xpath('//tei:author[.//tei:persName]', namespaces=tei)
    all_affiliations: List[str] = []
    for aff in xml.xpath('//tei:affiliation', namespaces=tei):
        t = aff_text_single(aff)
        t = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', t)
        if t and len(t) > 8 and not aff_is_duplicate(t, all_affiliations):
            all_affiliations.append(t)

    def name_components(name: str) -> Tuple[str, List[str]]:
        given, surname = split_name(name)
        return surname.lower(), initials_from_given(given)

    def names_match_loose(a_name: str, b_name: str) -> bool:
        if not a_name or not b_name:
            return False
        a_surn, a_inits = name_components(a_name)
        b_surn, b_inits = name_components(b_name)
        if not a_surn or not b_surn:
            return False
        if normalize_name_for_compare(a_surn) != normalize_name_for_compare(b_surn):
            return False
        if not a_inits or not b_inits:
            return True
        if a_inits[0] == b_inits[0]:
            return True
        if any(ch in b_inits for ch in a_inits):
            return True
        return False

    aff_map: Dict[str, List[str]] = {a: [] for a in authors}
    for nm in authors:
        for node in author_nodes_by_name.get(nm, []):
            for a in itxt_aff(node):
                if not aff_is_duplicate(a, aff_map[nm]):
                    aff_map[nm].append(a)
        if not aff_map[nm]:
            p = person_by_name.get(normalize_spaces(nm).lower())
            if p is not None:
                for a in itxt_aff(p):
                    if not aff_is_duplicate(a, aff_map[nm]):
                        aff_map[nm].append(a)
        if not aff_map[nm]:
            for node in authors_any:
                if author_name_from(node) == nm:
                    for a in itxt_aff(node):
                        if not aff_is_duplicate(a, aff_map[nm]):
                            aff_map[nm].append(a)
        if not aff_map[nm]:
            for p in persons:
                pn = person_name_from(p)
                if pn and names_match_loose(nm, pn):
                    for a in itxt_aff(p):
                        if not aff_is_duplicate(a, aff_map[nm]):
                            aff_map[nm].append(a)

    if len(authors) == 1 and not aff_map[authors[0]] and all_affiliations:
        aff_map[authors[0]] = all_affiliations

    md["affiliations"] = []
    for nm in authors:
        if aff_map.get(nm):
            md["affiliations"].append({"author": nm, "affiliations": aff_map[nm]})

    abs_node = xml.find('.//tei:text/tei:front/tei:abstract', namespaces=tei) \
            or xml.find('.//tei:profileDesc/tei:abstract', namespaces=tei)
    if abs_node is not None:
        # itertext() concatenates <p> siblings with no separator, gluing
        # "Line one." + "Line two." into "Line one.Line two.". Join blocks
        # with a space so the abstract matches the gold text.
        paras = [normalize_spaces(''.join(p.itertext()))
                 for p in abs_node.findall('.//tei:p', namespaces=tei)]
        paras = [p for p in paras if p]
        md["abstract"] = ' '.join(paras) if paras else itxt(abs_node)

    j = xml.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title', namespaces=tei)
    if j is not None: md["journal"] = itxt(j)
    for unit, key in [("volume","volume"),("issue","issue"),("page","pages")]:
        el = xml.find(f'.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:biblScope[@unit="{unit}"]', namespaces=tei)
        if el is not None: md[key] = itxt(el)
    yr = xml.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date', namespaces=tei)
    if yr is not None: md["year"] = (yr.get('when') or itxt(yr))
    doi = xml.find('.//tei:idno[@type="DOI"]', namespaces=tei)
    if doi is not None: md["doi"] = itxt(doi)

    return md

def extract_from_pdf_grobid_fallback(pdf_path: Path) -> Dict[str, Any]:
    text_head = pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
    text_all  = pdf_text_all(pdf_path)
    md = {
        "format": "pdf",
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
    head_lines = [normalize_spaces(ln) for ln in (text_head or text_all).splitlines() if normalize_spaces(ln)]
    md["title"] = head_lines[0] if head_lines else ""
    parts = re.split(r',|\band\b', by, flags=re.I)
    for chunk in parts:
        name = re.sub(r'[\*\d\u00B9\u00B2\u00B3\u2070-\u2079\u2020\u2021\u00A7\u00B6]+', '', chunk)
        name = normalize_spaces(name).strip(',:;')
        if 2 <= len(name.split()) <= 6 and re.search(r'\b[A-Z][a-z]', name):
            if name not in md["authors"]:
                md["authors"].append(name)
    m = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)(\n\s*[A-Z][^\n]{{0,80}}\n|{KEYWORDS_HDR}\s*:|^Introduction|\n\s*1\.\s*Introduction)', text_head or text_all, re.I|re.M)
    if m:
        md["abstract"] = normalize_spaces(m.group(1))[:3000]
    mdoi = DOI_RE.search(text_all or text_head or "")
    md["doi"] = mdoi.group(0).rstrip('.,);]') if mdoi else ""
    return md

def extract_grobid(pdf_path: Path, grobid_ok: bool) -> Dict[str, Any]:
    tei_or_bib = extract_with_grobid(pdf_path) if grobid_ok else None
    md_primary: Optional[Dict[str, Any]] = None
    if tei_or_bib:
        # A truncated or malformed TEI response (GROBID timeout, dropped
        # connection) makes lxml raise. Fall through to the PDF-text fallback
        # below instead of losing the paper entirely.
        try:
            if tei_or_bib.strip().startswith('@'):
                md_primary = parse_bibtex(tei_or_bib)
            elif tei_or_bib.strip().startswith('<'):
                md_primary = parse_tei(tei_or_bib)
        except Exception as e:
            print(f"⚠️  TEI parse failed for {pdf_path.name}: {type(e).__name__}; using PDF-text fallback")
            md_primary = None
    md_fallback = extract_from_pdf_grobid_fallback(pdf_path)
    return md_primary or md_fallback

# -------------------- LLM extraction functions --------------------


def build_llm_prompt_affiliation_only(head_text: str, byline_text: str) -> str:
    """
    Affiliation-only prompt for the hybrid pipeline.

    byline_text is carved out of head_text by byline_block(), so sending both
    shipped the same characters twice (~1.4k chars/paper on this corpus).
    We send the byline alone when it already covers the head window, and only
    append HEAD_TEXT when it carries material the byline does not.
    """
    sections = []
    if byline_text:
        sections.append(f"BYLINE_TEXT:\n{byline_text}")
    if head_text and aff_key(head_text) not in aff_key(byline_text or ""):
        sections.append(f"HEAD_TEXT:\n{head_text}")
    body = "\n\n".join(sections)
    return (
        "You are a scientific metadata extractor. Extract ONLY author affiliations from the text. "
        "Return STRICT JSON with keys: authors (array of strings, full names), "
        "affiliations (array of objects: {author: str, affiliations: array[str]}). "
        "If a field is unknown, use an empty string or empty array.\n\n"
        f"{body}\n\n"
        "Output JSON only."
    )

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    m = re.search(r"```json\s*(\{[\s\S]*\})\s*```", s, re.I)
    if m:
        s = m.group(1)
    if not s.strip().startswith('{'):
        m2 = re.search(r"(\{[\s\S]*\})", s)
        if m2:
            s = m2.group(1)
    try:
        return json.loads(s)
    except Exception:
        return None

def call_openai_extract(head_text: str, byline_text: str) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI API to extract affiliations and authors (AlphaMind Extractor).
    
    Args:
        head_text: First few pages of PDF text
        byline_text: Author byline section
    
    Returns:
        Parsed JSON response or None on error
    """
    api_key = get_llm_api_key()
    if not api_key:
        return None

    prompt = build_llm_prompt_affiliation_only(head_text, byline_text)

    result = _chat_completion(prompt, api_key)
    if result is not None:
        return result
    if LLM_PROVIDER == "createai":
        # Gateway may not expose the OpenAI-compatible surface; use CreateAI's
        # native /query endpoint before giving up.
        return _createai_query(prompt, api_key)
    return None


def _record_usage(usage: Any) -> None:
    """Pull prompt/completion counts off a response and add them to the run total."""
    if not usage:
        return
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    prompt_toks = get("prompt_tokens") or get("input_tokens")
    completion_toks = get("completion_tokens") or get("output_tokens")
    _update_usage(prompt_toks, completion_toks)


def _chat_completion(prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenAI-wire chat completion against whichever gateway is configured.

    CreateAI speaks the same wire format, so the only difference from calling
    api.openai.com is base_url and the bearer token.
    """
    try:
        from openai import OpenAI
    except Exception:
        return None

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if LLM_PROVIDER == "createai":
        kwargs["base_url"] = CREATEAI_BASE_URL
    try:
        client = OpenAI(**kwargs)
    except Exception:
        return None

    messages = [
        {"role": "system", "content": "You are a metadata extractor. Respond with JSON only."},
        {"role": "user", "content": prompt},
    ]
    # Try structured-JSON mode first; not every gateway model supports it.
    for extra in ({"response_format": {"type": "json_object"}}, {}):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0,
                timeout=OPENAI_TIMEOUT,
                **extra,
            )
        except Exception:
            continue
        _record_usage(getattr(resp, "usage", None))
        content = resp.choices[0].message.content if resp.choices else ''
        parsed = try_parse_json(content)
        if parsed is not None:
            return parsed
    return None


def _createai_query(prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    CreateAI's native endpoint: POST {"query": ...} to /query with a bearer token.

    Documented at https://docs.aiml.asu.edu/ . Response shape varies by model, so
    probe the common containers for the generated text rather than assuming one.
    """
    try:
        resp = requests.post(
            CREATEAI_QUERY_URL,
            json={"query": prompt, "model_provider": os.environ.get("CREATEAI_MODEL_PROVIDER", "openai"),
                  "model_name": OPENAI_MODEL},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=OPENAI_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if isinstance(data, dict):
        _record_usage(data.get("usage"))
        for key in ("response", "result", "output", "text", "content", "message"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return try_parse_json(val)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return try_parse_json(msg["content"])
        # Envelope we don't recognise. Returning it verbatim would look like a
        # successful extraction carrying zero authors and zero affiliations,
        # which suppresses the regex fallback. Signal failure instead.
        if not any(k in data for k in ("authors", "affiliations")):
            return None
    return None

def extract_llm(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract affiliations and authors using LLM (AlphaMind Extractor).
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with authors and affiliations
    """
    # Tier 1: the small window. Affiliations live above the Abstract on page 1;
    # measured on the gold corpus, this window holds the answer for ~90% of
    # papers at ~11% of the characters of the original 3-page window.
    narrow_text = pdf_text_byline_window(pdf_path)
    by = byline_block(narrow_text)

    md_default = {
        "format": "llm",
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

    result = call_openai_extract(narrow_text or '', by)

    # Tier 2: only the ~10% of papers whose affiliation is not in the small
    # window pay for the full head read (SCAN_FIRST_N_PAGES_FOR_BYLINE pages).
    #
    # Escalate only when the model answered but found no affiliation — that is
    # the case more text can fix. `result is None` means the call itself failed
    # (auth, timeout, unreachable gateway); resending a larger prompt would just
    # double the failures, so fall through to the regex fallback instead.
    if result is not None and not _has_affiliation(result):
        head_text = pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
        if aff_key(head_text) != aff_key(narrow_text):
            wide = call_openai_extract(head_text or '', byline_block(head_text))
            if _has_affiliation(wide):
                result = wide
            elif result is None:
                result = wide

    if not result:
        # Fallback: extract authors from byline using regex
        md = md_default.copy()
        parts = re.split(r',|\band\b', by, flags=re.I)
        for chunk in parts:
            name = re.sub(r'[\*\d\u00B9\u00B2\u00B3\u2070-\u2079\u2020\u2021\u00A7\u00B6]+', '', chunk)
            name = normalize_spaces(name).strip(',:;')
            if 2 <= len(name.split()) <= 6 and re.search(r'\b[A-Z][a-z]', name):
                if name not in md["authors"]:
                    md["authors"].append(name)
        return md

    # Extract authors and affiliations from LLM response
    md = md_default.copy()
    md["authors"] = [normalize_spaces(a) for a in (result.get("authors") or []) if normalize_spaces(a)]
    affs = []
    for item in result.get("affiliations") or []:
        try:
            a = normalize_spaces(item.get("author", ""))
            vals = [normalize_spaces(x) for x in (item.get("affiliations", []) or []) if normalize_spaces(x)]
            if a and vals:
                affs.append({"author": a, "affiliations": vals})
        except Exception:
            continue
    md["affiliations"] = affs
    return md

# -------------------- Confidence routing --------------------

# An affiliation string GROBID is trusted on must name an institution-level
# organisation (not just a department), be a single affiliation rather than a
# run-on list, and be a plausible length.
_INSTITUTION_RE = re.compile(
    r'\b(universit|institute of technology|college|polytechnic|academy|school)\w*', re.I)


def affiliation_is_confident(text: str) -> bool:
    """
    Would we trust this single affiliation string without asking the LLM?

    Measured on the 46 linked gold rows after the raw_affiliation fix:
    the gate accepts 37 rows and is correct on 35 of them -- 94.6% precision
    (95% CI 82.3% - 98.5%). Re-run tools/measure_gate.py if the corpus changes.
    """
    a = normalize_spaces(text or "")
    if not a or a.lower() in ("not specified", "nan"):
        return False
    if not (GATE_MIN_CHARS <= len(a) <= GATE_MAX_CHARS):
        return False
    if a.count(';') + 1 > GATE_MAX_SEGMENTS:
        # Several affiliations joined together -- GROBID has probably run two
        # authors' affiliations into one string. Let the LLM separate them.
        return False
    if _looks_self_repeating(a):
        # Defence in depth. If the raw_affiliation fix ever regresses we get
        # "Purdue University Purdue University" again, which would otherwise
        # sail through this gate and reach the spreadsheet unreviewed.
        return False
    return bool(_INSTITUTION_RE.search(a))


def _looks_self_repeating(text: str) -> bool:
    """True if the string is a phrase followed by a near-copy of itself."""
    words = aff_key(text).split()
    n = len(words)
    if n < 4:
        return False
    half = n // 2
    # exact doubling, and the odd-length case where one stray token trails
    for cut in (half, half + (n % 2)):
        if 2 <= cut <= n - 2 and words[:cut] == words[cut:2 * cut]:
            return True
    # containment: the first half reappears verbatim later in the string
    first = ' '.join(words[:half])
    rest = ' '.join(words[half:])
    return bool(first) and first in rest


def grobid_affiliation_is_confident(affiliations: List[Dict[str, Any]]) -> bool:
    """
    Paper-level decision: skip the LLM only if GROBID produced at least one
    affiliation and *every* one of them clears the bar.

    Requiring all of them (rather than a majority) keeps precision high: one
    doubtful affiliation on a paper sends the whole paper to the LLM, which is
    cheap insurance at ~20% call volume.
    """
    if not affiliations:
        return False
    saw_one = False
    for item in affiliations:
        vals = [v for v in (item.get("affiliations") or []) if normalize_spaces(v)]
        if not vals:
            continue
        for v in vals:
            if not affiliation_is_confident(v):
                return False
            saw_one = True
    return saw_one


# -------------------- Hybrid extraction --------------------

def extract_hybrid(pdf_path: Path, grobid_ok: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract metadata using hybrid approach:
    - Use GROBID for fields where GROBID F1 >= 0.85
    - Use LLM for fields where GROBID F1 < 0.85 (only Affiliation)
    
    Runs GROBID and LLM extraction in parallel for optimal performance.
    Uses optimized affiliation-only LLM prompt to reduce tokens and API time.
    
    Returns: (hybrid_md, llm_md) tuple to avoid re-extracting LLM data
    """
    if ROUTE_ON_GROBID_CONFIDENCE:
        # Confidence routing: run GROBID first and only pay for the LLM when
        # GROBID's affiliation does not clear the bar. Measured on the gold
        # corpus after the raw_affiliation fix, the gate is right 94.6% of the
        # time (35/37) and skips ~80% of papers. See docstring on
        # grobid_affiliation_is_confident().
        #
        # Sequential rather than parallel, but faster on average: GROBID always
        # runs (~1.8s) and the LLM call (~3s) now fires on ~20% of papers.
        grobid_md = extract_grobid(pdf_path, grobid_ok)
        grobid_affs = grobid_md.get("affiliations", []) or []

        if grobid_affiliation_is_confident(grobid_affs):
            hybrid_md = grobid_md.copy()
            hybrid_md["grobid_affiliations"] = grobid_affs
            hybrid_md["affiliations"] = grobid_affs
            hybrid_md["affiliation_source"] = "grobid"
            with _USAGE_LOCK:
                _ROUTING_COUNTS["grobid_only"] += 1
            # llm_md mirrors GROBID so the row builder's name matching still works
            llm_md = {"format": "grobid-confident",
                      "authors": list(grobid_md.get("authors", []) or []),
                      "affiliations": grobid_affs}
            return hybrid_md, llm_md

        llm_md = extract_llm(pdf_path)
    else:
        # Original behaviour: always call the LLM, both engines in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            grobid_future = executor.submit(extract_grobid, pdf_path, grobid_ok)
            llm_future = executor.submit(extract_llm, pdf_path)
            grobid_md = grobid_future.result()
            llm_md = llm_future.result()

    # Combine: use GROBID values for most fields, LLM value for Affiliation.
    # Keep GROBID's own affiliations so the row builder can fall back to them
    # when the LLM's author naming does not line up with GROBID's.
    hybrid_md = grobid_md.copy()
    hybrid_md["grobid_affiliations"] = grobid_md.get("affiliations", []) or []
    hybrid_md["affiliations"] = llm_md.get("affiliations", [])
    hybrid_md["affiliation_source"] = "llm"
    with _USAGE_LOCK:
        _ROUTING_COUNTS["llm_called"] += 1

    return hybrid_md, llm_md

# -------------------- Recursion & Row helpers --------------------

def gather_pdfs(root: Path) -> List[Path]:
    paths: List[Path] = []
    if root.is_file() and root.suffix.lower() == '.pdf':
        return [root]
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith('.pdf'):
                paths.append(Path(r) / fn)
    uniq: List[Path] = []
    seen = set()
    for p in sorted(paths):
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq

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

def clean_for_excel(text: str) -> str:
    if text is None:
        return ''
    return re.sub(r'[\x00-\x1F\x7F]', '', str(text))

# -------------------- Main aggregation --------------------

def main():
    argv = [a for a in sys.argv[1:] if a not in ("--consolidate-header", "--no-consolidate-header")]
    if "--no-consolidate-header" in sys.argv[1:]:
        os.environ["CONSOLIDATE_HEADER"] = "0"
    elif "--consolidate-header" in sys.argv[1:]:
        os.environ["CONSOLIDATE_HEADER"] = "1"
    root = ROOT_FOLDER
    if argv:
        root = Path(argv[0]).expanduser().resolve()
    targets = abbreviate_targets(TARGET_AUTHORS[:])
    if len(argv) >= 2:
        targets = abbreviate_targets([a.strip() for a in argv[1].split(",") if a.strip()])

    if not root.exists():
        print(f"❌ Root folder does not exist: {root}")
        sys.exit(1)

    grobid_ok = is_grobid_alive()
    print(f"🌐 GROBID reachable at {GROBID_URL}: {'YES' if grobid_ok else 'NO (will use fallback)'}")
    print(f"   service=/api/{GROBID_SERVICE}  consolidateHeader={os.environ.get('CONSOLIDATE_HEADER', CONSOLIDATE_HEADER)}")
    print(f"🔀 Confidence routing: {'ON (LLM only when GROBID is unsure)' if ROUTE_ON_GROBID_CONFIDENCE else 'OFF (LLM on every paper)'}")

    llm_ok = bool(get_llm_api_key())
    if llm_ok:
        target = CREATEAI_BASE_URL if LLM_PROVIDER == "createai" else "https://api.openai.com"
        print(f"🤖 LLM available: YES (provider={LLM_PROVIDER}, model={OPENAI_MODEL})")
        print(f"   endpoint={target}")
    else:
        want = "ASU_CREATE_AI_KEY" if LLM_PROVIDER == "createai" else "OPENAI_API_KEY"
        print(f"🤖 LLM available: NO — set {want} (will use regex fallback)")

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
        md, llm_md = extract_hybrid(pdf, grobid_ok)
        authors: List[str] = md.get('authors', []) or []  # GROBID authors
        if not authors:
            return out_rows
        # Attempt to match any target author
        found_pairs: List[Tuple[str, str]] = []
        if targets:
            for t in targets:
                for f in authors:
                    if robust_author_match(t, f):
                        found_pairs.append((t, f))
        else:
            found_pairs = [(f, f) for f in authors]
        if not found_pairs:
            return out_rows
        # Affiliation mapping (from LLM, but match LLM author names to GROBID author names)
        # Reuse LLM metadata from extract_hybrid (no need to extract again)
        llm_authors: List[str] = llm_md.get('authors', []) or []
        llm_affiliations = md.get('affiliations', []) or []  # Already from LLM via extract_hybrid
        
        # Build affiliation map keyed by LLM author names (as provided by LLM)
        llm_aff_map: Dict[str, List[str]] = {}
        for item in llm_affiliations:
            llm_author_name = item.get('author', '')
            vals = item.get('affiliations', []) or []
            if llm_author_name:
                llm_aff_map.setdefault(llm_author_name, [])
                for v in vals:
                    if v and v not in llm_aff_map[llm_author_name]:
                        llm_aff_map[llm_author_name].append(v)
        
        # GROBID's own affiliations, used as a fallback below.
        grobid_aff_map: Dict[str, List[str]] = {}
        for item in md.get('grobid_affiliations', []) or []:
            nm = item.get('author', '')
            vals = [v for v in (item.get('affiliations', []) or []) if v]
            if nm and vals:
                grobid_aff_map[nm] = vals

        # Map GROBID author names to LLM author names for affiliation lookup
        aff_map: Dict[str, List[str]] = {}
        for grobid_author in authors:
            # Try to find matching LLM author name
            matched_llm_author = None
            for llm_author in llm_authors:
                if robust_author_match(grobid_author, llm_author):
                    matched_llm_author = llm_author
                    break
            # If we found a match, use LLM's affiliations for that author
            if matched_llm_author and matched_llm_author in llm_aff_map:
                aff_map[grobid_author] = llm_aff_map[matched_llm_author].copy()
            elif grobid_author in llm_aff_map:
                # Direct lookup, in case the names are already identical
                aff_map[grobid_author] = llm_aff_map[grobid_author].copy()
            else:
                # The LLM named this author differently (or returned nothing for
                # them). Since the raw_affiliation fix, GROBID's own affiliation
                # scores F1 0.832 -- far better than emitting "Not specified".
                aff_map[grobid_author] = grobid_aff_map.get(grobid_author, []).copy()
        
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
    if ROUTE_ON_GROBID_CONFIDENCE:
        skipped = _ROUTING_COUNTS["grobid_only"]
        called = _ROUTING_COUNTS["llm_called"]
        total = skipped + called
        if total:
            print(f"🔀 Confidence routing: {skipped}/{total} papers "
                  f"({100*skipped/total:.0f}%) answered by GROBID, "
                  f"{called} sent to the LLM")

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

    if pd is None:
        print("pandas is required for Excel export. Please install with: pip install pandas openpyxl")
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

    # Persist runtime and LLM cost/usage for compare_pipelines.py
    try:
        stats_path = Path(RUN_STATS_JSON)
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {}
        with _USAGE_LOCK:
            usage_snapshot = dict(_USAGE)
        hybrid_stats = {
            "runtime_seconds": round(dt, 3),
            "cost_usd": round(usage_snapshot.get("cost_usd", 0.0), 6),
            "prompt_tokens": int(usage_snapshot.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_snapshot.get("completion_tokens", 0)),
        }
        stats["alphamind_extractor"] = hybrid_stats
        stats["joshcoley"] = hybrid_stats  # legacy key for compare_pipelines
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"💰 Recorded AlphaMind Extractor stats in {stats_path}")
    except Exception as e:
        print(f"⚠️ Failed to write AlphaMind Extractor stats: {e}")

if __name__ == "__main__":
    main()

