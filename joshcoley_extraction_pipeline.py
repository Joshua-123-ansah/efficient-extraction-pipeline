#!/usr/bin/env python3
"""
JoshColey Hybrid Extraction Pipeline
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
import time
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
    import fitz  # PyMuPDF
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

# GROBID imports
import requests
from lxml import etree

# -------------------- CONFIG --------------------

# GROBID
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT = 60

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "60"))

# Root folder variable (can be changed):
ROOT_FOLDER: Path = Path(os.environ.get("ROOT_PAPERS_DIR", "My Papers")).expanduser().resolve()

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
OUTPUT_XLSX: Path = Path(os.environ.get("OUTPUT_XLSX", "joshcoley_results.xlsx")).resolve()

# Concurrency: be conservative for API rate-limits
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "2"))

# Heuristics / limits
SCAN_FIRST_N_PAGES_FOR_BYLINE = int(os.environ.get("HEAD_PAGES", "3"))
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+\b', re.I)

ABSTRACT_HDR = r'(Abstract|Summary|Résumé|Resumé|Zusammenfassung|Resumen|Резюме|摘要|概要|초록|ملخص)'
KEYWORDS_HDR = r'(Keywords?|Index Terms|Mots[- ]clés|Palabras clave|Schlüsselwörter|Ключевые слова|关键词)'

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
    n = normalize_spaces(name)
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
    n = normalize_spaces(name)
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

def normalize_name_for_compare_robust(s: str) -> str:
    if not s:
        return ''
    t = normalize_spaces(s).lower()
    t = strip_accents(t)
    t = re.sub(r'[^\w\-\']', '', t)
    return t

def robust_author_match(target: str, found: str) -> bool:
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
    return t_first_norm == f_first_norm and t_last_norm == f_last_norm

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
                'consolidateHeader': '1',
                'consolidateCitations': '0',
                'includeRawCitations': '0',
                'includeRawAffiliations': '1',
                'teiCoordinates': '0'
            }
            resp = session.post(f"{GROBID_URL}/api/processFulltextDocument", files=files, data=data, timeout=GROBID_TIMEOUT)
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
            if nm not in authors:
                authors.append(nm)
            author_nodes_by_name.setdefault(nm, []).append(a)
    md["authors"] = authors

    # Affiliations mapping (Note: GROBID affiliations are extracted but replaced with LLM results)
    def itxt_aff(node) -> List[str]:
        out: List[str] = []
        for aff in node.findall('.//tei:affiliation', namespaces=tei):
            t = itxt(aff)
            if t:
                out.append(t)
            for ref in aff.xpath('.//@ref', namespaces=tei):
                if ref in org_by_id and org_by_id[ref]:
                    out.append(org_by_id[ref])
        seen, cleaned = set(), []
        for a in out:
            c = re.sub(r'\s+', ' ', a).strip()
            c = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', c)
            if c and c not in seen and len(c) > 8:
                seen.add(c)
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
        t = re.sub(r'\s+', ' ', itxt(aff)).strip()
        t = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', t)
        if t and len(t) > 8 and t not in all_affiliations:
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
                if a not in aff_map[nm]:
                    aff_map[nm].append(a)
        if not aff_map[nm]:
            p = person_by_name.get(normalize_spaces(nm).lower())
            if p is not None:
                for a in itxt_aff(p):
                    if a not in aff_map[nm]:
                        aff_map[nm].append(a)
        if not aff_map[nm]:
            for node in authors_any:
                if author_name_from(node) == nm:
                    for a in itxt_aff(node):
                        if a not in aff_map[nm]:
                            aff_map[nm].append(a)
        if not aff_map[nm]:
            for p in persons:
                pn = person_name_from(p)
                if pn and names_match_loose(nm, pn):
                    for a in itxt_aff(p):
                        if a not in aff_map[nm]:
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
        md["abstract"] = itxt(abs_node)

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
        if tei_or_bib.strip().startswith('@'):
            md_primary = parse_bibtex(tei_or_bib)
        elif tei_or_bib.strip().startswith('<'):
            md_primary = parse_tei(tei_or_bib)
    md_fallback = extract_from_pdf_grobid_fallback(pdf_path)
    return md_primary or md_fallback

# -------------------- LLM extraction functions --------------------


def build_llm_prompt_affiliation_only(head_text: str, byline_text: str) -> str:
    """
    Optimized prompt for JoshColey: only extract affiliations and authors (for matching).
    This reduces token usage and API response time significantly.
    """
    return (
        "You are a scientific metadata extractor. Extract ONLY author affiliations from the text. "
        "Return STRICT JSON with keys: authors (array of strings, full names), "
        "affiliations (array of objects: {author: str, affiliations: array[str]}). "
        "If a field is unknown, use an empty string or empty array.\n\n"
        f"BYLINE_TEXT:\n{byline_text}\n\n"
        f"HEAD_TEXT:\n{head_text}\n\n"
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
    Call OpenAI API to extract affiliations and authors (optimized for JoshColey).
    
    Args:
        head_text: First few pages of PDF text
        byline_text: Author byline section
    
    Returns:
        Parsed JSON response or None on error
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    
    prompt = build_llm_prompt_affiliation_only(head_text, byline_text)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=OPENAI_TIMEOUT,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ''
            return try_parse_json(content)
        except Exception:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=OPENAI_TIMEOUT,
            )
            content = resp.choices[0].message.content if resp.choices else ''
            return try_parse_json(content)
    except Exception:
        try:
            import openai
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                request_timeout=OPENAI_TIMEOUT,
            )
            content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ''
            return try_parse_json(content)
        except Exception:
            return None

def extract_llm(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract affiliations and authors using LLM (optimized for JoshColey).
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with authors and affiliations
    """
    head_text = pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
    all_text = pdf_text_all(pdf_path)
    by = byline_block(head_text or all_text)

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

    result = call_openai_extract(head_text or '', by)
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
    # Run GROBID and LLM extraction in parallel for better performance
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both extractions concurrently
        grobid_future = executor.submit(extract_grobid, pdf_path, grobid_ok)
        # Extract affiliations and authors using optimized LLM call
        llm_future = executor.submit(extract_llm, pdf_path)
        
        # Wait for both to complete
        grobid_md = grobid_future.result()
        llm_md = llm_future.result()
    
    # Combine: use GROBID values for most fields, LLM value for Affiliation
    hybrid_md = grobid_md.copy()
    hybrid_md["affiliations"] = llm_md.get("affiliations", [])
    
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

    grobid_ok = is_grobid_alive()
    print(f"🌐 GROBID reachable at {GROBID_URL}: {'YES' if grobid_ok else 'NO (will use fallback)'}")
    
    openai_ok = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    print(f"🤖 OpenAI available: {'YES (model=' + OPENAI_MODEL + ')' if openai_ok else 'NO (will use fallback)'}")

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
            else:
                # No match found, try direct lookup (in case names are exactly the same)
                if grobid_author in llm_aff_map:
                    aff_map[grobid_author] = llm_aff_map[grobid_author].copy()
                else:
                    aff_map[grobid_author] = []
        
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
    if MAX_WORKERS > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, p): p for p in candidates}
            iterator = tqdm(as_completed(futures), total=len(futures), desc='Processing', unit='pdf') if tqdm else as_completed(futures)
            for fut in iterator:
                try:
                    rs = fut.result()
                    rows.extend(rs)
                except Exception:
                    pass
                processed += 1
    else:
        iterable = tqdm(candidates, desc='Processing', unit='pdf') if tqdm else candidates
        for p in iterable:
            rows.extend(process_one(p))
            processed += 1

    dt = time.time() - t0
    rate = (processed / dt) if dt > 0 else 0
    print(f"⏱️ Done in {dt:.1f}s ({rate:.2f} pdf/s). Matched rows: {len(rows)}")

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

if __name__ == "__main__":
    main()

