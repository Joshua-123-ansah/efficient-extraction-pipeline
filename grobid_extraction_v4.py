#!/usr/bin/env python3
"""
GROBID Extraction Pipeline
- Recursively scan a root folder for PDFs
- Filter/process only papers containing any target authors
- Extract metadata using GROBID server
- Export Excel with columns: Title, Target Author, Found Author Name, Author's Position,
  Total Authors, Affiliation, Abstract, File Path, Paper Row
"""

import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import unicodedata
import requests
from lxml import etree
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional engines
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

# -------------------- CONFIG --------------------
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
TIMEOUT = 60

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
OUTPUT_XLSX: Path = Path(os.environ.get("OUTPUT_XLSX", "grobid_results_v4.xlsx")).resolve()
# Optional worker count for concurrency
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "4"))

# Heuristics / limits
SCAN_FIRST_N_PAGES_FOR_BYLINE = 3
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
    # This is what remove the punctuations. 
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
    """Return (first, middle_tokens, last) - simple and focused."""
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
    """Abbreviate middle names while preserving titles and complex last names."""
    first, middles, last = parse_name_for_abbrev(name)
    if not first or not last:
        return name
    
    # Don't abbreviate if it's already an initial
    if len(first) == 1 and first.endswith('.'):
        return name
    
    # Handle complex last names (keep full last name)
    if not middles:
        return f"{first} {last}"
    
    # Abbreviate middle names
    mid_initials = []
    for middle in middles:
        if middle and len(middle) > 1:
            mid_initials.append(middle[0].upper() + '.')
        else:
            mid_initials.append(middle)
    
    return f"{first} {' '.join(mid_initials)} {last}".strip()

def abbreviate_targets(targets: List[str]) -> List[str]:
    return [abbreviate_target_name(t) for t in (targets or [])]

# robust match: last names equal; first token/initial compatible

def extract_first_last(name: str) -> Tuple[str, str]:
    """Extract first and last name - simple and focused."""
    n = normalize_spaces(name)
    if not n:
        return '', ''
    
    # Handle "Last, First ..." form
    if ',' in n:
        parts = [p.strip() for p in n.split(',', 1)]
        if len(parts) == 2:
            last_part = parts[0]
            first_part = parts[1]
            # Extract first name from first part
            first_tokens = [t for t in re.split(r'[\s\-]+', first_part) if t]
            first = first_tokens[0] if first_tokens else ''
            # Extract last name from last part
            last_tokens = [t for t in re.split(r'[\s\-]+', last_part) if t]
            last = last_tokens[-1] if last_tokens else ''
            return first, last
    
    # Default "First ... Last" - take first and last tokens
    tokens = [t for t in re.split(r'[\s\-]+', n) if t]
    if not tokens:
        return '', ''
    if len(tokens) == 1:
        return tokens[0], ''
    return tokens[0], tokens[-1]

def normalize_name_for_compare_robust(s: str) -> str:
    """More robust normalization for name comparison."""
    if not s:
        return ''
    # Normalize spaces and convert to lowercase
    t = normalize_spaces(s).lower()
    # Strip accents
    t = strip_accents(t)
    # Remove punctuation but keep hyphens and apostrophes for compound names
    t = re.sub(r'[^\w\-\']', '', t)
    return t

def robust_author_match(target: str, found: str) -> bool:
    """Simple and focused: if first and last names match, it's a match."""
    if not target or not found:
        return False
    
    t_first, t_last = extract_first_last(target)
    f_first, f_last = extract_first_last(found)
    
    if not t_first or not t_last or not f_first or not f_last:
        return False
    
    # Normalize names for comparison
    t_first_norm = normalize_name_for_compare_robust(t_first)
    t_last_norm = normalize_name_for_compare_robust(t_last)
    f_first_norm = normalize_name_for_compare_robust(f_first)
    f_last_norm = normalize_name_for_compare_robust(f_last)
    
    # Simple rule: first AND last names must match exactly
    return t_first_norm == f_first_norm and t_last_norm == f_last_norm

# -------------------- GROBID path --------------------

def is_grobid_alive() -> bool:
    try:
        with requests.Session() as s:
            r = s.get(f"{GROBID_URL}/api/isalive", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def extract_with_grobid(pdf_path: Path) -> Optional[str]:
    # Reuse a session for connection pooling
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
            resp = session.post(f"{GROBID_URL}/api/processFulltextDocument", files=files, data=data, timeout=TIMEOUT)
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
    # Remove XML declaration if present
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

    # Affiliations mapping
    def itxt_aff(node) -> List[str]:
        out: List[str] = []
        for aff in node.findall('.//tei:affiliation', namespaces=tei):
            t = itxt(aff)
            if t:
                out.append(t)
            for ref in aff.xpath('.//@ref', namespaces=tei):
                if ref in org_by_id and org_by_id[ref]:
                    out.append(org_by_id[ref])
        # clean & dedupe
        seen, cleaned = set(), []
        for a in out:
            c = re.sub(r'\s+', ' ', a).strip()
            c = re.sub(r'^[\d\*\u2020\u2021\u00A7\u00B6]+\s*', '', c)
            if c and c not in seen and len(c) > 8:
                seen.add(c)
                cleaned.append(c)
        return cleaned

    # listPerson fallback
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

# -------------------- PDF fallback path --------------------

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

# Quick prefilter: check head text for any target author to avoid heavy GROBID on non-matches

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

def parse_authors_from_byline(byline: str) -> List[str]:
    parts = re.split(r',|\band\b', byline, flags=re.I)
    out: List[str] = []
    for chunk in parts:
        name = re.sub(r'[\*\d\u00B9\u00B2\u00B3\u2070-\u2079\u2020\u2021\u00A7\u00B6]+', '', chunk)
        name = normalize_spaces(name).strip(',:;')
        if 2 <= len(name.split()) <= 6 and re.search(r'\b[A-Z][a-z]', name):
            out.append(name)
    return [x for i, x in enumerate(out) if x and x not in out[:i]]

# -------------------- Orchestration --------------------

def extract_from_pdf(pdf_path: Path) -> Dict[str, Any]:
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
    authors = parse_authors_from_byline(by)
    md["authors"] = authors
    head_lines = [normalize_spaces(ln) for ln in (text_head or text_all).splitlines() if normalize_spaces(ln)]
    md["title"] = head_lines[0] if head_lines else ""
    # abstract
    m = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)(\n\s*[A-Z][^\n]{{0,80}}\n|{KEYWORDS_HDR}\s*:|^Introduction|\n\s*1\.\s*Introduction)', text_head or text_all, re.I|re.M)
    if m:
        md["abstract"] = normalize_spaces(m.group(1))[:3000]
    mdoi = DOI_RE.search(text_all or text_head or "")
    md["doi"] = mdoi.group(0).rstrip('.,);]') if mdoi else ""
    return md

def process_pdf(pdf_path: Path, grobid_ok: bool) -> Dict[str, Any]:
    tei_or_bib = extract_with_grobid(pdf_path) if grobid_ok else None
    md_primary: Optional[Dict[str, Any]] = None
    if tei_or_bib:
        if tei_or_bib.strip().startswith('@'):
            md_primary = parse_bibtex(tei_or_bib)
        elif tei_or_bib.strip().startswith('<'):
            md_primary = parse_tei(tei_or_bib)
    md_fallback = extract_from_pdf(pdf_path)
    return md_primary or md_fallback

# -------------------- Recursion & Row --------------------

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

@lru_cache(maxsize=1000)
def get_file_row_cached(file_path: str) -> int:
    """Get the row position of the file in its directory - CACHED VERSION."""
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
    # Allow optional CLI overrides: root folder and comma-separated authors
    root = ROOT_FOLDER
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1]).expanduser().resolve()
    # Abbreviate TARGET_AUTHORS (middle -> initial) before matching
    targets = abbreviate_targets(TARGET_AUTHORS[:])
    if len(sys.argv) >= 3:
        # authors passed as comma-separated string
        targets = abbreviate_targets([a.strip() for a in sys.argv[2].split(',') if a.strip()])

    if not root.exists():
        print(f"❌ Root folder does not exist: {root}")
        sys.exit(1)

    grobid_ok = is_grobid_alive()
    print(f"🌐 GROBID reachable at {GROBID_URL}: {'YES' if grobid_ok else 'NO (will use fallback)'}")
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
        md = process_pdf(pdf, grobid_ok)
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
    if MAX_WORKERS > 1 and len(candidates) > 1:
        iterator = None
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