#!/usr/bin/env python3
"""
LLM Extraction Pipeline (OpenAI)
- Recursively scan a root folder for PDFs
- Prefilter using same author heuristics as GROBID to reduce cost
- Use OpenAI LLM to extract metadata as strict JSON
- Export Excel with columns: Title, Target Author, Found Author Name, Author's Position,
  Total Authors, Affiliation, Abstract, File Path, Paper Row
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

# -------------------- CONFIG --------------------

# Root folder variable (can be changed):
ROOT_FOLDER: Path = Path(os.environ.get("ROOT_PAPERS_DIR", "My Papers")).expanduser().resolve()

# Target authors variable (can be changed): reuse your list for fair comparison
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
OUTPUT_XLSX: Path = Path(os.environ.get("OUTPUT_XLSX", "llm_results.xlsx")).resolve()

# Concurrency: be conservative for API rate-limits
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "2"))

# Heuristics / limits
SCAN_FIRST_N_PAGES_FOR_BYLINE = int(os.environ.get("HEAD_PAGES", "3"))
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+\b', re.I)

ABSTRACT_HDR = r'(Abstract|Summary|Résumé|Resumé|Zusammenfassung|Resumen|Резюме|摘要|概要|초록|ملخص)'
KEYWORDS_HDR = r'(Keywords?|Index Terms|Mots[- ]clés|Palabras clave|Schlüsselwörter|Ключевые слова|关键词)'

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "60"))

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

# -------------------- Prefilter (same as Grobid) --------------------

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
        if re.search(rf'\b{re.escape(first_lower[0])}\.\?\s+{re.escape(last_lower)}(?:\d+)?\b', hay):
            return True
        
        # Pattern 3: Last, First (e.g., "Miles, Monica")
        if re.search(rf'\b{re.escape(last_lower)}\s*,\s*{re.escape(first_lower)}\b', hay):
            return True
        
        # Pattern 4: Last, Initial (e.g., "Miles, M.")
        if re.search(rf'\b{re.escape(last_lower)}\s*,\s*{re.escape(first_lower[0])}\.\?\b', hay):
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

# -------------------- LLM helpers --------------------

def slice_abstract_full(text: str) -> str:
    if not text:
        return ''
    # Prefer Abstract .. Keywords; else Abstract .. Introduction
    m_abs_kw = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)\b{KEYWORDS_HDR}\b\s*[:\n]', text, re.I)
    if m_abs_kw:
        return normalize_spaces(m_abs_kw.group(1))
    m_abs_intro = re.search(rf'\b{ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)(^Introduction|\n\s*1\.\s*Introduction)', text, re.I | re.M)
    if m_abs_intro:
        return normalize_spaces(m_abs_intro.group(1))
    return ''

def build_llm_prompt(head_text: str, byline_text: str, abstract_candidate: str) -> str:
    return (
        "You are a meticulous scientific metadata extractor. "
        "Given raw text from the beginning of a scientific article, extract structured metadata. "
        "Return STRICT JSON with keys: title (string), authors (array of strings, full names), "
        "affiliations (array of objects: {author: str, affiliations: array[str]}), emails (array of objects: {author: str, emails: array[str]}), "
        "abstract (string, full, do not truncate), keywords (string), journal (string), volume (string), issue (string), pages (string), year (string), doi (string). "
        "If a field is unknown, use an empty string or empty array.\n\n"
        f"BYLINE_TEXT:\n{byline_text}\n\n"
        f"HEAD_TEXT:\n{head_text}\n\n"
        f"ABSTRACT_CANDIDATE_FROM_REGEX (use if plausible):\n{abstract_candidate}\n\n"
        "Output JSON only."
    )

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # Strip fences ```json ... ``` if present
    m = re.search(r"```json\s*(\{[\s\S]*\})\s*```", s, re.I)
    if m:
        s = m.group(1)
    # Fallback: first {...}
    if not s.strip().startswith('{'):
        m2 = re.search(r"(\{[\s\S]*\})", s)
        if m2:
            s = m2.group(1)
    try:
        return json.loads(s)
    except Exception:
        return None

def call_openai_extract(head_text: str, byline_text: str, abstract_candidate: str) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    prompt = build_llm_prompt(head_text, byline_text, abstract_candidate)
    # Prefer new SDK interface, fallback to legacy if needed
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
            # Try without response_format
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

# -------------------- LLM extraction core --------------------

def extract_from_pdf_llm(pdf_path: Path) -> Dict[str, Any]:
    head_text = pdf_text_first_n(pdf_path, SCAN_FIRST_N_PAGES_FOR_BYLINE)
    all_text = pdf_text_all(pdf_path)
    by = byline_block(head_text or all_text)
    abstract_candidate = slice_abstract_full(head_text or all_text)

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

    result = call_openai_extract(head_text or '', by, abstract_candidate)
    if not result:
        # Minimal fallback: regex-heuristic similar to NLP pipeline to avoid empty rows
        md = md_default.copy()
        # Title (first significant line)
        lines = [normalize_spaces(ln) for ln in (head_text or all_text).splitlines() if normalize_spaces(ln)]
        md["title"] = lines[0] if lines else ''
        # Authors from byline
        parts = re.split(r',|\band\b', by, flags=re.I)
        for chunk in parts:
            name = re.sub(r'[\*\d\u00B9\u00B2\u00B3\u2070-\u2079\u2020\u2021\u00A7\u00B6]+', '', chunk)
            name = normalize_spaces(name).strip(',:;')
            if 2 <= len(name.split()) <= 6 and re.search(r'\b[A-Z][a-z]', name):
                if name not in md["authors"]:
                    md["authors"].append(name)
        # Abstract candidate
        md["abstract"] = abstract_candidate
        # DOI
        mdoi = DOI_RE.search(all_text or head_text or '')
        md["doi"] = mdoi.group(0).rstrip('.,);]') if mdoi else ''
        return md

    # Normalize result into our structure
    md = md_default.copy()
    md["title"] = normalize_spaces(result.get("title", ""))
    md["authors"] = [normalize_spaces(a) for a in (result.get("authors") or []) if normalize_spaces(a)]
    # affiliations: list of {author, affiliations: [..]}
    affs = []
    for item in result.get("affiliations") or []:
        try:
            a = normalize_spaces(item.get("author", ""))
            vals = [normalize_spaces(x) for x in (item.get("affiliations") or []) if normalize_spaces(x)]
            if a and vals:
                affs.append({"author": a, "affiliations": vals})
        except Exception:
            continue
    md["affiliations"] = affs
    # emails: list of {author, emails: [..]}
    emails = []
    for item in result.get("emails") or []:
        try:
            a = normalize_spaces(item.get("author", ""))
            vals = [normalize_spaces(x) for x in (item.get("emails") or []) if EMAIL_RE.fullmatch(normalize_spaces(x))]
            if a and vals:
                emails.append({"author": a, "emails": vals})
        except Exception:
            continue
    md["emails"] = emails
    md["abstract"] = normalize_spaces(result.get("abstract", ""))
    md["keywords"] = normalize_spaces(result.get("keywords", ""))
    md["journal"] = normalize_spaces(result.get("journal", ""))
    md["volume"] = normalize_spaces(result.get("volume", ""))
    md["issue"] = normalize_spaces(result.get("issue", ""))
    md["pages"] = normalize_spaces(result.get("pages", ""))
    md["year"] = normalize_spaces(result.get("year", ""))
    md["doi"] = normalize_spaces(result.get("doi", ""))
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

    # Report OpenAI availability (single line, like Grobid)
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
        md = extract_from_pdf_llm(pdf)
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


