#!/usr/bin/env python3
"""
Preflight check: confirm GROBID and the ASU CreateAI gateway are reachable and
correctly configured before running a full extraction.

    python preflight_check.py

Checks, in order:
  1. GROBID is alive, and which service endpoint responds
  2. GROBID returns raw affiliations without the duplication bug
  3. Which CreateAI base URL actually accepts an OpenAI-style chat completion
  4. A real affiliation-extraction round trip

Nothing here writes results files; it only reports.
"""

import os
import sys
import json
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline_paths import MY_PAPERS

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import requests

GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070").rstrip('/')
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Candidate OpenAI-compatible roots to probe. The docs list the gateway host;
# the exact suffix is confirmed empirically here rather than guessed in code.
CANDIDATE_BASES = [
    os.environ.get("CREATEAI_BASE_URL", "").rstrip('/'),
    "https://api-main.aiml.asu.edu/v1",
    "https://api-main.aiml.asu.edu/openai/v1",
    "https://api-main.aiml.asu.edu",
]

OK, BAD, WARN = "  [ok]", "  [FAIL]", "  [warn]"


def get_key() -> str:
    for var in ("CREATEAI_API_KEY", "ASU_CREATE_AI_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(var, "").strip()
        if v:
            print(f"  using credential from {var}")
            return v
    return ""


def check_grobid() -> None:
    print("\n[1] GROBID")
    try:
        r = requests.get(f"{GROBID_URL}/api/isalive", timeout=5)
        print(f"{OK} isalive -> {r.status_code} {r.text.strip()[:20]}")
    except Exception as e:
        print(f"{BAD} cannot reach {GROBID_URL}: {type(e).__name__}")
        print("       start it with: docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-full")
        return
    try:
        v = requests.get(f"{GROBID_URL}/api/version", timeout=5)
        print(f"{OK} version -> {v.text.strip()}")
    except Exception:
        print(f"{WARN} /api/version did not respond (older GROBID builds omit it)")


def check_grobid_affiliations(pdf: Path) -> None:
    print(f"\n[2] GROBID affiliation parsing  ({pdf.name})")
    for service in ("processHeaderDocument", "processFulltextDocument"):
        try:
            t0 = time.time()
            with open(pdf, 'rb') as f:
                r = requests.post(
                    f"{GROBID_URL}/api/{service}",
                    files={'input': f},
                    data={'consolidateHeader': '0', 'includeRawAffiliations': '1'},
                    timeout=120,
                )
            dt = time.time() - t0
            if r.status_code != 200:
                print(f"{WARN} {service} -> HTTP {r.status_code}")
                continue
            tei = r.content.decode('utf-8', errors='ignore')
            has_raw = 'raw_affiliation' in tei
            print(f"{OK} {service}: {dt:5.2f}s, {len(tei):,} bytes, raw_affiliation={'yes' if has_raw else 'NO'}")
            if not has_raw:
                print(f"{WARN}   without raw_affiliation the parser falls back to structured orgName text")
        except Exception as e:
            print(f"{BAD} {service}: {type(e).__name__}: {e}")

    # Confirm the parser produces un-doubled strings
    try:
        import alphamind_extractor as jc
        md = jc.extract_grobid(pdf, True)
        affs = md.get("affiliations") or []
        if not affs:
            print(f"{WARN} parser returned no affiliations for this PDF")
        for item in affs[:3]:
            for a in item.get("affiliations", [])[:2]:
                words = a.lower().split()
                half = len(words) // 2
                doubled = half > 2 and words[:half] == words[half:2 * half]
                flag = BAD + " DOUBLED:" if doubled else OK
                print(f"{flag} {item['author']}: {a[:90]}")
    except Exception as e:
        print(f"{WARN} could not run parser check: {type(e).__name__}: {e}")


def check_createai(key: str) -> None:
    print("\n[3] CreateAI gateway")
    if not key:
        print(f"{BAD} no credential found (set ASU_CREATE_AI_KEY in .env)")
        return
    try:
        from openai import OpenAI
    except Exception:
        print(f"{BAD} openai SDK not installed: pip install openai")
        return

    working = None
    seen = set()
    for base in CANDIDATE_BASES:
        if not base or base in seen:
            continue
        seen.add(base)
        try:
            client = OpenAI(api_key=key, base_url=base, timeout=30)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0,
            )
            txt = (resp.choices[0].message.content or "").strip()[:40]
            u = getattr(resp, "usage", None)
            toks = f", tokens in/out={getattr(u,'prompt_tokens','?')}/{getattr(u,'completion_tokens','?')}" if u else ""
            print(f"{OK} {base} -> {txt!r}{toks}")
            working = working or base
        except Exception as e:
            msg = str(e).split('\n')[0][:110]
            print(f"{WARN} {base} -> {type(e).__name__}: {msg}")

    print("\n[4] Native /query endpoint")
    try:
        r = requests.post(
            os.environ.get("CREATEAI_QUERY_URL", "https://api-main.aiml.asu.edu/query"),
            json={"query": "Reply with the single word: ok"},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=30,
        )
        body = r.text[:200].replace('\n', ' ')
        print(f"{OK if r.status_code == 200 else WARN} HTTP {r.status_code}: {body}")
        if r.status_code == 200:
            try:
                print(f"       response keys: {list(r.json().keys())}")
            except Exception:
                pass
    except Exception as e:
        print(f"{WARN} {type(e).__name__}: {e}")

    print()
    if working:
        print(f"==> Set this in your .env:   CREATEAI_BASE_URL={working}")
    else:
        print("==> No OpenAI-compatible base URL accepted the request.")
        print("    Check the exact path at https://docs.aiml.asu.edu/openai-compatible")
        print("    and the model name at https://docs.aiml.asu.edu/models")
        print("    The pipeline will fall back to the native /query endpoint if that works.")


def main() -> None:
    print("=" * 66)
    print("Preflight check")
    print("=" * 66)
    key = get_key()
    check_grobid()

    root = Path(os.environ.get("ROOT_PAPERS_DIR", str(MY_PAPERS)))
    pdfs = sorted(root.rglob("*.pdf")) if root.exists() else []
    if pdfs:
        check_grobid_affiliations(pdfs[0])
    else:
        print(f"\n[2] skipped — no PDFs under {root}")

    check_createai(key)
    print("\nDone.")


if __name__ == "__main__":
    main()
