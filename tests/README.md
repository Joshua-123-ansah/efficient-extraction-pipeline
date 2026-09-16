# Test suite

```bash
pip install pytest
python -m pytest tests/ -q
```

No network required. GROBID and the LLM gateway are stubbed, so the suite runs
without a GROBID server or an API key. Tests that need PDFs read from
`data/corpora/my_papers/`; tests that need scoring read `data/gold/Gold Standard.xlsx` and the
`*_results.xlsx` files. Any of these missing causes a `skip`, not a failure.

## Layout

| File | Covers |
|---|---|
| `conftest.py` | Fixtures and a TEI builder (`build_tei`, `author_xml`) |
| `test_names.py` | Author-name parsing, suffixes, initials, abbreviation, matching |
| `test_tei_parsing.py` | GROBID TEI: authors, affiliations, abstract, malformed input |
| `test_text_extraction.py` | Abstract/keyword slicing, byline block, PDF windows |
| `test_llm_path.py` | Prompt construction, provider routing, two-tier escalation |
| `test_scoring.py` | Record linkage, TP/FP/FN accounting, report tables |
| `test_end_to_end.py` | Full `main()` runs, row building, failure reporting |
| `test_uncovered.py` | Network boundaries, SDK stubs, per-pipeline entry points |
| `test_uncovered_round2.py` | Loose author matching, `--run` path, NLP without spaCy |
| `test_fallback_branches.py` | Linkage fallbacks, cost estimation, missing PyMuPDF |

Name-handling tests run against all five modules that carry a copy of those
helpers, so the copies cannot drift apart.

## Regressions these tests pin

Each of these was a real defect found and fixed in this codebase:

1. **Affiliation doubling** — `includeRawAffiliations=1` nests a raw note beside
   the parsed `orgName`; `itertext()` over both produced
   `"Purdue University Purdue University"`, which drove measured affiliation
   F1 to 0.038.
2. **Author double-counting** — the same people appear under `<analytic>` and
   `<titleStmt>` in different renderings; exact-string dedup let variants
   through, inflating `Total Authors` and shifting `Author's Position`.
3. **Capturing header patterns** — `ABSTRACT_HDR`/`KEYWORDS_HDR` were capturing
   groups embedded in larger patterns, shifting every group index so extraction
   returned the literal word `"Abstract"`.
4. **Glued abstracts** — `itertext()` joined `<p>` siblings with no separator.
5. **Name matching** — initials (`B. Coley`), suffixes (`James Holly, Jr.`) and
   apostrophes (`Na'ilah`) all failed to match, silently dropping papers.
6. **Mangled targets** — `abbreviate_target_name` read `"James Holly, Jr."` as
   Last=`"James Holly"`, First=`"Jr."`, producing `"Jr. James Holly"`.
7. **Cross-wired scoring** — linkage matched on File Path alone, so on papers
   with two target authors one author's gold row was scored against the other's
   prediction.
8. **Swallowed failures** — the parallel path used `except Exception: pass` and
   the serial path had no guard at all.
9. **Malformed TEI** — a truncated GROBID response raised instead of falling
   back to the PDF-text path.
10. **Pointless escalation** — a failed API call was retried with a *larger*
    prompt, doubling failures.
11. **Mismatched pricing** — the two pipelines priced tokens differently, making
    the cost comparison meaningless.

## Coverage

```bash
pip install coverage
COVERAGE_FILE=/tmp/.coverage python -m coverage run --source=. \
  --omit="tests/*,.venv/*" -m pytest tests/ -q
COVERAGE_FILE=/tmp/.coverage python -m coverage report
```

Currently ~84% of statements; every function in the five main modules is
entered by at least one test. The remaining uncovered lines are import guards
(`fitz = None` when PyMuPDF is absent) and exception handlers.

## What is NOT covered

- Live GROBID. The HTTP call is stubbed; use `python scripts/preflight_check.py`
  against a running server.
- Live CreateAI. The SDK and `/query` endpoint are stubbed; `scripts/preflight_check.py`
  probes the real gateway.
- spaCy model paths, which skip unless `en_core_web_sm` is installed.
