"""
Whole-pipeline runs with GROBID and the LLM stubbed out.

Covers the row-building and reporting code that only executes inside main(),
including the failure paths that previously swallowed errors silently.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipeline_paths import MY_PAPERS
from conftest import build_tei, author_xml

EXPECTED_COLUMNS = ['Title', 'Target Author', 'Found Author Name', "Author's Position",
                    'Total Authors', 'Affiliation', 'Abstract', 'File Path', 'Paper Row']


@pytest.fixture
def corpus(tmp_path, repo):
    """Two real PDFs copied into a temp directory."""
    src = sorted(MY_PAPERS.rglob("*.pdf")) if MY_PAPERS.exists() else []
    if len(src) < 2:
        pytest.skip("need at least 2 PDFs under data/corpora/my_papers")
    d = tmp_path / "papers"
    d.mkdir()
    for p in src[:2]:
        (d / p.name).write_bytes(p.read_bytes())
    return d


@pytest.fixture
def good_tei():
    return build_tei(
        title="A Test Paper",
        analytic_authors=[
            author_xml("Brooke Coley", raw_affiliations=["Arizona State University"]),
            author_xml("Cid Dee", raw_affiliations=["Purdue University"]),
        ],
        abstract_paragraphs=["First part.", "Second part."])


def run_hybrid(jc, monkeypatch, tmp_path, corpus, tei_for, llm_response, workers=1):
    """Execute joshcoley main() end to end with the network stubbed."""
    out = tmp_path / "out.xlsx"
    stats = tmp_path / "stats.json"
    monkeypatch.setattr(jc, "OUTPUT_XLSX", out)
    monkeypatch.setattr(jc, "RUN_STATS_JSON", str(stats))
    monkeypatch.setattr(jc, "MAX_WORKERS", workers)
    monkeypatch.setattr(jc, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(jc, "extract_with_grobid", tei_for)
    monkeypatch.setattr(jc, "call_openai_extract", llm_response)
    monkeypatch.setattr(jc, "quick_author_hit", lambda text, targets: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    jc.main()
    return out, stats


def llm_ok(head, byline):
    return {"authors": ["Brooke Coley"],
            "affiliations": [{"author": "Brooke Coley",
                              "affiliations": ["Arizona State University"]}]}


# ------------------------------------------------------------- happy path

def test_writes_expected_schema(jc, monkeypatch, tmp_path, corpus, good_tei):
    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, lambda p: good_tei, llm_ok)
    df = pd.read_excel(out)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2


def test_row_values_are_correct(jc, monkeypatch, tmp_path, corpus, good_tei):
    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, lambda p: good_tei, llm_ok)
    row = pd.read_excel(out).iloc[0]
    assert row["Title"] == "A Test Paper"
    assert row["Found Author Name"] == "Brooke Coley"
    assert row["Author's Position"] == 1
    assert row["Total Authors"] == 2
    assert row["Affiliation"] == "Arizona State University"
    assert row["Abstract"] == "First part. Second part."


def test_stats_file_written(jc, monkeypatch, tmp_path, corpus, good_tei):
    _, stats = run_hybrid(jc, monkeypatch, tmp_path, corpus, lambda p: good_tei, llm_ok)
    data = json.loads(stats.read_text())
    assert "alphamind_extractor" in data
    assert data["alphamind_extractor"]["runtime_seconds"] >= 0
    assert data["alphamind_extractor"]["prompt_tokens"] >= 0
    assert "joshcoley" in data  # legacy stats key


def test_stats_file_merges_not_clobbers(jc, monkeypatch, tmp_path, corpus, good_tei):
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"grobid": {"runtime_seconds": 1.0, "cost_usd": 0.0}}))
    monkeypatch.setattr(jc, "OUTPUT_XLSX", tmp_path / "out.xlsx")
    monkeypatch.setattr(jc, "RUN_STATS_JSON", str(stats))
    monkeypatch.setattr(jc, "MAX_WORKERS", 1)
    monkeypatch.setattr(jc, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: good_tei)
    monkeypatch.setattr(jc, "call_openai_extract", llm_ok)
    monkeypatch.setattr(jc, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    jc.main()
    data = json.loads(stats.read_text())
    assert "grobid" in data and "alphamind_extractor" in data


# ------------------------------------------- affiliation fallback behaviour

def test_falls_back_to_grobid_when_llm_names_differ(jc, monkeypatch, tmp_path, corpus, good_tei):
    """
    The LLM naming the author differently must not produce "Not specified"
    when GROBID has a perfectly good affiliation.
    """
    def llm_different_name(head, byline):
        return {"authors": ["B. R. Coley"],
                "affiliations": [{"author": "Someone Else",
                                  "affiliations": ["Unrelated Place"]}]}
    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, lambda p: good_tei, llm_different_name)
    affs = pd.read_excel(out)["Affiliation"].tolist()
    assert all(a != "Not specified" for a in affs), affs
    assert all("Arizona State University" in str(a) for a in affs), affs


def test_regex_fallback_when_llm_unavailable(jc, monkeypatch, tmp_path, corpus, good_tei):
    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, lambda p: good_tei, lambda h, b: None)
    df = pd.read_excel(out)
    assert len(df) == 2, "papers must still be produced without the LLM"


# ----------------------------------------------------------- failure paths

def test_corrupt_pdf_does_not_abort_serial_run(jc, monkeypatch, tmp_path, corpus, good_tei, capsys):
    (corpus / "corrupt.pdf").write_text("this is not a pdf")

    def tei_for(p):
        # Match on the filename only: the pytest tmp dir name also contains
        # "corrupt", which would otherwise break every file in the corpus.
        if Path(p).name == "corrupt.pdf":
            return "<TEI><unclosed>"
        return good_tei

    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, tei_for, llm_ok, workers=1)
    assert out.exists(), "run must complete despite a corrupt input"
    assert len(pd.read_excel(out)) == 2


def test_corrupt_pdf_does_not_abort_parallel_run(jc, monkeypatch, tmp_path, corpus, good_tei):
    (corpus / "corrupt.pdf").write_text("this is not a pdf")

    def tei_for(p):
        # Match on the filename only: the pytest tmp dir name also contains
        # "corrupt", which would otherwise break every file in the corpus.
        if Path(p).name == "corrupt.pdf":
            return "<TEI><unclosed>"
        return good_tei

    out, _ = run_hybrid(jc, monkeypatch, tmp_path, corpus, tei_for, llm_ok, workers=2)
    assert out.exists()
    assert len(pd.read_excel(out)) == 2


def test_processing_failures_are_reported(jc, monkeypatch, tmp_path, corpus, good_tei, capsys):
    """Failures were previously swallowed by `except Exception: pass`."""
    def explode(pdf):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(jc, "OUTPUT_XLSX", tmp_path / "out.xlsx")
    monkeypatch.setattr(jc, "RUN_STATS_JSON", str(tmp_path / "s.json"))
    monkeypatch.setattr(jc, "MAX_WORKERS", 2)
    monkeypatch.setattr(jc, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: good_tei)
    monkeypatch.setattr(jc, "call_openai_extract", llm_ok)
    monkeypatch.setattr(jc, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(jc, "extract_hybrid", lambda p, ok: explode(p))
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    jc.main()
    captured = capsys.readouterr().out
    assert "failed to process" in captured
    assert "simulated extraction failure" in captured


def test_no_matching_authors_writes_nothing(jc, monkeypatch, tmp_path, corpus, good_tei, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Nonexistent Person"])
    out = tmp_path / "out.xlsx"
    monkeypatch.setattr(jc, "OUTPUT_XLSX", out)
    monkeypatch.setattr(jc, "RUN_STATS_JSON", str(tmp_path / "s.json"))
    monkeypatch.setattr(jc, "MAX_WORKERS", 1)
    monkeypatch.setattr(jc, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: good_tei)
    monkeypatch.setattr(jc, "call_openai_extract", llm_ok)
    monkeypatch.setattr(jc, "quick_author_hit", lambda t, x: True)
    jc.main()
    assert not out.exists()
    assert "nothing to write" in capsys.readouterr().out


# ------------------------------------------------------- GROBID pipeline

def test_grobid_pipeline_end_to_end(gr, monkeypatch, tmp_path, corpus, good_tei):
    out = tmp_path / "g.xlsx"
    stats = tmp_path / "g.json"
    monkeypatch.setattr(gr, "OUTPUT_XLSX", out)
    monkeypatch.setattr(gr, "RUN_STATS_JSON", str(stats))
    monkeypatch.setattr(gr, "MAX_WORKERS", 1)
    monkeypatch.setattr(gr, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(gr, "extract_with_grobid", lambda p: good_tei)
    monkeypatch.setattr(gr, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    gr.main()
    df = pd.read_excel(out)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.iloc[0]["Affiliation"] == "Arizona State University"


def test_grobid_pipeline_records_runtime(gr, monkeypatch, tmp_path, corpus, good_tei):
    """Without this the Time column in the comparison table is NaN."""
    stats = tmp_path / "g.json"
    monkeypatch.setattr(gr, "OUTPUT_XLSX", tmp_path / "g.xlsx")
    monkeypatch.setattr(gr, "RUN_STATS_JSON", str(stats))
    monkeypatch.setattr(gr, "MAX_WORKERS", 1)
    monkeypatch.setattr(gr, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(gr, "extract_with_grobid", lambda p: good_tei)
    monkeypatch.setattr(gr, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    gr.main()
    data = json.loads(stats.read_text())
    assert "grobid" in data
    assert data["grobid"]["runtime_seconds"] >= 0
