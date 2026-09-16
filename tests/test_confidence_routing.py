"""
Confidence routing: skip the LLM when GROBID's affiliation clears the bar.

This is the optimisation that takes cost from 70% to 94% below the pure-LLM
pipeline. It is only safe because the raw_affiliation fix landed first — before
that, GROBID's affiliations were doubled and the gate was right only ~80% of the
time. The measured-precision test below is the guard on that assumption.
"""

import math

import pandas as pd
import pytest

from conftest import build_tei, author_xml
from pipeline_paths import EVAL_DIR, MY_PAPERS


# ------------------------------------------------- the gate on single strings

@pytest.mark.parametrize("text", [
    "Purdue University",
    "Virginia Polytechnic Institute and State University",
    "School of Computing, Florida International University",
    "Rose-Hulman Institute of Technology, Terre Haute",
])
def test_gate_accepts_clean_institutions(jc, text):
    assert jc.affiliation_is_confident(text) is True


@pytest.mark.parametrize("text,why", [
    ("", "empty"),
    ("Not specified", "placeholder"),
    ("nan", "placeholder"),
    ("Purdue", "too short"),
    ("Department of Engineering Education", "department, no institution"),
    ("Dept. of Physics, 123 Main St", "department, no institution"),
    ("Purdue University; Virginia Tech", "two affiliations run together"),
    ("X" * 400, "implausibly long"),
])
def test_gate_rejects_doubtful_strings(jc, text, why):
    assert jc.affiliation_is_confident(text) is False, why


def test_gate_rejects_the_old_doubling_bug_signature(jc):
    """
    If the raw_affiliation fix ever regresses, doubled strings must not be
    trusted — that combination would send wrong answers straight to the output.
    """
    assert jc.affiliation_is_confident("Purdue University Purdue University") is False


# --------------------------------------------------- the paper-level decision

@pytest.mark.parametrize("affiliations,expected", [
    ([], False),
    ([{"author": "A", "affiliations": []}], False),
    ([{"author": "A", "affiliations": ["Purdue University"]}], True),
    ([{"author": "A", "affiliations": ["Department of Engineering"]}], False),
    ([{"author": "A", "affiliations": ["Purdue University"]},
      {"author": "B", "affiliations": ["Virginia Tech University"]}], True),
    # one doubtful affiliation sends the whole paper to the LLM
    ([{"author": "A", "affiliations": ["Purdue University"]},
      {"author": "B", "affiliations": ["Dept of Physics"]}], False),
])
def test_paper_level_gate(jc, affiliations, expected):
    assert jc.grobid_affiliation_is_confident(affiliations) is expected


# ------------------------------------------------- routing behaviour end-to-end

@pytest.fixture
def confident_tei():
    return build_tei(analytic_authors=[
        author_xml("Brooke Coley", raw_affiliations=["Arizona State University"])])


@pytest.fixture
def doubtful_tei():
    return build_tei(analytic_authors=[
        author_xml("Brooke Coley", raw_affiliations=["Department of Engineering Education"])])


def test_llm_not_called_when_grobid_is_confident(jc, monkeypatch, sample_pdf, confident_tei):
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: confident_tei)
    monkeypatch.setattr(jc, "call_openai_extract",
                        lambda h, b: pytest.fail("LLM must not be called"))
    md, _ = jc.extract_hybrid(sample_pdf, True)
    assert md["affiliation_source"] == "grobid"
    affs = [a for it in md["affiliations"] for a in it["affiliations"]]
    assert affs == ["Arizona State University"]


def test_llm_is_called_when_grobid_is_doubtful(jc, monkeypatch, sample_pdf, doubtful_tei):
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: doubtful_tei)
    called = []

    def llm(head, byline):
        called.append(1)
        return {"authors": ["Brooke Coley"],
                "affiliations": [{"author": "Brooke Coley",
                                  "affiliations": ["Arizona State University"]}]}

    monkeypatch.setattr(jc, "call_openai_extract", llm)
    md, _ = jc.extract_hybrid(sample_pdf, True)
    assert called, "doubtful GROBID output must escalate to the LLM"
    assert md["affiliation_source"] == "llm"


def test_routing_can_be_disabled(jc, monkeypatch, sample_pdf, confident_tei):
    """ROUTE_ON_GROBID_CONFIDENCE=0 must restore always-call-the-LLM behaviour."""
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", False)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: confident_tei)
    called = []
    monkeypatch.setattr(jc, "call_openai_extract", lambda h, b: called.append(1) or {
        "authors": ["Brooke Coley"],
        "affiliations": [{"author": "Brooke Coley", "affiliations": ["LLM Answer"]}]})
    md, _ = jc.extract_hybrid(sample_pdf, True)
    assert called, "with routing off the LLM must always be called"
    assert md["affiliation_source"] == "llm"


def test_grobid_affiliations_preserved_when_routing(jc, monkeypatch, sample_pdf, confident_tei):
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: confident_tei)
    monkeypatch.setattr(jc, "call_openai_extract", lambda h, b: None)
    md, _ = jc.extract_hybrid(sample_pdf, True)
    assert md.get("grobid_affiliations")


def test_routed_paper_still_produces_rows(jc, monkeypatch, tmp_path, repo, confident_tei):
    """A skipped paper must still reach the spreadsheet with its affiliation."""
    import sys
    src = sorted(MY_PAPERS.rglob("*.pdf"))
    if len(src) < 1:
        pytest.skip("no PDFs")
    d = tmp_path / "papers"
    d.mkdir()
    (d / src[0].name).write_bytes(src[0].read_bytes())

    out = tmp_path / "out.xlsx"
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", True)
    monkeypatch.setattr(jc, "OUTPUT_XLSX", out)
    monkeypatch.setattr(jc, "RUN_STATS_JSON", str(tmp_path / "s.json"))
    monkeypatch.setattr(jc, "MAX_WORKERS", 1)
    monkeypatch.setattr(jc, "is_grobid_alive", lambda: True)
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: confident_tei)
    monkeypatch.setattr(jc, "call_openai_extract",
                        lambda h, b: pytest.fail("LLM must not be called"))
    monkeypatch.setattr(jc, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(d), "Brooke Coley"])
    jc.main()
    df = pd.read_excel(out)
    assert len(df) == 1
    assert df.iloc[0]["Affiliation"] == "Arizona State University"


# ------------------------------------------- the precision claim, on real data

def test_gate_precision_meets_85_percent_on_gold_corpus(jc, cp, gold_df, repo):
    """
    The whole optimisation rests on this number. If a future change to the
    parser or the gate drops precision below 85%, this test fails and the
    routing must be re-tuned or switched off.
    """
    p = EVAL_DIR / "grobid_results_v4.xlsx"
    if not p.exists():
        pytest.skip("grobid_results_v4.xlsx not present")
    gr = cp.load_df(str(p))
    mapping = cp.link_records(gold_df, gr, cp.TITLE_MATCH_THRESHOLD)

    accepted = correct = 0
    for gi, pi in mapping.items():
        if pi is None:
            continue
        a = gr.iloc[pi]["Affiliation"]
        a = "" if pd.isna(a) else str(a)
        if not jc.affiliation_is_confident(a):
            continue
        accepted += 1
        if cp.is_correct("Affiliation", gold_df.loc[gi, "Affiliation"], a, cp.THRESHOLDS):
            correct += 1

    assert accepted >= 20, f"gate accepted only {accepted} rows; too few to judge"
    precision = correct / accepted
    assert precision >= 0.85, (
        f"gate precision {precision:.1%} ({correct}/{accepted}) is below the 85% floor")


def test_gate_skips_a_worthwhile_share_of_papers(jc, cp, gold_df, repo):
    """If the gate stops skipping papers, the cost saving is gone."""
    p = EVAL_DIR / "grobid_results_v4.xlsx"
    if not p.exists():
        pytest.skip("grobid_results_v4.xlsx not present")
    gr = cp.load_df(str(p))
    total = len(gr)
    skipped = sum(1 for a in gr["Affiliation"]
                  if jc.affiliation_is_confident("" if pd.isna(a) else str(a)))
    assert skipped / total >= 0.5, f"only {skipped}/{total} papers skip the LLM"
