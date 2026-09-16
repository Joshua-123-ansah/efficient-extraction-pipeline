"""
Round 3: the functions still unentered after test_uncovered.py.

Mostly fallback branches that only fire on awkward input, plus the remaining
main()/process_one() entry points.
"""

import json
import sys

import pytest

from pipeline_paths import EVAL_DIR, GOLD_XLSX, MY_PAPERS


# ------------------------------- loose author matching in the TEI fallback

LOOSE_TEI = (
    '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
    '<titleStmt><title level="a" type="main">T</title></titleStmt>'
    '<sourceDesc><biblStruct><analytic><title level="a" type="main">T</title>'
    # analytic author carries no affiliation and is rendered with an initial
    '<author><persName><forename>A</forename><surname>Bee</surname></persName></author>'
    '</analytic></biblStruct></sourceDesc></fileDesc>'
    '<profileDesc><particDesc><listPerson>'
    # listPerson spells the given name out, so only the loose matcher connects them
    '<person><persName><forename>Ann</forename><forename>Marie</forename>'
    '<surname>Bee</surname></persName>'
    '<affiliation key="p0"><note type="raw_affiliation">Loose Match University</note>'
    '</affiliation></person>'
    '</listPerson></particDesc></profileDesc></teiHeader></TEI>')


@pytest.fixture(params=["joshcoley", "grobid", "task1"])
def parser(request, tei_parsers):
    return tei_parsers[request.param]


def test_loose_name_matching_recovers_affiliation(parser):
    """
    names_match_loose() is the last fallback that pairs an analytic author with
    a listPerson entry spelled differently. Without it the affiliation is lost.
    """
    md = parser.parse_tei(LOOSE_TEI)
    affs = [a for it in md.get("affiliations", []) for a in it["affiliations"]]
    assert affs == ["Loose Match University"]


def test_loose_matching_does_not_pair_different_people(parser):
    """
    Two analytic authors, so the single-author "inherit any orphan affiliation"
    heuristic cannot fire. The listPerson surname matches neither author, so no
    affiliation should be attached.
    """
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        '<titleStmt><title level="a" type="main">T</title></titleStmt>'
        '<sourceDesc><biblStruct><analytic><title level="a" type="main">T</title>'
        '<author><persName><forename>A</forename><surname>Bee</surname></persName></author>'
        '<author><persName><forename>C</forename><surname>Dee</surname></persName></author>'
        '</analytic></biblStruct></sourceDesc></fileDesc>'
        '<profileDesc><particDesc><listPerson>'
        '<person><persName><forename>Ann</forename><surname>Zed</surname></persName>'
        '<affiliation key="p0"><note type="raw_affiliation">Unrelated University</note>'
        '</affiliation></person>'
        '</listPerson></particDesc></profileDesc></teiHeader></TEI>')
    md = parser.parse_tei(tei)
    affs = [a for it in md.get("affiliations", []) for a in it["affiliations"]]
    assert affs == []


# --------------------------- GROBID network boundary in grobid_extraction_v4

def test_grobid_module_is_alive_true(gr, monkeypatch):
    class R:
        status_code = 200
    class S:
        def get(self, *a, **k): return R()
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(gr.requests, "Session", lambda: S())
    assert gr.is_grobid_alive() is True


def test_grobid_module_is_alive_false(gr, monkeypatch):
    class S:
        def get(self, *a, **k): raise ConnectionError("refused")
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(gr.requests, "Session", lambda: S())
    assert gr.is_grobid_alive() is False


def test_grobid_module_extract_posts_raw_affiliations(gr, monkeypatch, sample_pdf):
    seen = {}

    class S:
        def post(self, url, files=None, data=None, timeout=None):
            seen["url"] = url
            seen["data"] = data
            class R:
                content = b"<TEI/>"
                def raise_for_status(self): pass
            return R()
        def close(self): pass

    monkeypatch.setattr(gr.requests, "Session", lambda: S())
    gr.extract_with_grobid(sample_pdf)
    assert seen["url"].endswith(f"/api/{gr.GROBID_SERVICE}")
    assert seen["data"]["includeRawAffiliations"] == "1"


def test_grobid_module_extract_returns_none_on_error(gr, monkeypatch, sample_pdf):
    class S:
        def post(self, *a, **k): raise ConnectionError("down")
        def close(self): pass
    monkeypatch.setattr(gr.requests, "Session", lambda: S())
    assert gr.extract_with_grobid(sample_pdf) is None


# ------------------------------------------------ LLM pipeline entry points

def test_llm_get_api_key(lm, monkeypatch):
    monkeypatch.setattr(lm, "LLM_PROVIDER", "createai")
    monkeypatch.setenv("ASU_CREATE_AI_KEY", "jwt")
    assert lm.get_llm_api_key() == "jwt"
    monkeypatch.setattr(lm, "LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert lm.get_llm_api_key() == "sk-x"


def test_llm_call_extract_without_key(lm, monkeypatch):
    monkeypatch.setattr(lm, "get_llm_api_key", lambda: "")
    assert lm.call_openai_extract("head", "byline") is None


def test_llm_call_extract_with_stubbed_sdk(lm, monkeypatch):
    class Completions:
        def create(self, **kw):
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 2})()
            msg = type("M", (), {"content": '{"authors": ["Ann Bee"]}'})()
            choice = type("C", (), {"message": msg})()
            return type("R", (), {"choices": [choice], "usage": usage})()

    class Client:
        def __init__(self, **kw):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": Client}))
    monkeypatch.setattr(lm, "get_llm_api_key", lambda: "key")
    assert lm.call_openai_extract("head", "byline") == {"authors": ["Ann Bee"]}


@pytest.fixture
def corpus(tmp_path, repo):
    src = sorted(MY_PAPERS.rglob("*.pdf")) if MY_PAPERS.exists() else []
    if len(src) < 2:
        pytest.skip("need at least 2 PDFs")
    d = tmp_path / "papers"
    d.mkdir()
    for p in src[:2]:
        (d / p.name).write_bytes(p.read_bytes())
    return d


def test_llm_main_end_to_end(lm, monkeypatch, tmp_path, corpus):
    import pandas as pd
    out = tmp_path / "llm.xlsx"
    monkeypatch.setattr(lm, "OUTPUT_XLSX", out)
    monkeypatch.setattr(lm, "RUN_STATS_JSON", str(tmp_path / "s.json"))
    monkeypatch.setattr(lm, "MAX_WORKERS", 1)
    monkeypatch.setattr(lm, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(lm, "call_openai_extract", lambda h, b: {
        "title": "A Test Paper",
        "authors": ["Brooke Coley", "Cid Dee"],
        "affiliations": [{"author": "Brooke Coley", "affiliations": ["Arizona State University"]}],
        "abstract": "An abstract.",
    })
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    lm.main()
    df = pd.read_excel(out)
    assert len(df) == 2
    assert df.iloc[0]["Found Author Name"] == "Brooke Coley"
    assert df.iloc[0]["Affiliation"] == "Arizona State University"


def test_llm_main_writes_stats(lm, monkeypatch, tmp_path, corpus):
    stats = tmp_path / "s.json"
    monkeypatch.setattr(lm, "OUTPUT_XLSX", tmp_path / "llm.xlsx")
    monkeypatch.setattr(lm, "RUN_STATS_JSON", str(stats))
    monkeypatch.setattr(lm, "MAX_WORKERS", 1)
    monkeypatch.setattr(lm, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(lm, "call_openai_extract", lambda h, b: {
        "title": "T", "authors": ["Brooke Coley"],
        "affiliations": [{"author": "Brooke Coley", "affiliations": ["ASU"]}],
        "abstract": "a"})
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    lm.main()
    assert "llm" in json.loads(stats.read_text())


# ------------------------------------------------ NLP pipeline entry points

def test_npl_extract_affiliations_without_spacy(npl):
    """Signature is (byline_text, head_text, nlp); must degrade when nlp is None."""
    out = npl.extract_affiliations(
        "Ann Bee", "Purdue University, West Lafayette, Indiana\nAbstract", None)
    assert isinstance(out, list)


def test_npl_extract_affiliations_empty_input(npl):
    assert npl.extract_affiliations("", "", None) == []


def test_npl_spacy_helpers_return_empty_without_model(npl):
    assert npl.spacy_persons("Ann Bee wrote this.", None) == []
    assert npl.spacy_orgs("Purdue University did this.", None) == []


def test_npl_extract_from_pdf_nlp_without_spacy(npl, sample_pdf):
    md = npl.extract_from_pdf_nlp(sample_pdf, None)
    assert isinstance(md, dict)
    assert "authors" in md


def test_npl_main_runs_without_spacy(npl, monkeypatch, tmp_path, corpus):
    monkeypatch.setattr(npl, "OUTPUT_XLSX", tmp_path / "nlp.xlsx")
    monkeypatch.setattr(npl, "RUN_STATS_JSON", str(tmp_path / "s.json"))
    monkeypatch.setattr(npl, "MAX_WORKERS", 1)
    monkeypatch.setattr(npl, "load_spacy_model", lambda: None)
    monkeypatch.setattr(npl, "quick_author_hit", lambda t, x: True)
    monkeypatch.setattr(sys, "argv", ["prog", str(corpus), "Brooke Coley"])
    npl.main()  # must not raise


# ------------------------------------------- compare_pipelines --run path

def test_compare_run_and_time(cp, monkeypatch, repo, tmp_path, capsys):
    """run_and_time is nested in main() and only fires under --run."""
    for name in ["Gold Standard.xlsx", "grobid_results_v4.xlsx", "nlp_results.xlsx",
                 "llm_results.xlsx", "joshcoley_results.xlsx"]:
        path = GOLD_XLSX if name == "Gold Standard.xlsx" else EVAL_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not present")
    calls = []

    env_used = {}

    def fake_run(cmd, **kw):
        calls.append(cmd)
        env_used.update(kw.get("env") or {})
        return type("R", (), {"returncode": 0})()

    monkeypatch.chdir(repo)
    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    monkeypatch.setattr(cp, "SUMMARY_XLSX", str(tmp_path / "s.xlsx"))
    monkeypatch.setattr(sys, "argv", ["compare_pipelines.py", "--run", "--run-which", "grobid"])
    cp.main()
    assert calls, "the pipeline subprocess should have been invoked"
    assert any("grobid" in " ".join(map(str, c)) for c in calls)
    assert env_used.get("CONSOLIDATE_HEADER") == "1"
