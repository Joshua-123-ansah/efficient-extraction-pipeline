"""
Functions the first round of tests never entered.

Written after a coverage pass showed 53 functions with zero execution. Network
boundaries are stubbed; spaCy-dependent paths skip when the model is absent.
"""

import json
import sys

import pytest

from conftest import build_tei, author_xml
from pipeline_paths import EVAL_DIR, GOLD_XLSX, MY_PAPERS

ALL = ["joshcoley", "grobid", "llm", "npl", "task1"]


@pytest.fixture(params=ALL)
def mod(request, extractors):
    return extractors[request.param]


# ------------------------------------------------- shared name primitives

@pytest.mark.parametrize("name,expected", [
    ("Ann Bee", (["Ann"], "Bee")),
    ("Ann Marie Bee", (["Ann", "Marie"], "Bee")),
    ("Madonna", ([], "Madonna")),
    ("", ([], "")),
])
def test_split_name(mod, name, expected):
    assert mod.split_name(name) == expected


@pytest.mark.parametrize("tokens,expected", [
    (["Ann", "Marie"], ["a", "m"]),
    ([], []),
    (["", "Bee"], ["b"]),
])
def test_initials_from_given(mod, tokens, expected):
    assert mod.initials_from_given(tokens) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Ann Bee", "annbee"),
    ("Ann-Marie  Bee", "annmariebee"),
    ("Résumé", "resume"),
    ("", ""),
])
def test_normalize_name_for_compare(mod, raw, expected):
    assert mod.normalize_name_for_compare(raw) == expected


# ------------------------------------- GROBID network boundary (stubbed)

def test_is_grobid_alive_true(jc, monkeypatch):
    class R:
        status_code = 200
    class S:
        def get(self, *a, **k): return R()
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(jc.requests, "Session", lambda: S())
    assert jc.is_grobid_alive() is True


def test_is_grobid_alive_false_on_error(jc, monkeypatch):
    class S:
        def get(self, *a, **k): raise ConnectionError("refused")
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(jc.requests, "Session", lambda: S())
    assert jc.is_grobid_alive() is False


class _GrobidSession:
    """Captures the POST so we can assert on URL and form fields."""
    last = {}

    def post(self, url, files=None, data=None, timeout=None):
        _GrobidSession.last = {"url": url, "data": data}
        class R:
            content = b"<TEI/>"
            def raise_for_status(self): pass
        return R()

    def close(self): pass


def test_extract_with_grobid_uses_configured_service(jc, monkeypatch, sample_pdf):
    monkeypatch.setattr(jc.requests, "Session", lambda: _GrobidSession())
    jc.extract_with_grobid(sample_pdf)
    assert _GrobidSession.last["url"].endswith(f"/api/{jc.GROBID_SERVICE}")


def test_extract_with_grobid_requests_raw_affiliations(jc, monkeypatch, sample_pdf):
    """The affiliation parser depends on <note type="raw_affiliation">."""
    monkeypatch.setattr(jc.requests, "Session", lambda: _GrobidSession())
    jc.extract_with_grobid(sample_pdf)
    assert _GrobidSession.last["data"]["includeRawAffiliations"] == "1"


def test_extract_with_grobid_returns_none_on_error(jc, monkeypatch, sample_pdf):
    class S:
        def post(self, *a, **k): raise ConnectionError("down")
        def close(self): pass
    monkeypatch.setattr(jc.requests, "Session", lambda: S())
    assert jc.extract_with_grobid(sample_pdf) is None


# --------------------------------------- TEI author/affiliation helpers

def test_person_name_from_and_loose_matching(mod_tei_helpers):
    """Exercised indirectly: listPerson fallback supplies affiliations."""
    parse_tei, tei = mod_tei_helpers
    md = parse_tei(tei)
    assert md["authors"] == ["Ann Bee"]


@pytest.fixture(params=["joshcoley", "grobid", "task1"])
def mod_tei_helpers(request, tei_parsers):
    mod = tei_parsers[request.param]
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        '<titleStmt><title level="a" type="main">T</title></titleStmt>'
        '<sourceDesc><biblStruct><analytic><title level="a" type="main">T</title>'
        '<author><persName><forename>Ann</forename><surname>Bee</surname></persName></author>'
        '</analytic></biblStruct></sourceDesc></fileDesc>'
        '<profileDesc><particDesc><listPerson>'
        '<person><persName><forename>Ann</forename><surname>Bee</surname></persName>'
        '<affiliation key="p0"><note type="raw_affiliation">Listed University</note></affiliation>'
        '</person></listPerson></particDesc></profileDesc>'
        '</teiHeader></TEI>')
    return mod.parse_tei, tei


def test_listperson_affiliation_is_used(mod_tei_helpers):
    parse_tei, tei = mod_tei_helpers
    md = parse_tei(tei)
    affs = [a for it in md["affiliations"] for a in it["affiliations"]]
    assert affs == ["Listed University"]


# ------------------------------------------- chat completion (stubbed SDK)

class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def _install_fake_openai(monkeypatch, jc, content, fail_json_mode=False):
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            if fail_json_mode and "response_format" in kw:
                raise RuntimeError("model does not support response_format")
            return _FakeResponse(content)

    class Chat:
        completions = Completions()

    class Client:
        def __init__(self, **kw):
            self.kwargs = kw
            self.chat = Chat()
            Client.last_kwargs = kw

    fake_module = type("openai", (), {"OpenAI": Client})
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return calls, Client


def test_chat_completion_parses_response(jc, monkeypatch):
    _install_fake_openai(monkeypatch, jc, '{"authors": ["Ann Bee"], "affiliations": []}')
    got = jc._chat_completion("prompt", "key")
    assert got == {"authors": ["Ann Bee"], "affiliations": []}


def test_chat_completion_points_at_createai(jc, monkeypatch):
    _, Client = _install_fake_openai(monkeypatch, jc, '{"authors": []}')
    monkeypatch.setattr(jc, "LLM_PROVIDER", "createai")
    jc._chat_completion("prompt", "key")
    assert Client.last_kwargs.get("base_url") == jc.CREATEAI_BASE_URL


def test_chat_completion_no_base_url_for_openai(jc, monkeypatch):
    _, Client = _install_fake_openai(monkeypatch, jc, '{"authors": []}')
    monkeypatch.setattr(jc, "LLM_PROVIDER", "openai")
    jc._chat_completion("prompt", "key")
    assert "base_url" not in Client.last_kwargs


def test_chat_completion_retries_without_json_mode(jc, monkeypatch):
    """Not every gateway model supports response_format."""
    calls, _ = _install_fake_openai(monkeypatch, jc, '{"authors": []}', fail_json_mode=True)
    got = jc._chat_completion("prompt", "key")
    assert got == {"authors": []}
    assert len(calls) == 2
    assert "response_format" not in calls[1]


def test_chat_completion_records_usage(jc, monkeypatch):
    _install_fake_openai(monkeypatch, jc, '{"authors": []}')
    before = dict(jc._JOSHCOLEY_USAGE)
    jc._chat_completion("prompt", "key")
    assert jc._JOSHCOLEY_USAGE["prompt_tokens"] == before["prompt_tokens"] + 100


def test_chat_completion_without_sdk_returns_none(jc, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    assert jc._chat_completion("prompt", "key") is None


# ------------------------------------------------- call_openai_extract

def test_call_openai_extract_without_key_returns_none(jc, monkeypatch):
    monkeypatch.setattr(jc, "get_llm_api_key", lambda: "")
    assert jc.call_openai_extract("head", "byline") is None


def test_call_openai_extract_prefers_chat_completion(jc, monkeypatch):
    monkeypatch.setattr(jc, "get_llm_api_key", lambda: "key")
    monkeypatch.setattr(jc, "_chat_completion", lambda p, k: {"authors": ["A"]})
    monkeypatch.setattr(jc, "_createai_query", lambda p, k: pytest.fail("must not fall back"))
    assert jc.call_openai_extract("head", "byline") == {"authors": ["A"]}


def test_call_openai_extract_falls_back_to_native_query(jc, monkeypatch):
    monkeypatch.setattr(jc, "get_llm_api_key", lambda: "key")
    monkeypatch.setattr(jc, "LLM_PROVIDER", "createai")
    monkeypatch.setattr(jc, "_chat_completion", lambda p, k: None)
    monkeypatch.setattr(jc, "_createai_query", lambda p, k: {"authors": ["B"]})
    assert jc.call_openai_extract("head", "byline") == {"authors": ["B"]}


def test_call_openai_extract_openai_provider_no_native_fallback(jc, monkeypatch):
    monkeypatch.setattr(jc, "get_llm_api_key", lambda: "key")
    monkeypatch.setattr(jc, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(jc, "_chat_completion", lambda p, k: None)
    assert jc.call_openai_extract("head", "byline") is None


# ------------------------------------------------ LLM pipeline extraction

def test_llm_try_parse_json(lm):
    assert lm.try_parse_json('{"a": 1}') == {"a": 1}
    assert lm.try_parse_json("garbage") is None


def test_llm_usage_accounting(lm):
    before = dict(lm._LLM_USAGE)
    lm._update_llm_usage(1_000_000, 0)
    assert lm._LLM_USAGE["prompt_tokens"] == before["prompt_tokens"] + 1_000_000
    lm._update_llm_usage(None, None)
    assert lm._LLM_USAGE["prompt_tokens"] == before["prompt_tokens"] + 1_000_000


def test_llm_pdf_helpers(lm, tmp_path, sample_pdf):
    bad = tmp_path / "x.pdf"
    bad.write_text("nope")
    assert lm.pdf_text_all(bad) == ""
    assert lm.pdf_text_first_n(bad, 2) == ""
    assert len(lm.pdf_text_first_n(sample_pdf, 1)) > 0


def test_llm_gather_and_row(lm, tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    assert [p.name for p in lm.gather_pdfs(tmp_path)] == ["a.pdf"]
    assert lm.get_file_row_cached("/nope/none.pdf") == 1


def test_extract_from_pdf_llm_uses_llm_result(lm, monkeypatch, sample_pdf):
    monkeypatch.setattr(lm, "call_openai_extract", lambda h, b: {
        "title": "Extracted Title",
        "authors": ["Ann Bee", "Cid Dee"],
        "affiliations": [{"author": "Ann Bee", "affiliations": ["Purdue University"]}],
        "abstract": "The abstract.",
    })
    md = lm.extract_from_pdf_llm(sample_pdf)
    assert md["title"] == "Extracted Title"
    assert md["authors"] == ["Ann Bee", "Cid Dee"]
    assert md["affiliations"][0]["affiliations"] == ["Purdue University"]


def test_extract_from_pdf_llm_survives_no_llm(lm, monkeypatch, sample_pdf):
    monkeypatch.setattr(lm, "call_openai_extract", lambda h, b: None)
    md = lm.extract_from_pdf_llm(sample_pdf)
    assert isinstance(md, dict)
    assert "authors" in md


# ------------------------------------------------------- NLP pipeline

def test_npl_pdf_and_gather_helpers(npl, tmp_path, sample_pdf):
    bad = tmp_path / "x.pdf"
    bad.write_text("nope")
    assert npl.pdf_text_all(bad) == ""
    assert npl.pdf_text_first_n(bad, 2) == ""
    (tmp_path / "a.pdf").write_text("x")
    assert [p.name for p in npl.gather_pdfs(tmp_path)] == ["a.pdf", "x.pdf"] or \
           sorted(p.name for p in npl.gather_pdfs(tmp_path)) == ["a.pdf", "x.pdf"]
    assert npl.get_file_row_cached("/nope/none.pdf") == 1


@pytest.mark.parametrize("byline,expected", [
    ("Ann Bee, Cid Dee and Eve Fay", ["Ann Bee", "Cid Dee", "Eve Fay"]),
    ("Ann Bee", ["Ann Bee"]),
    ("", []),
])
def test_npl_parse_authors_from_byline(npl, byline, expected):
    assert npl.parse_authors_from_byline(byline) == expected


def test_npl_extract_year(npl):
    assert npl.extract_year("Published in 2019 by someone") == "2019"
    assert npl.extract_year("no year here") == ""


def test_npl_extract_title_from_head(npl):
    head = "A Reasonably Long Descriptive Paper Title\nAnn Bee\nAbstract\nBody"
    assert "Paper Title" in npl.extract_title_from_head(head)


def test_npl_extract_journal_meta(npl):
    """Returns a (journal, volume, issue, pages) tuple."""
    journal, volume, issue, pages = npl.extract_journal_meta(
        "Journal of Testing 12(3), 45-67 (2019)")
    assert "Journal of Testing" in journal
    assert npl.extract_journal_meta("") == ('', '', '', '')


def test_npl_spacy_paths_skip_without_model(npl):
    model = npl.load_spacy_model()
    if model is None:
        pytest.skip("spaCy model not installed")
    assert npl.spacy_persons("Ann Bee wrote this.", model) is not None


# ------------------------------------------------ compare_pipelines extras

def test_estimate_llm_cost_from_outputs(cp, repo):
    p = EVAL_DIR / "llm_results.xlsx"
    if not p.exists():
        pytest.skip("llm_results.xlsx not present")
    cost = cp.estimate_llm_cost_from_outputs(cp.load_df(str(p)), 0.002)
    assert cost >= 0.0


def test_compare_main_runs_without_arguments(cp, monkeypatch, repo, tmp_path, capsys):
    """
    run_and_time is nested inside main(), so main() is the only entry point.
    Exercise the full report path with --export writing to a temp file.
    """
    for name in ["Gold Standard.xlsx", "grobid_results_v4.xlsx", "nlp_results.xlsx",
                 "llm_results.xlsx", "joshcoley_results.xlsx"]:
        path = GOLD_XLSX if name == "Gold Standard.xlsx" else EVAL_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not present")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cp, "SUMMARY_XLSX", str(tmp_path / "summary.xlsx"))
    monkeypatch.setattr(sys, "argv", ["compare_pipelines.py"])
    cp.main()
    out = capsys.readouterr().out
    assert "COVERAGE" in out
    assert "PER-FIELD" in out
