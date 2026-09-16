"""
LLM call path: prompt construction, provider routing, response parsing,
and the two-tier escalation that produces the cost saving.

No network. The HTTP boundary and the OpenAI SDK are both stubbed.
"""

import json

import pytest

from conftest import build_tei, author_xml


# ------------------------------------------------------------------- prompt

def test_byline_not_duplicated_when_identical(jc):
    """byline_block() is carved out of head_text; sending both paid twice."""
    prompt = jc.build_llm_prompt_affiliation_only("HEADTEXT", "HEADTEXT")
    assert prompt.count("HEADTEXT") == 1


def test_head_text_included_when_it_adds_content(jc):
    prompt = jc.build_llm_prompt_affiliation_only("HEAD EXTRA MATERIAL", "HEAD")
    assert "EXTRA MATERIAL" in prompt


def test_prompt_requests_json(jc):
    prompt = jc.build_llm_prompt_affiliation_only("head", "byline")
    assert "JSON" in prompt.upper()


def test_prompt_shorter_than_full_metadata_prompt(jc, lm):
    """The hybrid asks for 1 field; the pure-LLM pipeline asks for 12."""
    head, byline = "head text", "byline text"
    assert len(jc.build_llm_prompt_affiliation_only(head, byline)) < \
           len(lm.build_llm_prompt(head, byline))


# ------------------------------------------------------------ JSON recovery

@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go: {"a": 1} hope that helps', {"a": 1}),
    ('', None),
    ('not json at all', None),
    ('{"a": ', None),
])
def test_try_parse_json(jc, raw, expected):
    assert jc.try_parse_json(raw) == expected


# ------------------------------------------------------- affiliation guard

@pytest.mark.parametrize("payload,expected", [
    (None, False),
    ({}, False),
    ({"affiliations": []}, False),
    ({"affiliations": [{"author": "A", "affiliations": []}]}, False),
    ({"affiliations": [{"author": "A", "affiliations": ["   "]}]}, False),
    ({"affiliations": [{"author": "A", "affiliations": ["Purdue"]}]}, True),
])
def test_has_affiliation(jc, payload, expected):
    assert jc._has_affiliation(payload) is expected


# ------------------------------------------------------ provider selection

def test_createai_key_preferred(jc, monkeypatch):
    monkeypatch.setattr(jc, "LLM_PROVIDER", "createai")
    monkeypatch.setenv("ASU_CREATE_AI_KEY", "jwt-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert jc.get_llm_api_key() == "jwt-token"


def test_falls_back_to_openai_key(jc, monkeypatch):
    monkeypatch.setattr(jc, "LLM_PROVIDER", "createai")
    monkeypatch.delenv("ASU_CREATE_AI_KEY", raising=False)
    monkeypatch.delenv("CREATEAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert jc.get_llm_api_key() == "sk-openai"


def test_openai_provider_ignores_createai_key(jc, monkeypatch):
    monkeypatch.setattr(jc, "LLM_PROVIDER", "openai")
    monkeypatch.setenv("ASU_CREATE_AI_KEY", "jwt-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert jc.get_llm_api_key() == "sk-openai"


def test_no_key_returns_empty(jc, monkeypatch):
    monkeypatch.setattr(jc, "LLM_PROVIDER", "createai")
    for var in ("ASU_CREATE_AI_KEY", "CREATEAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert jc.get_llm_api_key() == ""


# ------------------------------------------------ native /query response shapes

class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.mark.parametrize("payload,expected", [
    ({"response": '{"authors": ["X"], "affiliations": []}'},
     {"authors": ["X"], "affiliations": []}),
    ({"choices": [{"message": {"content": '{"authors": ["Y"]}'}}]},
     {"authors": ["Y"]}),
    ({"result": '{"authors": ["Z"]}'}, {"authors": ["Z"]}),
    ({"unrecognised_envelope": 1}, None),
    ({}, None),
])
def test_createai_query_shapes(jc, monkeypatch, payload, expected):
    monkeypatch.setattr(jc.requests, "post", lambda *a, **k: _Resp(payload))
    assert jc._createai_query("prompt", "key") == expected


def test_createai_query_network_error_returns_none(jc, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(jc.requests, "post", boom)
    assert jc._createai_query("prompt", "key") is None


# --------------------------------------------------------- usage accounting

def test_usage_ignores_non_integers(jc):
    before = dict(jc._JOSHCOLEY_USAGE)
    jc._update_joshcoley_usage(None, None)
    jc._update_joshcoley_usage("100", 5)
    assert jc._JOSHCOLEY_USAGE["prompt_tokens"] == before["prompt_tokens"]


def test_usage_accumulates_and_prices(jc):
    before = dict(jc._JOSHCOLEY_USAGE)
    jc._update_joshcoley_usage(1_000_000, 0)
    assert jc._JOSHCOLEY_USAGE["prompt_tokens"] == before["prompt_tokens"] + 1_000_000
    delta = jc._JOSHCOLEY_USAGE["cost_usd"] - before["cost_usd"]
    assert delta == pytest.approx(jc.OPENAI_INPUT_PRICE_PER_M)


def test_pricing_matches_across_pipelines(jc, lm):
    """A cost comparison between two pipelines priced differently is meaningless."""
    assert jc.OPENAI_INPUT_PRICE_PER_M == lm.OPENAI_INPUT_PRICE_PER_M
    assert jc.OPENAI_OUTPUT_PRICE_PER_M == lm.OPENAI_OUTPUT_PRICE_PER_M


# ------------------------------------------------------ two-tier escalation

@pytest.fixture
def hybrid_env(jc, monkeypatch, sample_pdf):
    """
    Stub GROBID; record every prompt size the LLM is called with.

    Confidence routing is disabled here on purpose: these tests are about the
    escalation logic *inside* the LLM path. Routing itself is covered in
    test_confidence_routing.py, and with it on the LLM would never be reached.
    """
    monkeypatch.setattr(jc, "ROUTE_ON_GROBID_CONFIDENCE", False)
    tei = build_tei(analytic_authors=[
        author_xml("Brooke Coley", raw_affiliations=["Arizona State University"])])
    monkeypatch.setattr(jc, "extract_with_grobid", lambda p: tei)
    calls = []
    return sample_pdf, calls


def test_narrow_window_only_when_answer_found(jc, monkeypatch, hybrid_env):
    pdf, calls = hybrid_env

    def llm(head, byline):
        calls.append(len(head))
        return {"authors": ["Brooke Coley"],
                "affiliations": [{"author": "Brooke Coley",
                                  "affiliations": ["Arizona State University"]}]}
    monkeypatch.setattr(jc, "call_openai_extract", llm)
    md, _ = jc.extract_hybrid(pdf, True)
    assert len(calls) == 1, "must not escalate when the answer was found"


def test_escalates_when_answered_but_empty(jc, monkeypatch, hybrid_env):
    pdf, calls = hybrid_env
    state = {"n": 0}

    def llm(head, byline):
        state["n"] += 1
        calls.append(len(head))
        if state["n"] == 1:
            return {"authors": ["Brooke Coley"], "affiliations": []}
        return {"authors": ["Brooke Coley"],
                "affiliations": [{"author": "Brooke Coley",
                                  "affiliations": ["Arizona State University"]}]}
    monkeypatch.setattr(jc, "call_openai_extract", llm)
    jc.extract_hybrid(pdf, True)
    assert len(calls) == 2, "must escalate when the model answered with no affiliation"
    assert calls[1] > calls[0], "escalation must send MORE text"


def test_does_not_escalate_when_call_failed(jc, monkeypatch, hybrid_env):
    """
    result is None means the request itself failed (auth, timeout). Resending a
    larger prompt cannot help and doubles the failures.
    """
    pdf, calls = hybrid_env

    def llm(head, byline):
        calls.append(len(head))
        return None
    monkeypatch.setattr(jc, "call_openai_extract", llm)
    jc.extract_hybrid(pdf, True)
    assert len(calls) == 1


def test_hybrid_uses_llm_affiliation(jc, monkeypatch, hybrid_env):
    pdf, _ = hybrid_env
    monkeypatch.setattr(jc, "call_openai_extract", lambda h, b: {
        "authors": ["Brooke Coley"],
        "affiliations": [{"author": "Brooke Coley", "affiliations": ["LLM Answer"]}]})
    md, _ = jc.extract_hybrid(pdf, True)
    assert md["affiliations"] == [{"author": "Brooke Coley", "affiliations": ["LLM Answer"]}]


def test_hybrid_preserves_grobid_affiliations_for_fallback(jc, monkeypatch, hybrid_env):
    pdf, _ = hybrid_env
    monkeypatch.setattr(jc, "call_openai_extract", lambda h, b: None)
    md, _ = jc.extract_hybrid(pdf, True)
    assert md.get("grobid_affiliations"), \
        "GROBID affiliations must survive so the row builder can fall back"
