"""
Round 4: fallback branches that only fire when the primary path cannot match.

These are the paths that run on messy real-world input, which is exactly where
silent wrong answers hide.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_paths import EVAL_DIR


COLUMNS = ["Title", "Target Author", "Found Author Name", "Author's Position",
           "Total Authors", "Affiliation", "Abstract", "File Path"]


def frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------ linkage fallback passes

def test_links_by_title_when_file_path_absent(cp):
    """Second fallback: no usable paths, so titles must carry the match."""
    gold = frame([["Gendered Interests in Engineering", "Ann Bee", "Ann Bee",
                   1, 1, "Purdue", "abs", ""]])
    pred = frame([["Gendered Interests in Engineering", "Ann Bee", "Ann Bee",
                   1, 1, "Purdue", "abs", ""]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] == 0


def test_title_fallback_rejects_dissimilar_titles(cp):
    gold = frame([["A Study of Engineering Identity", "Ann Bee", "Ann Bee", 1, 1, "P", "a", ""]])
    pred = frame([["Completely Unrelated Chemistry Paper", "Ann Bee", "Ann Bee", 1, 1, "P", "a", ""]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] is None


def test_links_by_filename_when_author_missing_from_pred(cp):
    """
    Pass 0 needs both file and author. When the author column is blank the
    filename pass must still connect the rows.
    """
    gold = frame([["T", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/old/dir/paper.pdf"]])
    pred = frame([["T", "", "Ann Bee", 1, 1, "P", "a", "/new/dir/paper.pdf"]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] == 0


def test_gold_row_without_path_still_links_by_title(cp):
    gold = frame([["Engineering Identity Formation", "Ann Bee", "Ann Bee", 1, 1, "P", "a", ""]])
    pred = frame([["Engineering Identity Formation", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/x/y.pdf"]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] == 0


def test_nan_title_does_not_crash_linkage(cp):
    gold = frame([[np.nan, "Ann Bee", "Ann Bee", 1, 1, "P", "a", ""]])
    pred = frame([["Something", "Ann Bee", "Ann Bee", 1, 1, "P", "a", ""]])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert mapping[0] in (0, None)


def test_more_gold_rows_than_predictions(cp):
    gold = frame([
        ["T One", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/p/1.pdf"],
        ["T Two", "Cid Dee", "Cid Dee", 1, 1, "Q", "b", "/p/2.pdf"],
    ])
    pred = frame([["T One", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/p/1.pdf"]])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert mapping[0] == 0
    assert mapping[1] is None


def test_more_predictions_than_gold_rows(cp):
    gold = frame([["T One", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/p/1.pdf"]])
    pred = frame([
        ["T One", "Ann Bee", "Ann Bee", 1, 1, "P", "a", "/p/1.pdf"],
        ["T Two", "Cid Dee", "Cid Dee", 1, 1, "Q", "b", "/p/2.pdf"],
    ])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert mapping[0] == 0
    assert len(mapping) == 1


# ---------------------------------------------------------- load_df guards

def test_load_df_fills_missing_fields(cp, tmp_path):
    p = tmp_path / "sparse.xlsx"
    pd.DataFrame([{"Title": "T", "File Path": "/x.pdf"}]).to_excel(p, index=False)
    df = cp.load_df(str(p))
    for field in cp.FIELDS:
        assert field in df.columns


def test_load_df_requires_id_column(cp, tmp_path):
    p = tmp_path / "no_title.xlsx"
    pd.DataFrame([{"Affiliation": "P"}]).to_excel(p, index=False)
    with pytest.raises(ValueError):
        cp.load_df(str(p))


# ---------------------------------------------------------- cost estimation

def test_estimate_cost_zero_for_empty_frame(cp):
    assert cp.estimate_llm_cost_from_outputs(frame([]), 0.002) == 0.0


def test_estimate_cost_scales_with_text_volume(cp):
    small = frame([["T", "A", "A", 1, 1, "P", "short", "/x.pdf"]])
    large = frame([["T" * 500, "A", "A", 1, 1, "P" * 500, "long " * 500, "/x.pdf"]])
    assert cp.estimate_llm_cost_from_outputs(large, 0.002) > \
           cp.estimate_llm_cost_from_outputs(small, 0.002)


def _hybrid_xlsx_name() -> str:
    for name in ("alphamind_extractor_results.xlsx", "joshcoley_results.xlsx"):
        if (EVAL_DIR / name).exists():
            return name
    pytest.skip("alphamind_extractor_results.xlsx / joshcoley_results.xlsx not present")
    return "alphamind_extractor_results.xlsx"


def test_time_cost_table_uses_measured_times(cp, repo):
    hybrid = _hybrid_xlsx_name()
    for n in ["grobid_results_v4.xlsx", "nlp_results.xlsx", "llm_results.xlsx", hybrid]:
        if not (EVAL_DIR / n).exists():
            pytest.skip(f"{n} not present")
    dfs = [cp.load_df(str(EVAL_DIR / n)) for n in
           ["grobid_results_v4.xlsx", "nlp_results.xlsx", "llm_results.xlsx", hybrid]]
    t = cp.time_cost_table(*dfs, measured_times={"grobid": 11.0, "nlp": 22.0})
    row = t[t.Pipeline == "GROBID"].iloc[0]
    assert row["Time (s)"] == 11.0


def test_time_cost_table_all_pipelines_present(cp, repo):
    hybrid = _hybrid_xlsx_name()
    for n in ["grobid_results_v4.xlsx", "nlp_results.xlsx", "llm_results.xlsx", hybrid]:
        if not (EVAL_DIR / n).exists():
            pytest.skip(f"{n} not present")
    dfs = [cp.load_df(str(EVAL_DIR / n)) for n in
           ["grobid_results_v4.xlsx", "nlp_results.xlsx", "llm_results.xlsx", hybrid]]
    t = cp.time_cost_table(*dfs)
    assert set(t["Pipeline"]) == {"GROBID", "NLP/ML", "LLM", cp.HYBRID_PIPELINE_LABEL}
    assert (t["Cost (USD)"] >= 0).all()


# ------------------------------------------- name parsing edge branches

@pytest.fixture(params=["joshcoley", "grobid", "llm", "npl", "task1"])
def mod(request, extractors):
    return extractors[request.param]


@pytest.mark.parametrize("raw", ["", "   ", ","])
def test_parse_name_for_abbrev_degenerate(mod, raw):
    first, middle, last = mod.parse_name_for_abbrev(raw)
    assert isinstance(first, str) and isinstance(last, str)


def test_parse_name_for_abbrev_single_token(mod):
    first, middle, last = mod.parse_name_for_abbrev("Madonna")
    assert first == "Madonna"
    assert middle == []


def test_abbreviate_leaves_existing_initial_alone(mod):
    assert mod.abbreviate_target_name("Ann B. Cee") == "Ann B. Cee"


def test_abbreviate_handles_single_token(mod):
    assert mod.abbreviate_target_name("Madonna") == "Madonna"


def test_strip_accents_non_string_safe(mod):
    assert mod.strip_accents("plain") == "plain"


# ---------------------------------------------- PDF fallback when fitz absent

def test_pdf_text_all_falls_back_without_fitz(jc, monkeypatch, sample_pdf):
    monkeypatch.setattr(jc, "fitz", None)
    out = jc.pdf_text_all(sample_pdf)
    assert isinstance(out, str)


def test_byline_window_falls_back_without_fitz(jc, monkeypatch, sample_pdf):
    monkeypatch.setattr(jc, "fitz", None)
    out = jc.pdf_text_byline_window(sample_pdf)
    assert isinstance(out, str)


def test_pdf_text_first_n_falls_back_without_fitz(jc, monkeypatch, sample_pdf):
    monkeypatch.setattr(jc, "fitz", None)
    assert isinstance(jc.pdf_text_first_n(sample_pdf, 2), str)


def test_byline_window_on_empty_pdf(jc, monkeypatch, sample_pdf):
    """A zero-page document must return '' rather than IndexError."""
    class EmptyDoc:
        def __len__(self): return 0
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(jc.fitz, "open", lambda p: EmptyDoc())
    assert jc.pdf_text_byline_window(sample_pdf) == ""
