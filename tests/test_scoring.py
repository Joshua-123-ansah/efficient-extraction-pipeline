"""
Record linkage and scoring in compare_pipelines.py.

A defect here corrupts the reported F1 without touching extraction quality,
so these are the numbers that end up in a paper or on a CV.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pipeline_paths import EVAL_DIR


# ------------------------------------------------------------- primitives

@pytest.mark.parametrize("raw,expected", [
    ("A, B!", "a b"),
    ("  Multiple   spaces  ", "multiple spaces"),
    ("", ""),
])
def test_normalize_text(cp, raw, expected):
    assert cp.normalize_text(raw) == expected


def test_normalize_text_handles_nan(cp):
    assert cp.normalize_text(np.nan) == ""


def test_to_int_or_nan(cp):
    assert cp.to_int_or_nan("5") == 5
    assert np.isnan(cp.to_int_or_nan("abc"))
    assert np.isnan(cp.to_int_or_nan(None))


def test_field_similarity_numeric_fields(cp):
    assert cp.field_similarity("Author's Position", 3, 3) == 1.0
    assert cp.field_similarity("Author's Position", 3, 4) == 0.0
    assert cp.field_similarity("Total Authors", 5, "x") == 0.0


def test_field_similarity_text(cp):
    assert cp.field_similarity("Title", "A Paper", "A Paper") == 1.0
    assert cp.field_similarity("Title", "A Paper", "Totally Different") < 0.5


def test_affiliation_ignores_postal_verbosity(cp):
    gold = "VirginiaTech"
    pred = "Virginia Tech, 416-B Hutcheson Hall, MC 0349, 250 Drillfield Drive, Blacksburg, VA, 24061"
    assert cp.field_similarity("Affiliation", gold, pred) >= cp.THRESHOLDS["Affiliation"]
    assert cp.is_correct("Affiliation", gold, pred, cp.THRESHOLDS)


def test_affiliation_still_rejects_wrong_institution(cp):
    assert cp.field_similarity("Affiliation", "Purdue University", "Somewhere Else Entirely") < 0.5


def test_is_correct_respects_thresholds(cp):
    assert cp.is_correct("Title", "A Paper", "A Paper", cp.THRESHOLDS) is True
    assert cp.is_correct("Title", "A Paper", "Unrelated Text", cp.THRESHOLDS) is False


# ------------------------------------------------------- person identity

@pytest.mark.parametrize("a,b,expected", [
    ("jerrod a henderson", "jerrod henderson", True),
    ("j henderson", "jerrod henderson", True),
    ("brooke coley", "brooke coley", True),
    ("kelly cross", "david cross", False),
    ("brooke coley", "brooke smith", False),
    ("", "", False),
    ("", "ann bee", False),
])
def test_same_person(cp, a, b, expected):
    assert cp._same_person(a, b) is expected


# ------------------------------------------------------------- linkage

def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "Title", "Target Author", "Found Author Name", "Author's Position",
        "Total Authors", "Affiliation", "Abstract", "File Path"])


def test_linkage_binds_file_and_author(cp):
    """
    Papers with two target-author rows must not cross-wire: matching on File
    Path alone let the first gold row claim the first prediction row.
    """
    gold = _frame([
        ["T", "Ebony O. McGee", "Ebony McGee", 1, 2, "A", "abs", "/p/paper.pdf"],
        ["T", "Monica L. Miles", "Monica Miles", 2, 2, "B", "abs", "/p/paper.pdf"],
    ])
    pred = _frame([
        ["T", "Monica L. Miles", "Monica Miles", 2, 2, "B", "abs", "/p/paper.pdf"],
        ["T", "Ebony O. McGee", "Ebony McGee", 1, 2, "A", "abs", "/p/paper.pdf"],
    ])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    for gi, pi in mapping.items():
        assert pi is not None
        assert cp._same_person(cp.normalize_text(gold.loc[gi, "Target Author"]),
                               cp.normalize_text(pred.iloc[pi]["Target Author"]))


def test_linkage_tolerates_name_variants(cp):
    gold = _frame([["T", "Jerrod A. Henderson", "J Henderson", 1, 1, "A", "x", "/p/a.pdf"]])
    pred = _frame([["T", "Jerrod Henderson", "J Henderson", 1, 1, "A", "x", "/p/a.pdf"]])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert mapping[0] == 0


def test_linkage_never_reuses_a_prediction_row(cp):
    gold = _frame([
        ["T", "Ann Bee", "Ann Bee", 1, 2, "A", "x", "/p/a.pdf"],
        ["T", "Cid Dee", "Cid Dee", 2, 2, "B", "x", "/p/a.pdf"],
    ])
    pred = _frame([["T", "Ann Bee", "Ann Bee", 1, 2, "A", "x", "/p/a.pdf"]])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    used = [v for v in mapping.values() if v is not None]
    assert len(used) == len(set(used))


def test_linkage_matches_by_filename_when_directories_differ(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/old/dir/a.pdf"]])
    pred = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/new/dir/a.pdf"]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] == 0


def test_linkage_unmatched_gold_maps_to_none(cp):
    gold = _frame([["Completely Unrelated Title", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/p/a.pdf"]])
    pred = _frame([["Nothing Alike Whatsoever", "Cid Dee", "Cid Dee", 1, 1, "B", "y", "/p/b.pdf"]])
    assert cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)[0] is None


def test_linkage_empty_prediction_frame(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/p/a.pdf"]])
    pred = _frame([])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert all(v is None for v in mapping.values())


def test_linkage_without_target_author_column(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/p/a.pdf"]])
    pred = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "A", "x", "/p/a.pdf"]]).drop(columns=["Target Author"])
    mapping = cp.link_records(gold, pred, cp.TITLE_MATCH_THRESHOLD)
    assert mapping[0] == 0


# ------------------------------------------------------------ evaluation

def test_perfect_prediction_scores_one(cp):
    rows = [["T", "Ann Bee", "Ann Bee", 1, 1, "Purdue University", "An abstract.", "/p/a.pdf"]]
    df = _frame(rows)
    metrics, summary = cp.evaluate_pipeline(df, df.copy(), cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    assert summary["coverage"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert metrics["FP"].sum() == 0
    assert metrics["FN"].sum() == 0


def test_missing_prediction_counts_as_false_negative(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "Purdue", "abs", "/p/a.pdf"]])
    pred = _frame([])
    metrics, summary = cp.evaluate_pipeline(gold, pred, cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    assert summary["coverage"] == 0.0
    assert metrics["FN"].sum() == len(cp.FIELDS)


def test_blank_predicted_field_counts_as_false_negative(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "Purdue University", "abs", "/p/a.pdf"]])
    pred = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "", "abs", "/p/a.pdf"]])
    metrics, _ = cp.evaluate_pipeline(gold, pred, cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    aff = metrics[metrics.Field == "Affiliation"].iloc[0]
    assert aff.FN == 1 and aff.TP == 0


def test_wrong_predicted_field_counts_as_false_positive(cp):
    gold = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "Purdue University", "abs", "/p/a.pdf"]])
    pred = _frame([["T", "Ann Bee", "Ann Bee", 1, 1, "Somewhere Else Entirely", "abs", "/p/a.pdf"]])
    metrics, _ = cp.evaluate_pipeline(gold, pred, cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    aff = metrics[metrics.Field == "Affiliation"].iloc[0]
    assert aff.FP == 1 and aff.TP == 0


def _hybrid_results_path() -> Path:
    for name in ("alphamind_extractor_results.xlsx", "joshcoley_results.xlsx"):
        p = EVAL_DIR / name
        if p.exists():
            return p
    pytest.skip("alphamind_extractor_results.xlsx / joshcoley_results.xlsx not present")
    return EVAL_DIR / "alphamind_extractor_results.xlsx"  # unreachable


def test_metrics_are_bounded(cp, gold_df, repo):
    for name in ["grobid_results_v4.xlsx", "llm_results.xlsx"]:
        p = EVAL_DIR / name
        if not p.exists():
            pytest.skip(f"{name} not present")
        metrics, summary = cp.evaluate_pipeline(
            gold_df, cp.load_df(str(p)), cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
        for col in ["Precision", "Recall", "F1"]:
            assert metrics[col].between(0.0, 1.0).all(), f"{name} {col} out of range"
        for k in ["coverage", "macro_f1", "micro_f1"]:
            assert 0.0 <= summary[k] <= 1.0
    p = _hybrid_results_path()
    metrics, summary = cp.evaluate_pipeline(
        gold_df, cp.load_df(str(p)), cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    for col in ["Precision", "Recall", "F1"]:
        assert metrics[col].between(0.0, 1.0).all(), f"{p.name} {col} out of range"
    for k in ["coverage", "macro_f1", "micro_f1"]:
        assert 0.0 <= summary[k] <= 1.0


def test_no_cross_wired_linkage_on_real_results(cp, gold_df, repo):
    for name in ["grobid_results_v4.xlsx", "llm_results.xlsx"]:
        p = EVAL_DIR / name
        if not p.exists():
            pytest.skip(f"{name} not present")
        pred = cp.load_df(str(p))
        mapping = cp.link_records(gold_df, pred, cp.TITLE_MATCH_THRESHOLD)
        for gi, pi in mapping.items():
            if pi is None:
                continue
            g = cp.normalize_text(gold_df.loc[gi, "Target Author"])
            q = cp.normalize_text(pred.iloc[pi]["Target Author"])
            assert cp._same_person(g, q), f"{name}: {g!r} linked to {q!r}"
    p = _hybrid_results_path()
    pred = cp.load_df(str(p))
    mapping = cp.link_records(gold_df, pred, cp.TITLE_MATCH_THRESHOLD)
    for gi, pi in mapping.items():
        if pi is None:
            continue
        g = cp.normalize_text(gold_df.loc[gi, "Target Author"])
        q = cp.normalize_text(pred.iloc[pi]["Target Author"])
        assert cp._same_person(g, q), f"{p.name}: {g!r} linked to {q!r}"


# --------------------------------------------------------- table builders

@pytest.fixture
def real_metrics(cp, gold_df, repo):
    hybrid_path = _hybrid_results_path()
    files = {"GROBID": "grobid_results_v4.xlsx", "NLP/ML": "nlp_results.xlsx",
             "LLM": "llm_results.xlsx", cp.HYBRID_PIPELINE_LABEL: hybrid_path.name}
    M, S, D = {}, {}, {}
    for name, fn in files.items():
        p = EVAL_DIR / fn
        if not p.exists():
            pytest.skip(f"{fn} not present")
        D[name] = cp.load_df(str(p))
        M[name], S[name] = cp.evaluate_pipeline(gold_df, D[name], cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    return M, S, D


def test_side_by_side_omits_missing_pipeline(cp):
    rows = [["T", "Ann Bee", "Ann Bee", 1, 1, "Purdue University", "An abstract.", "/p/a.pdf"]]
    df = _frame(rows)
    metrics, summary = cp.evaluate_pipeline(df, df.copy(), cp.FIELDS, cp.TITLE_MATCH_THRESHOLD)
    sxs = cp.side_by_side_table(metrics, None, None, None, cp.FIELDS)
    pipes = set(sxs.columns.get_level_values(1))
    assert pipes == {"GROBID"}
    overall = cp.overall_table(metrics, None, None, None, summary, None, None, None)
    assert list(overall.columns) == ["GROBID"]
    sentence = cp.overall_sentence(overall)
    assert "GROBID" in sentence
    assert "compared to" not in sentence


def test_side_by_side_table(cp, real_metrics):
    M, _, _ = real_metrics
    t = cp.side_by_side_table(M["GROBID"], M["NLP/ML"], M["LLM"], M[cp.HYBRID_PIPELINE_LABEL], cp.FIELDS)
    assert len(t) == len(cp.FIELDS)


def test_overall_table_and_sentence(cp, real_metrics):
    M, S, _ = real_metrics
    t = cp.overall_table(M["GROBID"], M["NLP/ML"], M["LLM"], M[cp.HYBRID_PIPELINE_LABEL],
                         S["GROBID"], S["NLP/ML"], S["LLM"], S[cp.HYBRID_PIPELINE_LABEL])
    assert not t.empty
    assert isinstance(cp.overall_sentence(t), str)


def test_per_field_sentences(cp, real_metrics):
    M, _, _ = real_metrics
    sxs = cp.side_by_side_table(M["GROBID"], M["NLP/ML"], M["LLM"], M[cp.HYBRID_PIPELINE_LABEL], cp.FIELDS)
    out = cp.per_field_sentences(sxs)
    assert isinstance(out, list) and out


def test_time_cost_table(cp, real_metrics):
    _, _, D = real_metrics
    t = cp.time_cost_table(D["GROBID"], D["NLP/ML"], D["LLM"], D[cp.HYBRID_PIPELINE_LABEL])
    assert list(t["Pipeline"]) == ["GROBID", "NLP/ML", "LLM", cp.HYBRID_PIPELINE_LABEL]


def test_add_micro_macro_rows(cp, real_metrics):
    M, S, _ = real_metrics
    t = cp.add_micro_macro_rows(M["GROBID"].copy(), S["GROBID"])
    assert len(t) > len(M["GROBID"])


def test_safe_read_json_missing_file(cp):
    assert cp.safe_read_json("/nonexistent/stats.json") == {}
