"""
Text-level extraction: abstract/keyword slicing, byline detection, PDF windows.

The ABSTRACT_HDR / KEYWORDS_HDR patterns are embedded inside larger regexes
whose group(1) is meant to be the body text. If they are capturing groups every
index shifts and extraction returns the literal word "Abstract".
"""

import re

import pytest

ALL = ["joshcoley", "grobid", "llm", "npl", "task1"]


@pytest.fixture(params=ALL)
def mod(request, extractors):
    return extractors[request.param]


# ------------------------------------------------- header patterns are safe

def test_abstract_hdr_is_non_capturing(mod):
    assert re.compile(mod.ABSTRACT_HDR).groups == 0, \
        "ABSTRACT_HDR must not capture; it shifts group indices in composed patterns"


def test_keywords_hdr_is_non_capturing(mod):
    assert re.compile(mod.KEYWORDS_HDR).groups == 0, \
        "KEYWORDS_HDR must not capture; it shifts group indices in composed patterns"


@pytest.mark.parametrize("header", ["Abstract", "Summary", "Resumen", "摘要"])
def test_abstract_hdr_matches_known_headers(mod, header):
    assert re.fullmatch(mod.ABSTRACT_HDR, header, re.I)


@pytest.mark.parametrize("header", ["Keywords", "Keyword", "Index Terms", "关键词"])
def test_keywords_hdr_matches_known_headers(mod, header):
    assert re.fullmatch(mod.KEYWORDS_HDR, header, re.I)


def test_composed_pattern_group1_is_body(mod):
    """The exact composition used by the PDF-text fallback paths."""
    body = "Some Title\nAnn Bee\nAbstract\nThe real abstract body here.\nKeywords: x\n"
    m = re.search(
        rf'\b{mod.ABSTRACT_HDR}\b[:\s\n]*([\s\S]+?)\b{mod.KEYWORDS_HDR}\b',
        body, re.I | re.M)
    assert m is not None
    assert mod.normalize_spaces(m.group(1)) == "The real abstract body here."


# ---------------------------------------------------------- abstract slicing

def test_slice_abstract_full_returns_body(lm):
    text = "Title\nAuthors\nAbstract\nThis is the abstract body.\nKeywords: a, b\nIntroduction"
    assert lm.slice_abstract_full(text) == "This is the abstract body."


def test_slice_abstract_full_intro_delimiter(lm):
    text = "Title\nAbstract\nBody sentence here.\nIntroduction\nMore text"
    assert lm.slice_abstract_full(text) == "Body sentence here."


@pytest.mark.parametrize("text", ["", "no headers at all"])
def test_slice_abstract_full_empty(lm, text):
    assert lm.slice_abstract_full(text) == ""


def test_extract_keywords_returns_keywords(npl):
    assert npl.extract_keywords("Keywords: alpha, beta") == "alpha, beta"


def test_extract_keywords_absent(npl):
    assert npl.extract_keywords("nothing here") == ""


# ------------------------------------------------------------- byline block

def test_byline_block_stops_at_abstract(mod):
    text = "A Title\nAnn Bee, Cid Dee\nAbstract\nBody text that must not appear."
    got = mod.byline_block(text)
    assert "Ann Bee" in got
    assert "Body text" not in got


def test_byline_block_empty(mod):
    assert mod.byline_block("") == ""


# --------------------------------------------------------------- text utils

def test_normalize_spaces_handles_none(mod):
    assert mod.normalize_spaces(None) == ""


def test_clean_for_excel_strips_control_chars(mod):
    assert mod.clean_for_excel("a\x00b\x1fc") == "abc"
    assert mod.clean_for_excel(None) == ""


def test_strip_accents(mod):
    assert mod.strip_accents("Résumé") == "Resume"


# --------------------------------------------------------- PDF text helpers

def test_pdf_helpers_tolerate_non_pdf(jc, tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_text("this is not a pdf")
    assert jc.pdf_text_all(bad) == ""
    assert jc.pdf_text_first_n(bad, 3) == ""
    assert jc.pdf_text_byline_window(bad) == ""


def test_byline_window_is_smaller_than_three_pages(jc, sample_pdf):
    """
    The narrow window is the whole point of the cost reduction. If it stops
    being smaller than the 3-page read, the saving is gone.
    """
    narrow = jc.pdf_text_byline_window(sample_pdf)
    wide = jc.pdf_text_first_n(sample_pdf, 3)
    assert len(narrow) > 0
    assert len(narrow) < len(wide)


def test_byline_window_respects_cap(jc, sample_pdf):
    got = jc.pdf_text_byline_window(sample_pdf)
    assert len(got) <= jc.BYLINE_WINDOW_CHARS + 900 + 1


def test_gather_pdfs(jc, tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.PDF").write_text("x")
    (tmp_path / "c.txt").write_text("x")
    names = sorted(p.name for p in jc.gather_pdfs(tmp_path))
    assert names == ["a.pdf", "b.PDF"]


def test_gather_pdfs_single_file(jc, tmp_path):
    f = tmp_path / "only.pdf"
    f.write_text("x")
    assert [p.name for p in jc.gather_pdfs(f)] == ["only.pdf"]


def test_gather_pdfs_missing_directory(jc, tmp_path):
    assert jc.gather_pdfs(tmp_path / "does_not_exist") == []


def test_get_file_row_cached_bad_path(jc):
    assert jc.get_file_row_cached("/nonexistent/dir/file.pdf") == 1


# ------------------------------------------------------------- prefilter

def test_quick_author_hit(mod):
    assert mod.quick_author_hit("anything", []) is True
    assert mod.quick_author_hit("written by Brooke Coley", ["Brooke Coley"]) is True
    assert mod.quick_author_hit("nothing relevant", ["Brooke Coley"]) is False
    assert mod.quick_author_hit("", ["Brooke Coley"]) is False
