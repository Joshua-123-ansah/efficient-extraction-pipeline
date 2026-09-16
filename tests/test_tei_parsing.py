"""
GROBID TEI parsing: authors, affiliations, abstract.

Every field asserted here is scored against the gold standard, so a regression
in this file moves the reported F1 directly.
"""

import pytest

from conftest import build_tei, author_xml

PARSERS = ["joshcoley", "grobid", "task1"]


@pytest.fixture(params=PARSERS)
def mod(request, tei_parsers):
    return tei_parsers[request.param]


def affiliations_of(md):
    """Flatten parse_tei output to a list of affiliation strings."""
    return [a for item in (md.get("affiliations") or []) for a in item.get("affiliations", [])]


# --------------------------------------------------------------- affiliations

def test_raw_and_orgname_not_doubled(mod):
    """
    includeRawAffiliations=1 nests a raw_affiliation note beside the parsed
    orgName. itertext() over both produced "Purdue University Purdue University".
    """
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee", affiliations=["Purdue University"],
                   raw_affiliations=["Purdue University"])])
    assert affiliations_of(mod.parse_tei(tei)) == ["Purdue University"]


def test_raw_affiliation_label_stripped(mod):
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee", affiliations=["Purdue University"],
                   raw_affiliations=["<label>c</label> Purdue University"])])
    assert affiliations_of(mod.parse_tei(tei)) == ["Purdue University"]


def test_raw_affiliation_preferred_over_structured(mod):
    """The raw note keeps the punctuation the gold standard uses."""
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee",
                   affiliations=["Florida International University"],
                   raw_affiliations=["School of Computing, Florida International University"])])
    assert affiliations_of(mod.parse_tei(tei)) == [
        "School of Computing, Florida International University"]


def test_orgname_used_when_no_raw_note(mod):
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee", affiliations=["Purdue University"])])
    assert affiliations_of(mod.parse_tei(tei)) == ["Purdue University"]


def test_empty_affiliation_dropped(mod):
    tei = build_tei(analytic_authors=[
        '<author><persName><forename>Ann</forename><surname>Bee</surname></persName>'
        '<affiliation key="a0"></affiliation></author>'])
    assert affiliations_of(mod.parse_tei(tei)) == []


def test_two_distinct_affiliations_both_kept(mod):
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee", raw_affiliations=[
            "Purdue University, West Lafayette, Indiana",
            "Virginia Tech, Blacksburg, Virginia"])])
    assert affiliations_of(mod.parse_tei(tei)) == [
        "Purdue University, West Lafayette, Indiana",
        "Virginia Tech, Blacksburg, Virginia"]


def test_no_trailing_separator(mod):
    tei = build_tei(analytic_authors=[
        author_xml("Ann Bee", raw_affiliations=["Shanghai Jiao Tong University ,"])])
    got = affiliations_of(mod.parse_tei(tei))
    assert got == ["Shanghai Jiao Tong University"]


def test_single_author_inherits_orphan_affiliation(mod):
    tei = build_tei(analytic_authors=["Ann Bee"],
                    orphan_affiliations=["Some Big University"])
    assert affiliations_of(mod.parse_tei(tei)) == ["Some Big University"]


def test_multiple_authors_do_not_inherit_orphan_affiliation(mod):
    tei = build_tei(analytic_authors=["Ann Bee", "Cid Dee"],
                    orphan_affiliations=["Some Big University"])
    assert affiliations_of(mod.parse_tei(tei)) == []


# -------------------------------------------------------------------- authors

def test_authors_in_order(mod):
    tei = build_tei(analytic_authors=["Ann Bee", "Cid Dee", "Eve Fay"])
    assert mod.parse_tei(tei)["authors"] == ["Ann Bee", "Cid Dee", "Eve Fay"]


@pytest.mark.parametrize("analytic,titlestmt,expected", [
    (["Jiabin Zhu", "Yi Li"], ["Jiabin Zhu", "Yi Li"], 2),   # identical
    (["Jiabin Zhu", "Yi Li"], ["J. Zhu", "Y. Li"], 2),        # initials variant
    (["Monica F. Cox"], ["Monica F Cox"], 1),                 # punctuation variant
    (["Jiabin Zhu", "Yi Li"], [], 2),                         # titleStmt empty
    (["Jiabin Zhu"], ["Jiabin Zhu", "Ed Itor"], 2),           # genuine extra person
    (["Anna Smith", "Brian Smith"], [], 2),                   # distinct, same surname
])
def test_authors_not_double_counted(mod, analytic, titlestmt, expected):
    """
    GROBID lists the same people under <analytic> and <titleStmt>, often
    rendered differently. Exact-string dedup let variants through, inflating
    Total Authors and shifting Author's Position.
    """
    tei = build_tei(analytic_authors=analytic, titlestmt_authors=titlestmt)
    assert len(mod.parse_tei(tei)["authors"]) == expected


def test_affiliation_follows_the_kept_author_rendering(mod):
    """Deduping must not orphan the affiliation attached to the dropped node."""
    tei = build_tei(
        analytic_authors=[author_xml("Jiabin Zhu", raw_affiliations=["Shanghai Jiao Tong University"])],
        titlestmt_authors=["J. Zhu"])
    md = mod.parse_tei(tei)
    assert md["authors"] == ["Jiabin Zhu"]
    assert affiliations_of(md) == ["Shanghai Jiao Tong University"]


# ------------------------------------------------------------------- abstract

def test_multi_paragraph_abstract_is_space_joined(mod):
    """itertext() glued <p> siblings into 'Line one.Line two.'"""
    tei = build_tei(abstract_paragraphs=["Line one.", "Line two."])
    assert mod.parse_tei(tei)["abstract"] == "Line one. Line two."


def test_abstract_without_paragraphs(mod):
    tei = build_tei(abstract_text="Bare text abstract.")
    assert mod.parse_tei(tei)["abstract"] == "Bare text abstract."


def test_missing_abstract_is_empty_string(mod):
    tei = build_tei(analytic_authors=["Ann Bee"])
    assert mod.parse_tei(tei)["abstract"] == ""


def test_title_extracted(mod):
    tei = build_tei(title="Repositioning Race and Gender")
    assert mod.parse_tei(tei)["title"] == "Repositioning Race and Gender"


# ---------------------------------------------------------------- robustness

def test_tei_without_authors_returns_empty_list(mod):
    assert mod.parse_tei(build_tei())["authors"] == []


def test_parse_bibtex_variants(mod):
    assert mod.parse_bibtex("")["authors"] == []
    assert mod.parse_bibtex("garbage")["authors"] == []
    md = mod.parse_bibtex("@article{k, title={T}, author={Ann Bee and Cid Dee}}")
    assert md["authors"] == ["Ann Bee", "Cid Dee"]
    assert md["title"] == "T"


def test_malformed_tei_does_not_propagate(mod, sample_pdf, monkeypatch):
    """
    A truncated TEI (GROBID timeout, dropped connection) must fall back to the
    PDF-text path rather than losing the paper.
    """
    monkeypatch.setattr(mod, "extract_with_grobid", lambda p: "<TEI><unclosed>")
    # joshcoley exposes extract_grobid(); grobid_extraction_v4 exposes process_pdf()
    entry = getattr(mod, "extract_grobid", None) or mod.process_pdf
    md = entry(sample_pdf, True)
    assert isinstance(md, dict)
    assert "authors" in md


def test_bibtex_response_routed_to_bibtex_parser(mod, sample_pdf, monkeypatch):
    monkeypatch.setattr(mod, "extract_with_grobid",
                        lambda p: "@article{k, title={T}, author={Ann Bee}}")
    entry = getattr(mod, "extract_grobid", None) or mod.process_pdf
    md = entry(sample_pdf, True)
    assert md["authors"] == ["Ann Bee"]


def test_grobid_unavailable_falls_back_to_pdf_text(mod, sample_pdf):
    """grobid_ok=False must still return a usable dict, not raise."""
    entry = getattr(mod, "extract_grobid", None) or mod.process_pdf
    md = entry(sample_pdf, False)
    assert isinstance(md, dict)
    assert "authors" in md
