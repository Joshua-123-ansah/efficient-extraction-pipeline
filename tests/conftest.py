"""Shared fixtures and TEI builders for the pipeline test suite."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline_paths import EVAL_DIR, GOLD_XLSX, MY_PAPERS

TEI_NS = 'http://www.tei-c.org/ns/1.0'


def author_xml(name: str, affiliations=None, raw_affiliations=None) -> str:
    """
    Build a <author> element.

    affiliations:     list of orgName institution strings
    raw_affiliations: list of raw_affiliation note strings (paired by index)
    """
    given, surname = name.rsplit(' ', 1)
    forenames = ''.join(f'<forename>{g}</forename>' for g in given.split())
    affiliations = affiliations or []
    raw_affiliations = raw_affiliations or []
    n = max(len(affiliations), len(raw_affiliations))
    aff = ''
    for i in range(n):
        raw = f'<note type="raw_affiliation">{raw_affiliations[i]}</note>' if i < len(raw_affiliations) else ''
        org = f'<orgName type="institution">{affiliations[i]}</orgName>' if i < len(affiliations) else ''
        aff += f'<affiliation key="aff{i}">{raw}{org}</affiliation>'
    return f'<author><persName>{forenames}<surname>{surname}</surname></persName>{aff}</author>'


def build_tei(title="A Test Paper", analytic_authors=(), titlestmt_authors=(),
              abstract_paragraphs=(), abstract_text=None, orphan_affiliations=()):
    """Assemble a minimal but structurally faithful GROBID TEI document."""
    analytic = ''.join(a if a.startswith('<author') else author_xml(a) for a in analytic_authors)
    titlestmt = ''.join(a if a.startswith('<author') else author_xml(a) for a in titlestmt_authors)
    orphans = ''.join(
        f'<affiliation key="orphan{i}"><note type="raw_affiliation">{o}</note></affiliation>'
        for i, o in enumerate(orphan_affiliations)
    )
    if abstract_text is not None:
        abstract = f'<abstract>{abstract_text}</abstract>'
    elif abstract_paragraphs:
        paras = ''.join(f'<p>{p}</p>' for p in abstract_paragraphs)
        abstract = f'<abstract><div>{paras}</div></abstract>'
    else:
        abstract = ''
    profile = f'<profileDesc>{abstract}</profileDesc>' if abstract else ''
    return (
        f'<TEI xmlns="{TEI_NS}"><teiHeader><fileDesc>'
        f'<titleStmt><title level="a" type="main">{title}</title>{titlestmt}</titleStmt>'
        f'<sourceDesc><biblStruct><analytic>'
        f'<title level="a" type="main">{title}</title>{analytic}{orphans}'
        f'</analytic></biblStruct></sourceDesc>'
        f'</fileDesc>{profile}</teiHeader></TEI>'
    )


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def jc():
    import alphamind_extractor
    return alphamind_extractor


@pytest.fixture(scope="session")
def gr():
    import grobid_pipeline
    return grobid_pipeline


@pytest.fixture(scope="session")
def lm():
    import LLM_extraction_pipeline
    return LLM_extraction_pipeline


@pytest.fixture(scope="session")
def npl():
    import NPL_extraction_pipeline
    return NPL_extraction_pipeline


@pytest.fixture(scope="session")
def task1():
    import grobid_pipeline
    return grobid_pipeline


@pytest.fixture(scope="session")
def cp():
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_pipelines", REPO / "compare_pipelines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def extractors(jc, gr, lm, npl, task1):
    """Every module that carries a copy of the name-handling helpers."""
    return {
        "alphamind_extractor": jc,
        "joshcoley": jc,  # legacy test alias
        "grobid": gr,
        "llm": lm,
        "npl": npl,
        "task1": task1,
    }


@pytest.fixture(scope="session")
def tei_parsers(jc, gr, task1):
    """Modules that expose parse_tei."""
    return {"alphamind_extractor": jc, "joshcoley": jc, "grobid": gr, "task1": task1}


@pytest.fixture
def sample_pdf(repo):
    """A real PDF from the corpus, or skip if the corpus is absent."""
    pdfs = sorted(MY_PAPERS.rglob("*.pdf")) if MY_PAPERS.exists() else []
    if not pdfs:
        pytest.skip("no PDFs under data/corpora/my_papers")
    return pdfs[0]


@pytest.fixture
def gold_df(cp, repo):
    if not GOLD_XLSX.exists():
        pytest.skip("Gold Standard.xlsx not present")
    return cp.load_df(str(GOLD_XLSX))
