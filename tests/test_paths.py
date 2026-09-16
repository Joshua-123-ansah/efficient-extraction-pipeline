"""Corpus path resolution."""

from pathlib import Path

import pytest

from pipeline_paths import CORPORA_DIR, MY_PAPERS


def test_env_override_wins(monkeypatch, tmp_path):
    target = tmp_path / "papers"
    target.mkdir()
    monkeypatch.setenv("ROOT_PAPERS_DIR", str(target))
    from importlib import reload
    import pipeline_paths as pp
    try:
        reload(pp)
        assert pp.MY_PAPERS.resolve() == target.resolve()
    finally:
        monkeypatch.delenv("ROOT_PAPERS_DIR", raising=False)
        reload(pp)


def test_default_corpus_is_local_my_papers():
    assert MY_PAPERS == CORPORA_DIR / "my_papers" or Path(MY_PAPERS).name == "my_papers"


def test_gold_corpus_is_present():
    pdfs = list(MY_PAPERS.rglob("*.pdf")) if MY_PAPERS.exists() else []
    if not pdfs:
        pytest.skip(f"no PDFs under {MY_PAPERS}")
    assert len(pdfs) >= 43, f"expected gold PDFs under {MY_PAPERS}, found {len(pdfs)}"
