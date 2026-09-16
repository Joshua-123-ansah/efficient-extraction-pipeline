"""Default paths for the efficient-extraction-pipeline package (no app coupling)."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINES_DIR = PACKAGE_ROOT  # pipeline modules live at package root

DATA_DIR = PACKAGE_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
GOLD_DIR = DATA_DIR / "gold"
EVAL_DIR = DATA_DIR / "evaluation"
OUTPUT_DIR = PACKAGE_ROOT / "output"


def _xlsx(env_name: str, *candidates: Path) -> Path:
    env = os.environ.get(env_name)
    if env:
        return Path(env).expanduser()
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


MY_PAPERS = Path(os.environ.get("ROOT_PAPERS_DIR", CORPORA_DIR / "my_papers")).expanduser()
JOURNALS = Path(os.environ.get("JOURNALS_DIR", CORPORA_DIR / "journals")).expanduser()

GOLD_XLSX = Path(os.environ.get("GOLD_XLSX", GOLD_DIR / "Gold Standard.xlsx"))
GROBID_RESULTS_XLSX = Path(os.environ.get("GROBID_XLSX", EVAL_DIR / "grobid_results_v4.xlsx"))
NLP_RESULTS_XLSX = Path(os.environ.get("NLP_XLSX", EVAL_DIR / "nlp_results.xlsx"))
LLM_RESULTS_XLSX = Path(os.environ.get("LLM_XLSX", EVAL_DIR / "llm_results.xlsx"))
ALPHAMIND_EXTRACTOR_RESULTS_XLSX = _xlsx(
    "ALPHAMIND_XLSX",
    EVAL_DIR / "alphamind_extractor_results.xlsx",
    EVAL_DIR / "joshcoley_results.xlsx",
)
JOSHCOLEY_RESULTS_XLSX = ALPHAMIND_EXTRACTOR_RESULTS_XLSX  # legacy alias
EVAL_SUMMARY_XLSX = Path(
    os.environ.get("SUMMARY_XLSX", EVAL_DIR / "pipeline_evaluation_summary.xlsx")
)
PIPELINE_RUN_STATS_JSON = Path(
    os.environ.get("RUN_STATS_JSON", OUTPUT_DIR / "pipeline_run_stats.json")
)


def ensure_runtime_dirs() -> None:
    for d in (EVAL_DIR, OUTPUT_DIR, GOLD_DIR, CORPORA_DIR, MY_PAPERS):
        d.mkdir(parents=True, exist_ok=True)
