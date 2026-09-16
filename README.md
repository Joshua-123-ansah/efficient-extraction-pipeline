# Efficient Extraction Pipeline

Installable Python package for scholarly PDF metadata extraction (GROBID, spaCy NLP, OpenAI/CreateAI LLM, and the AlphaMind hybrid extractor).

## Install

Use a project-local virtualenv. Do not install into Anaconda `base` or any shared environment.

```bash
# Python 3.12+ (macOS /usr/bin/python3 is often 3.9 — do not use that)
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

After that, keep `.venv` activated (or call `.venv/bin/python` / `.venv/bin/pytest`) for all work in this repo.

From GitHub (other projects can depend on this):

```bash
python -m pip install "git+https://github.com/Joshua-123-ansah/efficient-extraction-pipeline.git@main"
```

## Run

```bash
# Hybrid AlphaMind extractor
python alphamind_extractor.py [papers_dir] [authors]

# GROBID-only
python grobid_pipeline.py [papers_dir]

# Compare against gold standard (uses existing Excel under data/evaluation/)
python scripts/compare_pipelines.py
# or generate GROBID results first, then compare:
python scripts/compare_pipelines.py --run --run-which grobid
```

Defaults look under `data/corpora/my_papers/` (local PDFs, not in git) and write Excel under `data/evaluation/`. Override with `ROOT_PAPERS_DIR`, `OUTPUT_XLSX`, `GOLD_XLSX`, etc.

## GROBID server

Use the **full** 0.9.1 image (Deep Learning header + affiliation models). The small CRF image is weaker on author lists.

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-full
```

Header consolidation (CrossRef / biblio-glutton) is on by default. Pass `--no-consolidate-header` for PDF-only metadata.

```bash
python scripts/compare_pipelines.py --run --run-which grobid
python scripts/compare_pipelines.py --run --run-which grobid --no-consolidate-header
```

## Consumers

Apps such as the Space Lab / AlphaMind web platform should depend on this package rather than vendoring these files. When you push updates here, those apps pick them up with `pip install -U …`.
