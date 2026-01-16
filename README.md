# Academic Paper Metadata Extraction System

A multi-pipeline system for extracting metadata (authors, affiliations, titles, abstracts, etc.) from academic PDFs using GROBID, spaCy NLP, OpenAI LLM, and a hybrid approach.

## Features

- **Four extraction pipelines**: GROBID (ML-based), spaCy NLP (rule-based), OpenAI LLM (GPT-4), and hybrid "JoshColey" pipeline
- **Intelligent hybrid approach**: Combines GROBID and LLM based on field-specific F1 performance thresholds
- **Comprehensive evaluation**: Precision, recall, F1-score (macro/micro), and coverage metrics
- **Cost optimization**: Reduces LLM API costs by 85% through selective usage
- **Parallel processing**: ThreadPoolExecutor for improved performance

## Prerequisites

1. **Python 3.8+**
2. **GROBID Server**: Install and run GROBID locally or use a remote instance
   - Download from: https://github.com/kermitt2/grobid
   - Default URL: `http://localhost:8070`
3. **OpenAI API Key** (optional, for LLM pipeline): Set `OPENAI_API_KEY` environment variable

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd spacelabproject
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install spaCy language model for NLP pipeline:
```bash
python -m spacy download en_core_web_sm
```

4. Set up GROBID:
   - Follow GROBID installation instructions: https://grobid.readthedocs.io/en/latest/Install-Grobid/
   - Start GROBID server: `./gradlew run` (from grobid directory)
   - Or use Docker: `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0`

## Usage

### Running Extraction Pipelines

Each pipeline can be run independently:

#### GROBID Pipeline
```bash
python grobid_extraction_v4.py [root_folder] [target_authors]
```

#### NLP Pipeline (spaCy)
```bash
python NPL_extraction_pipeline.py [root_folder] [target_authors]
```

#### LLM Pipeline (OpenAI)
```bash
export OPENAI_API_KEY="your-api-key"
python LLM_extraction_pipeline.py [root_folder] [target_authors]
```

#### Hybrid Pipeline (JoshColey)
```bash
export OPENAI_API_KEY="your-api-key"
python joshcoley_extraction_pipeline.py [root_folder] [target_authors]
```

**Parameters:**
- `root_folder`: Path to directory containing PDFs (default: "My Papers")
- `target_authors`: Comma-separated list of author names to filter (optional)

### Comparing Pipelines

Compare pipeline outputs against a gold standard:

```bash
python compare_pipelines.py [--gold GOLD_XLSX] [--grobid GROBID_XLSX] [--nlp NLP_XLSX] [--llm LLM_XLSX] [--joshcoley JOSHCOLEY_XLSX]
```

**Options:**
- `--gold`: Path to gold standard Excel file (default: "Golden Standard.xlsx")
- `--grobid`: Path to GROBID results (default: "grobid_results_v4.xlsx")
- `--nlp`: Path to NLP results (default: "nlp_results.xlsx")
- `--llm`: Path to LLM results (default: "llm_results.xlsx")
- `--joshcoley`: Path to JoshColey results (default: "joshcoley_results.xlsx")
- `--export`: Export comparison results to Excel
- `--run`: Run a specific pipeline before comparison
- `--run-which`: Which pipeline to run (grobid, nlp, llm, joshcoley)

**Example:**
```bash
# Compare all pipelines
python compare_pipelines.py --export

# Run and compare only GROBID
python compare_pipelines.py --run --run-which grobid --export
```

## Configuration

Set environment variables or modify defaults in each script:

- `GROBID_URL`: GROBID server URL (default: `http://localhost:8070`)
- `OPENAI_API_KEY`: OpenAI API key (required for LLM and JoshColey pipelines)
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o-mini`)
- `ROOT_PAPERS_DIR`: Default root folder for PDFs (default: `My Papers`)

## Output Format

All pipelines export Excel files with the following columns:
- Title
- Target Author
- Found Author Name
- Author's Position
- Total Authors
- Affiliation
- Abstract
- File Path
- Paper Row

## Evaluation Metrics

The comparison script computes:
- **Precision, Recall, F1-score**: Per-field and macro/micro averages
- **Coverage**: Percentage of gold standard papers successfully matched
- **Record Linkage**: Matches records via file path (exact match) and title similarity (fallback)

## Project Structure

```
.
├── compare_pipelines.py          # Evaluation and comparison script
├── grobid_extraction_v4.py        # GROBID extraction pipeline
├── NPL_extraction_pipeline.py    # spaCy NLP extraction pipeline
├── LLM_extraction_pipeline.py    # OpenAI LLM extraction pipeline
├── joshcoley_extraction_pipeline.py  # Hybrid pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

**Note**: PDF files are not included in the repository. Users should provide their own PDFs in a directory structure of their choice.

## Notes

- GROBID must be running before using GROBID or hybrid pipelines
- LLM pipeline requires valid OpenAI API key and will incur costs
- Hybrid pipeline intelligently uses GROBID for high-performing fields (F1 ≥ 0.85) and LLM for low-performing fields (F1 < 0.85)
- All pipelines support parallel processing for improved performance

## License

[Add your license here]

