#!/usr/bin/env python3
"""
Compare Pipelines (Gold vs GROBID / NLP / LLM / AlphaMind Extractor)

What this script does:
- Reads your gold-standard Excel and pipeline outputs (same/similar columns).
- Matches records by File Path first (exact match), then Title similarity as fallback.
- Computes Coverage (#found / total gold) and per-field TP/FP/FN -> Precision/Recall/F1.
- Produces side-by-side tables (per-field & overall macro/micro).
- Prints a short narrative summary.
- Optionally exports results to Excel.
- Optionally reads time/cost stats; if unavailable, estimates LLM cost from outputs.

Assumptions:
- Columns in pipeline outputs: Title, Target Author, Found Author Name, Author's Position,
  Total Authors, Affiliation, Abstract, File Path, Paper Row

You can customize inputs via env vars or CLI (see CONFIG section).
"""

import os
import sys
import re
import json
import time
import subprocess
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline_paths import (  # noqa: E402
    EVAL_DIR,
    EVAL_SUMMARY_XLSX as _EVAL_SUMMARY_XLSX,
    GOLD_XLSX as _GOLD_XLSX,
    GROBID_RESULTS_XLSX as _GROBID_RESULTS_XLSX,
    ALPHAMIND_EXTRACTOR_RESULTS_XLSX as _ALPHAMIND_EXTRACTOR_RESULTS_XLSX,
    JOSHCOLEY_RESULTS_XLSX as _JOSHCOLEY_RESULTS_XLSX,
    LLM_RESULTS_XLSX as _LLM_RESULTS_XLSX,
    MY_PAPERS,
    NLP_RESULTS_XLSX as _NLP_RESULTS_XLSX,
    OUTPUT_DIR,
    PIPELINES_DIR,
    PIPELINE_RUN_STATS_JSON as _PIPELINE_RUN_STATS_JSON,
    ensure_runtime_dirs,
)
ensure_runtime_dirs()

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# -----------------------------
# 0) CONFIG
# -----------------------------

# Defaults (can override via env or CLI args)
GOLD_XLSX   = os.environ.get("GOLD_XLSX",   str(_GOLD_XLSX))
GROBID_XLSX = os.environ.get("GROBID_XLSX", str(_GROBID_RESULTS_XLSX))
NLP_XLSX    = os.environ.get("NLP_XLSX",    str(_NLP_RESULTS_XLSX))
LLM_XLSX    = os.environ.get("LLM_XLSX",    str(_LLM_RESULTS_XLSX))
HYBRID_PIPELINE_LABEL = "AlphaMind Extractor"
_HYBRID_RUN_ALIASES = frozenset({"joshcoley", "alphamind_extractor", "alphamind"})

JOSHCOLEY_XLSX = os.environ.get(
    "ALPHAMIND_EXTRACTOR_XLSX",
    os.environ.get("JOSHCOLEY_XLSX", str(_ALPHAMIND_EXTRACTOR_RESULTS_XLSX)),
)

# Column used to match records across files
ID_COL = os.environ.get("MATCH_ID_COL", "Title")

# Fields to evaluate - aligned with your pipeline outputs (Keywords dropped)
FIELDS: List[str] = [
    "Title",
    "Target Author",
    "Found Author Name",
    "Author's Position",
    "Total Authors",
    "Affiliation",
    "Abstract",
]

# Similarity thresholds per field (tune as needed)
THRESHOLDS: Dict[str, float] = {
    "Title":             0.93,
    "Target Author":     0.90,
    "Found Author Name": 0.90,
    "Author's Position": 1.00,  # exact numeric match
    "Total Authors":     1.00,  # exact numeric match
    "Affiliation":       0.90,
    "Abstract":          0.92,  # stricter for longer text
}
TITLE_MATCH_THRESHOLD = float(os.environ.get("TITLE_MATCH_THRESHOLD", "0.93"))

# Optional export
EXPORT_SUMMARY = os.environ.get("COMPARE_EXPORT", "1") not in ("0", "false", "False")
SUMMARY_XLSX = os.environ.get("SUMMARY_XLSX", str(_EVAL_SUMMARY_XLSX))

# Optional time/cost stats file (JSON). If absent, scripts estimate/minimize.
RUN_STATS_JSON = os.environ.get("RUN_STATS_JSON", str(_PIPELINE_RUN_STATS_JSON))

# Cost estimation fallback (only used if stats missing for LLM)
# Very rough: tokens ~= chars/4. Price per 1K tokens default.
LLM_COST_PER_1K_TOKENS = float(os.environ.get("LLM_COST_PER_1K_TOKENS", "0.002"))


# -----------------------------
# 1) HELPERS
# -----------------------------

_AFFIL_STOP = frozenset({
    "a", "an", "and", "at", "in", "of", "the", "to", "for",
    "usa", "us", "uk", "ca", "ny", "tx", "fl", "mi",
    "correspondence",
})
_GLUED_ORG = re.compile(
    r"([a-z])(university|universiteit|tech|college|institute|state)\b"
)
_CORRESPONDENCE = re.compile(r"\bcorrespondence\b.*$", re.I | re.S)
_EMAIL = re.compile(r"\S+@\S+")
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_MAIL_CODE = re.compile(r"\b(?:mc|mail\s*code)\s*\d+\b", re.I)


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2013", "-").replace("\u2014", "-")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_affiliation(s: str) -> str:
    """Scoring-only: drop postal/correspondence noise; keep institution wording."""
    if pd.isna(s):
        return ""
    text = unicodedata.normalize("NFKC", str(s))
    text = _CORRESPONDENCE.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MAIL_CODE.sub(" ", text)
    text = _ZIP.sub(" ", text)
    text = normalize_text(text)
    text = _GLUED_ORG.sub(r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_coverage(gold: str, pred: str) -> float:
    gt = {t for t in gold.split() if t not in _AFFIL_STOP}
    pt = {t for t in pred.split() if t not in _AFFIL_STOP}
    if not gt:
        return 1.0 if not pt else 0.0
    return len(gt & pt) / len(gt)

def to_int_or_nan(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def ratio(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def field_similarity(field: str, gold_val, pred_val) -> float:
    if field in ["Author's Position", "Total Authors"]:
        g = to_int_or_nan(gold_val)
        p = to_int_or_nan(pred_val)
        return 1.0 if (not np.isnan(g) and not np.isnan(p) and g == p) else 0.0
    if field == "Affiliation":
        g = normalize_affiliation(gold_val)
        p = normalize_affiliation(pred_val)
        if not g and not p:
            return 1.0
        if not g or not p:
            return 0.0
        return max(SequenceMatcher(None, g, p).ratio(), _token_coverage(g, p))
    return ratio(gold_val, pred_val)

def is_correct(field: str, gold_val, pred_val, thresholds: Dict[str, float]) -> bool:
    sim = field_similarity(field, gold_val, pred_val)
    return sim >= thresholds.get(field, 0.90)

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{path} must contain '{ID_COL}' column.")
    # Ensure all eval fields exist (fill missing with empty)
    for f in FIELDS:
        if f not in df.columns:
            df[f] = ""
    # Build keep list without duplicates (avoid double 'Title')
    keep = [ID_COL] + [f for f in FIELDS if f != ID_COL]
    # Also keep File Path if it exists (for better matching)
    if "File Path" in df.columns and "File Path" not in keep:
        keep.append("File Path")
    return df[keep].copy()


# -----------------------------
# 2) RECORD LINKAGE (File Path first, then Title fallback)
# -----------------------------

def _same_person(a: str, b: str) -> bool:
    """
    Loose author identity for record linkage only (not for scoring).

    Gold and pipelines render the same person differently ("Jerrod A. Henderson"
    vs "Jerrod Henderson"), so require the surname to agree and the given names
    to agree or be initial-compatible.
    """
    at, bt = a.split(), b.split()
    if not at or not bt:
        return False
    if at[-1] != bt[-1]:
        return False
    af, bf = at[0], bt[0]
    if af == bf:
        return True
    return af[0] == bf[0] and (len(af) == 1 or len(bf) == 1)


def link_records(gold_df: pd.DataFrame, pred_df: pd.DataFrame, title_threshold: float) -> Dict[int, int]:
    """
    Link records using File Path first (exact match), then Title similarity as fallback.
    This is more robust than title-only matching.
    """
    mapping: Dict[int, int] = {}
    used_pred = set()
    
    # Normalize file paths for comparison
    def normalize_path(p: str) -> str:
        if pd.isna(p):
            return ""
        p = str(p).strip()
        if not p:
            return ""
        # Normalize separators first
        p = p.replace("\\", "/")
        try:
            # Try to resolve if it's a valid path
            path_obj = Path(p)
            if path_obj.exists():
                return str(path_obj.resolve())
            elif path_obj.is_absolute():
                return str(path_obj)  # Keep absolute paths as-is
            else:
                # For relative paths, just normalize
                return p
        except Exception:
            # If path resolution fails, just return normalized string
            return p
    
    # First pass: Try File Path matching (most reliable)
    has_file_path = "File Path" in gold_df.columns and "File Path" in pred_df.columns
    if has_file_path:
        gold_paths = gold_df["File Path"].fillna("").apply(normalize_path).tolist()
        pred_paths = pred_df["File Path"].fillna("").apply(normalize_path).tolist()

        # Pass 0: File Path AND Target Author.
        #
        # 8 of the 43 gold papers carry more than one target-author row. Matching
        # on File Path alone lets the first gold row for a paper claim whichever
        # pred row for that paper comes first, so e.g. Ebony O. McGee's gold row
        # gets scored against Monica L. Miles's prediction. Bind the author too.
        if "Target Author" in gold_df.columns and "Target Author" in pred_df.columns:
            gold_auth = gold_df["Target Author"].fillna("").apply(normalize_text).tolist()
            pred_auth = pred_df["Target Author"].fillna("").apply(normalize_text).tolist()
            for gi in range(len(gold_df)):
                gp = gold_paths[gi] if gi < len(gold_paths) else ""
                ga = gold_auth[gi] if gi < len(gold_auth) else ""
                if not gp or not ga:
                    continue
                for pi in range(len(pred_df)):
                    if pi in used_pred:
                        continue
                    pp = pred_paths[pi] if pi < len(pred_paths) else ""
                    pa = pred_auth[pi] if pi < len(pred_auth) else ""
                    same_file = (pp and gp == pp) or (
                        pp and Path(gp).name.lower() == Path(pp).name.lower())
                    if same_file and pa and _same_person(ga, pa):
                        mapping[gi] = pi
                        used_pred.add(pi)
                        break

        # Try exact path match first
        for gi in range(len(gold_df)):
            gold_path = gold_paths[gi] if gi < len(gold_paths) else ""
            if gold_path:  # Only match if gold has a path
                for pi in range(len(pred_df)):
                    if pi not in used_pred:
                        pred_path = pred_paths[pi] if pi < len(pred_paths) else ""
                        if pred_path and gold_path == pred_path:
                            mapping[gi] = pi
                            used_pred.add(pi)
                            break
        
        # Second pass: Try filename (basename) matching for unmatched records
        # This handles cases where paths are different but files are the same
        for gi in range(len(gold_df)):
            if gi in mapping:  # Already matched
                continue
            gold_path = gold_paths[gi] if gi < len(gold_paths) else ""
            if gold_path:
                gold_filename = Path(gold_path).name.lower()
                for pi in range(len(pred_df)):
                    if pi not in used_pred:
                        pred_path = pred_paths[pi] if pi < len(pred_paths) else ""
                        if pred_path:
                            pred_filename = Path(pred_path).name.lower()
                            if gold_filename == pred_filename and gold_filename:  # Match by filename
                                mapping[gi] = pi
                                used_pred.add(pi)
                                break
    
    # Second pass: Try Title matching for unmatched records (fallback)
    pred_titles = pred_df[ID_COL].fillna("").tolist()
    for gi in range(len(gold_df)):
        if gi in mapping:  # Already matched by File Path
            continue
        gold_title = gold_df.loc[gi, ID_COL] if ID_COL in gold_df.columns else ""
        if pd.isna(gold_title):
            gold_title = ""
        else:
            gold_title = str(gold_title)
        
        best_pi, best_score = None, -1.0
        for pi, pred_title in enumerate(pred_titles):
            if pi in used_pred:
                continue
            sc = ratio(gold_title, pred_title)
            if sc > best_score:
                best_score = sc
                best_pi = pi
        if best_score >= title_threshold:
            mapping[gi] = best_pi  # type: ignore
            used_pred.add(best_pi)  # type: ignore
        else:
            mapping[gi] = None  # type: ignore
    
    return mapping



# -----------------------------
# 3) EVALUATION (TP/FP/FN + METRICS)
# -----------------------------

def evaluate_pipeline(gold_df: pd.DataFrame, pred_df: pd.DataFrame, fields: List[str], title_match_threshold: float):
    mapping = link_records(gold_df, pred_df, title_match_threshold)
    total_gold = len(gold_df)
    found = sum(1 for v in mapping.values() if v is not None)
    coverage = found / total_gold if total_gold else 0.0

    counts = {f: {"TP": 0, "FP": 0, "FN": 0} for f in fields}

    for gi, pred_idx in mapping.items():
        gold_row = gold_df.loc[gi]
        if pred_idx is None:
            for f in fields:
                counts[f]["FN"] += 1
            continue
        pred_row = pred_df.iloc[pred_idx]
        for f in fields:
            gval = gold_row.get(f, np.nan)
            pval = pred_row.get(f, np.nan)
            if (isinstance(pval, float) and np.isnan(pval)) or (isinstance(pval, str) and pval.strip() == ""):
                counts[f]["FN"] += 1
                continue
            if is_correct(f, gval, pval, THRESHOLDS):
                counts[f]["TP"] += 1
            else:
                counts[f]["FP"] += 1

    rows = []
    for f, c in counts.items():
        TP, FP, FN = c["TP"], c["FP"], c["FN"]
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "Field": f, "TP": TP, "FP": FP, "FN": FN,
            "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4)
        })
    metrics_df = pd.DataFrame(rows).sort_values("Field")

    macro_precision = metrics_df["Precision"].mean()
    macro_recall    = metrics_df["Recall"].mean()
    macro_f1        = metrics_df["F1"].mean()

    TP_sum = metrics_df["TP"].sum()
    FP_sum = metrics_df["FP"].sum()
    FN_sum = metrics_df["FN"].sum()
    micro_precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) else 0.0
    micro_recall    = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) else 0.0
    micro_f1        = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    summary = {
      "coverage": round(coverage, 4),
      "macro_precision": round(macro_precision, 4),
      "macro_recall": round(macro_recall, 4),
      "macro_f1": round(macro_f1, 4),
      "micro_precision": round(micro_precision, 4),
      "micro_recall": round(micro_recall, 4),
      "micro_f1": round(micro_f1, 4),
    }
    return metrics_df, summary


# -----------------------------
# 4) REPORTING TABLES
# -----------------------------

def add_micro_macro_rows(metrics_df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = metrics_df.copy()
    macro = {
        "Field": "Average (Macro)",
        "TP": df["TP"].sum(), "FP": df["FP"].sum(), "FN": df["FN"].sum(),
        "Precision": df["Precision"].mean(),
        "Recall": df["Recall"].mean(),
        "F1": df["F1"].mean()
    }
    TP, FP, FN = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    micro_prec = TP / (TP + FP) if (TP + FP) else 0.0
    micro_rec  = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f1   = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0
    micro = {
        "Field": "Average (Micro)",
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": micro_prec, "Recall": micro_rec, "F1": micro_f1
    }
    out = pd.concat([df, pd.DataFrame([macro, micro])], ignore_index=True)
    out["Pipeline"] = name
    for c in ["Precision", "Recall", "F1"]:
        out[c] = out[c].round(4)
    return out

def side_by_side_table(grobid_metrics: pd.DataFrame, nlp_metrics: pd.DataFrame, llm_metrics: pd.DataFrame, joshcoley_metrics: pd.DataFrame, fields_order: List[str]) -> pd.DataFrame:
    def prep(df, name):
        keep = df[["Field", "Precision", "Recall", "F1"]].copy()
        keep["Pipeline"] = name
        return keep
    chunks = [
        prep(df, name)
        for df, name in (
            (grobid_metrics, "GROBID"),
            (nlp_metrics, "NLP/ML"),
            (llm_metrics, "LLM"),
            (joshcoley_metrics, HYBRID_PIPELINE_LABEL),
        )
        if df is not None
    ]
    if not chunks:
        return pd.DataFrame()
    long = pd.concat(chunks, ignore_index=True)
    wide = (long
            .pivot(index="Field", columns="Pipeline", values=["Precision", "Recall", "F1"]))
    if fields_order:
        wide = wide.reindex(fields_order)
    return wide.round(3)

def overall_table(grobid_metrics: pd.DataFrame, nlp_metrics: pd.DataFrame, llm_metrics: pd.DataFrame, joshcoley_metrics: pd.DataFrame,
                  grobid_summary: dict, nlp_summary: dict, llm_summary: dict, joshcoley_summary: dict) -> pd.DataFrame:
    def overall_from(df, summary, name):
        TP, FP, FN = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
        micro_prec = TP / (TP + FP) if (TP + FP) else 0.0
        micro_rec  = TP / (TP + FN) if (TP + FN) else 0.0
        micro_f1   = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0
        return pd.Series({
            ("Coverage", name):          summary["coverage"],
            ("Macro-Precision", name):   df["Precision"].mean(),
            ("Macro-Recall", name):      df["Recall"].mean(),
            ("Macro-F1", name):          df["F1"].mean(),
            ("Micro-Precision", name):   micro_prec,
            ("Micro-Recall", name):      micro_rec,
            ("Micro-F1", name):          micro_f1
        })
    series = []
    for df, summary, name in (
        (grobid_metrics, grobid_summary, "GROBID"),
        (nlp_metrics, nlp_summary, "NLP/ML"),
        (llm_metrics, llm_summary, "LLM"),
        (joshcoley_metrics, joshcoley_summary, HYBRID_PIPELINE_LABEL),
    ):
        if df is not None and summary is not None:
            series.append(overall_from(df, summary, name))
    if not series:
        return pd.DataFrame()
    row = pd.concat(series)
    out = (row.to_frame(name="value")
              .reset_index()
              .rename(columns={"level_0": "Metric", "level_1": "Pipeline"})
              .pivot(index="Metric", columns="Pipeline", values="value")
              .round(3))
    return out


# -----------------------------
# 5) NARRATIVE GENERATORS
# -----------------------------

def overall_sentence(overall_tbl: pd.DataFrame) -> str:
    def g(metric, pipe):
        v = overall_tbl.loc[metric, pipe]
        return f"{v:.3f}"
    
    # Find the pipeline with the best Macro-F1 score
    pipelines = ["GROBID", "LLM", "NLP/ML", HYBRID_PIPELINE_LABEL]
    best_pipeline = None
    best_f1 = -1.0
    for pipe in pipelines:
        if pipe in overall_tbl.columns:
            f1 = overall_tbl.loc["Macro-F1", pipe]
            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipe
    
    if best_pipeline is None:
        cols = [p for p in pipelines if p in overall_tbl.columns]
        if not cols:
            return "No pipeline metrics to summarize."
        best_pipeline = cols[0]
    
    # Build the sentence with best pipeline first
    parts = []
    for pipe in pipelines:
        if pipe in overall_tbl.columns:
            parts.append(f"{pipe} at {g('Macro-Precision',pipe)}/{g('Macro-Recall',pipe)}/{g('Macro-F1',pipe)}")
    
    score = (
        f"{g('Macro-Precision',best_pipeline)}/{g('Macro-Recall',best_pipeline)}/{g('Macro-F1',best_pipeline)}"
    )
    others = [p for p in parts if not p.startswith(best_pipeline)]
    if not others:
        return f"Overall, {best_pipeline} was best with Macro Precision/Recall/F1 of {score}."
    return (
        f"Overall, {best_pipeline} was best with Macro Precision/Recall/F1 of {score}, "
        f"compared to " + ", ".join(others) + "."
    )

def per_field_sentences(sxs: pd.DataFrame) -> List[str]:
    lines = []
    for field in sxs.index:
        row = sxs.loc[field]
        pr = row.loc["Precision"].astype(float)
        rc = row.loc["Recall"].astype(float)
        f1 = row.loc["F1"].astype(float)
        
        # Find all pipelines with the maximum F1 score (handle ties)
        max_f1 = f1.max()
        leaders = f1[f1 == max_f1].index.tolist()
        
        if len(leaders) == 1:
            # Single leader
            leader = leaders[0]
            lines.append(
                f"For {field}, {leader} led with F1={f1[leader]:.3f} (Precision={pr[leader]:.3f}, Recall={rc[leader]:.3f})."
            )
        else:
            # Multiple leaders (tie) - show all tied pipelines
            leader_names = " and ".join(leaders) if len(leaders) == 2 else ", ".join(leaders[:-1]) + f", and {leaders[-1]}"
            # Use the first leader's precision/recall for display (they're likely very similar)
            lines.append(
                f"For {field}, {leader_names} tied for best with F1={max_f1:.3f} (Precision={pr[leaders[0]]:.3f}, Recall={rc[leaders[0]]:.3f})."
            )
    return lines


# -----------------------------
# 6) TIME & COST (OPTIONAL)
# -----------------------------

def safe_read_json(path: str) -> dict:
    try:
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def estimate_llm_cost_from_outputs(df: pd.DataFrame, rate_per_1k_tokens: float) -> float:
    if df is None or df.empty:
        return 0.0
    # Rough chars count using Title + Abstract + Affiliation
    char_cols = [c for c in ["Title", "Abstract", "Affiliation"] if c in df.columns]
    if not char_cols:
        return 0.0
    chars = 0
    for c in char_cols:
        chars += df[c].astype(str).map(len).sum()
    tokens = chars / 4.0
    return round((tokens / 1000.0) * rate_per_1k_tokens, 4)

def time_cost_table(grobid_df: pd.DataFrame, nlp_df: pd.DataFrame, llm_df: pd.DataFrame, joshcoley_df: pd.DataFrame, measured_times: dict = None) -> pd.DataFrame:
    stats = safe_read_json(RUN_STATS_JSON)
    grobid = stats.get("grobid", {})
    nlp    = stats.get("nlp", {})
    llm    = stats.get("llm", {})
    hybrid = stats.get("alphamind_extractor") or stats.get("joshcoley", {})
    # Defaults
    grobid_time = grobid.get("runtime_seconds", np.nan)
    nlp_time    = nlp.get("runtime_seconds", np.nan)
    llm_time    = llm.get("runtime_seconds", np.nan)
    joshcoley_time = hybrid.get("runtime_seconds", np.nan)

    # Override with measured times if provided
    if measured_times:
        if "grobid" in measured_times:
            grobid_time = measured_times.get("grobid", grobid_time)
        if "nlp" in measured_times:
            nlp_time = measured_times.get("nlp", nlp_time)
        if "llm" in measured_times:
            llm_time = measured_times.get("llm", llm_time)
        if "alphamind_extractor" in measured_times or "joshcoley" in measured_times:
            joshcoley_time = measured_times.get(
                "alphamind_extractor", measured_times.get("joshcoley", joshcoley_time)
            )

    grobid_cost = grobid.get("cost_usd", 0.0)
    nlp_cost    = nlp.get("cost_usd", 0.0)
    if "cost_usd" in llm:
        llm_cost = llm.get("cost_usd", 0.0)
    else:
        # Estimate if not present
        llm_cost = estimate_llm_cost_from_outputs(llm_df, LLM_COST_PER_1K_TOKENS)
    
    # Estimate AlphaMind Extractor cost: GROBID (free) + LLM cost for Affiliation only
    # Since the hybrid uses LLM only for Affiliation, estimate cost as a fraction of full LLM cost
    # Rough estimate: if LLM extracts all fields, hybrid uses LLM for 1/7 fields (Affiliation only)
    # But LLM prompt includes all fields, so estimate based on proportional usage
    # Conservative estimate: ~30% of LLM cost (since we only need Affiliation but still run full LLM extraction)
    if "cost_usd" in hybrid:
        joshcoley_cost = hybrid.get("cost_usd", 0.0)
    else:
        # Estimate: GROBID is free, LLM cost for Affiliation only
        # Since we run full LLM extraction but only use Affiliation, estimate as ~30% of LLM cost
        joshcoley_cost = llm_cost * 0.3
    
    # Estimate hybrid time: GROBID time + LLM time (since we run both)
    if pd.isna(joshcoley_time):
        if not pd.isna(grobid_time) and not pd.isna(llm_time):
            # Conservative: sum of both (sequential processing)
            joshcoley_time = grobid_time + llm_time
        elif not pd.isna(grobid_time):
            joshcoley_time = grobid_time
        elif not pd.isna(llm_time):
            joshcoley_time = llm_time

    tbl = pd.DataFrame({
        "Pipeline": ["GROBID", "NLP/ML", "LLM", HYBRID_PIPELINE_LABEL],
        "Time (s)": [grobid_time, nlp_time, llm_time, joshcoley_time],
        "Cost (USD)": [grobid_cost, nlp_cost, llm_cost, joshcoley_cost],
    })
    return tbl


# -----------------------------
# 7) MAIN
# -----------------------------

def main():
    # CLI flags
    parser = argparse.ArgumentParser(description="Compare gold vs Grobid/NLP/LLM outputs with optional timed runs.")
    parser.add_argument("gold", nargs="?", help="Path to gold Excel", default=None)
    parser.add_argument("grobid", nargs="?", help="Path to grobid Excel", default=None)
    parser.add_argument("nlp", nargs="?", help="Path to nlp Excel", default=None)
    parser.add_argument("llm", nargs="?", help="Path to llm Excel", default=None)
    parser.add_argument("joshcoley", nargs="?", help="Path to AlphaMind Extractor Excel (legacy arg name)", default=None)
    parser.add_argument("--run", action="store_true", help="Run pipelines before comparing and measure time")
    parser.add_argument("--run-which", default="all", help="Which pipelines to run: all or comma list (grobid,nlp,llm,alphamind_extractor)")
    parser.add_argument("--root", default=None, help="Root folder to pass to pipelines (optional)")
    parser.add_argument("--authors", default=None, help="Comma-separated authors to pass to pipelines (optional)")
    parser.add_argument("--llm-ok", action="store_true", help="Confirm permission to run the LLM pipeline (costs may apply)")
    cons = parser.add_mutually_exclusive_group()
    cons.add_argument(
        "--consolidate-header",
        action="store_true",
        help="Kept for compatibility; CrossRef/biblio-glutton header consolidation is already the default",
    )
    cons.add_argument(
        "--no-consolidate-header",
        action="store_true",
        help="Disable GROBID header consolidation (PDF-only metadata)",
    )
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("PIPELINE_RUN_TIMEOUT", "3600")), help="Per-pipeline timeout in seconds")
    args_ns = parser.parse_args()

    gold_path   = Path(args_ns.gold).expanduser().resolve() if args_ns.gold else Path(GOLD_XLSX).resolve()
    grobid_path = Path(args_ns.grobid).expanduser().resolve() if args_ns.grobid else Path(GROBID_XLSX).resolve()
    nlp_path    = Path(args_ns.nlp).expanduser().resolve()    if args_ns.nlp else Path(NLP_XLSX).resolve()
    llm_path    = Path(args_ns.llm).expanduser().resolve()    if args_ns.llm else Path(LLM_XLSX).resolve()
    joshcoley_path = Path(args_ns.joshcoley).expanduser().resolve() if args_ns.joshcoley else Path(JOSHCOLEY_XLSX).resolve()

    # Optional: run pipelines and measure time
    measured_times: Dict[str, float] = {}
    if args_ns.run:
        which = args_ns.run_which.strip().lower()
        if which == "all":
            to_run = {"grobid", "nlp", "llm", "alphamind_extractor"}
        else:
            to_run = set()
            for w in which.split(","):
                tag = w.strip().lower()
                if not tag:
                    continue
                to_run.add("alphamind_extractor" if tag in _HYBRID_RUN_ALIASES else tag)
        # helper to run and time
        def run_and_time(tag: str, script: Path, env: Optional[dict] = None) -> None:
            cmd: List[str] = [sys.executable, str(script)]
            if args_ns.root:
                cmd.append(args_ns.root)
            if args_ns.authors:
                cmd.append(args_ns.authors)
            t0 = time.perf_counter()
            run_env = os.environ.copy()
            if env:
                run_env.update(env)
            try:
                print(f"▶ Running {tag}…")
                proc = subprocess.run(cmd, check=False, timeout=args_ns.timeout, cwd=str(_ROOT), env=run_env)
            except subprocess.TimeoutExpired:
                print(f"⏱️ {tag} timed out after {args_ns.timeout}s")
                measured_times[tag] = np.nan
                return
            except Exception as e:
                print(f"❌ {tag} run failed: {e}")
                measured_times[tag] = np.nan
                return
            if getattr(proc, "returncode", 0) not in (0, None):
                print(f"❌ {tag} exited with code {proc.returncode}")
                measured_times[tag] = np.nan
                return
            dt = time.perf_counter() - t0
            measured_times[tag] = round(dt, 3)
            print(f"✅ {tag} finished in {dt:.1f}s")

        project_root = _ROOT
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("ROOT_PAPERS_DIR", str(MY_PAPERS))
        env.setdefault("OUTPUT_XLSX", str(_GROBID_RESULTS_XLSX))
        if args_ns.no_consolidate_header:
            env["CONSOLIDATE_HEADER"] = "0"
            print("ℹ️ GROBID header consolidation OFF (PDF only)")
        else:
            env["CONSOLIDATE_HEADER"] = "1"
            print("ℹ️ GROBID header consolidation ON (CrossRef / biblio-glutton)")
        if "grobid" in to_run:
            run_and_time("grobid", PIPELINES_DIR / "grobid_pipeline.py", env=env)
        env["OUTPUT_XLSX"] = str(_NLP_RESULTS_XLSX)
        if "nlp" in to_run:
            run_and_time("nlp", PIPELINES_DIR / "NPL_extraction_pipeline.py", env=env)
        env["OUTPUT_XLSX"] = str(_LLM_RESULTS_XLSX)
        if "llm" in to_run:
            if args_ns.llm_ok:
                run_and_time("llm", PIPELINES_DIR / "LLM_extraction_pipeline.py", env=env)
            else:
                print("ℹ️ Skipping LLM run (missing --llm-ok)")
        env["OUTPUT_XLSX"] = str(_ALPHAMIND_EXTRACTOR_RESULTS_XLSX)
        if "alphamind_extractor" in to_run:
            if args_ns.llm_ok:
                run_and_time("alphamind_extractor", PIPELINES_DIR / "alphamind_extractor.py", env=env)
            else:
                print("ℹ️ Skipping AlphaMind Extractor run (missing --llm-ok, requires LLM)")

    if not gold_path.exists():
        print(f"❌ Missing gold standard: {gold_path}")
        sys.exit(1)

    gold = load_df(str(gold_path))
    empty = pd.DataFrame()

    def _load_optional(label: str, path: Path):
        if path.exists():
            df = load_df(str(path))
            metrics, summary = evaluate_pipeline(gold, df, FIELDS, TITLE_MATCH_THRESHOLD)
            return df, metrics, summary
        print(f"ℹ️ Skipping {label}: missing {path}")
        return None, None, None

    grobid, grobid_metrics, grobid_summary = _load_optional("GROBID", grobid_path)
    nlp, nlp_metrics, nlp_summary = _load_optional("NLP/ML", nlp_path)
    llm, llm_metrics, llm_summary = _load_optional("LLM", llm_path)
    joshcoley, joshcoley_metrics, joshcoley_summary = _load_optional(
        HYBRID_PIPELINE_LABEL, joshcoley_path
    )

    present = [
        name for name, df in (
            ("GROBID", grobid),
            ("NLP/ML", nlp),
            ("LLM", llm),
            (HYBRID_PIPELINE_LABEL, joshcoley),
        )
        if df is not None
    ]
    if not present:
        print("❌ No pipeline result files found under data/evaluation/. Run a pipeline first.")
        sys.exit(1)

    grobid_full = add_micro_macro_rows(grobid_metrics, "GROBID") if grobid_metrics is not None else None
    nlp_full = add_micro_macro_rows(nlp_metrics, "NLP/ML") if nlp_metrics is not None else None
    llm_full = add_micro_macro_rows(llm_metrics, "LLM") if llm_metrics is not None else None
    joshcoley_full = (
        add_micro_macro_rows(joshcoley_metrics, HYBRID_PIPELINE_LABEL)
        if joshcoley_metrics is not None else None
    )

    sxs = side_by_side_table(grobid_metrics, nlp_metrics, llm_metrics, joshcoley_metrics, fields_order=FIELDS)
    overall_tbl = overall_table(grobid_metrics, nlp_metrics, llm_metrics, joshcoley_metrics,
                                grobid_summary, nlp_summary, llm_summary, joshcoley_summary)
    tc_tbl = time_cost_table(
        grobid if grobid is not None else empty,
        nlp if nlp is not None else empty,
        llm if llm is not None else empty,
        joshcoley if joshcoley is not None else empty,
        measured_times=measured_times if measured_times else None,
    )

    coverage_row = {}
    if grobid_summary is not None:
        coverage_row["GROBID"] = [grobid_summary["coverage"]]
    if nlp_summary is not None:
        coverage_row["NLP/ML"] = [nlp_summary["coverage"]]
    if llm_summary is not None:
        coverage_row["LLM"] = [llm_summary["coverage"]]
    if joshcoley_summary is not None:
        coverage_row[HYBRID_PIPELINE_LABEL] = [joshcoley_summary["coverage"]]

    print("=== COVERAGE (papers found / total gold) ===")
    print(pd.DataFrame(coverage_row, index=["Coverage"]).round(3))
    print()

    print("=== OVERALL (Macro & Micro) ===")
    print(overall_tbl)
    print()

    print("=== PER-FIELD (Precision/Recall/F1) ===")
    print(sxs)
    print()

    print("=== TIME & COST (Optional/Estimated) ===")
    print(tc_tbl)
    print()

    print("=== SUMMARY SENTENCE ===")
    print(overall_sentence(overall_tbl))
    print()

    print("=== PER-FIELD SENTENCES ===")
    for line in per_field_sentences(sxs):
        print("-", line)

    if EXPORT_SUMMARY:
        try:
            with pd.ExcelWriter(SUMMARY_XLSX) as writer:
                pd.DataFrame({
                    "Pipeline": list(coverage_row.keys()),
                    "Coverage": [v[0] for v in coverage_row.values()],
                }).to_excel(writer, sheet_name="Coverage", index=False)
                if grobid_full is not None:
                    grobid_full.to_excel(writer, sheet_name="GROBID_per_field", index=False)
                if nlp_full is not None:
                    nlp_full.to_excel(writer, sheet_name="NLP_per_field", index=False)
                if llm_full is not None:
                    llm_full.to_excel(writer, sheet_name="LLM_per_field", index=False)
                if joshcoley_full is not None:
                    joshcoley_full.to_excel(writer, sheet_name="AlphaMind_Extractor_per_field", index=False)
                sxs.to_excel(writer, sheet_name="SideBySide_PerField")
                overall_tbl.to_excel(writer, sheet_name="Overall_MacroMicro")
                tc_tbl.to_excel(writer, sheet_name="Time_Cost", index=False)
            print(f"✅ Wrote summary to {SUMMARY_XLSX}")
        except Exception as e:
            print(f"❌ Failed to write summary Excel: {e}")


if __name__ == "__main__":
    main()


