#!/usr/bin/env python3
"""
Compare Pipelines (Gold vs GROBID / NLP / LLM / JoshColey)

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
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# -----------------------------
# 0) CONFIG
# -----------------------------

# Defaults (can override via env or CLI args)
GOLD_XLSX   = os.environ.get("GOLD_XLSX",   "Golden Standard.xlsx")
GROBID_XLSX = os.environ.get("GROBID_XLSX", "grobid_results_v4.xlsx")
NLP_XLSX    = os.environ.get("NLP_XLSX",    "nlp_results.xlsx")
LLM_XLSX    = os.environ.get("LLM_XLSX",    "llm_results.xlsx")
JOSHCOLEY_XLSX = os.environ.get("JOSHCOLEY_XLSX", "joshcoley_results.xlsx")

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
SUMMARY_XLSX = os.environ.get("SUMMARY_XLSX", "pipeline_evaluation_summary.xlsx")

# Optional time/cost stats file (JSON). If absent, scripts estimate/minimize.
RUN_STATS_JSON = os.environ.get("RUN_STATS_JSON", "pipeline_run_stats.json")

# Cost estimation fallback (only used if stats missing for LLM)
# Very rough: tokens ~= chars/4. Price per 1K tokens default.
LLM_COST_PER_1K_TOKENS = float(os.environ.get("LLM_COST_PER_1K_TOKENS", "0.002"))


# -----------------------------
# 1) HELPERS
# -----------------------------

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_int_or_nan(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def ratio(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    return SequenceMatcher(None, a, b).ratio()


def field_similarity(field: str, gold_val, pred_val) -> float:
    if field in ["Author's Position", "Total Authors"]:
        g = to_int_or_nan(gold_val)
        p = to_int_or_nan(pred_val)
        return 1.0 if (not np.isnan(g) and not np.isnan(p) and g == p) else 0.0
    if field == "Abstract":
        return ratio(gold_val, pred_val)
    # Default for text fields
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
    long = pd.concat([
        prep(grobid_metrics, "GROBID"),
        prep(nlp_metrics,    "NLP/ML"),
        prep(llm_metrics,    "LLM"),
        prep(joshcoley_metrics, "JoshColey")
    ], ignore_index=True)
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
    row = pd.concat([
        overall_from(grobid_metrics, grobid_summary, "GROBID"),
        overall_from(nlp_metrics,    nlp_summary,    "NLP/ML"),
        overall_from(llm_metrics,    llm_summary,    "LLM"),
        overall_from(joshcoley_metrics, joshcoley_summary, "JoshColey"),
    ])
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
    pipelines = ["GROBID", "LLM", "NLP/ML", "JoshColey"]
    best_pipeline = None
    best_f1 = -1.0
    for pipe in pipelines:
        if pipe in overall_tbl.columns:
            f1 = overall_tbl.loc["Macro-F1", pipe]
            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipe
    
    if best_pipeline is None:
        best_pipeline = "GROBID"  # Fallback
    
    # Build the sentence with best pipeline first
    parts = []
    for pipe in pipelines:
        if pipe in overall_tbl.columns:
            parts.append(f"{pipe} at {g('Macro-Precision',pipe)}/{g('Macro-Recall',pipe)}/{g('Macro-F1',pipe)}")
    
    return (
        f"Overall, {best_pipeline} was best with Macro Precision/Recall/F1 of "
        f"{g('Macro-Precision',best_pipeline)}/{g('Macro-Recall',best_pipeline)}/{g('Macro-F1',best_pipeline)}, "
        f"compared to " + ", ".join([p for p in parts if not p.startswith(best_pipeline)]) + ".")

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
    joshcoley = stats.get("joshcoley", {})
    # Defaults
    grobid_time = grobid.get("runtime_seconds", np.nan)
    nlp_time    = nlp.get("runtime_seconds", np.nan)
    llm_time    = llm.get("runtime_seconds", np.nan)
    joshcoley_time = joshcoley.get("runtime_seconds", np.nan)

    # Override with measured times if provided
    if measured_times:
        if "grobid" in measured_times:
            grobid_time = measured_times.get("grobid", grobid_time)
        if "nlp" in measured_times:
            nlp_time = measured_times.get("nlp", nlp_time)
        if "llm" in measured_times:
            llm_time = measured_times.get("llm", llm_time)
        if "joshcoley" in measured_times:
            joshcoley_time = measured_times.get("joshcoley", joshcoley_time)

    grobid_cost = grobid.get("cost_usd", 0.0)
    nlp_cost    = nlp.get("cost_usd", 0.0)
    if "cost_usd" in llm:
        llm_cost = llm.get("cost_usd", 0.0)
    else:
        # Estimate if not present
        llm_cost = estimate_llm_cost_from_outputs(llm_df, LLM_COST_PER_1K_TOKENS)
    
    # Estimate JoshColey cost: GROBID (free) + LLM cost for Affiliation only
    # Since JoshColey uses LLM only for Affiliation, estimate cost as a fraction of full LLM cost
    # Rough estimate: if LLM extracts all fields, JoshColey uses LLM for 1/7 fields (Affiliation only)
    # But LLM prompt includes all fields, so estimate based on proportional usage
    # Conservative estimate: ~30% of LLM cost (since we only need Affiliation but still run full LLM extraction)
    if "cost_usd" in joshcoley:
        joshcoley_cost = joshcoley.get("cost_usd", 0.0)
    else:
        # Estimate: GROBID is free, LLM cost for Affiliation only
        # Since we run full LLM extraction but only use Affiliation, estimate as ~30% of LLM cost
        joshcoley_cost = llm_cost * 0.3
    
    # Estimate JoshColey time: GROBID time + LLM time (since we run both)
    if pd.isna(joshcoley_time):
        if not pd.isna(grobid_time) and not pd.isna(llm_time):
            # Conservative: sum of both (sequential processing)
            joshcoley_time = grobid_time + llm_time
        elif not pd.isna(grobid_time):
            joshcoley_time = grobid_time
        elif not pd.isna(llm_time):
            joshcoley_time = llm_time

    tbl = pd.DataFrame({
        "Pipeline": ["GROBID", "NLP/ML", "LLM", "JoshColey"],
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
    parser.add_argument("joshcoley", nargs="?", help="Path to joshcoley Excel", default=None)
    parser.add_argument("--run", action="store_true", help="Run pipelines before comparing and measure time")
    parser.add_argument("--run-which", default="all", help="Which pipelines to run: all or comma list (grobid,nlp,llm,joshcoley)")
    parser.add_argument("--root", default=None, help="Root folder to pass to pipelines (optional)")
    parser.add_argument("--authors", default=None, help="Comma-separated authors to pass to pipelines (optional)")
    parser.add_argument("--llm-ok", action="store_true", help="Confirm permission to run the LLM pipeline (costs may apply)")
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
        to_run = {"grobid", "nlp", "llm", "joshcoley"} if which == "all" else {w.strip() for w in which.split(",") if w.strip()}
        # helper to run and time
        def run_and_time(tag: str, script: Path) -> None:
            cmd: List[str] = [sys.executable, str(script)]
            if args_ns.root:
                cmd.append(args_ns.root)
            if args_ns.authors:
                cmd.append(args_ns.authors)
            t0 = time.perf_counter()
            try:
                print(f"▶ Running {tag}…")
                subprocess.run(cmd, check=False, timeout=args_ns.timeout)
            except subprocess.TimeoutExpired:
                print(f"⏱️ {tag} timed out after {args_ns.timeout}s")
                measured_times[tag] = np.nan
                return
            except Exception as e:
                print(f"❌ {tag} run failed: {e}")
                measured_times[tag] = np.nan
                return
            dt = time.perf_counter() - t0
            measured_times[tag] = round(dt, 3)
            print(f"✅ {tag} finished in {dt:.1f}s")

        project_root = Path(__file__).parent
        if "grobid" in to_run:
            run_and_time("grobid", project_root / "grobid_extraction_v4.py")
        if "nlp" in to_run:
            run_and_time("nlp", project_root / "NPL_extraction_pipeline.py")
        if "llm" in to_run:
            if args_ns.llm_ok:
                run_and_time("llm", project_root / "LLM_extraction_pipeline.py")
            else:
                print("ℹ️ Skipping LLM run (missing --llm-ok)")
        if "joshcoley" in to_run:
            if args_ns.llm_ok:
                run_and_time("joshcoley", project_root / "joshcoley_extraction_pipeline.py")
            else:
                print("ℹ️ Skipping JoshColey run (missing --llm-ok, requires LLM)")

    for p in [gold_path, grobid_path, nlp_path, llm_path, joshcoley_path]:
        if not p.exists():
            print(f"❌ Missing input: {p}")
            sys.exit(1)

    gold   = load_df(str(gold_path))
    grobid = load_df(str(grobid_path))
    nlp    = load_df(str(nlp_path))
    llm    = load_df(str(llm_path))
    joshcoley = load_df(str(joshcoley_path))

    grobid_metrics, grobid_summary = evaluate_pipeline(gold, grobid, FIELDS, TITLE_MATCH_THRESHOLD)
    nlp_metrics,    nlp_summary    = evaluate_pipeline(gold, nlp,    FIELDS, TITLE_MATCH_THRESHOLD)
    llm_metrics,    llm_summary    = evaluate_pipeline(gold, llm,    FIELDS, TITLE_MATCH_THRESHOLD)
    joshcoley_metrics, joshcoley_summary = evaluate_pipeline(gold, joshcoley, FIELDS, TITLE_MATCH_THRESHOLD)

    # Add macro/micro rows (per-pipeline sheets)
    grobid_full = add_micro_macro_rows(grobid_metrics, "GROBID")
    nlp_full    = add_micro_macro_rows(nlp_metrics,    "NLP/ML")
    llm_full    = add_micro_macro_rows(llm_metrics,    "LLM")
    joshcoley_full = add_micro_macro_rows(joshcoley_metrics, "JoshColey")

    # Side-by-side per-field
    sxs = side_by_side_table(grobid_metrics, nlp_metrics, llm_metrics, joshcoley_metrics, fields_order=FIELDS)

    # Overall compact
    overall_tbl = overall_table(grobid_metrics, nlp_metrics, llm_metrics, joshcoley_metrics,
                                grobid_summary, nlp_summary, llm_summary, joshcoley_summary)

    # Time & Cost
    tc_tbl = time_cost_table(grobid, nlp, llm, joshcoley, measured_times=measured_times if measured_times else None)

    # PRINT REPORT
    print("=== COVERAGE (papers found / total gold) ===")
    print(pd.DataFrame({
        "GROBID": [grobid_summary["coverage"]],
        "NLP/ML": [nlp_summary["coverage"]],
        "LLM":    [llm_summary["coverage"]],
        "JoshColey": [joshcoley_summary["coverage"]],
    }, index=["Coverage"]).round(3))
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

    # EXPORT
    if EXPORT_SUMMARY:
        try:
            with pd.ExcelWriter(SUMMARY_XLSX) as writer:
                # Coverage
                pd.DataFrame({
                    "Pipeline": ["GROBID", "NLP/ML", "LLM", "JoshColey"],
                    "Coverage": [grobid_summary["coverage"], nlp_summary["coverage"], llm_summary["coverage"], joshcoley_summary["coverage"]],
                }).to_excel(writer, sheet_name="Coverage", index=False)

                # Per-pipeline per-field (with averages)
                grobid_full.to_excel(writer, sheet_name="GROBID_per_field", index=False)
                nlp_full.to_excel(writer,    sheet_name="NLP_per_field",    index=False)
                llm_full.to_excel(writer,    sheet_name="LLM_per_field",    index=False)
                joshcoley_full.to_excel(writer, sheet_name="JoshColey_per_field", index=False)

                # Side-by-side per-field
                sxs.to_excel(writer, sheet_name="SideBySide_PerField")

                # Overall macro/micro compact table
                overall_tbl.to_excel(writer, sheet_name="Overall_MacroMicro")

                # Time & Cost
                tc_tbl.to_excel(writer, sheet_name="Time_Cost", index=False)
            print(f"✅ Wrote summary to {SUMMARY_XLSX}")
        except Exception as e:
            print(f"❌ Failed to write summary Excel: {e}")


if __name__ == "__main__":
    main()


