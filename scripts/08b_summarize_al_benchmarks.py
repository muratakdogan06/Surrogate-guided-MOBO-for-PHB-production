#!/usr/bin/env python3
# ============================================================================
# 08b_summarize_al_benchmarks.py
# Summarize multi-seed AL benchmark results with stronger metrics
# ============================================================================
"""
Computes benchmark summary metrics for multi-seed AL runs:
- Hypervolume AUC
- Best viable PHA AUC
- Time-to-threshold
- Low-budget snapshots

Outputs
-------
results/tables/benchmark_auc_summary.csv
results/tables/benchmark_time_to_threshold.csv
results/tables/benchmark_low_budget_snapshots.csv
"""

from __future__ import annotations

from pathlib import Path
import glob
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

METHOD_PATTERNS = {
    "al": "al_history_seed*.csv",
    "random": "random_baseline_seed*.csv",
    "greedy": "greedy_baseline_seed*.csv",
    "exploitation": "exploitation_baseline_seed*.csv",
}

# Thresholds: adjust if needed
THRESHOLDS = {
    "hypervolume": [0.60, 0.64],
    "best_viable_pha": [2.1620, 2.1630, 2.1634],
    "pareto_discovered": [5, 6],
}

LOW_BUDGET_EVALS = [20, 30, 50]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_seed(path: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def load_runs(method_name: str, pattern: str) -> List[pd.DataFrame]:
    files = sorted(glob.glob(str(TABLE_DIR / pattern)))
    if not files:
        print(f"[WARN] No files found for {method_name}: {pattern}")
        return []

    runs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.loc[:, ~df.columns.str.lower().str.startswith("index")].copy()
        df["seed"] = extract_seed(f)
        df["method"] = method_name
        df["source_file"] = Path(f).name
        runs.append(df)

    print(f"[INFO] Loaded {len(runs)} runs for method '{method_name}'")
    return runs


def pick_x_col(df: pd.DataFrame) -> str:
    for c in ["n_evaluated", "evaluated", "iteration"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized x-axis column found in columns: {list(df.columns)}")


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")

    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if _trap is None:
        from scipy.integrate import trapezoid as _trap
    return float(_trap(y=y, x=x))


def first_eval_reaching_threshold(
    df: pd.DataFrame,
    metric: str,
    threshold: float,
    x_col: str,
) -> Optional[float]:
    if metric not in df.columns:
        return None
    hit = df[df[metric] >= threshold]
    if hit.empty:
        return None
    return float(hit.iloc[0][x_col])


def value_at_or_before_eval(
    df: pd.DataFrame,
    metric: str,
    eval_point: float,
    x_col: str,
) -> Optional[float]:
    if metric not in df.columns:
        return None
    sub = df[df[x_col] <= eval_point]
    if sub.empty:
        return None
    return float(sub.iloc[-1][metric])


# ---------------------------------------------------------------------------
# AUC summary
# ---------------------------------------------------------------------------

def summarize_auc(all_runs: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for method_name, runs in all_runs.items():
        for df in runs:
            x_col = pick_x_col(df)
            seed = df["seed"].iloc[0]

            row = {
                "method": method_name,
                "seed": seed,
            }

            if "hypervolume" in df.columns:
                row["hypervolume_auc"] = trapezoid_auc(df[x_col].values, df["hypervolume"].values)

            if "best_viable_pha" in df.columns:
                row["best_viable_pha_auc"] = trapezoid_auc(df[x_col].values, df["best_viable_pha"].values)

            if "pareto_discovered" in df.columns:
                row["pareto_discovered_auc"] = trapezoid_auc(df[x_col].values, df["pareto_discovered"].values)

            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = TABLE_DIR / "benchmark_auc_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")

    if not out.empty:
        agg = out.groupby("method").agg(["mean", "std"])
        print("\n=== AUC SUMMARY (mean ± std) ===")
        print(agg)

    return out


# ---------------------------------------------------------------------------
# Time-to-threshold summary
# ---------------------------------------------------------------------------

def summarize_time_to_threshold(all_runs: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for method_name, runs in all_runs.items():
        for df in runs:
            x_col = pick_x_col(df)
            seed = df["seed"].iloc[0]

            for metric, thresholds in THRESHOLDS.items():
                if metric not in df.columns:
                    continue

                for thr in thresholds:
                    hit_eval = first_eval_reaching_threshold(df, metric, thr, x_col)
                    rows.append({
                        "method": method_name,
                        "seed": seed,
                        "metric": metric,
                        "threshold": thr,
                        "hit_evaluation": hit_eval,
                        "reached": int(hit_eval is not None),
                    })

    out = pd.DataFrame(rows)
    out_path = TABLE_DIR / "benchmark_time_to_threshold.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")

    if not out.empty:
        print("\n=== TIME-TO-THRESHOLD SUMMARY ===")
        summary = (
            out.groupby(["method", "metric", "threshold"])
            .agg(
                reach_rate=("reached", "mean"),
                mean_hit_evaluation=("hit_evaluation", "mean"),
                std_hit_evaluation=("hit_evaluation", "std"),
            )
        )
        print(summary)

    return out


# ---------------------------------------------------------------------------
# Low-budget snapshots
# ---------------------------------------------------------------------------

def summarize_low_budget(all_runs: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    metrics_of_interest = ["hypervolume", "best_viable_pha", "pareto_discovered"]
    rows = []

    for method_name, runs in all_runs.items():
        for df in runs:
            x_col = pick_x_col(df)
            seed = df["seed"].iloc[0]

            for eval_point in LOW_BUDGET_EVALS:
                row = {
                    "method": method_name,
                    "seed": seed,
                    "evaluation_budget": eval_point,
                }

                for metric in metrics_of_interest:
                    row[metric] = value_at_or_before_eval(df, metric, eval_point, x_col)

                rows.append(row)

    out = pd.DataFrame(rows)
    out_path = TABLE_DIR / "benchmark_low_budget_snapshots.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")

    if not out.empty:
        print("\n=== LOW-BUDGET SNAPSHOTS (mean ± std) ===")
        summary = (
            out.groupby(["method", "evaluation_budget"])
            .agg(["mean", "std"])
        )
        print(summary)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_runs: Dict[str, List[pd.DataFrame]] = {}

    for method_name, pattern in METHOD_PATTERNS.items():
        all_runs[method_name] = load_runs(method_name, pattern)

    if not any(all_runs.values()):
        print("[ERROR] No benchmark files found.")
        return

    auc_df = summarize_auc(all_runs)
    ttt_df = summarize_time_to_threshold(all_runs)
    low_df = summarize_low_budget(all_runs)

    print("\n============================================================")
    print("Benchmark summarization complete.")
    print("Generated files:")
    print(" - results/tables/benchmark_auc_summary.csv")
    print(" - results/tables/benchmark_time_to_threshold.csv")
    print(" - results/tables/benchmark_low_budget_snapshots.csv")
    print("============================================================")


if __name__ == "__main__":
    main()