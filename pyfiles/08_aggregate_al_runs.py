#!/usr/bin/env python3
# ============================================================================
# 08_aggregate_al_runs.py — Aggregate multi-seed AL benchmark results
# ============================================================================
"""
Reads multiple seed-specific benchmark CSVs, computes mean/std trajectories,
saves aggregated tables, and generates comparison figures.

Expected input file patterns
----------------------------
results/tables/al_history_seed*.csv
results/tables/random_baseline_seed*.csv
results/tables/greedy_baseline_seed*.csv
results/tables/exploitation_baseline_seed*.csv

Expected columns
----------------
Common trajectory columns typically include:
- iteration
- n_evaluated
- hypervolume
- best_viable_pha
- pareto_discovered

Outputs
-------
results/tables/al_agg.csv
results/tables/random_agg.csv
results/tables/greedy_agg.csv
results/tables/exploitation_agg.csv

results/figures/benchmark_hypervolume.png
results/figures/benchmark_best_viable_pha.png
results/figures/benchmark_pareto_discovered.png   (if available)
"""

from __future__ import annotations

from pathlib import Path
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHOD_PATTERNS = {
    "al": "al_history_seed*.csv",
    "random": "random_baseline_seed*.csv",
    "greedy": "greedy_baseline_seed*.csv",
    "exploitation": "exploitation_baseline_seed*.csv",
}

# Metrics to aggregate if present
METRICS = [
    "hypervolume",
    "best_viable_pha",
    "best_pha",
    "pareto_discovered",
    "pareto_size",
    "mean_pha",
    "mean_biomass",
]

# x-axis preference
X_CANDIDATES = ["n_evaluated", "evaluated", "iteration"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_seed_from_filename(path: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def pick_x_column(df: pd.DataFrame) -> str:
    for c in X_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find any x-axis column among {X_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def load_method_runs(method_name: str, pattern: str) -> List[pd.DataFrame]:
    files = sorted(glob.glob(str(TABLE_DIR / pattern)))
    if not files:
        print(f"[WARN] No files found for method '{method_name}' using pattern: {pattern}")
        return []

    runs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.copy()
        seed = find_seed_from_filename(f)
        df["seed"] = seed
        df["source_file"] = Path(f).name
        runs.append(df)

    print(f"[INFO] Loaded {len(runs)} runs for method '{method_name}'")
    return runs


def aggregate_runs(runs: List[pd.DataFrame], method_name: str) -> Optional[pd.DataFrame]:
    if not runs:
        return None

    x_col = pick_x_column(runs[0])

    # Keep only common columns across runs
    common_cols = set(runs[0].columns)
    for df in runs[1:]:
        common_cols &= set(df.columns)
    common_cols = list(common_cols)

    # Combine
    big = pd.concat([df[common_cols].copy() for df in runs], ignore_index=True)

    # Drop accidental index-like columns that can break repeated merges
    bad_cols = [c for c in big.columns if c.lower().startswith("index")]
    if bad_cols:
        big = big.drop(columns=bad_cols, errors="ignore")

    present_metrics = [m for m in METRICS if m in big.columns]
    if not present_metrics:
        raise ValueError(
            f"No supported metric columns found for method '{method_name}'. "
            f"Available columns: {list(big.columns)}"
        )

    # Start from unique x-axis values
    agg_df = (
        big[[x_col]]
        .drop_duplicates()
        .sort_values(by=x_col)
        .reset_index(drop=True)
    )
    agg_df["n_runs"] = big.groupby(x_col).size().values
    agg_df["method"] = method_name

    # Aggregate each metric safely
    for metric in present_metrics:
        stats = (
            big.groupby(x_col)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
            })
        )

        # Safety: remove any accidental index-like columns before merge
        bad_stats_cols = [c for c in stats.columns if c != x_col and c.lower().startswith("index")]
        if bad_stats_cols:
            stats = stats.drop(columns=bad_stats_cols, errors="ignore")

        agg_df = agg_df.merge(stats, on=x_col, how="left")

    # Reorder
    ordered_cols = [x_col, "n_runs", "method"]
    for metric in present_metrics:
        ordered_cols.extend([f"{metric}_mean", f"{metric}_std"])
    agg_df = agg_df[ordered_cols]

    return agg_df


def save_agg(df: pd.DataFrame, method_name: str) -> None:
    out = TABLE_DIR / f"{method_name}_agg.csv"
    df.to_csv(out, index=False)
    print(f"[INFO] Saved: {out}")


def plot_metric(
    agg_map: Dict[str, pd.DataFrame],
    metric: str,
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    plt.figure(figsize=(7, 5))

    plotted_any = False

    for method_name, df in agg_map.items():
        if df is None:
            continue

        x_col = pick_x_column(df)
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        x = df[x_col].values
        y = df[mean_col].values
        s = df[std_col].fillna(0.0).values if std_col in df.columns else np.zeros_like(y)

        plt.plot(x, y, label=method_name, linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.2)
        plotted_any = True

    if not plotted_any:
        print(f"[WARN] Metric '{metric}' not available for any method. Skipping plot.")
        plt.close()
        return

    plt.xlabel("FBA evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {outfile}")


def print_summary(agg_map: Dict[str, pd.DataFrame]) -> None:
    print("\n================ FINAL SUMMARY ================")
    for method_name, df in agg_map.items():
        if df is None or df.empty:
            continue

        last = df.iloc[-1]
        x_col = pick_x_column(df)

        msg = [f"{method_name:>12} | final {x_col}={last[x_col]}"]

        for metric in ["hypervolume", "best_viable_pha", "pareto_discovered", "best_pha"]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in df.columns:
                mean_val = last[mean_col]
                std_val = last[std_col] if std_col in df.columns and pd.notna(last[std_col]) else 0.0
                msg.append(f"{metric}={mean_val:.6f}±{std_val:.6f}")

        print(" | ".join(msg))
    print("==============================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    agg_map: Dict[str, Optional[pd.DataFrame]] = {}

    for method_name, pattern in METHOD_PATTERNS.items():
        runs = load_method_runs(method_name, pattern)
        agg_df = aggregate_runs(runs, method_name) if runs else None
        agg_map[method_name] = agg_df
        if agg_df is not None:
            save_agg(agg_df, method_name)

    print_summary(agg_map)

    # Figures
    plot_metric(
        agg_map=agg_map,
        metric="hypervolume",
        title="Hypervolume vs FBA evaluations",
        ylabel="Pareto hypervolume",
        outfile=FIG_DIR / "benchmark_hypervolume.png",
    )

    plot_metric(
        agg_map=agg_map,
        metric="best_viable_pha",
        title="Best viable PHA vs FBA evaluations",
        ylabel="Best viable PHA flux",
        outfile=FIG_DIR / "benchmark_best_viable_pha.png",
    )

    plot_metric(
        agg_map=agg_map,
        metric="pareto_discovered",
        title="Pareto solutions discovered vs FBA evaluations",
        ylabel="Pareto points discovered",
        outfile=FIG_DIR / "benchmark_pareto_discovered.png",
    )


if __name__ == "__main__":
    main()