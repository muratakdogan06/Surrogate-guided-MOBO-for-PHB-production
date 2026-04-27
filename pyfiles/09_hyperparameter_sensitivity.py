#!/usr/bin/env python3
"""
09_hyperparameter_sensitivity.py — One-at-a-Time (OAT) Sensitivity Analysis
=============================================================================
Quality Improvement Step 4 — Bioresource Technology submission

Tests robustness of the Bayesian active-learning framework to key
hyperparameter choices using one-at-a-time (OAT) sensitivity analysis.

Parameters tested:
  - UCB kappa (κ): exploration-exploitation trade-off
  - Diversity penalty (λ_diversity): batch diversity enforcement
  - Biomass penalty (λ_biomass): growth-compatible design filtering
  - Biomass viability threshold: minimum acceptable predicted growth

Outputs
-------
  results/tables/hyperparameter_sensitivity_results.csv
  results/tables/hyperparameter_sensitivity_summary.csv
  results/figures/hyperparameter_tornado_chart.png
  results/figures/hyperparameter_sensitivity_curves.png

References
----------
  Shahriari et al. (2016) Proc IEEE 104:148-175
  Heckmann et al. (2023) ACS Synth Biol. DOI:10.1021/acssynbio.3c00186
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "results/logs/09_sensitivity.log"),
    ],
)
logger = logging.getLogger("pipeline.09")

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGS_DIR   = PROJECT_ROOT / "results" / "figures"
for d in [TABLES_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_al_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "active_learning.yaml") as f:
        return yaml.safe_load(f)


class SyntheticFitnessLandscape:
    """
    Synthetic fitness landscape for AL benchmarking when cobra is unavailable.

    Models a multi-modal PHA production landscape with:
    - 3 local optima and 1 global optimum
    - Biomass-PHA trade-off (negative correlation)
    - Noise representing biological variability
    """
    def __init__(self, n_features: int = 62, seed: int = 42):
        self.n_features = n_features
        rng = np.random.default_rng(seed)

        # True top designs (global optima in design space)
        self.top_designs = {
            "opt1": rng.integers(0, 2, n_features),
            "opt2": rng.integers(0, 2, n_features),
            "opt3": rng.integers(0, 2, n_features),
        }
        self.opt_pha_values   = {"opt1": 2.1634, "opt2": 2.0891, "opt3": 1.9743}
        self.opt_bio_values   = {"opt1": 0.412,  "opt2": 0.398,  "opt3": 0.389}

    def evaluate(self, design: np.ndarray, seed: int = 42) -> tuple[float, float]:
        """Return (pha_flux, biomass_flux) for a given design vector."""
        rng = np.random.default_rng(seed + hash(design.tobytes()) % 10000)

        # Compute similarity to each optimum
        similarities = {}
        for name, opt in self.top_designs.items():
            sim = 1 - np.sum(np.abs(design - opt)) / self.n_features
            similarities[name] = sim

        # Weighted average of optima
        total_w = sum(np.exp(5 * s) for s in similarities.values())
        pha = sum(self.opt_pha_values[n] * np.exp(5 * s) for n, s in similarities.items()) / total_w
        bio = sum(self.opt_bio_values[n] * np.exp(5 * s) for n, s in similarities.items()) / total_w

        # Add noise
        pha += rng.normal(0, 0.02)
        bio += rng.normal(0, 0.005)
        return max(0, float(pha)), max(0, float(bio))


class SimpleSurrogate:
    """Lightweight gradient-boosting surrogate for sensitivity testing."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 4, seed: int = 42):
        from sklearn.ensemble import GradientBoostingRegressor
        self.models_pha = [
            GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=seed + i)
            for i in range(3)
        ]
        self.models_bio = [
            GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=seed + 100 + i)
            for i in range(3)
        ]
        self.fitted = False

    def fit(self, X: np.ndarray, y_pha: np.ndarray, y_bio: np.ndarray):
        for m in self.models_pha: m.fit(X, y_pha)
        for m in self.models_bio: m.fit(X, y_bio)
        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pha_preds = np.array([m.predict(X) for m in self.models_pha])
        bio_preds = np.array([m.predict(X) for m in self.models_bio])
        return (pha_preds.mean(0), pha_preds.std(0),
                bio_preds.mean(0), bio_preds.std(0))


def ucb_acquisition(
    pha_mean: np.ndarray,
    pha_std:  np.ndarray,
    bio_mean: np.ndarray,
    bio_std:  np.ndarray,
    kappa:            float,
    diversity_lambda: float,
    biomass_penalty:  float,
    biomass_threshold: float,
    evaluated_designs: np.ndarray | None,
    candidate_designs: np.ndarray,
    batch_size: int,
) -> list[int]:
    """UCB acquisition with diversity and biomass penalties."""
    scores = pha_mean + kappa * pha_std

    # Biomass penalty
    bio_violation = bio_mean < biomass_threshold
    scores -= biomass_penalty * bio_violation.astype(float)

    selected_indices: list[int] = []
    for _ in range(batch_size):
        if len(selected_indices) > 0 and diversity_lambda > 0:
            selected_arr = candidate_designs[selected_indices]
            distances = np.array([
                np.sum(np.abs(candidate_designs - s), axis=1)
                for s in selected_arr
            ]).min(axis=0)
            diversity_bonus = diversity_lambda * distances / (candidate_designs.shape[1] + 1e-8)
            effective_scores = scores + diversity_bonus
        else:
            effective_scores = scores.copy()

        effective_scores[selected_indices] = -np.inf
        selected_indices.append(int(np.argmax(effective_scores)))

    return selected_indices


def run_al_with_config(
    config: dict,
    landscape: SyntheticFitnessLandscape,
    seed: int,
    n_designs: int = 300,
    n_iterations: int = 15,
    batch_size: int = 5,
    n_initial: int = 5,
) -> dict:
    """Run one AL session with given config, return metrics trajectory."""
    rng = np.random.default_rng(seed)
    designs = rng.integers(0, 2, (n_designs, landscape.n_features))

    kappa     = config["ucb_kappa"]
    div_lam   = config["diversity_lambda"]
    bio_pen   = config["biomass_penalty_lambda"]
    bio_thr   = config["biomass_min_for_acq"]
    bio_frac  = config.get("biomass_fraction_required", 0.30)

    # Evaluate initial random pool
    evaluated = list(rng.choice(n_designs, size=n_initial, replace=False))
    X_eval, y_pha_eval, y_bio_eval = [], [], []
    for idx in evaluated:
        pha, bio = landscape.evaluate(designs[idx], seed=seed + idx)
        X_eval.append(designs[idx]); y_pha_eval.append(pha); y_bio_eval.append(bio)

    surrogate = SimpleSurrogate(seed=seed)
    history = []
    best_viable_pha = max(y for y, b in zip(y_pha_eval, y_bio_eval) if b >= bio_thr * 0.5) if y_pha_eval else 0.0

    for iteration in range(n_iterations):
        if len(X_eval) >= 5:
            surrogate.fit(np.array(X_eval), np.array(y_pha_eval), np.array(y_bio_eval))
            unevaluated = [i for i in range(n_designs) if i not in evaluated]
            if not unevaluated:
                break
            cands = designs[unevaluated]
            pha_m, pha_s, bio_m, bio_s = surrogate.predict(cands)
            sel_rel = ucb_acquisition(
                pha_m, pha_s, bio_m, bio_s,
                kappa, div_lam, bio_pen, bio_thr,
                None, cands, batch_size,
            )
            sel_abs = [unevaluated[i] for i in sel_rel]
        else:
            unevaluated = [i for i in range(n_designs) if i not in evaluated]
            sel_abs = list(rng.choice(unevaluated, size=min(batch_size, len(unevaluated)), replace=False))

        for idx in sel_abs:
            pha, bio = landscape.evaluate(designs[idx], seed=seed + idx)
            X_eval.append(designs[idx]); y_pha_eval.append(pha); y_bio_eval.append(bio)
            evaluated.append(idx)
            if bio >= bio_thr * 0.5:
                best_viable_pha = max(best_viable_pha, pha)

        # Compute hypervolume proxy (dominated area relative to reference)
        viable_pha = [y for y, b in zip(y_pha_eval, y_bio_eval) if b >= bio_thr * 0.5]
        hv_proxy = np.sum(np.array(viable_pha) - 1.5) if viable_pha else 0.0
        hv_proxy = max(0, hv_proxy)

        history.append({
            "iteration":        iteration + 1,
            "n_evaluated":      len(evaluated),
            "best_viable_pha":  best_viable_pha,
            "hypervolume":      hv_proxy,
        })

    final_hv  = history[-1]["hypervolume"] if history else 0.0
    final_pha = history[-1]["best_viable_pha"] if history else 0.0
    x_vals    = [h["n_evaluated"] for h in history]
    hv_vals   = [h["hypervolume"] for h in history]
    pha_vals  = [h["best_viable_pha"] for h in history]

    # AUC via trapezoid
    hv_auc  = float(np.trapezoid(hv_vals,  x_vals)) if len(x_vals) > 1 else 0.0
    pha_auc = float(np.trapezoid(pha_vals, x_vals)) if len(x_vals) > 1 else 0.0

    return {
        "final_hypervolume":     final_hv,
        "final_best_pha":        final_pha,
        "hypervolume_auc":       hv_auc,
        "best_viable_pha_auc":   pha_auc,
        "history":               history,
    }


def run_sensitivity_analysis(al_cfg: dict) -> pd.DataFrame:
    """Run OAT sensitivity analysis across all parameters."""
    sens_cfg = al_cfg.get("sensitivity_analysis", {})
    params   = sens_cfg.get("parameters", {})
    n_seeds  = sens_cfg.get("n_seeds_per_config", 3)

    landscape = SyntheticFitnessLandscape(n_features=62, seed=42)

    # Default config
    default_cfg = {
        "ucb_kappa":              al_cfg["ucb_kappa"],
        "diversity_lambda":       al_cfg["diversity_lambda"],
        "biomass_penalty_lambda": al_cfg["biomass_penalty_lambda"],
        "biomass_min_for_acq":    al_cfg["biomass_min_for_acq"],
        "biomass_fraction_required": al_cfg["biomass_fraction_required"],
    }

    rows = []
    total_runs = sum(len(p["grid"]) * n_seeds for p in params.values())
    logger.info("Sensitivity analysis: %d configurations × %d seeds = %d total runs",
                len(params), n_seeds, total_runs // n_seeds * n_seeds)

    run_count = 0
    for param_name, param_info in params.items():
        grid    = param_info["grid"]
        default = param_info["default"]

        for value in grid:
            test_cfg = default_cfg.copy()
            test_cfg[param_name] = value
            seed_results = []

            for seed in range(n_seeds):
                t0 = time.time()
                res = run_al_with_config(test_cfg, landscape, seed=seed)
                seed_results.append(res)
                run_count += 1
                if run_count % 10 == 0:
                    logger.info("  Run %d/%d — param=%s value=%.3f seed=%d HV=%.4f (%.1fs)",
                                run_count, total_runs, param_name, value, seed,
                                res["hypervolume_auc"], time.time()-t0)

            rows.append({
                "parameter":             param_name,
                "value":                 value,
                "is_default":            abs(value - default) < 1e-9,
                "n_seeds":               n_seeds,
                "hv_auc_mean":           np.mean([r["hypervolume_auc"] for r in seed_results]),
                "hv_auc_std":            np.std( [r["hypervolume_auc"] for r in seed_results]),
                "pha_auc_mean":          np.mean([r["best_viable_pha_auc"] for r in seed_results]),
                "pha_auc_std":           np.std( [r["best_viable_pha_auc"] for r in seed_results]),
                "final_hv_mean":         np.mean([r["final_hypervolume"] for r in seed_results]),
                "final_pha_mean":        np.mean([r["final_best_pha"] for r in seed_results]),
                "rationale":             param_info.get("rationale", ""),
            })

    return pd.DataFrame(rows)


def compute_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sensitivity range for each parameter (tornado chart data)."""
    rows = []
    for param in df["parameter"].unique():
        subset  = df[df["parameter"] == param]
        default = subset[subset.is_default]["hv_auc_mean"].values
        default_val = float(default[0]) if len(default) > 0 else np.nan
        min_val = subset["hv_auc_mean"].min()
        max_val = subset["hv_auc_mean"].max()
        rang = max_val - min_val
        rows.append({
            "parameter":      param,
            "default_hv_auc": default_val,
            "min_hv_auc":     min_val,
            "max_hv_auc":     max_val,
            "range_hv_auc":   rang,
            "sensitivity":    "HIGH" if rang > 0.15 * default_val else
                              ("MEDIUM" if rang > 0.05 * default_val else "LOW"),
        })
    return pd.DataFrame(rows).sort_values("range_hv_auc", ascending=False)


# ── FIGURES ──────────────────────────────────────────────────────────────────

def plot_tornado_chart(summary_df: pd.DataFrame, full_df: pd.DataFrame, outpath: Path) -> None:
    """Tornado chart showing sensitivity range for each parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_sorted = summary_df.sort_values("range_hv_auc", ascending=True)
    params  = summary_sorted["parameter"].values
    default = summary_sorted["default_hv_auc"].values
    mins    = summary_sorted["min_hv_auc"].values
    maxs    = summary_sorted["max_hv_auc"].values

    colors = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#27ae60"}
    sens_map = dict(zip(summary_sorted["parameter"], summary_sorted["sensitivity"]))

    for i, (p, d, lo, hi) in enumerate(zip(params, default, mins, maxs)):
        color = colors.get(sens_map.get(p, "LOW"), "#95a5a6")
        ax.barh(i, hi - d, left=d,  color=color, alpha=0.7, height=0.6)
        ax.barh(i, d - lo, left=lo, color=color, alpha=0.7, height=0.6)
        ax.text(hi + 0.003, i, f"{hi:.3f}", va="center", fontsize=8)
        ax.text(lo - 0.003, i, f"{lo:.3f}", va="center", ha="right", fontsize=8)

    ax.axvline(x=np.nanmean(default), color="black", linestyle="--", lw=1.5,
               label="Default configuration")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in params], fontsize=10)
    ax.set_xlabel("Hypervolume AUC", fontsize=11)
    ax.set_title("Hyperparameter Sensitivity — Tornado Chart\n"
                 "(One-at-a-Time Analysis, 3 seeds per configuration)",
                 fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=c, alpha=0.7, label=f"{s} sensitivity")
               for s, c in colors.items()]
    patches.append(plt.Line2D([0], [0], color="black", ls="--", label="Default value"))
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved tornado chart: %s", outpath)


def plot_sensitivity_curves(df: pd.DataFrame, outpath: Path) -> None:
    """Plot sensitivity curves for each parameter."""
    params = df["parameter"].unique()
    n_params = len(params)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cmap = cm.get_cmap("tab10")
    for ax_idx, param in enumerate(params):
        if ax_idx >= len(axes):
            break
        subset = df[df["parameter"] == param].sort_values("value")
        ax = axes[ax_idx]
        ax.errorbar(
            subset["value"], subset["hv_auc_mean"],
            yerr=subset["hv_auc_std"],
            marker="o", color=cmap(ax_idx), linewidth=2, markersize=7,
            capsize=4, label="HV AUC"
        )

        # Mark default
        default_row = subset[subset.is_default]
        if not default_row.empty:
            ax.axvline(x=default_row["value"].values[0], color="red",
                       linestyle="--", alpha=0.7, label="Default value")
            ax.scatter(default_row["value"], default_row["hv_auc_mean"],
                       color="red", s=100, zorder=5)

        ax.set_xlabel(param.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Hypervolume AUC (mean ± SD)", fontsize=10)
        ax.set_title(f"Sensitivity: {param.replace('_', ' ').title()}", fontsize=11)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("One-at-a-Time Hyperparameter Sensitivity Analysis\n"
                 "Bayesian Active Learning for PHA Production in P. megaterium",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved sensitivity curves: %s", outpath)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STEP 4 (IMPROVEMENT) — Hyperparameter Sensitivity Analysis")
    logger.info("Quality Improvement Step 4 — Bioresource Technology")
    logger.info("=" * 70)

    al_cfg = load_al_config()

    # Run OAT sensitivity analysis
    logger.info("Running one-at-a-time sensitivity analysis...")
    df = run_sensitivity_analysis(al_cfg)
    df.to_csv(TABLES_DIR / "hyperparameter_sensitivity_results.csv", index=False)
    logger.info("Saved: %s", TABLES_DIR / "hyperparameter_sensitivity_results.csv")

    # Compute summary
    summary_df = compute_sensitivity_summary(df)
    summary_df.to_csv(TABLES_DIR / "hyperparameter_sensitivity_summary.csv", index=False)
    logger.info("Saved: %s", TABLES_DIR / "hyperparameter_sensitivity_summary.csv")

    # Generate figures
    plot_tornado_chart(summary_df, df, FIGS_DIR / "hyperparameter_tornado_chart.png")
    plot_sensitivity_curves(df, FIGS_DIR / "hyperparameter_sensitivity_curves.png")

    logger.info("Done in %.1f s", time.time() - t0)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)
    print(summary_df[["parameter", "range_hv_auc", "sensitivity"]].to_string(index=False))
    print("=" * 70)
    print("Interpretation:")
    print("  HIGH sensitivity  → parameter choice matters; justify default carefully")
    print("  MEDIUM sensitivity → moderate effect; current default is reasonable")
    print("  LOW sensitivity   → robust to parameter choice; default is safe")


if __name__ == "__main__":
    main()
