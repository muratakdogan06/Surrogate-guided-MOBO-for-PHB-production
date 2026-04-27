#!/usr/bin/env python3
"""
09_hyperparameter_sensitivity.py — One-at-a-Time (OAT) Sensitivity Analysis
=============================================================================
Quality Improvement Step 4 — Bioresource Technology submission

Tests robustness of the MC-EHVI Bayesian active-learning framework to key
hyperparameter choices using one-at-a-time (OAT) sensitivity analysis.
Each parameter is varied across its grid (defined in active_learning.yaml)
while all other parameters are held at their default values.

Parameters tested (from configs/active_learning.yaml → sensitivity_analysis):
  - n_mc_samples       : Monte Carlo fidelity for EHVI approximation
  - diversity_lambda   : Batch diversity enforcement
  - biomass_min_for_acq: Biomass viability threshold for tracking metric
  - ensemble_size      : GBR ensemble size controlling uncertainty width

The AL loop uses the real GEM-based MC-EHVI pipeline from
``phaopt.active_learning_ehvi`` with epsilon-constraint FBA evaluations.

Outputs
-------
  results/tables/hyperparameter_sensitivity_results.csv
  results/tables/hyperparameter_sensitivity_summary.csv
  results/figures/hyperparameter_tornado_chart.png
  results/figures/hyperparameter_sensitivity_curves.png

References
----------
  Daulton S et al. (2020) NeurIPS 33:9851–9864
  Shahriari B et al. (2016) Proc IEEE 104:148–175
  Heckmann D et al. (2023) ACS Synth Biol. DOI:10.1021/acssynbio.3c00186
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.active_learning import (
    _build_global_pareto_reference,
    run_active_learning,
)
from phaopt.io import load_sbml_model
from phaopt.perturbation_space import generate_designs
from phaopt.utils import (
    load_al_config,
    load_candidate_reactions,
    load_conditions,
    load_model_config,
)

(PROJECT_ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

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

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ── Metric extraction ────────────────────────────────────────────────────────

def _extract_run_metrics(res: Dict[str, Any]) -> Dict[str, float]:
    """Extract AUC and final-value metrics from a single AL run result."""
    history = res.get("history", [])
    if not history:
        return {
            "final_hypervolume":   0.0,
            "final_best_pha":      0.0,
            "hypervolume_auc":     0.0,
            "best_viable_pha_auc": 0.0,
        }

    x_vals  = [h["n_evaluated"]    for h in history]
    hv_vals = [h["hypervolume"]    for h in history]
    pha_vals = [h["best_viable_pha"] for h in history]

    hv_auc  = float(_trapezoid(hv_vals,  x_vals)) if len(x_vals) > 1 else 0.0
    pha_auc = float(_trapezoid(pha_vals, x_vals)) if len(x_vals) > 1 else 0.0

    return {
        "final_hypervolume":   history[-1]["hypervolume"],
        "final_best_pha":      history[-1]["best_viable_pha"],
        "hypervolume_auc":     hv_auc,
        "best_viable_pha_auc": pha_auc,
    }


# ── Core OAT loop ────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    al_cfg: dict,
    model,
    designs: List[Dict[str, Any]],
    cands: List[str],
    cfg: dict,
    conditions_cfg: dict,
    candidates_cfg: dict,
    global_pareto_set: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run OAT sensitivity analysis using the real MC-EHVI AL pipeline.

    For each parameter in the sensitivity grid, every grid value is tested
    while all other parameters remain at their defaults.  Each configuration
    is repeated over ``n_seeds_per_config`` random seeds.
    """
    sens_cfg = al_cfg.get("sensitivity_analysis", {})
    params   = sens_cfg.get("parameters", {})
    n_seeds  = sens_cfg.get("n_seeds_per_config", 3)

    total_grid_points = sum(len(p["grid"]) for p in params.values())
    total_runs = total_grid_points * n_seeds
    logger.info(
        "Sensitivity analysis: %d parameters, %d grid points, "
        "%d seeds/config → %d total AL runs",
        len(params), total_grid_points, n_seeds, total_runs,
    )

    rows: List[Dict[str, Any]] = []
    run_count = 0

    for param_name, param_info in params.items():
        grid    = param_info["grid"]
        default = param_info["default"]
        logger.info(
            "── Parameter: %s  (default=%s, grid=%s)",
            param_name, default, grid,
        )

        for value in grid:
            test_al_cfg = al_cfg.copy()
            test_al_cfg[param_name] = value

            seed_results: List[Dict[str, float]] = []

            for seed_idx in range(n_seeds):
                test_al_cfg["random_seed"] = seed_idx
                t0 = time.time()

                try:
                    res = run_active_learning(
                        model=model,
                        designs=designs,
                        candidate_reaction_ids=cands,
                        al_cfg=test_al_cfg,
                        cfg=cfg,
                        conditions_cfg=conditions_cfg,
                        candidates_cfg=candidates_cfg,
                        global_pareto_set=global_pareto_set,
                    )
                    metrics = _extract_run_metrics(res)
                except Exception as exc:
                    logger.warning(
                        "  FAILED: param=%s value=%s seed=%d — %s",
                        param_name, value, seed_idx, exc,
                    )
                    metrics = {
                        "final_hypervolume":   0.0,
                        "final_best_pha":      0.0,
                        "hypervolume_auc":     0.0,
                        "best_viable_pha_auc": 0.0,
                    }

                seed_results.append(metrics)
                run_count += 1
                elapsed = time.time() - t0

                logger.info(
                    "  Run %3d/%d | %s=%s | seed=%d | "
                    "HV_AUC=%.4f | best_PHA=%.4f | %.1fs",
                    run_count, total_runs, param_name, value, seed_idx,
                    metrics["hypervolume_auc"],
                    metrics["final_best_pha"],
                    elapsed,
                )

            hv_aucs  = [r["hypervolume_auc"]     for r in seed_results]
            pha_aucs = [r["best_viable_pha_auc"] for r in seed_results]
            final_hvs  = [r["final_hypervolume"] for r in seed_results]
            final_phas = [r["final_best_pha"]    for r in seed_results]

            rows.append({
                "parameter":      param_name,
                "value":          value,
                "is_default":     abs(float(value) - float(default)) < 1e-9,
                "n_seeds":        n_seeds,
                "hv_auc_mean":    float(np.mean(hv_aucs)),
                "hv_auc_std":     float(np.std(hv_aucs)),
                "pha_auc_mean":   float(np.mean(pha_aucs)),
                "pha_auc_std":    float(np.std(pha_aucs)),
                "final_hv_mean":  float(np.mean(final_hvs)),
                "final_pha_mean": float(np.mean(final_phas)),
                "rationale":      param_info.get("rationale", ""),
            })

    return pd.DataFrame(rows)


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sensitivity range for each parameter (tornado chart data)."""
    rows = []
    for param in df["parameter"].unique():
        subset      = df[df["parameter"] == param]
        default_arr = subset[subset["is_default"]]["hv_auc_mean"].values
        default_val = float(default_arr[0]) if len(default_arr) > 0 else np.nan
        min_val     = subset["hv_auc_mean"].min()
        max_val     = subset["hv_auc_mean"].max()
        rang        = max_val - min_val

        if np.isnan(default_val) or default_val == 0:
            sensitivity = "LOW"
        elif rang > 0.15 * abs(default_val):
            sensitivity = "HIGH"
        elif rang > 0.05 * abs(default_val):
            sensitivity = "MEDIUM"
        else:
            sensitivity = "LOW"

        rows.append({
            "parameter":      param,
            "default_hv_auc": default_val,
            "min_hv_auc":     min_val,
            "max_hv_auc":     max_val,
            "range_hv_auc":   rang,
            "sensitivity":    sensitivity,
        })

    return pd.DataFrame(rows).sort_values("range_hv_auc", ascending=False)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_tornado_chart(
    summary_df: pd.DataFrame,
    full_df: pd.DataFrame,
    outpath: Path,
) -> None:
    """Tornado chart showing sensitivity range for each parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_sorted = summary_df.sort_values("range_hv_auc", ascending=True)
    params  = summary_sorted["parameter"].values
    default = summary_sorted["default_hv_auc"].values
    mins    = summary_sorted["min_hv_auc"].values
    maxs    = summary_sorted["max_hv_auc"].values

    colors   = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#27ae60"}
    sens_map = dict(zip(summary_sorted["parameter"], summary_sorted["sensitivity"]))

    for i, (p, d, lo, hi) in enumerate(zip(params, default, mins, maxs)):
        color = colors.get(sens_map.get(p, "LOW"), "#95a5a6")
        ax.barh(i, hi - d, left=d,  color=color, alpha=0.7, height=0.6)
        ax.barh(i, d - lo, left=lo, color=color, alpha=0.7, height=0.6)
        ax.text(hi + 0.003, i, f"{hi:.3f}", va="center", fontsize=8)
        ax.text(lo - 0.003, i, f"{lo:.3f}", va="center", ha="right", fontsize=8)

    ax.axvline(
        x=np.nanmean(default), color="black", linestyle="--", lw=1.5,
        label="Default configuration",
    )
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(
        [p.replace("_", " ").title() for p in params], fontsize=10,
    )
    ax.set_xlabel("Hypervolume AUC", fontsize=11)
    ax.set_title(
        "MC-EHVI Hyperparameter Sensitivity — Tornado Chart\n"
        f"(One-at-a-Time Analysis, {full_df['n_seeds'].iloc[0]} seeds per configuration)",
        fontsize=12, fontweight="bold",
    )

    patches = [
        mpatches.Patch(color=c, alpha=0.7, label=f"{s} sensitivity")
        for s, c in colors.items()
    ]
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
    params   = df["parameter"].unique()
    n_params = len(params)
    n_cols   = 2
    n_rows   = max(1, (n_params + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
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
            capsize=4, label="HV AUC",
        )

        default_row = subset[subset["is_default"]]
        if not default_row.empty:
            ax.axvline(
                x=default_row["value"].values[0], color="red",
                linestyle="--", alpha=0.7, label="Default value",
            )
            ax.scatter(
                default_row["value"], default_row["hv_auc_mean"],
                color="red", s=100, zorder=5,
            )

        ax.set_xlabel(param.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Hypervolume AUC (mean ± SD)", fontsize=10)
        ax.set_title(f"Sensitivity: {param.replace('_', ' ').title()}", fontsize=11)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "One-at-a-Time Hyperparameter Sensitivity Analysis\n"
        "MC-EHVI Bayesian Active Learning for PHA Production in P. megaterium",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved sensitivity curves: %s", outpath)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0_global = time.time()
    logger.info("=" * 70)
    logger.info("STEP 4 — MC-EHVI Hyperparameter Sensitivity Analysis (OAT)")
    logger.info("Quality Improvement Step 4 — Bioresource Technology")
    logger.info("=" * 70)

    # ── Load configs ──────────────────────────────────────────────────────
    cfg            = load_model_config()
    al_cfg         = load_al_config()
    candidates_cfg = load_candidate_reactions()
    conditions_cfg = load_conditions()

    sens_cfg = al_cfg.get("sensitivity_analysis", {})
    if not sens_cfg.get("enabled", False):
        logger.info("Sensitivity analysis is disabled in config. Exiting.")
        return

    # ── Load COBRA model and generate designs ─────────────────────────────
    logger.info("Loading COBRA model from: %s", cfg["extended_model_path"])
    model   = load_sbml_model(cfg["extended_model_path"])
    rxn_ids = {r.id for r in model.reactions}
    designs = generate_designs(rxn_ids, candidates_cfg)

    cands = sorted({
        rid
        for grp in candidates_cfg["candidate_groups"].values()
        for rid in grp["reaction_ids"]
        if rid in rxn_ids
    })

    condition_name = al_cfg.get("condition_name", "base")
    biomass_fraction_required = float(al_cfg.get("biomass_fraction_required", 0.30))

    logger.info("Condition         : %s", condition_name)
    logger.info("Candidate designs : %d", len(designs))
    logger.info("Candidate rxns    : %d", len(cands))
    logger.info("Acquisition       : %s", al_cfg.get("acquisition", "mc_ehvi"))

    # ── Build global Pareto reference once (shared across all runs) ───────
    global_pareto_set = None
    if al_cfg.get("use_global_pareto_reference", True):
        pha_rxn    = cfg["pha_reaction_id"]
        bio_rxn    = cfg["biomass_reaction_id"]
        upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
        overrides  = (
            conditions_cfg["conditions"][condition_name].get("overrides") or {}
        )

        logger.info("Building global Pareto reference set …")
        global_pareto_set = _build_global_pareto_reference(
            cobra_model=model,
            all_designs=designs,
            pha_rxn=pha_rxn,
            bio_rxn=bio_rxn,
            overrides=overrides,
            upreg_fold=upreg_fold,
            biomass_fraction_required=biomass_fraction_required,
            max_designs=al_cfg.get("pareto_n_points", 500),
        )
        logger.info("Global Pareto reference size: %d", len(global_pareto_set))

    # ── Run OAT sensitivity analysis ──────────────────────────────────────
    logger.info("Running one-at-a-time sensitivity analysis …")
    df = run_sensitivity_analysis(
        al_cfg=al_cfg,
        model=model,
        designs=designs,
        cands=cands,
        cfg=cfg,
        conditions_cfg=conditions_cfg,
        candidates_cfg=candidates_cfg,
        global_pareto_set=global_pareto_set,
    )
    df.to_csv(TABLES_DIR / "hyperparameter_sensitivity_results.csv", index=False)
    logger.info("Saved: %s", TABLES_DIR / "hyperparameter_sensitivity_results.csv")

    # ── Compute summary ───────────────────────────────────────────────────
    summary_df = compute_sensitivity_summary(df)
    summary_df.to_csv(
        TABLES_DIR / "hyperparameter_sensitivity_summary.csv", index=False,
    )
    logger.info("Saved: %s", TABLES_DIR / "hyperparameter_sensitivity_summary.csv")

    # ── Generate figures ──────────────────────────────────────────────────
    plot_tornado_chart(
        summary_df, df,
        FIGS_DIR / "hyperparameter_tornado_chart.png",
    )
    plot_sensitivity_curves(
        df,
        FIGS_DIR / "hyperparameter_sensitivity_curves.png",
    )

    elapsed = time.time() - t0_global
    logger.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)

    print("\n" + "=" * 70)
    print("MC-EHVI HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)
    print(summary_df[["parameter", "range_hv_auc", "sensitivity"]].to_string(index=False))
    print("=" * 70)
    print("Interpretation:")
    print("  HIGH sensitivity   → parameter choice matters; justify default carefully")
    print("  MEDIUM sensitivity → moderate effect; current default is reasonable")
    print("  LOW sensitivity    → robust to parameter choice; default is safe")


if __name__ == "__main__":
    main()
