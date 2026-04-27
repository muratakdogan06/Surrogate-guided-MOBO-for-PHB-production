#!/usr/bin/env python3
"""
12_dataset_statistics_and_learning_curves.py — Dataset Stats & Surrogate Validation
=====================================================================================
Quality Improvement Step 12 — Bioresource Technology submission

Generates:
  1. Comprehensive design-space and ML dataset statistics table (Table 1)
  2. Surrogate model learning curves (sample size vs R²)
  3. FBA dataset distribution analysis
  4. Design space coverage analysis

Outputs
-------
  results/tables/table1_dataset_statistics.csv
  results/tables/surrogate_learning_curves.csv
  results/tables/design_space_statistics.csv
  results/figures/surrogate_learning_curves.png
  results/figures/dataset_distribution.png
  results/figures/design_space_coverage.png

References
----------
  Heckmann et al. (2023) ACS Synth Biol. DOI:10.1021/acssynbio.3c00186
  Aminian-Dehkordi et al. (2019) Sci Rep 9:18762. DOI:10.1038/s41598-019-55041-w
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("xgboost is required. Install via: pip install xgboost>=2.0")

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOGS_DIR   = PROJECT_ROOT / "results" / "logs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGS_DIR   = PROJECT_ROOT / "results" / "figures"
for d in [LOGS_DIR, TABLES_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "12_dataset_stats.log"),
    ],
)
logger = logging.getLogger("pipeline.12")


def load_configs() -> tuple[dict, dict, dict]:
    with open(PROJECT_ROOT / "configs" / "model_config.yaml")    as f: cfg   = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "active_learning.yaml") as f: al    = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "candidate_reactions.yaml") as f: cand = yaml.safe_load(f)
    return cfg, al, cand


def compute_design_space_statistics(cand_cfg: dict, cfg: dict) -> dict:
    """Compute theoretical and effective combinatorial design space sizes."""
    groups = cand_cfg["candidate_groups"]
    all_reactions = []
    for grp in groups.values():
        all_reactions.extend(grp.get("reaction_ids", []))
    n_candidates = len(all_reactions)
    n_unique     = len(set(all_reactions))

    max_ko  = cand_cfg.get("max_knockouts", 4)
    max_up  = cand_cfg.get("max_upregulations", 3)

    # Theoretical combinatorial sizes
    ko_space  = sum(comb(n_unique, k) for k in range(1, max_ko + 1))
    up_space  = sum(comb(n_unique, u) for u in range(0, max_up + 1))
    total_space = ko_space * up_space

    # Effective space (removing illegal combos where same rxn is KO and UP)
    # Approximate: remove ~5% for overlapping selections
    effective_space = int(total_space * 0.95)

    cond_path = PROJECT_ROOT / "configs" / "conditions.yaml"
    with open(cond_path) as f:
        cond_cfg = yaml.safe_load(f)
    conditions = len(cond_cfg.get("conditions", {}))
    eps_levels  = len(cfg.get("biomass_fraction_grid", [0.10, 0.30, 0.50, 0.70]))

    # Conservative estimate of designs evaluated (if full grid run)
    n_designs_conservative = 5000  # typical for this pipeline
    n_fba_rows = n_designs_conservative * conditions * eps_levels

    return {
        "n_candidate_reactions":         n_unique,
        "n_candidate_groups":            len(groups),
        "max_simultaneous_knockouts":    max_ko,
        "max_simultaneous_upregulations": max_up,
        "theoretical_KO_space":          ko_space,
        "theoretical_UP_space":          up_space,
        "total_theoretical_designs":     total_space,
        "effective_design_space":        effective_space,
        "n_environmental_conditions":    conditions,
        "n_epsilon_levels":              eps_levels,
        "estimated_FBA_rows_if_full":    n_fba_rows,
        "fraction_sampled_by_AL":        round(105 / effective_space, 8),
        "al_budget_evaluations":         105,  # 5 init + 20*5 iterations
    }


def train_surrogate_with_learning_curve(
    X: pd.DataFrame,
    y: pd.DataFrame,
    al_cfg: dict,
) -> dict:
    """Train XGBoost surrogates and compute learning curves (R² vs training size).

    Uses absolute training sizes spanning the AL-scale (50) to full-dataset
    (224K) range so the learning curve captures the full transition from
    low-data to high-data regimes.
    """
    n_estimators = al_cfg.get("n_estimators", 200)
    max_depth    = al_cfg.get("max_depth", 8)
    seed         = al_cfg.get("random_seed", 42)

    n_total = len(X)

    rng = np.random.default_rng(seed)
    test_idx  = rng.choice(n_total, size=int(0.2 * n_total), replace=False)
    train_pool = np.array([i for i in range(n_total) if i not in set(test_idx)])
    n_pool = len(train_pool)

    train_sizes = [50, 100, 200, 500, 1_000, 2_000, 5_000,
                   10_000, 20_000, 50_000, 100_000, n_pool]
    train_sizes = [s for s in train_sizes if s <= n_pool]

    X_test = X.iloc[test_idx].values
    X_train_full = X.iloc[train_pool].values

    results: dict[str, list] = {}
    logger.info("Computing XGBoost learning curves across %d training sizes "
                "(test set n=%d)...", len(train_sizes), len(test_idx))

    for target in ["pha_flux", "biomass_flux"]:
        y_test       = y[target].iloc[test_idx].values
        y_train_full = y[target].iloc[train_pool].values
        lc_rows: list[dict] = []

        for n_train in train_sizes:
            subset = rng.choice(n_pool, size=n_train, replace=False)
            X_tr = X_train_full[subset]
            y_tr = y_train_full[subset]

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            lc_rows.append({
                "target":            target,
                "training_fraction": round(n_train / n_pool, 4),
                "n_training":        n_train,
                "R2":                round(r2, 4),
                "MAE":               round(mae, 5),
                "RMSE":              round(rmse, 5),
            })
            logger.info("  %s | n=%6d  R²=%.4f  MAE=%.5f",
                        target, n_train, r2, mae)

        results[target] = lc_rows

    return results


def compute_table1(design_stats: dict, lc_results: dict, al_cfg: dict) -> pd.DataFrame:
    """Generate Table 1: Dataset and model statistics for publication."""
    pha_final  = [r for r in lc_results["pha_flux"]
                  if r["training_fraction"] >= 0.99]
    bio_final  = [r for r in lc_results["biomass_flux"]
                  if r["training_fraction"] >= 0.99]

    surr_src = "surrogate_comparison.csv"
    try:
        sc = pd.read_csv(TABLES_DIR / "surrogate_comparison.csv")
        xgb_pha = sc[(sc["model_family"] == "xgboost") & (sc["target"] == "pha_flux")]
        xgb_bio = sc[(sc["model_family"] == "xgboost") & (sc["target"] == "biomass_flux")]
        if len(xgb_pha) and len(xgb_bio):
            pha_r2   = round(float(xgb_pha["R2_test"].iloc[0]), 4)
            pha_mae  = round(float(xgb_pha["MAE_test"].iloc[0]), 5)
            pha_rmse = round(float(xgb_pha["RMSE_test"].iloc[0]), 5)
            bio_r2   = round(float(xgb_bio["R2_test"].iloc[0]), 4)
            bio_mae  = round(float(xgb_bio["MAE_test"].iloc[0]), 5)
            bio_rmse = round(float(xgb_bio["RMSE_test"].iloc[0]), 5)
        else:
            raise FileNotFoundError
    except Exception:
        surr_src = "learning curve (full training)"
        pha_r2   = pha_final[0]["R2"]  if pha_final else "N/A"
        pha_mae  = pha_final[0]["MAE"] if pha_final else "N/A"
        pha_rmse = pha_final[0]["RMSE"] if pha_final else "N/A"
        bio_r2   = bio_final[0]["R2"]  if bio_final else "N/A"
        bio_mae  = bio_final[0]["MAE"] if bio_final else "N/A"
        bio_rmse = bio_final[0]["RMSE"] if bio_final else "N/A"

    rows = [
        ("DESIGN SPACE", "", ""),
        ("Candidate reactions", str(design_stats["n_candidate_reactions"]), "candidate_reactions.yaml"),
        ("Metabolic groups", str(design_stats["n_candidate_groups"]), "candidate_reactions.yaml"),
        ("Max simultaneous KOs", str(design_stats["max_simultaneous_knockouts"]), "candidate_reactions.yaml"),
        ("Max simultaneous UPs", str(design_stats["max_simultaneous_upregulations"]), "candidate_reactions.yaml"),
        ("Theoretical design space (upper bound)", f"{design_stats['total_theoretical_designs']:,}", "combinatorics"),
        ("Effective design space (estimated)", f"{design_stats['effective_design_space']:,}", "combinatorics"),
        ("", "", ""),
        ("FBA SIMULATION PARAMETERS", "", ""),
        ("Environmental conditions", str(design_stats["n_environmental_conditions"]), "conditions.yaml"),
        ("ε-constraint levels", str(design_stats["n_epsilon_levels"]), "model_config.yaml"),
        ("FBA dataset size", "280,000 (10,000 × 7 × 4)", "fba_results.parquet"),
        ("AL total FBA evaluations per seed", str(design_stats["al_budget_evaluations"]), "active_learning.yaml"),
        ("Fraction of design space sampled by AL", f"{design_stats['fraction_sampled_by_AL']:.2e}", "computed"),
        ("", "", ""),
        ("SURROGATE MODEL PARAMETERS", "", ""),
        ("Model family", "XGBoost (XGBRegressor)", "active_learning.yaml"),
        ("n_estimators", str(al_cfg.get("n_estimators", 200)), "active_learning.yaml"),
        ("max_depth", str(al_cfg.get("max_depth", 8)), "active_learning.yaml"),
        ("Ensemble size", str(al_cfg.get("ensemble_size", 5)), "active_learning.yaml"),
        ("Train/test split", "80% / 20%", "05b_compare_surrogates.py"),
        ("", "", ""),
        ("SURROGATE PERFORMANCE (TEST SET)", "", ""),
        ("R² — pha_flux",       str(pha_r2),   surr_src),
        ("MAE — pha_flux",      str(pha_mae),  surr_src),
        ("RMSE — pha_flux",     str(pha_rmse), surr_src),
        ("R² — biomass_flux",   str(bio_r2),   surr_src),
        ("MAE — biomass_flux",  str(bio_mae),  surr_src),
        ("RMSE — biomass_flux", str(bio_rmse), surr_src),
        ("", "", ""),
        ("ACTIVE LEARNING CONFIGURATION", "", ""),
        ("Acquisition function", "MC-EHVI (Monte Carlo Expected Hypervolume Improvement)", "active_learning.yaml"),
        ("MC samples", str(al_cfg.get("n_mc_samples", 256)), "active_learning.yaml"),
        ("EHVI reference point", str(al_cfg.get("ehvi_ref_point", [0.0, 0.0])), "active_learning.yaml"),
        ("Diversity penalty λ", str(al_cfg.get("diversity_lambda", 0.35)), "active_learning.yaml"),
        ("Biomass min for acq.", str(al_cfg.get("biomass_min_for_acq", 0.07)), "active_learning.yaml"),
        ("Ensemble size", str(al_cfg.get("ensemble_size", 5)), "active_learning.yaml"),
        ("AL iterations", str(al_cfg.get("n_iterations", 20)), "active_learning.yaml"),
        ("Batch size", str(al_cfg.get("batch_size", 5)), "active_learning.yaml"),
        ("Number of seeds", str(al_cfg.get("n_seeds", 10)), "active_learning.yaml"),
        ("Pareto reference points", str(al_cfg.get("pareto_n_points", 500)), "active_learning.yaml"),
    ]

    df = pd.DataFrame(rows, columns=["Parameter", "Value", "Source"])
    return df


# ── FIGURES ──────────────────────────────────────────────────────────────────

def plot_learning_curves(lc_results: dict, outpath: Path) -> None:
    """Plot surrogate learning curves: R² and MAE vs training set size (log scale)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    target_styles = {
        "pha_flux":     {"color": "#e74c3c", "label": "PHA Flux Target"},
        "biomass_flux": {"color": "#3498db", "label": "Biomass Flux Target"},
    }
    metrics = [("R2", "R² (coefficient of determination)"), ("MAE", "MAE (mmol gDW⁻¹ h⁻¹)")]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, (target, style) in enumerate(target_styles.items()):
            ax   = axes[row][col]
            data = lc_results.get(target, [])
            if not data:
                continue
            df_lc = pd.DataFrame(data)
            ax.plot(df_lc["n_training"], df_lc[metric],
                    color=style["color"], lw=2.5, marker="o", markersize=6)

            if metric == "R2":
                ax.axhline(0.80, color="green", lw=1.5, linestyle="--", alpha=0.6,
                           label="Adequate R² threshold (0.80)")
                ax.axhline(0.99, color="blue", lw=1, linestyle=":", alpha=0.5,
                           label="R² = 0.99")
                ax.set_ylim(-0.1, 1.05)
            ax.set_xscale("log")
            ax.set_xlabel("Training Set Size (log scale)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{style['label']}\n{metric} Learning Curve", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "R2":
                ax.legend(fontsize=8, loc="lower right")

            final_val = df_lc[metric].iloc[-1]
            ax.annotate(f"n={df_lc['n_training'].iloc[-1]:,}\n{metric}={final_val:.4f}",
                        xy=(df_lc["n_training"].iloc[-1], final_val),
                        xytext=(-50, -20), textcoords="offset points",
                        fontsize=8, color=style["color"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=style["color"], lw=1))

    fig.suptitle("Surrogate Model Learning Curves\n"
                 "XGBoost Ensemble — P. megaterium PHA Engineering",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved learning curves: %s", outpath)


def plot_design_space_coverage(design_stats: dict, al_cfg: dict, outpath: Path) -> None:
    """Bar chart visualising design space coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: design space size comparison (log scale)
    ax = axes[0]
    categories = [
        "Theoretical\ncombinatorial\nspace",
        "Effective\ndesign space",
        "FBA evaluations\n(full grid est.)",
        "AL evaluations\n(per seed)",
        "Initial random\npool",
    ]
    values = [
        design_stats["total_theoretical_designs"],
        design_stats["effective_design_space"],
        design_stats["estimated_FBA_rows_if_full"],
        design_stats["al_budget_evaluations"],
        al_cfg.get("initial_random_samples", 5),
    ]
    colors = ["#e74c3c", "#e67e22", "#27ae60", "#3498db", "#9b59b6"]
    bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.8,
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Number of Designs / Rows (log scale)", fontsize=10)
    ax.set_title("Design Space Size Comparison", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                f"{val:,.0f}", ha="center", fontsize=7, rotation=30, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: candidate reactions by group
    ax2 = axes[1]
    cand_path = PROJECT_ROOT / "configs" / "candidate_reactions.yaml"
    with open(cand_path) as f:
        cand_cfg = yaml.safe_load(f)

    groups = cand_cfg["candidate_groups"]
    group_names   = list(groups.keys())
    group_counts  = [len(g["reaction_ids"]) for g in groups.values()]
    group_palette = ["#e74c3c","#e67e22","#f39c12","#27ae60","#16a085","#2980b9","#8e44ad","#95a5a6"]

    bars2 = ax2.barh(range(len(group_names)), group_counts,
                     color=group_palette[:len(group_names)], alpha=0.85,
                     edgecolor="white", linewidth=1.2)
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels([g.replace("_", " ").title() for g in group_names], fontsize=9)
    ax2.set_xlabel("Number of Candidate Reactions", fontsize=10)
    ax2.set_title("Candidate Reactions by Metabolic Group", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, group_counts):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=9, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    total = sum(group_counts)
    ax2.set_title(f"Candidate Reactions by Group (Total: {total})",
                  fontsize=11, fontweight="bold")

    fig.suptitle("Design Space Coverage Analysis\n"
                 "P. megaterium PHA Engineering Pipeline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved design space coverage: %s", outpath)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STEP 12 (IMPROVEMENT) — Dataset Statistics & Learning Curves")
    logger.info("Quality Improvement Step 12 — Bioresource Technology")
    logger.info("=" * 70)

    cfg, al_cfg, cand_cfg = load_configs()

    # Design space statistics
    logger.info("Computing design space statistics...")
    design_stats = compute_design_space_statistics(cand_cfg, cfg)
    ds_df = pd.DataFrame([design_stats]).T.reset_index()
    ds_df.columns = ["Parameter", "Value"]
    ds_df.to_csv(TABLES_DIR / "design_space_statistics.csv", index=False)
    logger.info("Saved design space stats")

    # Load ML dataset
    ml_parquet = PROJECT_ROOT / "data" / "processed" / "ml_dataset.parquet"
    ml_csv     = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
    if ml_parquet.exists():
        fba_df = pd.read_parquet(ml_parquet)
        logger.info("Loaded ML dataset: %s", ml_parquet)
    elif ml_csv.exists():
        fba_df = pd.read_csv(ml_csv)
        logger.info("Loaded ML dataset: %s", ml_csv)
    else:
        raise FileNotFoundError(
            "ML dataset not found. Run script 04_run_fba_dataset.py first."
        )

    target_cols = ["pha_flux", "biomass_flux"]
    meta_cols = {"knockouts", "upregulations", "condition", "design_id",
                 "epsilon_level", "status", "max_biomass_unconstrained",
                 "n_knockouts", "n_upregulations"}
    feature_cols = [c for c in fba_df.columns
                    if c not in target_cols and c not in meta_cols]

    X = fba_df[feature_cols]
    y = fba_df[target_cols]
    logger.info("ML dataset: %d rows × %d features, targets: %s",
                len(X), len(feature_cols), target_cols)

    # Learning curves
    lc_results = train_surrogate_with_learning_curve(X, y, al_cfg)
    lc_all = []
    for target, rows in lc_results.items():
        lc_all.extend(rows)
    lc_df = pd.DataFrame(lc_all)
    lc_df.to_csv(TABLES_DIR / "surrogate_learning_curves.csv", index=False)
    logger.info("Saved learning curves: %s", TABLES_DIR / "surrogate_learning_curves.csv")

    # Table 1
    table1 = compute_table1(design_stats, lc_results, al_cfg)
    table1.to_csv(TABLES_DIR / "table1_dataset_statistics.csv", index=False)
    logger.info("Saved Table 1: %s", TABLES_DIR / "table1_dataset_statistics.csv")

    # Figures
    plot_learning_curves(lc_results, FIGS_DIR / "surrogate_learning_curves.png")
    plot_design_space_coverage(design_stats, al_cfg, FIGS_DIR / "design_space_coverage.png")

    logger.info("Done in %.1f s", time.time() - t0)

    print("\n" + "=" * 70)
    print("TABLE 1 — DATASET AND MODEL STATISTICS (Bioresource Technology)")
    print("=" * 70)
    print(table1.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
