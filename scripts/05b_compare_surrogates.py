#!/usr/bin/env python3
"""
Step 5b — Compare Multiple ML Surrogate Models

Trains and evaluates 4 model families on the same FBA dataset:
  1. Gradient Boosting (GBR)
  2. Random Forest (RF)
  3. XGBoost
  4. Elastic Net (linear baseline)

Outputs
-------
  results/tables/surrogate_comparison.csv
  results/figures/surrogate_model_comparison.png
  results/figures/surrogate_parity_plots.png
  models/surrogate_<best_family>__<target>.joblib

Usage
-----
    python scripts/05b_compare_surrogates.py
    python scripts/05b_compare_surrogates.py --families gradient_boosting random_forest xgboost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_dataframe
from phaopt.train import (
    compare_surrogates,
    plot_model_comparison,
    plot_parity,
    save_train_results,
)
from phaopt.utils import load_al_config, resolve_path, setup_logging

logger = setup_logging("pipeline.05b")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5b: Compare ML surrogates")
    parser.add_argument(
        "--families", nargs="+", default=None,
        help="Model families to compare (default: all available)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STEP 5b — Compare Multiple ML Surrogate Models")
    logger.info("=" * 70)

    # Load data
    df = load_dataframe("data/processed/ml_dataset.parquet")
    logger.info("Loaded ML dataset: %d x %d", *df.shape)

    # Run comparison
    results = compare_surrogates(
        df,
        families=args.families,
        seed=args.seed,
        test_size=args.test_size,
        n_cv_splits=args.n_cv,
    )

    # Save results
    save_train_results(results)

    # Generate figures
    fig_dir = resolve_path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(
        results["comparison"],
        save_path=str(fig_dir / "surrogate_model_comparison.png"),
    )

    plot_parity(
        results["comparison"],
        df,
        top_n=4,
        seed=args.seed,
        test_size=args.test_size,
        save_path=str(fig_dir / "surrogate_parity_plots.png"),
        train_idx=results.get("train_idx"),
        test_idx=results.get("test_idx"),
    )

    # Print summary
    comp = results["comparison"]
    print("\n" + "=" * 90)
    print("SURROGATE MODEL COMPARISON")
    print("=" * 90)
    print(comp[["model_family", "target", "R2_test", "MAE_test", "RMSE_test",
                "R2_cv", "train_time_s"]].to_string(index=False))
    print("=" * 90)

    print("\nBEST MODEL PER TARGET:")
    for target, family in results["best_family"].items():
        row = comp[(comp["target"] == target) & (comp["model_family"] == family)].iloc[0]
        print(f"  {target}: {family}  (R²={row['R2_test']:.4f}, MAE={row['MAE_test']:.6f})")
    print("=" * 90)


if __name__ == "__main__":
    main()
