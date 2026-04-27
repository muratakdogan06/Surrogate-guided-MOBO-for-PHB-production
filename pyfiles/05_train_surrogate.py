#!/usr/bin/env python3
"""
05_train_surrogate.py — Train ML surrogate models (Step 6).

Outputs
-------
  results/tables/surrogate_metrics.parquet / .csv
  models/surrogate_*.joblib
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_dataframe
from phaopt.train import save_train_results, train_surrogates
from phaopt.utils import load_al_config, setup_logging

logger = setup_logging("pipeline.05")


def main() -> None:
    logger.info("=" * 65)
    logger.info("STEP 6 — Train surrogate models")
    logger.info("=" * 65)

    al_cfg = load_al_config()
    df = load_dataframe("data/processed/ml_dataset.parquet")
    logger.info("Loaded ML dataset: %d × %d", *df.shape)

    results = train_surrogates(
        df,
        n_estimators=al_cfg["n_estimators"],
        max_depth=al_cfg["max_depth"],
        seed=al_cfg["random_seed"],
    )
    save_train_results(results)

    logger.info("\nSurrogate metrics:\n%s", results["metrics"].to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
