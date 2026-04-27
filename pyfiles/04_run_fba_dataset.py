#!/usr/bin/env python3
"""
04_run_fba_dataset.py — FBA simulations → ML dataset (Steps 4 + 5).

Outputs
-------
  data/processed/fba_results.parquet / .csv
  data/processed/ml_dataset.parquet  / .csv

Now supports multi-threshold epsilon-constraint simulation:
  design × condition × biomass_fraction_required
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.dataset import build_ml_dataset, save_ml_dataset
from phaopt.io import load_sbml_model, save_dataframe
from phaopt.perturbation_space import generate_designs
from phaopt.simulation import run_all_simulations
from phaopt.utils import (
    load_candidate_reactions,
    load_conditions,
    load_model_config,
    setup_logging,
)

logger = setup_logging("pipeline.04")


def main() -> None:
    logger.info("=" * 65)
    logger.info("STEP 4/5 — FBA simulations & ML dataset")
    logger.info("=" * 65)

    cfg = load_model_config()
    model = load_sbml_model(cfg["extended_model_path"])
    candidates_cfg = load_candidate_reactions()
    conditions_cfg = load_conditions()

    biomass_fractions = cfg.get("biomass_fraction_grid", [0.10, 0.30, 0.50, 0.70])
    logger.info("Biomass-fraction grid: %s", biomass_fractions)
    logger.info("PHA target: %s", cfg["pha_reaction_id"])
    logger.info("Biomass reaction (configured): %s", cfg["biomass_reaction_id"])

    designs = generate_designs({r.id for r in model.reactions}, candidates_cfg)
    logger.info("Generated %d designs for simulation.", len(designs))

    sim_df = run_all_simulations(
        model=model,
        designs=designs,
        cfg=cfg,
        conditions_cfg=conditions_cfg,
        candidates_cfg=candidates_cfg,
    )
    save_dataframe(sim_df, "data/processed/fba_results")
    logger.info("FBA results: %d rows", len(sim_df))

    ml_df = build_ml_dataset(sim_df)
    save_ml_dataset(ml_df)
    logger.info("ML dataset: %d rows × %d cols", *ml_df.shape)

    if "pha_flux" in ml_df.columns and "biomass_flux" in ml_df.columns:
        logger.info(
            "Target diversity — pha_flux unique: %d | biomass_flux unique: %d",
            ml_df["pha_flux"].nunique(),
            ml_df["biomass_flux"].nunique(),
        )
        logger.info(
            "PHA range: [%.6f, %.6f] | Biomass range: [%.6f, %.6f]",
            float(ml_df["pha_flux"].min()),
            float(ml_df["pha_flux"].max()),
            float(ml_df["biomass_flux"].min()),
            float(ml_df["biomass_flux"].max()),
        )


if __name__ == "__main__":
    main()