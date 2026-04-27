#!/usr/bin/env python3
"""
03_generate_design_space.py — Combinatorial metabolic designs.

Outputs
-------
  data/processed/design_space.parquet / .csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_sbml_model
from phaopt.perturbation_space import generate_designs, save_designs
from phaopt.utils import load_candidate_reactions, load_model_config, setup_logging

logger = setup_logging("pipeline.03")


def main() -> None:
    logger.info("=" * 65)
    logger.info("STEP 3 — Generate metabolic design space")
    logger.info("=" * 65)

    cfg = load_model_config()
    model = load_sbml_model(cfg["extended_model_path"])
    rxn_ids = {r.id for r in model.reactions}

    candidates_cfg = load_candidate_reactions()
    designs = generate_designs(rxn_ids, candidates_cfg)
    df = save_designs(designs)
    logger.info("Saved %d designs.", len(df))


if __name__ == "__main__":
    main()
