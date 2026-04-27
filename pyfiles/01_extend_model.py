#!/usr/bin/env python3
"""
01_extend_model.py — Add PHA objective reaction to the GEM.

Outputs
-------
  models/model_with_PHA.xml
  models/model_extension_report.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.model_extension import extend_model
from phaopt.utils import load_model_config, setup_logging

logger = setup_logging("pipeline.01")


def main() -> None:
    logger.info("=" * 65)
    logger.info("STEP 1 — Extend model with PHA objective")
    logger.info("=" * 65)

    cfg = load_model_config()
    model, report = extend_model(cfg=cfg, export=True)

    logger.info("  Precursor : %s", report["precursor_metabolite_id"])
    logger.info("  Reaction  : %s — %s", report["pha_reaction_id"],
                report["pha_reaction_string"])
    logger.info("  Objective : %s", report["objective_reaction"])
    logger.info("  Bounds    : [%.1f, %.1f]", report["lower_bound"],
                report["upper_bound"])
    logger.info("  Exported  : %s", report.get("exported_model", "n/a"))
    logger.info("Done.")


if __name__ == "__main__":
    main()
