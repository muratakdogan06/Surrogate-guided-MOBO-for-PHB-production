#!/usr/bin/env python3
"""
02_validate_model.py — Validate the extended GEM.

Outputs
-------
  results/tables/validation_report.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_sbml_model
from phaopt.model_validation import save_validation_report, validate_model
from phaopt.utils import load_model_config, resolve_path, setup_logging

logger = setup_logging("pipeline.02")


def main() -> None:
    logger.info("=" * 65)
    logger.info("STEP 2 — Validate extended model")
    logger.info("=" * 65)

    cfg = load_model_config()
    model = load_sbml_model(cfg["extended_model_path"])

    report = validate_model(model, cfg)
    save_validation_report(report)

    for name, info in report["checks"].items():
        if "found" in info:
            tag = "OK" if info["found"] else "MISSING"
        elif "added" in info:
            tag = "ADDED" if info["added"] else "NOT_ADDED"
        else:
            tag = "INFO"
        logger.info("  [%s] %-20s %s", tag, name, info.get("id"))

    bfba = report["baseline_fba"]
    logger.info("  Max biomass : %s", bfba.get("max_biomass"))
    logger.info("  Max PHA     : %s", bfba.get("max_pha"))
    logger.info("  Blocked rxns: %s", report["blocked_reactions"]["count"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
