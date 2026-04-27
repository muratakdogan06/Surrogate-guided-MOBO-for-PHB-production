"""
phaopt.model_validation — Validate the extended GEM.

Functions
---------
validate_model        : Run structural checks and baseline FBA.
save_validation_report: Write the report dict to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from phaopt.utils import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def validate_model(model, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the extended COBRA model.

    Checks performed:
    1. PHA reaction presence
    2. Biomass reaction presence
    3. Glucose exchange presence
    4. Oxygen exchange presence
    5. Baseline FBA for max biomass and max PHA
    6. Blocked-reaction count (via FVA where possible)

    Parameters
    ----------
    model : cobra.Model
    cfg : dict
        Model configuration (from ``load_model_config()``).

    Returns
    -------
    dict
        Validation report with keys ``checks``, ``baseline_fba``,
        ``blocked_reactions``.
    """
    import cobra

    rxn_ids = {r.id for r in model.reactions}
    met_ids = {m.id for m in model.metabolites}

    checks: Dict[str, Dict[str, Any]] = {}

    # --- PHA reaction ---
    pha_rxn_id = cfg.get("pha_reaction_id", "DM_POLHYBU_c")
    checks["pha_reaction"] = {
        "id": pha_rxn_id,
        "found": pha_rxn_id in rxn_ids,
    }

    # --- Biomass ---
    bio_rxn_id = cfg.get("biomass_reaction_id", "biomass")
    bio_found = bio_rxn_id in rxn_ids
    if not bio_found:
        # Try search patterns
        for pattern in cfg.get("biomass_search_patterns", []):
            matches = [r.id for r in model.reactions if pattern.lower() in r.id.lower()]
            if matches:
                bio_rxn_id = matches[0]
                bio_found = True
                break
    checks["biomass_reaction"] = {"id": bio_rxn_id, "found": bio_found}

    # --- Glucose exchange ---
    glc_id = cfg.get("glucose_exchange_id", "Re_bm01273")
    glc_found = glc_id in rxn_ids
    if not glc_found:
        for pattern in cfg.get("glucose_exchange_patterns", []):
            if pattern in rxn_ids:
                glc_id = pattern
                glc_found = True
                break
    checks["glucose_exchange"] = {"id": glc_id, "found": glc_found}

    # --- Oxygen exchange ---
    o2_id = cfg.get("oxygen_exchange_id", "Re_bm01302")
    o2_found = o2_id in rxn_ids
    if not o2_found:
        for pattern in cfg.get("oxygen_exchange_patterns", []):
            if pattern in rxn_ids:
                o2_id = pattern
                o2_found = True
                break
    checks["oxygen_exchange"] = {"id": o2_id, "found": o2_found}

    # --- Baseline FBA: max biomass ---
    baseline_fba: Dict[str, Any] = {}
    try:
        with model as m:
            m.objective = bio_rxn_id
            sol = m.optimize()
            baseline_fba["max_biomass"] = round(sol.objective_value, 6) if sol.status == "optimal" else None
    except Exception as exc:
        baseline_fba["max_biomass"] = None
        logger.warning("Biomass FBA failed: %s", exc)

    # --- Baseline FBA: max PHA ---
    try:
        with model as m:
            m.objective = pha_rxn_id
            sol = m.optimize()
            baseline_fba["max_pha"] = round(sol.objective_value, 6) if sol.status == "optimal" else None
    except Exception as exc:
        baseline_fba["max_pha"] = None
        logger.warning("PHA FBA failed: %s", exc)

    # --- Blocked reactions (quick heuristic via FVA on a small subset) ---
    blocked_count = 0
    try:
        from cobra.flux_analysis import find_blocked_reactions
        blocked = find_blocked_reactions(model, open_exchanges=False)
        blocked_count = len(blocked)
    except Exception:
        # Fallback: count reactions with zero bounds
        blocked_count = sum(
            1 for r in model.reactions
            if r.lower_bound == 0 and r.upper_bound == 0
        )

    report = {
        "checks": checks,
        "baseline_fba": baseline_fba,
        "blocked_reactions": {"count": blocked_count},
    }
    return report


def save_validation_report(report: Dict[str, Any], path: str | None = None) -> Path:
    """
    Save the validation report to JSON.

    Parameters
    ----------
    report : dict
    path : str, optional
        Output path.  Defaults to ``results/tables/validation_report.json``.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    if path is None:
        path = "results/tables/validation_report.json"

    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Saved validation report: %s", resolved)
    return resolved
