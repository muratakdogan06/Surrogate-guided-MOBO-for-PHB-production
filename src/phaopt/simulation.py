"""
phaopt.simulation — FBA simulations across designs, conditions, and epsilon levels.

Functions
---------
run_all_simulations : Simulate all designs under all conditions with
                      multi-threshold epsilon-constraint FBA.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)


def _apply_overrides(model, overrides: Dict[str, Any]) -> None:
    """Apply exchange-reaction bound overrides to a COBRA model context."""
    rxn_ids = {r.id for r in model.reactions}
    for rxn_id, (lb, ub) in overrides.items():
        if rxn_id in rxn_ids:
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = lb
            rxn.upper_bound = ub
        else:
            logger.debug("Override reaction %s not found in model; skipping.", rxn_id)


def _apply_design(
    model,
    design: Dict[str, Any],
    upreg_fold: float,
    candidates_cfg: Dict[str, Any],
) -> None:
    """Apply knockouts and upregulations to a COBRA model context."""
    rxn_ids = {r.id for r in model.reactions}

    for rid in design.get("knockouts", []):
        if rid in rxn_ids:
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = 0.0
            rxn.upper_bound = 0.0

    for rid in design.get("upregulations", []):
        if rid in rxn_ids:
            rxn = model.reactions.get_by_id(rid)
            # Upregulate by widening bounds by fold_change
            if rxn.upper_bound > 0:
                rxn.upper_bound *= upreg_fold
            if rxn.lower_bound < 0:
                rxn.lower_bound *= upreg_fold


def simulate_single(
    model,
    design: Dict[str, Any],
    condition_name: str,
    overrides: Dict[str, Any],
    biomass_fraction: float,
    cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single epsilon-constraint FBA:

    1. Fix biomass >= biomass_fraction * max_biomass.
    2. Maximise PHA flux.

    Returns a result dict.
    """
    pha_rxn_id = cfg["pha_reaction_id"]
    bio_rxn_id = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]

    with model as m:
        # Apply condition overrides
        _apply_overrides(m, overrides)
        # Apply design perturbations
        _apply_design(m, design, upreg_fold, candidates_cfg)

        # Step 1: find max biomass under this design + condition
        m.objective = bio_rxn_id
        sol_bio = m.optimize()
        max_biomass = sol_bio.objective_value if sol_bio.status == "optimal" else 0.0

        # Step 2: constrain biomass >= fraction * max
        bio_rxn = m.reactions.get_by_id(bio_rxn_id)
        bio_rxn.lower_bound = biomass_fraction * max_biomass

        # Step 3: maximise PHA
        m.objective = pha_rxn_id
        sol_pha = m.optimize()

        if sol_pha.status == "optimal":
            pha_flux = sol_pha.objective_value
            biomass_flux = sol_pha.fluxes.get(bio_rxn_id, 0.0)
        else:
            pha_flux = 0.0
            biomass_flux = 0.0

    return {
        "condition": condition_name,
        "biomass_fraction_required": biomass_fraction,
        "knockouts": "|".join(design.get("knockouts", [])),
        "upregulations": "|".join(design.get("upregulations", [])),
        "n_knockouts": len(design.get("knockouts", [])),
        "n_upregulations": len(design.get("upregulations", [])),
        "max_biomass_unconstrained": round(max_biomass, 8),
        "pha_flux": round(pha_flux, 8),
        "biomass_flux": round(biomass_flux, 8),
        "status": sol_pha.status if sol_pha else "infeasible",
    }


def run_all_simulations(
    model,
    designs: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    conditions_cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Run epsilon-constraint FBA for every (design x condition x biomass_fraction).

    Parameters
    ----------
    model : cobra.Model
    designs : list[dict]
        Each dict has ``knockouts`` and ``upregulations`` lists.
    cfg : dict
        Model config (from ``load_model_config()``).
    conditions_cfg : dict
        Conditions config (from ``load_conditions()``).
    candidates_cfg : dict
        Candidate reactions config (from ``load_candidate_reactions()``).

    Returns
    -------
    pd.DataFrame
        One row per (design, condition, biomass_fraction).
    """
    biomass_fractions = cfg.get("biomass_fraction_grid", [0.10, 0.30, 0.50, 0.70])
    conditions = conditions_cfg.get("conditions", {"base": {"overrides": {}}})

    total = len(designs) * len(conditions) * len(biomass_fractions)
    logger.info(
        "Starting FBA simulations: %d designs x %d conditions x %d epsilon levels = %d total",
        len(designs),
        len(conditions),
        len(biomass_fractions),
        total,
    )

    results: List[Dict[str, Any]] = []
    done = 0

    for cond_name, cond_data in conditions.items():
        overrides = cond_data.get("overrides") or {}
        for bf in biomass_fractions:
            for design in designs:
                try:
                    row = simulate_single(
                        model, design, cond_name, overrides, bf, cfg, candidates_cfg
                    )
                    results.append(row)
                except Exception as exc:
                    results.append({
                        "condition": cond_name,
                        "biomass_fraction_required": bf,
                        "knockouts": "|".join(design.get("knockouts", [])),
                        "upregulations": "|".join(design.get("upregulations", [])),
                        "n_knockouts": len(design.get("knockouts", [])),
                        "n_upregulations": len(design.get("upregulations", [])),
                        "max_biomass_unconstrained": 0.0,
                        "pha_flux": 0.0,
                        "biomass_flux": 0.0,
                        "status": f"error: {exc}",
                    })
                done += 1
                if done % 500 == 0:
                    logger.info("  Progress: %d / %d (%.1f%%)", done, total, 100 * done / total)

    df = pd.DataFrame(results)
    logger.info("Simulations complete: %d rows", len(df))
    return df
