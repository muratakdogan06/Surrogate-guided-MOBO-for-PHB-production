"""
phaopt.model_extension — Add PHA objective reaction to a genome-scale model.

Functions
---------
extend_model : Add PHA demand reaction and return (model, report).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from phaopt.utils import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def extend_model(
    cfg: Dict[str, Any],
    export: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Add a PHA demand/sink reaction to the GEM and optionally export.

    Parameters
    ----------
    cfg : dict
        Model configuration (from ``load_model_config()``).
    export : bool
        If *True*, write the extended model to ``cfg["extended_model_path"]``.

    Returns
    -------
    model : cobra.Model
        Extended COBRA model with the PHA reaction added.
    report : dict
        Summary dict with keys:
        ``precursor_metabolite_id``, ``pha_reaction_id``,
        ``pha_reaction_string``, ``objective_reaction``,
        ``lower_bound``, ``upper_bound``, ``exported_model``.
    """
    import cobra

    # Load the base model
    model_path = resolve_path(cfg["model_path"])
    logger.info("Loading base model: %s", model_path)
    model = cobra.io.read_sbml_model(str(model_path))

    # --- Find PHA metabolite in model --------------------------------------
    pha_met_id = cfg.get("pha_metabolite_id", "POLHYBU[c]")
    pha_met = None
    for mid in [pha_met_id, pha_met_id.replace("[", "_").replace("]", "")]:
        try:
            pha_met = model.metabolites.get_by_id(mid)
            break
        except KeyError:
            continue

    if pha_met is None:
        pha_met = cobra.Metabolite(
            id=pha_met_id,
            name=cfg.get("pha_metabolite_name", "Polyhydroxybutyrate"),
            compartment=cfg.get("pha_compartment", "c"),
        )
        logger.info("Created PHA metabolite: %s", pha_met.id)
    else:
        logger.info("Found PHA metabolite in model: %s (%s)", pha_met.id, pha_met.name)

    # --- Create PHA demand reaction (sink) --------------------------------
    pha_rxn_id = cfg.get("pha_reaction_id", "DM_POLHYBU_c")

    if pha_rxn_id in {r.id for r in model.reactions}:
        logger.info("PHA reaction %s already exists; skipping addition.", pha_rxn_id)
        pha_rxn = model.reactions.get_by_id(pha_rxn_id)
    else:
        pha_rxn = cobra.Reaction(pha_rxn_id)
        pha_rxn.name = "PHA demand (sink for PHB polymer)"
        pha_rxn.lower_bound = cfg.get("pha_reaction_lower_bound", 0.0)
        pha_rxn.upper_bound = cfg.get("pha_reaction_upper_bound", 1000.0)

        # Demand reaction: POLHYBU[c] -->  (removes PHB from system)
        pha_rxn.add_metabolites({pha_met: -1.0})
        model.add_reactions([pha_rxn])
        logger.info("Added PHA demand reaction: %s  (%s)", pha_rxn_id, pha_rxn.reaction)

    # --- Build report dict -------------------------------------------------
    report: Dict[str, Any] = {
        "precursor_metabolite_id": pha_met.id,
        "pha_reaction_id": pha_rxn.id,
        "pha_reaction_string": pha_rxn.reaction,
        "objective_reaction": pha_rxn.id,
        "lower_bound": pha_rxn.lower_bound,
        "upper_bound": pha_rxn.upper_bound,
    }

    # --- Export ------------------------------------------------------------
    if export:
        out_path = resolve_path(cfg.get("extended_model_path", "models/model_with_PHA.xml"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cobra.io.write_sbml_model(model, str(out_path))
        report["exported_model"] = str(out_path)
        logger.info("Exported extended model to %s", out_path)

    return model, report
