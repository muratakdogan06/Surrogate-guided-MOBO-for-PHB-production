#!/usr/bin/env python3
"""
06_run_active_learning_option_c.py — True multi-objective Bayesian optimisation (Step 7).

Seed-aware version for multi-run benchmarking.

Outputs
-------
results/tables/al_history_seed{seed}.csv
results/tables/al_evaluated_seed{seed}.parquet
results/tables/al_best_design_seed{seed}.json
results/tables/random_baseline_seed{seed}.csv
results/tables/greedy_baseline_seed{seed}.csv
results/tables/exploitation_baseline_seed{seed}.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.active_learning import (
    _build_global_pareto_reference,
    run_active_learning,
    run_exploitation_baseline,
    run_greedy_baseline,
    run_random_baseline,
)
from phaopt.io import load_sbml_model, save_dataframe, save_json
from phaopt.perturbation_space import generate_designs
from phaopt.utils import (
    load_al_config,
    load_candidate_reactions,
    load_conditions,
    load_model_config,
    setup_logging,
)

logger = setup_logging("pipeline.06")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true multi-objective Bayesian optimisation with a specific random seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible benchmark runs.",
    )
    return parser.parse_args()


def log_active_learning_config(al_cfg: dict) -> None:
    logger.info("-" * 65)
    logger.info("Active learning configuration")
    logger.info("-" * 65)
    logger.info("initial_random_samples : %s", al_cfg.get("initial_random_samples"))
    logger.info("n_iterations           : %s", al_cfg.get("n_iterations"))
    logger.info("batch_size             : %s", al_cfg.get("batch_size"))

    logger.info("acquisition            : %s", al_cfg.get("acquisition"))
    logger.info("n_mc_samples           : %s", al_cfg.get("n_mc_samples"))
    logger.info("ehvi_ref_point         : %s", al_cfg.get("ehvi_ref_point"))

    logger.info("surrogate_model        : %s", al_cfg.get("surrogate_model"))
    logger.info("n_estimators           : %s", al_cfg.get("n_estimators"))
    logger.info("max_depth              : %s", al_cfg.get("max_depth"))
    logger.info("ensemble_size          : %s", al_cfg.get("ensemble_size"))

    logger.info("diversity_lambda       : %s", al_cfg.get("diversity_lambda"))

    logger.info("condition_name         : %s", al_cfg.get("condition_name"))
    logger.info("biomass_fraction_req   : %s", al_cfg.get("biomass_fraction_required"))

    logger.info("biomass_min_for_acq    : %s", al_cfg.get("biomass_min_for_acq"))
    logger.info("biomass_penalty_lambda : %s", al_cfg.get("biomass_penalty_lambda"))

    logger.info("run_random_baseline    : %s", al_cfg.get("run_random_baseline"))
    logger.info("run_greedy_baseline    : %s", al_cfg.get("run_greedy_baseline"))
    logger.info("run_exploitation_base  : %s", al_cfg.get("run_exploitation_baseline"))

    logger.info("convergence_window     : %s", al_cfg.get("convergence_window"))
    logger.info("convergence_tol        : %s", al_cfg.get("convergence_tol"))

    logger.info("pareto_n_points        : %s", al_cfg.get("pareto_n_points"))
    logger.info("random_seed            : %s", al_cfg.get("random_seed"))
    logger.info("use_global_pareto_ref  : %s", al_cfg.get("use_global_pareto_reference"))
    logger.info("-" * 65)


def main() -> None:
    args = parse_args()
    seed = int(args.seed)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    logger.info("=" * 65)
    logger.info("STEP 7 — True multi-objective Bayesian optimisation")
    logger.info("=" * 65)
    logger.info("Seed: %d", seed)
    logger.info("Config source: configs/active_learning.yaml")

    cfg = load_model_config()
    al_cfg = load_al_config()
    candidates_cfg = load_candidate_reactions()
    conditions_cfg = load_conditions()

    # Override config seed from CLI
    al_cfg["random_seed"] = seed

    log_active_learning_config(al_cfg)

    condition_name = al_cfg.get("condition_name", "base")
    biomass_fraction_required = float(al_cfg.get("biomass_fraction_required", 0.30))
    use_global_pareto_reference = bool(al_cfg.get("use_global_pareto_reference", True))

    model = load_sbml_model(cfg["extended_model_path"])
    rxn_ids = {r.id for r in model.reactions}
    designs = generate_designs(rxn_ids, candidates_cfg)

    cands = sorted({
        rid
        for grp in candidates_cfg["candidate_groups"].values()
        for rid in grp["reaction_ids"]
        if rid in rxn_ids
    })

    global_pareto_set = None
    if use_global_pareto_reference:
        pha_rxn = cfg["pha_reaction_id"]
        bio_rxn = cfg["biomass_reaction_id"]
        upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
        overrides = conditions_cfg["conditions"][condition_name].get("overrides") or {}

        logger.info("Building global Pareto reference set for AL benchmarking …")
        global_pareto_set = _build_global_pareto_reference(
            cobra_model=model,
            all_designs=designs,
            pha_rxn=pha_rxn,
            bio_rxn=bio_rxn,
            overrides=overrides,
            upreg_fold=upreg_fold,
            biomass_fraction_required=biomass_fraction_required,
        )
        logger.info("Global Pareto reference size: %d", len(global_pareto_set))

    logger.info("Condition: %s", condition_name)
    logger.info("Biomass fraction required: %.2f", biomass_fraction_required)
    logger.info("Candidate designs: %d", len(designs))
    logger.info("Candidate reactions: %d", len(cands))
    logger.info("Surrogate: %s", al_cfg.get("surrogate_model", "xgboost"))
    logger.info("Multi-objective strategy: MC-EHVI (Daulton et al. 2020)")

    # ------------------------------------------------------------------
    # Active Learning
    # ------------------------------------------------------------------
    logger.info("Running multi-objective Bayesian optimisation …")
    al_res = run_active_learning(
        model,
        designs,
        cands,
        al_cfg,
        cfg,
        conditions_cfg,
        candidates_cfg,
        global_pareto_set=global_pareto_set,
    )

    # Seed-aware outputs
    al_history_csv = PROJECT_ROOT / f"results/tables/al_history_seed{seed}.csv"
    al_evaluated_base = f"results/tables/al_evaluated_seed{seed}"
    al_best_json = PROJECT_ROOT / f"results/tables/al_best_design_seed{seed}.json"

    pd.DataFrame(al_res["history"]).to_csv(al_history_csv, index=False)
    save_dataframe(al_res["evaluated"], al_evaluated_base)
    save_json(al_res["best"], str(al_best_json))

    logger.info("Saved: %s", al_history_csv)
    logger.info("Saved: %s.parquet", al_evaluated_base)
    logger.info("Saved: %s", al_best_json)
    logger.info("AL best PHA: %.6f", al_res["best"]["pha_flux"])
    logger.info("AL best biomass: %.6f", al_res["best"]["biomass_flux"])

    # history may use either n_evaluated or evaluated depending on implementation
    last_hist = al_res["history"][-1]
    if "n_evaluated" in last_hist:
        n_al = int(last_hist["n_evaluated"])
    else:
        n_al = int(last_hist.get("evaluated", 0))

    # ------------------------------------------------------------------
    # Random baseline
    # ------------------------------------------------------------------
    if al_cfg.get("run_random_baseline", True):
        logger.info("Running random baseline (%d evals) …", n_al)
        rand_res = run_random_baseline(
            model,
            designs,
            n_al,
            cfg,
            conditions_cfg,
            candidates_cfg,
            seed=seed,
            condition_name=condition_name,
            biomass_fraction_required=biomass_fraction_required,
            global_pareto_set=global_pareto_set,
        )

        rand_csv = PROJECT_ROOT / f"results/tables/random_baseline_seed{seed}.csv"
        pd.DataFrame(rand_res["history"]).to_csv(rand_csv, index=False)
        logger.info("Saved: %s", rand_csv)
        logger.info(
            "Random best viable PHA: %.6f",
            max(row["best_viable_pha"] for row in rand_res["history"]),
        )

    # ------------------------------------------------------------------
    # Greedy baseline
    # ------------------------------------------------------------------
    if al_cfg.get("run_greedy_baseline", True):
        logger.info("Running greedy baseline (%d evals) …", n_al)
        greedy_res = run_greedy_baseline(
            model,
            designs,
            cands,
            n_al,
            cfg,
            conditions_cfg,
            candidates_cfg,
            seed=seed,
            condition_name=condition_name,
            biomass_fraction_required=biomass_fraction_required,
            global_pareto_set=global_pareto_set,
        )

        greedy_csv = PROJECT_ROOT / f"results/tables/greedy_baseline_seed{seed}.csv"
        pd.DataFrame(greedy_res["history"]).to_csv(greedy_csv, index=False)
        logger.info("Saved: %s", greedy_csv)
        logger.info(
            "Greedy best viable PHA: %.6f",
            max(row["best_viable_pha"] for row in greedy_res["history"]),
        )

    # ------------------------------------------------------------------
    # Exploitation baseline
    # ------------------------------------------------------------------
    if al_cfg.get("run_exploitation_baseline", True):
        logger.info("Running exploitation baseline (%d evals) …", n_al)
        exploit_res = run_exploitation_baseline(
            model,
            designs,
            cands,
            n_al,
            cfg,
            conditions_cfg,
            candidates_cfg,
            seed=seed,
            condition_name=condition_name,
            biomass_fraction_required=biomass_fraction_required,
            global_pareto_set=global_pareto_set,
        )

        exploit_csv = PROJECT_ROOT / f"results/tables/exploitation_baseline_seed{seed}.csv"
        pd.DataFrame(exploit_res["history"]).to_csv(exploit_csv, index=False)
        logger.info("Saved: %s", exploit_csv)
        logger.info(
            "Exploitation best viable PHA: %.6f",
            max(row["best_viable_pha"] for row in exploit_res["history"]),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()