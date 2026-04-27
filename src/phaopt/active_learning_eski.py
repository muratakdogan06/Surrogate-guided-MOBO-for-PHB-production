"""
phaopt.active_learning — Multi-objective Bayesian optimisation via active learning.

Functions
---------
run_active_learning          : Main AL loop (UCB acquisition, diversity, biomass penalty).
run_random_baseline          : Random-sampling baseline.
run_greedy_baseline          : Greedy single-objective baseline.
run_exploitation_baseline    : Pure-exploitation (no exploration) baseline.
_build_global_pareto_reference : Pre-compute Pareto-optimal reference set from all designs.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_design(
    model,
    design: Dict[str, Any],
    pha_rxn: str,
    bio_rxn: str,
    overrides: Dict[str, Any],
    upreg_fold: float,
    biomass_fraction_required: float,
) -> Dict[str, float]:
    """
    Run an epsilon-constraint FBA for a single design.

    Returns dict with ``pha_flux`` and ``biomass_flux``.
    """
    with model as m:
        rxn_ids = {r.id for r in m.reactions}

        # Apply condition overrides
        for rxn_id, bounds in (overrides or {}).items():
            if rxn_id in rxn_ids:
                rxn = m.reactions.get_by_id(rxn_id)
                rxn.lower_bound, rxn.upper_bound = bounds[0], bounds[1]

        # Apply knockouts
        for rid in design.get("knockouts", []):
            if rid in rxn_ids:
                rxn = m.reactions.get_by_id(rid)
                rxn.lower_bound = 0.0
                rxn.upper_bound = 0.0

        # Apply upregulations
        for rid in design.get("upregulations", []):
            if rid in rxn_ids:
                rxn = m.reactions.get_by_id(rid)
                if rxn.upper_bound > 0:
                    rxn.upper_bound *= upreg_fold
                if rxn.lower_bound < 0:
                    rxn.lower_bound *= upreg_fold

        # Step 1: max biomass
        m.objective = bio_rxn
        sol = m.optimize()
        max_bio = sol.objective_value if sol.status == "optimal" else 0.0

        # Step 2: constrain biomass
        m.reactions.get_by_id(bio_rxn).lower_bound = biomass_fraction_required * max_bio

        # Step 3: max PHA
        m.objective = pha_rxn
        sol2 = m.optimize()
        if sol2.status == "optimal":
            return {
                "pha_flux": sol2.objective_value,
                "biomass_flux": sol2.fluxes.get(bio_rxn, 0.0),
            }
        return {"pha_flux": 0.0, "biomass_flux": 0.0}


def _design_to_vector(
    design: Dict[str, Any],
    all_candidates: List[str],
) -> np.ndarray:
    """Encode a design as a binary vector over all candidate reactions."""
    ko_set = set(design.get("knockouts", []))
    up_set = set(design.get("upregulations", []))
    vec = np.zeros(len(all_candidates) * 2, dtype=np.float64)
    for i, rid in enumerate(all_candidates):
        if rid in ko_set:
            vec[i] = 1.0
        if rid in up_set:
            vec[len(all_candidates) + i] = 1.0
    return vec


def _compute_hypervolume_2d(
    points: np.ndarray,
    ref: np.ndarray,
) -> float:
    """Compute 2-D hypervolume (dominated area) relative to a reference point."""
    if len(points) == 0:
        return 0.0
    # Filter out dominated & below-ref points
    valid = points[(points[:, 0] > ref[0]) & (points[:, 1] > ref[1])]
    if len(valid) == 0:
        return 0.0
    # Sort by first objective descending
    sorted_pts = valid[valid[:, 0].argsort()[::-1]]
    hv = 0.0
    prev_y = ref[1]
    for pt in sorted_pts:
        if pt[1] > prev_y:
            hv += (pt[0] - ref[0]) * (pt[1] - prev_y)
            prev_y = pt[1]
    return float(hv)


def _pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Extract the 2-D Pareto front (maximisation on both objectives)."""
    if len(points) == 0:
        return points
    sorted_pts = points[points[:, 0].argsort()[::-1]]
    front = [sorted_pts[0]]
    for pt in sorted_pts[1:]:
        if pt[1] > front[-1][1]:
            front.append(pt)
    return np.array(front)


def _count_pareto_discovered(
    evaluated_points: np.ndarray,
    global_pareto: Optional[np.ndarray],
    tol: float = 1e-4,
) -> int:
    """Count how many global Pareto points have been discovered."""
    if global_pareto is None or len(global_pareto) == 0:
        return 0
    count = 0
    for gp in global_pareto:
        dists = np.linalg.norm(evaluated_points - gp, axis=1)
        if dists.min() < tol:
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Global Pareto reference
# ═══════════════════════════════════════════════════════════════════════════

def _build_global_pareto_reference(
    cobra_model,
    all_designs: List[Dict[str, Any]],
    pha_rxn: str,
    bio_rxn: str,
    overrides: Dict[str, Any],
    upreg_fold: float,
    biomass_fraction_required: float,
    max_designs: int = 500,
) -> np.ndarray:
    """
    Pre-compute the Pareto-optimal reference set by evaluating a subset
    of designs via FBA.

    Parameters
    ----------
    cobra_model : cobra.Model
    all_designs : list[dict]
    pha_rxn, bio_rxn : str
    overrides : dict
    upreg_fold : float
    biomass_fraction_required : float
    max_designs : int
        Maximum number of designs to evaluate for the reference.

    Returns
    -------
    np.ndarray, shape (n_pareto, 2)
        Columns are [pha_flux, biomass_flux].
    """
    rng = np.random.default_rng(42)
    if len(all_designs) > max_designs:
        indices = rng.choice(len(all_designs), size=max_designs, replace=False)
        subset = [all_designs[i] for i in indices]
    else:
        subset = all_designs

    points = []
    for design in subset:
        try:
            res = _simulate_design(
                cobra_model, design, pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction_required,
            )
            points.append([res["pha_flux"], res["biomass_flux"]])
        except Exception:
            continue

    if not points:
        return np.empty((0, 2))

    pts = np.array(points)
    front = _pareto_front_2d(pts)
    return front


# ═══════════════════════════════════════════════════════════════════════════
# Active Learning (UCB + diversity + biomass penalty)
# ═══════════════════════════════════════════════════════════════════════════

def run_active_learning(
    model,
    designs: List[Dict[str, Any]],
    candidate_reaction_ids: List[str],
    al_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
    conditions_cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
    global_pareto_set: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run the multi-objective Bayesian active learning loop.

    Parameters
    ----------
    model : cobra.Model
    designs : list[dict]
    candidate_reaction_ids : list[str]
    al_cfg : dict
    cfg : dict
    conditions_cfg : dict
    candidates_cfg : dict
    global_pareto_set : np.ndarray, optional

    Returns
    -------
    dict
        ``history`` (list of dicts per iteration),
        ``evaluated`` (DataFrame of all evaluated designs),
        ``best`` (dict with best design info).
    """
    seed = al_cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    n_initial = al_cfg.get("initial_random_samples", 5)
    n_iterations = al_cfg.get("n_iterations", 20)
    batch_size = al_cfg.get("batch_size", 5)
    n_estimators = al_cfg.get("n_estimators", 200)
    max_depth = al_cfg.get("max_depth", 8)
    ensemble_size = al_cfg.get("ensemble_size", 5)
    kappa = al_cfg.get("ucb_kappa", 2.5)
    diversity_lambda = al_cfg.get("diversity_lambda", 0.35)
    biomass_min = al_cfg.get("biomass_min_for_acq", 0.07)
    biomass_penalty = al_cfg.get("biomass_penalty_lambda", 15.0)
    convergence_window = al_cfg.get("convergence_window", 10)
    convergence_tol = al_cfg.get("convergence_tol", 0.0002)

    condition_name = al_cfg.get("condition_name", "base")
    biomass_fraction = float(al_cfg.get("biomass_fraction_required", 0.30))
    pha_rxn = cfg["pha_reaction_id"]
    bio_rxn = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides = conditions_cfg["conditions"][condition_name].get("overrides") or {}

    cands = sorted(candidate_reaction_ids)

    # Vectorise all designs
    all_vecs = np.array([_design_to_vector(d, cands) for d in designs])

    # --- Initial random sampling -------------------------------------------
    n_total = len(designs)
    initial_indices = list(rng.choice(n_total, size=min(n_initial, n_total), replace=False))

    evaluated_indices: List[int] = []
    evaluated_pha: List[float] = []
    evaluated_bio: List[float] = []

    for idx in initial_indices:
        try:
            res = _simulate_design(
                model, designs[idx], pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction,
            )
        except Exception:
            res = {"pha_flux": 0.0, "biomass_flux": 0.0}
        evaluated_indices.append(idx)
        evaluated_pha.append(res["pha_flux"])
        evaluated_bio.append(res["biomass_flux"])

    history: List[Dict[str, Any]] = []
    ref_point = np.array([0.0, 0.0])

    # --- AL iterations -----------------------------------------------------
    for iteration in range(n_iterations):
        X_train = all_vecs[evaluated_indices]
        y_pha = np.array(evaluated_pha)
        y_bio = np.array(evaluated_bio)

        # Train ensemble of surrogates
        pha_models = []
        bio_models = []
        for e in range(ensemble_size):
            m_pha = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=seed + e,
            )
            m_bio = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=seed + 100 + e,
            )
            m_pha.fit(X_train, y_pha)
            m_bio.fit(X_train, y_bio)
            pha_models.append(m_pha)
            bio_models.append(m_bio)

        # Predict on unevaluated designs
        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            logger.info("All designs evaluated; stopping.")
            break

        X_cand = all_vecs[unevaluated]
        pha_preds = np.array([m.predict(X_cand) for m in pha_models])
        bio_preds = np.array([m.predict(X_cand) for m in bio_models])

        pha_mean = pha_preds.mean(axis=0)
        pha_std = pha_preds.std(axis=0)
        bio_mean = bio_preds.mean(axis=0)

        # --- UCB acquisition with biomass penalty --------------------------
        scores = pha_mean + kappa * pha_std
        # Penalise designs with predicted biomass below threshold
        below_threshold = bio_mean < biomass_min
        scores -= biomass_penalty * below_threshold.astype(float)

        # --- Diversity-aware batch selection --------------------------------
        selected_rel: List[int] = []
        for _ in range(min(batch_size, len(unevaluated))):
            if selected_rel and diversity_lambda > 0:
                sel_vecs = X_cand[selected_rel]
                dists = np.array([
                    np.sum(np.abs(X_cand - sv), axis=1) for sv in sel_vecs
                ]).min(axis=0)
                diversity_bonus = diversity_lambda * dists / (X_cand.shape[1] + 1e-8)
                eff_scores = scores + diversity_bonus
            else:
                eff_scores = scores.copy()

            # Mask already-selected
            for si in selected_rel:
                eff_scores[si] = -np.inf

            selected_rel.append(int(np.argmax(eff_scores)))

        # Convert relative indices to absolute
        selected_abs = [unevaluated[i] for i in selected_rel]

        # --- Evaluate selected designs via FBA -----------------------------
        for idx in selected_abs:
            try:
                res = _simulate_design(
                    model, designs[idx], pha_rxn, bio_rxn,
                    overrides, upreg_fold, biomass_fraction,
                )
            except Exception:
                res = {"pha_flux": 0.0, "biomass_flux": 0.0}
            evaluated_indices.append(idx)
            evaluated_pha.append(res["pha_flux"])
            evaluated_bio.append(res["biomass_flux"])

        # --- Compute metrics -----------------------------------------------
        eval_points = np.column_stack([evaluated_pha, evaluated_bio])
        hv = _compute_hypervolume_2d(eval_points, ref_point)

        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable_pha = float(np.max(np.array(evaluated_pha)[viable_mask])) if viable_mask.any() else 0.0

        pareto_discovered = _count_pareto_discovered(eval_points, global_pareto_set)

        history.append({
            "iteration": iteration + 1,
            "n_evaluated": len(evaluated_indices),
            "hypervolume": round(hv, 6),
            "best_viable_pha": round(best_viable_pha, 6),
            "pareto_discovered": pareto_discovered,
            "batch_mean_pha": round(float(np.mean([evaluated_pha[-batch_size:]])), 6),
        })

        logger.info(
            "  Iter %d | evaluated=%d | HV=%.4f | best_viable_pha=%.4f | pareto=%d",
            iteration + 1, len(evaluated_indices), hv, best_viable_pha, pareto_discovered,
        )

        # --- Convergence check ---------------------------------------------
        if len(history) >= convergence_window:
            recent_hv = [h["hypervolume"] for h in history[-convergence_window:]]
            if max(recent_hv) - min(recent_hv) < convergence_tol:
                logger.info("Converged after %d iterations.", iteration + 1)
                break

    # --- Build outputs -----------------------------------------------------
    eval_rows = []
    for i, idx in enumerate(evaluated_indices):
        d = designs[idx]
        eval_rows.append({
            "design_index": idx,
            "knockouts": "|".join(d.get("knockouts", [])),
            "upregulations": "|".join(d.get("upregulations", [])),
            "pha_flux": evaluated_pha[i],
            "biomass_flux": evaluated_bio[i],
        })
    evaluated_df = pd.DataFrame(eval_rows)

    # Best viable design
    viable_mask = np.array(evaluated_bio) >= biomass_min
    if viable_mask.any():
        viable_pha = np.array(evaluated_pha)
        viable_pha[~viable_mask] = -np.inf
        best_idx_local = int(np.argmax(viable_pha))
        best_design_idx = evaluated_indices[best_idx_local]
        best = {
            "design_index": best_design_idx,
            "knockouts": designs[best_design_idx].get("knockouts", []),
            "upregulations": designs[best_design_idx].get("upregulations", []),
            "pha_flux": evaluated_pha[best_idx_local],
            "biomass_flux": evaluated_bio[best_idx_local],
        }
    else:
        best = {"design_index": -1, "knockouts": [], "upregulations": [],
                "pha_flux": 0.0, "biomass_flux": 0.0}

    return {"history": history, "evaluated": evaluated_df, "best": best}


# ═══════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════

def run_random_baseline(
    model,
    designs: List[Dict[str, Any]],
    n_evaluations: int,
    cfg: Dict[str, Any],
    conditions_cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
    seed: int = 0,
    condition_name: str = "base",
    biomass_fraction_required: float = 0.30,
    global_pareto_set: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Random-sampling baseline with the same evaluation budget as AL."""
    rng = np.random.default_rng(seed)
    pha_rxn = cfg["pha_reaction_id"]
    bio_rxn = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = cfg.get("biomass_min_for_acq", 0.07) if "biomass_min_for_acq" in cfg else 0.07

    indices = list(rng.choice(len(designs), size=min(n_evaluations, len(designs)), replace=False))

    evaluated_pha, evaluated_bio = [], []
    history = []
    ref_point = np.array([0.0, 0.0])

    for step, idx in enumerate(indices):
        try:
            res = _simulate_design(
                model, designs[idx], pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction_required,
            )
        except Exception:
            res = {"pha_flux": 0.0, "biomass_flux": 0.0}
        evaluated_pha.append(res["pha_flux"])
        evaluated_bio.append(res["biomass_flux"])

        eval_pts = np.column_stack([evaluated_pha, evaluated_bio])
        hv = _compute_hypervolume_2d(eval_pts, ref_point)
        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable = float(np.max(np.array(evaluated_pha)[viable_mask])) if viable_mask.any() else 0.0
        pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

        history.append({
            "iteration": step + 1,
            "n_evaluated": step + 1,
            "hypervolume": round(hv, 6),
            "best_viable_pha": round(best_viable, 6),
            "pareto_discovered": pareto_disc,
        })

    return {"history": history}


def run_greedy_baseline(
    model,
    designs: List[Dict[str, Any]],
    candidate_reaction_ids: List[str],
    n_evaluations: int,
    cfg: Dict[str, Any],
    conditions_cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
    seed: int = 0,
    condition_name: str = "base",
    biomass_fraction_required: float = 0.30,
    global_pareto_set: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Greedy baseline: train a surrogate, always pick the top-predicted
    (no exploration bonus), re-train, repeat.
    """
    rng = np.random.default_rng(seed)
    pha_rxn = cfg["pha_reaction_id"]
    bio_rxn = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = 0.07

    cands = sorted(candidate_reaction_ids)
    all_vecs = np.array([_design_to_vector(d, cands) for d in designs])
    n_total = len(designs)

    # Initial random
    n_init = min(5, n_evaluations)
    initial = list(rng.choice(n_total, size=n_init, replace=False))

    evaluated_indices, evaluated_pha, evaluated_bio = [], [], []
    for idx in initial:
        try:
            res = _simulate_design(
                model, designs[idx], pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction_required,
            )
        except Exception:
            res = {"pha_flux": 0.0, "biomass_flux": 0.0}
        evaluated_indices.append(idx)
        evaluated_pha.append(res["pha_flux"])
        evaluated_bio.append(res["biomass_flux"])

    history = []
    ref_point = np.array([0.0, 0.0])

    while len(evaluated_indices) < n_evaluations:
        X_train = all_vecs[evaluated_indices]
        y_train = np.array(evaluated_pha)

        m = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=seed)
        m.fit(X_train, y_train)

        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            break

        X_cand = all_vecs[unevaluated]
        preds = m.predict(X_cand)
        best_rel = int(np.argmax(preds))
        best_abs = unevaluated[best_rel]

        try:
            res = _simulate_design(
                model, designs[best_abs], pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction_required,
            )
        except Exception:
            res = {"pha_flux": 0.0, "biomass_flux": 0.0}
        evaluated_indices.append(best_abs)
        evaluated_pha.append(res["pha_flux"])
        evaluated_bio.append(res["biomass_flux"])

        eval_pts = np.column_stack([evaluated_pha, evaluated_bio])
        hv = _compute_hypervolume_2d(eval_pts, ref_point)
        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable = float(np.max(np.array(evaluated_pha)[viable_mask])) if viable_mask.any() else 0.0
        pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

        history.append({
            "iteration": len(history) + 1,
            "n_evaluated": len(evaluated_indices),
            "hypervolume": round(hv, 6),
            "best_viable_pha": round(best_viable, 6),
            "pareto_discovered": pareto_disc,
        })

    return {"history": history}


def run_exploitation_baseline(
    model,
    designs: List[Dict[str, Any]],
    candidate_reaction_ids: List[str],
    n_evaluations: int,
    cfg: Dict[str, Any],
    conditions_cfg: Dict[str, Any],
    candidates_cfg: Dict[str, Any],
    seed: int = 0,
    condition_name: str = "base",
    biomass_fraction_required: float = 0.30,
    global_pareto_set: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Exploitation baseline: UCB with kappa=0 (pure mean prediction,
    no exploration bonus, no diversity penalty).
    """
    rng = np.random.default_rng(seed)
    pha_rxn = cfg["pha_reaction_id"]
    bio_rxn = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = 0.07

    cands = sorted(candidate_reaction_ids)
    all_vecs = np.array([_design_to_vector(d, cands) for d in designs])
    n_total = len(designs)

    n_init = min(5, n_evaluations)
    initial = list(rng.choice(n_total, size=n_init, replace=False))

    evaluated_indices, evaluated_pha, evaluated_bio = [], [], []
    for idx in initial:
        try:
            res = _simulate_design(
                model, designs[idx], pha_rxn, bio_rxn,
                overrides, upreg_fold, biomass_fraction_required,
            )
        except Exception:
            res = {"pha_flux": 0.0, "biomass_flux": 0.0}
        evaluated_indices.append(idx)
        evaluated_pha.append(res["pha_flux"])
        evaluated_bio.append(res["biomass_flux"])

    history = []
    ref_point = np.array([0.0, 0.0])
    batch_size = 5

    while len(evaluated_indices) < n_evaluations:
        X_train = all_vecs[evaluated_indices]
        y_pha = np.array(evaluated_pha)

        m = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=seed)
        m.fit(X_train, y_pha)

        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            break

        X_cand = all_vecs[unevaluated]
        preds = m.predict(X_cand)

        # Select top-batch_size by predicted mean (no exploration)
        n_select = min(batch_size, len(unevaluated), n_evaluations - len(evaluated_indices))
        top_rel = np.argsort(preds)[::-1][:n_select]

        for ri in top_rel:
            abs_idx = unevaluated[ri]
            try:
                res = _simulate_design(
                    model, designs[abs_idx], pha_rxn, bio_rxn,
                    overrides, upreg_fold, biomass_fraction_required,
                )
            except Exception:
                res = {"pha_flux": 0.0, "biomass_flux": 0.0}
            evaluated_indices.append(abs_idx)
            evaluated_pha.append(res["pha_flux"])
            evaluated_bio.append(res["biomass_flux"])

        eval_pts = np.column_stack([evaluated_pha, evaluated_bio])
        hv = _compute_hypervolume_2d(eval_pts, ref_point)
        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable = float(np.max(np.array(evaluated_pha)[viable_mask])) if viable_mask.any() else 0.0
        pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

        history.append({
            "iteration": len(history) + 1,
            "n_evaluated": len(evaluated_indices),
            "hypervolume": round(hv, 6),
            "best_viable_pha": round(best_viable, 6),
            "pareto_discovered": pareto_disc,
        })

    return {"history": history}
