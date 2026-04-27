"""
phaopt.active_learning — Multi-objective Bayesian optimisation via active learning.
REVISED: UCB replaced with Monte Carlo Expected Hypervolume Improvement (MC-EHVI)
following Daulton et al. (2020) NeurIPS — Differentiable Expected Hypervolume
Improvement for Parallel Multi-Objective Bayesian Optimization.

Acquisition function:
    EHVI(x) = E[ HV( F_evaluated ∪ {f(x)} | r ) − HV( F_evaluated | r ) ]

where HV is the 2-D hypervolume dominated by the evaluated Pareto front
relative to a reference point r, and f(x) = (PHA_flux, Biomass_flux).

The expectation is approximated by Monte Carlo sampling from the GBR
ensemble predictive distributions for both objectives jointly.

Functions
---------
run_active_learning          : Main AL loop (MC-EHVI acquisition, diversity).
run_random_baseline          : Random-sampling baseline.
run_greedy_baseline          : Greedy single-objective baseline.
run_exploitation_baseline    : Pure-exploitation (no exploration) baseline.
_build_global_pareto_reference : Pre-compute Pareto-optimal reference set.

References
----------
Daulton S, Balandat M, Bakshy E (2020) Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization.
    NeurIPS 33:9851–9864.
Emmerich MTM, Giannakoglou KC, Naujoks B (2006) Single- and multiobjective
    evolutionary optimization assisted by Gaussian random field metamodels.
    IEEE Trans Evol Comput 10:421–439.
Shahriari B et al. (2016) Taking the human out of the loop: a review of
    Bayesian optimization. Proc IEEE 104:148–175.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)


def _build_regressor(surrogate_model: str, n_estimators: int, max_depth: int, seed: int):
    """Factory: build a regressor based on config string."""
    if surrogate_model == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.1, tree_method="hist",
            n_jobs=-1, verbosity=0, random_state=seed,
        )
    return GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
    )

# Number of Monte Carlo samples for EHVI approximation.
# 256 balances accuracy and computational cost for GBR ensembles.
# Increase to 512 or 1024 for higher-fidelity estimates at greater cost.
MC_SAMPLES: int = 256


# ═══════════════════════════════════════════════════════════════════════════
# Core helpers (unchanged from original)
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
    Returns dict with pha_flux and biomass_flux.
    """
    with model as m:
        rxn_ids = {r.id for r in m.reactions}

        for rxn_id, bounds in (overrides or {}).items():
            if rxn_id in rxn_ids:
                rxn = m.reactions.get_by_id(rxn_id)
                rxn.lower_bound, rxn.upper_bound = bounds[0], bounds[1]

        for rid in design.get("knockouts", []):
            if rid in rxn_ids:
                rxn = m.reactions.get_by_id(rid)
                rxn.lower_bound = 0.0
                rxn.upper_bound = 0.0

        for rid in design.get("upregulations", []):
            if rid in rxn_ids:
                rxn = m.reactions.get_by_id(rid)
                if rxn.upper_bound > 0:
                    rxn.upper_bound *= upreg_fold
                if rxn.lower_bound < 0:
                    rxn.lower_bound *= upreg_fold

        m.objective = bio_rxn
        sol = m.optimize()
        max_bio = sol.objective_value if sol.status == "optimal" else 0.0

        m.reactions.get_by_id(bio_rxn).lower_bound = biomass_fraction_required * max_bio

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
    """Compute 2-D hypervolume dominated area relative to a reference point."""
    if len(points) == 0:
        return 0.0
    valid = points[(points[:, 0] > ref[0]) & (points[:, 1] > ref[1])]
    if len(valid) == 0:
        return 0.0
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
# MC-EHVI acquisition function — REPLACES UCB
# ═══════════════════════════════════════════════════════════════════════════

def _compute_mc_ehvi(
    pha_mean: np.ndarray,
    pha_std: np.ndarray,
    bio_mean: np.ndarray,
    bio_std: np.ndarray,
    current_pareto_points: np.ndarray,
    ref_point: np.ndarray,
    n_samples: int = MC_SAMPLES,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute Monte Carlo Expected Hypervolume Improvement for each candidate.

    For each candidate design x, EHVI is estimated as:

        EHVI(x) ≈ (1/S) Σ_s [ HV( F ∪ {f_s(x)} | r ) − HV( F | r ) ]

    where:
        f_s(x) = (PHA_s, BIO_s) is the s-th Monte Carlo sample drawn from
                 the independent Gaussian predictive distributions
                 PHA ~ N(pha_mean, pha_std²) and
                 BIO ~ N(bio_mean, bio_std²)
        F       = current evaluated Pareto front points
        r       = reference point (origin by default)
        HV(·|r) = 2-D hypervolume dominated relative to r

    Parameters
    ----------
    pha_mean, pha_std : np.ndarray, shape (n_candidates,)
        Ensemble mean and std for PHB flux predictions.
    bio_mean, bio_std : np.ndarray, shape (n_candidates,)
        Ensemble mean and std for biomass flux predictions.
    current_pareto_points : np.ndarray, shape (n_pareto, 2)
        Current Pareto front from all previously evaluated designs.
    ref_point : np.ndarray, shape (2,)
        Reference point for hypervolume computation.
    n_samples : int
        Number of Monte Carlo samples per candidate.
    rng : np.random.Generator, optional

    Returns
    -------
    ehvi : np.ndarray, shape (n_candidates,)
        MC-EHVI estimate for each candidate design.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_candidates = len(pha_mean)

    # Current hypervolume without any new candidate
    baseline_hv = _compute_hypervolume_2d(current_pareto_points, ref_point)

    # Draw S samples jointly for all candidates simultaneously
    # Shape: (n_samples, n_candidates)
    pha_samples = rng.normal(
        loc=pha_mean[np.newaxis, :],
        scale=np.maximum(pha_std[np.newaxis, :], 1e-8),
        size=(n_samples, n_candidates),
    )
    bio_samples = rng.normal(
        loc=bio_mean[np.newaxis, :],
        scale=np.maximum(bio_std[np.newaxis, :], 1e-8),
        size=(n_samples, n_candidates),
    )

    # Clip to non-negative flux values
    pha_samples = np.maximum(pha_samples, 0.0)
    bio_samples = np.maximum(bio_samples, 0.0)

    ehvi = np.zeros(n_candidates)

    for c in range(n_candidates):
        hvi_sum = 0.0
        for s in range(n_samples):
            # Candidate point for this sample
            candidate_pt = np.array([[pha_samples[s, c], bio_samples[s, c]]])

            # Augmented point set: current Pareto front + candidate sample
            if len(current_pareto_points) > 0:
                augmented = np.vstack([current_pareto_points, candidate_pt])
            else:
                augmented = candidate_pt

            # Extract Pareto front of augmented set
            augmented_pareto = _pareto_front_2d(augmented)

            # Hypervolume improvement for this sample
            augmented_hv = _compute_hypervolume_2d(augmented_pareto, ref_point)
            hvi_sum += max(0.0, augmented_hv - baseline_hv)

        ehvi[c] = hvi_sum / n_samples

    return ehvi


# ═══════════════════════════════════════════════════════════════════════════
# Global Pareto reference (unchanged)
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
    Pre-compute the Pareto-optimal reference set by evaluating a random
    subset of designs via FBA. Used for hypervolume benchmarking only.
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
    return _pareto_front_2d(pts)


# ═══════════════════════════════════════════════════════════════════════════
# Main Active Learning Loop — MC-EHVI acquisition
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
    Run the multi-objective Bayesian active learning loop using MC-EHVI.

    At each iteration:
      1. Re-fit the GBR ensemble for both PHB and biomass flux.
      2. Compute MC-EHVI for all unevaluated candidates jointly over
         both objectives.
      3. Select a diversity-penalised batch of designs.
      4. Evaluate selected designs by epsilon-constraint FBA oracle.
      5. Update the evaluated set and track Pareto hypervolume.

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
    dict with keys: history, evaluated, best
    """
    seed = al_cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    n_initial     = al_cfg.get("initial_random_samples", 5)
    n_iterations  = al_cfg.get("n_iterations", 20)
    batch_size    = al_cfg.get("batch_size", 5)
    n_estimators   = al_cfg.get("n_estimators", 200)
    max_depth      = al_cfg.get("max_depth", 8)
    ensemble_size  = al_cfg.get("ensemble_size", 5)
    surrogate_model = al_cfg.get("surrogate_model", "gradient_boosting")
    diversity_lambda   = al_cfg.get("diversity_lambda", 0.35)
    biomass_min        = al_cfg.get("biomass_min_for_acq", 0.07)
    convergence_window = al_cfg.get("convergence_window", 10)
    convergence_tol    = al_cfg.get("convergence_tol", 0.0002)
    n_mc_samples       = al_cfg.get("n_mc_samples", MC_SAMPLES)

    # NOTE: kappa is no longer used — EHVI replaces UCB.
    # biomass_penalty_lambda is no longer used as a hard penalty —
    # biomass is now a co-optimised objective within EHVI.
    # Both parameters are retained in the config for backward compatibility
    # with baseline comparisons but are not applied in the EHVI acquisition.

    condition_name    = al_cfg.get("condition_name", "base")
    biomass_fraction  = float(al_cfg.get("biomass_fraction_required", 0.30))
    pha_rxn    = cfg["pha_reaction_id"]
    bio_rxn    = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides  = conditions_cfg["conditions"][condition_name].get("overrides") or {}

    cands    = sorted(candidate_reaction_ids)
    all_vecs = np.array([_design_to_vector(d, cands) for d in designs])
    n_total  = len(designs)

    # Reference point for hypervolume: set to origin (0, 0)
    # Both objectives are non-negative fluxes so origin is valid
    ref_point = np.array([0.0, 0.0])

    # --- Initial random sampling -------------------------------------------
    initial_indices = list(
        rng.choice(n_total, size=min(n_initial, n_total), replace=False)
    )

    evaluated_indices: List[int] = []
    evaluated_pha: List[float]   = []
    evaluated_bio: List[float]   = []

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

    # --- MC-EHVI AL iterations ---------------------------------------------
    for iteration in range(n_iterations):
        X_train = all_vecs[evaluated_indices]
        y_pha   = np.array(evaluated_pha)
        y_bio   = np.array(evaluated_bio)

        # Train ensemble: separate models for each objective
        pha_models = []
        bio_models = []
        for e in range(ensemble_size):
            m_pha = _build_regressor(surrogate_model, n_estimators, max_depth, seed + e)
            m_bio = _build_regressor(surrogate_model, n_estimators, max_depth, seed + 100 + e)
            m_pha.fit(X_train, y_pha)
            m_bio.fit(X_train, y_bio)
            pha_models.append(m_pha)
            bio_models.append(m_bio)

        # Predict on unevaluated candidates
        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            logger.info("All designs evaluated; stopping.")
            break

        X_cand = all_vecs[unevaluated]

        # Ensemble predictions — shape (ensemble_size, n_candidates)
        pha_preds = np.array([m.predict(X_cand) for m in pha_models])
        bio_preds = np.array([m.predict(X_cand) for m in bio_models])

        pha_mean = pha_preds.mean(axis=0)
        pha_std  = pha_preds.std(axis=0)
        bio_mean = bio_preds.mean(axis=0)
        bio_std  = bio_preds.std(axis=0)

        # Current Pareto front from all evaluated designs
        eval_pts_current = np.column_stack([evaluated_pha, evaluated_bio])
        current_pareto   = _pareto_front_2d(eval_pts_current)

        # --- MC-EHVI acquisition over both objectives ----------------------
        ehvi_scores = _compute_mc_ehvi(
            pha_mean=pha_mean,
            pha_std=pha_std,
            bio_mean=bio_mean,
            bio_std=bio_std,
            current_pareto_points=current_pareto,
            ref_point=ref_point,
            n_samples=n_mc_samples,
            rng=rng,
        )

        # --- Diversity-aware greedy batch selection ------------------------
        selected_rel: List[int] = []
        for _ in range(min(batch_size, len(unevaluated))):
            if selected_rel and diversity_lambda > 0:
                sel_vecs = X_cand[selected_rel]
                dists = np.array([
                    np.sum(np.abs(X_cand - sv), axis=1) for sv in sel_vecs
                ]).min(axis=0)
                diversity_bonus = diversity_lambda * dists / (X_cand.shape[1] + 1e-8)
                eff_scores = ehvi_scores + diversity_bonus
            else:
                eff_scores = ehvi_scores.copy()

            for si in selected_rel:
                eff_scores[si] = -np.inf

            selected_rel.append(int(np.argmax(eff_scores)))

        selected_abs = [unevaluated[i] for i in selected_rel]

        # --- Oracle FBA evaluation of selected designs --------------------
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

        # --- Metrics -------------------------------------------------------
        eval_points = np.column_stack([evaluated_pha, evaluated_bio])
        hv = _compute_hypervolume_2d(eval_points, ref_point)

        viable_mask      = np.array(evaluated_bio) >= biomass_min
        best_viable_pha  = (
            float(np.max(np.array(evaluated_pha)[viable_mask]))
            if viable_mask.any() else 0.0
        )
        pareto_discovered = _count_pareto_discovered(eval_points, global_pareto_set)

        history.append({
            "iteration":        iteration + 1,
            "n_evaluated":      len(evaluated_indices),
            "hypervolume":      round(hv, 6),
            "best_viable_pha":  round(best_viable_pha, 6),
            "pareto_discovered": pareto_discovered,
            "batch_mean_pha":   round(float(np.mean(evaluated_pha[-batch_size:])), 6),
            "mean_ehvi":        round(float(np.mean(ehvi_scores)), 8),
            "max_ehvi":         round(float(np.max(ehvi_scores)), 8),
        })

        logger.info(
            "  Iter %d | evaluated=%d | HV=%.4f | best_viable_pha=%.4f"
            " | pareto=%d | max_EHVI=%.6f",
            iteration + 1, len(evaluated_indices), hv,
            best_viable_pha, pareto_discovered,
            float(np.max(ehvi_scores)),
        )

        # --- Convergence check --------------------------------------------
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
            "design_index":  idx,
            "knockouts":     "|".join(d.get("knockouts", [])),
            "upregulations": "|".join(d.get("upregulations", [])),
            "pha_flux":      evaluated_pha[i],
            "biomass_flux":  evaluated_bio[i],
        })
    evaluated_df = pd.DataFrame(eval_rows)

    viable_mask = np.array(evaluated_bio) >= biomass_min
    if viable_mask.any():
        viable_pha = np.array(evaluated_pha).copy()
        viable_pha[~viable_mask] = -np.inf
        best_idx_local   = int(np.argmax(viable_pha))
        best_design_idx  = evaluated_indices[best_idx_local]
        best = {
            "design_index":  best_design_idx,
            "knockouts":     designs[best_design_idx].get("knockouts", []),
            "upregulations": designs[best_design_idx].get("upregulations", []),
            "pha_flux":      evaluated_pha[best_idx_local],
            "biomass_flux":  evaluated_bio[best_idx_local],
        }
    else:
        best = {
            "design_index": -1, "knockouts": [], "upregulations": [],
            "pha_flux": 0.0, "biomass_flux": 0.0,
        }

    return {"history": history, "evaluated": evaluated_df, "best": best}


# ═══════════════════════════════════════════════════════════════════════════
# Baselines (unchanged — use original UCB/greedy logic for comparison)
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
    """Random-sampling baseline — uniform design selection without surrogate."""
    rng       = np.random.default_rng(seed)
    pha_rxn   = cfg["pha_reaction_id"]
    bio_rxn   = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides  = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = 0.07

    indices = list(
        rng.choice(len(designs), size=min(n_evaluations, len(designs)), replace=False)
    )

    evaluated_pha, evaluated_bio = [], []
    history  = []
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

        eval_pts    = np.column_stack([evaluated_pha, evaluated_bio])
        hv          = _compute_hypervolume_2d(eval_pts, ref_point)
        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable = (
            float(np.max(np.array(evaluated_pha)[viable_mask]))
            if viable_mask.any() else 0.0
        )
        pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

        history.append({
            "iteration":        step + 1,
            "n_evaluated":      step + 1,
            "hypervolume":      round(hv, 6),
            "best_viable_pha":  round(best_viable, 6),
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
    Greedy baseline — ranks candidates by surrogate PHB mean prediction
    without iterative model updating between batches. Uses single-objective
    surrogate mean (no uncertainty, no EHVI) for comparison against MC-EHVI.
    """
    rng        = np.random.default_rng(seed)
    pha_rxn    = cfg["pha_reaction_id"]
    bio_rxn    = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides  = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = 0.07
    cands      = sorted(candidate_reaction_ids)
    all_vecs   = np.array([_design_to_vector(d, cands) for d in designs])
    n_total    = len(designs)
    ref_point  = np.array([0.0, 0.0])

    n_initial = 5
    initial_indices = list(
        rng.choice(n_total, size=min(n_initial, n_total), replace=False)
    )

    evaluated_indices: List[int] = []
    evaluated_pha: List[float]   = []
    evaluated_bio: List[float]   = []

    for idx in initial_indices:
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

    # Train once on initial data — no iterative updating (greedy)
    m_pha = _build_regressor("xgboost", 200, 8, seed)
    m_pha.fit(all_vecs[evaluated_indices], np.array(evaluated_pha))

    history     = []
    budget_used = n_initial
    batch_size  = 5

    while budget_used < n_evaluations:
        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            break

        X_cand     = all_vecs[unevaluated]
        pha_pred   = m_pha.predict(X_cand)
        this_batch = min(batch_size, n_evaluations - budget_used, len(unevaluated))
        top_rel    = np.argsort(pha_pred)[::-1][:this_batch]
        top_abs    = [unevaluated[i] for i in top_rel]

        for idx in top_abs:
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
            budget_used += 1

            eval_pts    = np.column_stack([evaluated_pha, evaluated_bio])
            hv          = _compute_hypervolume_2d(eval_pts, ref_point)
            viable_mask = np.array(evaluated_bio) >= biomass_min
            best_viable = (
                float(np.max(np.array(evaluated_pha)[viable_mask]))
                if viable_mask.any() else 0.0
            )
            pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

            history.append({
                "iteration":        budget_used,
                "n_evaluated":      budget_used,
                "hypervolume":      round(hv, 6),
                "best_viable_pha":  round(best_viable, 6),
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
    Exploitation baseline — iterative surrogate updating but κ = 0
    (pure surrogate mean, no uncertainty exploration).
    Uses single-objective PHB mean prediction for comparison against MC-EHVI.
    """
    rng        = np.random.default_rng(seed)
    pha_rxn    = cfg["pha_reaction_id"]
    bio_rxn    = cfg["biomass_reaction_id"]
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    overrides  = conditions_cfg["conditions"][condition_name].get("overrides") or {}
    biomass_min = 0.07
    cands      = sorted(candidate_reaction_ids)
    all_vecs   = np.array([_design_to_vector(d, cands) for d in designs])
    n_total    = len(designs)
    ref_point  = np.array([0.0, 0.0])

    n_initial   = 5
    n_iter      = 20
    batch_size  = 5
    n_estimators = 200
    max_depth    = 8
    ensemble_size = 5

    initial_indices = list(
        rng.choice(n_total, size=min(n_initial, n_total), replace=False)
    )

    evaluated_indices: List[int] = []
    evaluated_pha: List[float]   = []
    evaluated_bio: List[float]   = []

    for idx in initial_indices:
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

    for iteration in range(n_iter):
        X_train = all_vecs[evaluated_indices]
        y_pha   = np.array(evaluated_pha)

        # Ensemble — pure exploitation: use mean only (kappa = 0)
        pha_models = [
            _build_regressor("xgboost", n_estimators, max_depth, seed + e).fit(X_train, y_pha)
            for e in range(ensemble_size)
        ]

        unevaluated = [i for i in range(n_total) if i not in set(evaluated_indices)]
        if not unevaluated:
            break

        X_cand   = all_vecs[unevaluated]
        pha_mean = np.array([m.predict(X_cand) for m in pha_models]).mean(axis=0)

        # κ = 0: rank by mean only
        top_rel = np.argsort(pha_mean)[::-1][:min(batch_size, len(unevaluated))]
        top_abs = [unevaluated[i] for i in top_rel]

        for idx in top_abs:
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

        eval_pts    = np.column_stack([evaluated_pha, evaluated_bio])
        hv          = _compute_hypervolume_2d(eval_pts, ref_point)
        viable_mask = np.array(evaluated_bio) >= biomass_min
        best_viable = (
            float(np.max(np.array(evaluated_pha)[viable_mask]))
            if viable_mask.any() else 0.0
        )
        pareto_disc = _count_pareto_discovered(eval_pts, global_pareto_set)

        history.append({
            "iteration":         iteration + 1,
            "n_evaluated":       len(evaluated_indices),
            "hypervolume":       round(hv, 6),
            "best_viable_pha":   round(best_viable, 6),
            "pareto_discovered": pareto_disc,
        })

    return {"history": history}
