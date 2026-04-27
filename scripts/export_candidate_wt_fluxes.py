#!/usr/bin/env python3
"""
Export wild-type FBA fluxes through each candidate reaction (Table S1 / methods support).

Footnote (Table S1 / Methods) — suggested wording
---------------------------------------------------
ᵃ Wild-type (WT) flux values were computed by FBA under aerobic, glucose-replete
conditions (objective: biomass maximisation; no ε-constraint). Reactions with zero
WT flux were retained if carrying non-zero flux under alternative carbon source or
low-oxygen conditions tested in the screening framework.

Implementation
--------------
* Primary column ``wt_flux_max_growth`` = biomass-maximum FBA on the ``base``
  condition (``configs/conditions.yaml``: aerobic, glucose-replete; empty overrides).
* ``alt_conditions_nonzero_flux`` lists other condition keys (same objective, no ε)
  where |flux| > ``--flux-tol`` for that reaction.
* ``max_abs_wt_flux_all_screening_conditions`` = max |flux| over *all* conditions
  in ``conditions.yaml`` (including ``base``), for sorting / screening overview.

Optional: ``--with-pha-epsilon 0.30`` adds a column for the PHB-maximum WT state at
that ε (pipeline-consistent; *not* part of footnote ᵃ).

Requires: COBRA (cobrapy), ``configs/model_config.yaml`` → extended_model_path.

Output: results/tables/candidate_reactions_wt_fluxes.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_sbml_model  # noqa: E402
from phaopt.simulation import _apply_design, _apply_overrides  # noqa: E402
from phaopt.utils import (  # noqa: E402
    load_candidate_reactions,
    load_conditions,
    load_model_config,
    resolve_path,
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RBG_PATH = TABLES_DIR / "reactions_by_group.csv"
OUT_PATH = TABLES_DIR / "candidate_reactions_wt_fluxes.csv"


def _flatten_candidates(candidates_cfg: dict) -> list[tuple[str, str]]:
    """(reaction_id, group_key) preserving YAML order; first occurrence wins."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for group_key, block in candidates_cfg.get("candidate_groups", {}).items():
        for rid in block.get("reaction_ids", []):
            if not isinstance(rid, str) or not rid.startswith("bm"):
                continue
            if rid in seen:
                continue
            seen.add(rid)
            rows.append((rid, group_key))
    return rows


def _flux_max_growth(model, bio_rxn: str, overrides: dict) -> dict[str, float] | None:
    with model as m:
        _apply_overrides(m, overrides)
        m.objective = bio_rxn
        sol = m.optimize()
        if sol.status != "optimal":
            return None
        return dict(sol.fluxes)


def _flux_pha_optimum(
    model,
    bio_rxn: str,
    pha_rxn: str,
    overrides: dict,
    candidates_cfg: dict,
    biomass_fraction: float,
) -> dict[str, float] | None:
    upreg_fold = candidates_cfg["perturbation_types"]["upregulation"]["fold_change"]
    design = {"knockouts": [], "upregulations": []}
    with model as m:
        _apply_overrides(m, overrides)
        _apply_design(m, design, upreg_fold, candidates_cfg)
        m.objective = bio_rxn
        sol_bio = m.optimize()
        if sol_bio.status != "optimal":
            return None
        max_bio = float(sol_bio.objective_value)
        bio = m.reactions.get_by_id(bio_rxn)
        bio.lower_bound = float(biomass_fraction) * max_bio
        m.objective = pha_rxn
        sol = m.optimize()
        if sol.status != "optimal":
            return None
        return dict(sol.fluxes)


def _wt_flux_by_condition(
    model,
    bio_rxn: str,
    conditions_cfg: dict,
) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for name, block in conditions_cfg.get("conditions", {}).items():
        ov = block.get("overrides") or {}
        out[name] = _flux_max_growth(model, bio_rxn, ov)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="WT FBA fluxes for candidate reactions (Table S1).")
    ap.add_argument(
        "--flux-tol",
        type=float,
        default=1e-9,
        help="Treat |flux| below this as zero for alt-condition flags.",
    )
    ap.add_argument(
        "--with-pha-epsilon",
        type=float,
        default=None,
        metavar="EPS",
        help="If set, add WT PHB-max flux column at this ε (base condition only).",
    )
    args = ap.parse_args()

    cfg = load_model_config()
    cond_cfg = load_conditions()
    cand_cfg = load_candidate_reactions()
    conditions = cond_cfg.get("conditions", {})
    if "base" not in conditions:
        raise SystemExit("conditions.yaml must define a 'base' condition.")

    model_path = resolve_path(cfg["extended_model_path"])
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = load_sbml_model(str(model_path))
    bio_rxn = cfg["biomass_reaction_id"]
    pha_rxn = cfg["pha_reaction_id"]

    flux_by_cond = _wt_flux_by_condition(model, bio_rxn, cond_cfg)
    flux_base = flux_by_cond.get("base")
    if flux_base is None:
        raise SystemExit("WT max-growth FBA failed on 'base' condition.")

    flux_pha_base: dict[str, float] | None = None
    if args.with_pha_epsilon is not None:
        base_ov = conditions["base"].get("overrides") or {}
        flux_pha_base = _flux_pha_optimum(
            model, bio_rxn, pha_rxn, base_ov, cand_cfg, args.with_pha_epsilon
        )
        if flux_pha_base is None:
            print(
                "Warning: WT PHB-optimum FBA on base failed; PHB column will be NaN.",
                file=sys.stderr,
            )

    names: dict[str, str] = {}
    if RBG_PATH.is_file():
        rbg = pd.read_csv(RBG_PATH)
        names = dict(zip(rbg["reaction_id"].astype(str), rbg["reaction_name"].astype(str)))

    rxn_ids_model = {r.id for r in model.reactions}
    rows_out: list[dict] = []

    for rid, group in _flatten_candidates(cand_cfg):
        if rid not in rxn_ids_model:
            row = {
                "reaction_id": rid,
                "candidate_group": group,
                "reaction_name": names.get(rid, ""),
                "in_model": False,
                "wt_flux_max_growth": None,
                "alt_conditions_nonzero_flux": "",
                "max_abs_wt_flux_all_screening_conditions": None,
            }
            if args.with_pha_epsilon is not None:
                row[f"wt_flux_pha_opt_eps{args.with_pha_epsilon:g}"] = None
            rows_out.append(row)
            continue

        v_base = float(flux_base.get(rid, 0.0))
        alts: list[str] = []
        abs_max = 0.0
        for cname, fd in flux_by_cond.items():
            if fd is None:
                continue
            v = float(fd.get(rid, 0.0))
            abs_max = max(abs_max, abs(v))
            if cname != "base" and abs(v) > args.flux_tol:
                alts.append(cname)

        row = {
            "reaction_id": rid,
            "candidate_group": group,
            "reaction_name": names.get(rid, ""),
            "in_model": True,
            "wt_flux_max_growth": round(v_base, 8),
            "alt_conditions_nonzero_flux": ";".join(sorted(alts)),
            "max_abs_wt_flux_all_screening_conditions": round(abs_max, 8),
        }
        if args.with_pha_epsilon is not None:
            vp = float(flux_pha_base[rid]) if flux_pha_base and rid in flux_pha_base else float("nan")
            row[f"wt_flux_pha_opt_eps{args.with_pha_epsilon:g}"] = (
                round(vp, 8) if vp == vp else None
            )
        rows_out.append(row)

    out = pd.DataFrame(rows_out)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
