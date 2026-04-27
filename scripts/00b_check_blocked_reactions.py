#!/usr/bin/env python3
"""
00b_check_blocked_reactions.py — Check which candidate reactions are blocked.

Uses cobra's find_blocked_reactions (FVA-based) and cross-references
with the candidate_reactions.yaml pool.
"""

import sys
from pathlib import Path

import cobra
import yaml
from cobra.flux_analysis import find_blocked_reactions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

with open(PROJECT_ROOT / "configs" / "model_config.yaml") as f:
    model_cfg = yaml.safe_load(f)

with open(PROJECT_ROOT / "configs" / "candidate_reactions.yaml") as f:
    cand_cfg = yaml.safe_load(f)


def main() -> None:
    model = cobra.io.read_sbml_model(str(PROJECT_ROOT / model_cfg["extended_model_path"]))
    print(f"Model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")

    print("Running find_blocked_reactions (FVA)...")
    blocked = find_blocked_reactions(model, open_exchanges=False)
    print(f"Total blocked reactions: {len(blocked)}")

    rxn_to_group = {}
    for group_name, group_data in cand_cfg["candidate_groups"].items():
        for rxn_id in group_data.get("reaction_ids", []):
            rxn_to_group[rxn_id] = group_name

    blocked_in_candidates = []
    blocked_outside_candidates = []

    for rxn_id in blocked:
        if rxn_id in rxn_to_group:
            blocked_in_candidates.append({
                "reaction_id": rxn_id,
                "group": rxn_to_group[rxn_id]
            })
        else:
            blocked_outside_candidates.append(rxn_id)

    print(f"\nBlocked reactions within candidate pool: {len(blocked_in_candidates)}")
    print(f"Blocked reactions outside candidate pool: {len(blocked_outside_candidates)}")

    if blocked_in_candidates:
        print("\nWARNING: These candidate reactions are blocked and should be excluded:")
        for r in blocked_in_candidates:
            print(f"  {r['reaction_id']} — {r['group']}")
    else:
        print("\nAll candidate reactions carry flux — no exclusions needed.")


if __name__ == "__main__":
    main()
