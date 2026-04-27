#!/usr/bin/env python3
"""
Write reaction_id, gene_reaction_rule, and reaction name for all candidate reactions.

Reads: configs/candidate_reactions.yaml, models/model_with_PHA.xml (via model_config).

Output: results/tables/candidate_reactions_gpr.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import cobra
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.utils import load_model_config, resolve_path  # noqa: E402

OUT_PATH = PROJECT_ROOT / "results" / "tables" / "candidate_reactions_gpr.csv"


def _ordered_unique_candidate_ids(cfg: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in cfg.get("candidate_groups", {}).values():
        for rid in block.get("reaction_ids", []):
            if isinstance(rid, str) and rid.startswith("bm") and rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


def main() -> None:
    mcfg = load_model_config()
    model_path = resolve_path(mcfg["extended_model_path"])
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    with open(PROJECT_ROOT / "configs" / "candidate_reactions.yaml") as f:
        cfg = yaml.safe_load(f)

    model = cobra.io.read_sbml_model(str(model_path))
    rxn_ids = {r.id for r in model.reactions}

    rows: list[dict[str, str]] = []
    for rid in _ordered_unique_candidate_ids(cfg):
        if rid not in rxn_ids:
            rows.append({"reaction_id": rid, "gene_reaction_rule": "", "reaction_name": "MISSING_IN_MODEL"})
            continue
        rxn = model.reactions.get_by_id(rid)
        grr = (rxn.gene_reaction_rule or "").replace("\n", " ").strip()
        name = (rxn.name or "").replace("\n", " ").strip()
        rows.append({"reaction_id": rid, "gene_reaction_rule": grr, "reaction_name": name})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
