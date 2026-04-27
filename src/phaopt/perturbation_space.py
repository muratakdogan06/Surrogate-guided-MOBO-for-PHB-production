"""
phaopt.perturbation_space — Combinatorial metabolic design generation.

Functions
---------
generate_designs : Build list of design dicts from candidate reactions config.
save_designs     : Persist design list to parquet/CSV and return DataFrame.

Changes vs original
-------------------
- Stratified sampling: KO/UP order is drawn uniformly (not biased toward
  high-order combinations), then reactions are sampled within that order.
  This ensures single-KO designs are as likely as 4-KO designs in the sample.
- Added design_type column to output for downstream stratified analysis.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, List, Set

import pandas as pd

from phaopt.io import save_dataframe
from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)


def generate_designs(
    model_rxn_ids: Set[str],
    candidates_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate combinatorial knockout / upregulation designs.

    Each design is a dict with keys:
      - ``knockouts``     : list[str] — reaction IDs to knock out
      - ``upregulations`` : list[str] — reaction IDs to upregulate

    Sampling strategy
    -----------------
    If the full Cartesian product exceeds ``max_total * 2``, stratified
    random sampling is used:
      1. Draw KO order k ~ Uniform(0, max_ko)
      2. Draw UP order u ~ Uniform(0, max_up)
      3. Sample k reactions for KO and u (disjoint) reactions for UP
    This avoids the bias of uniform-over-combinations, which over-represents
    high-order designs because C(n,k) grows rapidly with k.

    Parameters
    ----------
    model_rxn_ids : set[str]
        Reaction IDs present in the COBRA model (used to filter candidates).
    candidates_cfg : dict
        Parsed ``candidate_reactions.yaml``.

    Returns
    -------
    list[dict]
    """
    max_ko = candidates_cfg.get("max_knockouts", 4)
    max_up = candidates_cfg.get("max_upregulations", 3)
    max_total = candidates_cfg.get("max_total", 100000)

    # Collect candidate reactions present in the model
    all_candidates: List[str] = []
    for grp in candidates_cfg["candidate_groups"].values():
        for rid in grp.get("reaction_ids", []):
            if rid in model_rxn_ids and rid not in all_candidates:
                all_candidates.append(rid)

    all_candidates.sort()
    n_cands = len(all_candidates)
    n_cfg = sum(
        len(g.get("reaction_ids", []))
        for g in candidates_cfg["candidate_groups"].values()
    )
    logger.info("Candidate reactions in model: %d / %d", n_cands, n_cfg)

    # Build full enumeration lists
    ko_combos: List[tuple] = [()]
    for k in range(1, max_ko + 1):
        ko_combos.extend(itertools.combinations(all_candidates, k))

    up_combos: List[tuple] = [()]
    for u in range(1, max_up + 1):
        up_combos.extend(itertools.combinations(all_candidates, u))

    total_possible = len(ko_combos) * len(up_combos)

    if total_possible <= max_total * 2:
        # Small enough: full enumeration then subsample if needed
        designs: List[Dict[str, Any]] = []
        for ko_set, up_set in itertools.product(ko_combos, up_combos):
            if set(ko_set) & set(up_set):
                continue
            if not ko_set and not up_set:
                continue
            designs.append({
                "knockouts": list(ko_set),
                "upregulations": list(up_set),
            })
        if len(designs) > max_total:
            rng = random.Random(42)
            designs = rng.sample(designs, max_total)

    else:
        # Large space: STRATIFIED sampling to avoid high-order bias
        logger.info(
            "Design space too large (%d x %d = %.2e); "
            "stratified sampling of %d designs (seed=42)",
            len(ko_combos), len(up_combos), total_possible, max_total,
        )
        rng = random.Random(42)
        designs = []
        seen: set = set()
        attempts = 0
        max_attempts = max_total * 50

        # Pre-build index: ko_by_order[k] = list of combos of length k
        ko_by_order: Dict[int, List[tuple]] = {k: [] for k in range(0, max_ko + 1)}
        for combo in ko_combos:
            ko_by_order[len(combo)].append(combo)

        up_by_order: Dict[int, List[tuple]] = {u: [] for u in range(0, max_up + 1)}
        for combo in up_combos:
            up_by_order[len(combo)].append(combo)

        while len(designs) < max_total and attempts < max_attempts:
            attempts += 1

            # Draw orders uniformly (stratified)
            k = rng.randint(0, max_ko)
            u = rng.randint(0, max_up)

            if k == 0 and u == 0:
                continue

            if not ko_by_order[k] or not up_by_order[u]:
                continue

            ko_set = rng.choice(ko_by_order[k])
            up_set = rng.choice(up_by_order[u])

            if set(ko_set) & set(up_set):
                continue

            key = (ko_set, up_set)
            if key in seen:
                continue

            seen.add(key)
            designs.append({
                "knockouts": list(ko_set),
                "upregulations": list(up_set),
            })

        if len(designs) < max_total:
            logger.warning(
                "Only generated %d / %d designs after %d attempts. "
                "Consider reducing max_total or increasing candidate reactions.",
                len(designs), max_total, attempts,
            )

    logger.info(
        "Generated %d unique designs (max_ko=%d, max_up=%d)",
        len(designs), max_ko, max_up,
    )
    return designs


def save_designs(
    designs: List[Dict[str, Any]],
    base_path: str = "data/processed/design_space",
) -> pd.DataFrame:
    """
    Convert design list to DataFrame and save to parquet + CSV.

    Columns
    -------
    design_id       : int  — sequential index
    knockouts       : str  — pipe-separated reaction IDs (empty = no KO)
    upregulations   : str  — pipe-separated reaction IDs (empty = no UP)
    n_knockouts     : int
    n_upregulations : int
    design_type     : str  — 'KO_only' | 'UP_only' | 'KO_UP' | 'WT'

    Parameters
    ----------
    designs : list[dict]
    base_path : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for i, d in enumerate(designs):
        n_ko = len(d["knockouts"])
        n_up = len(d["upregulations"])

        if   n_ko > 0 and n_up == 0: dtype = "KO_only"
        elif n_ko == 0 and n_up > 0: dtype = "UP_only"
        elif n_ko > 0 and n_up > 0:  dtype = "KO_UP"
        else:                         dtype = "WT"

        rows.append({
            "design_id":      i,
            "knockouts":      "|".join(d["knockouts"]),
            "upregulations":  "|".join(d["upregulations"]),
            "n_knockouts":    n_ko,
            "n_upregulations": n_up,
            "design_type":    dtype,
        })

    df = pd.DataFrame(rows)
    save_dataframe(df, base_path)
    logger.info("Saved %d designs to %s", len(df), base_path)

    # Log design type breakdown
    type_counts = df["design_type"].value_counts().to_dict()
    logger.info("Design types: %s", type_counts)
    ko_dist = df["n_knockouts"].value_counts().sort_index().to_dict()
    up_dist = df["n_upregulations"].value_counts().sort_index().to_dict()
    logger.info("KO order distribution: %s", ko_dist)
    logger.info("UP order distribution: %s", up_dist)

    return df