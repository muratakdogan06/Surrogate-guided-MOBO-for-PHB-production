"""
phaopt.dataset — Build ML-ready dataset from FBA simulation results.

Functions
---------
build_ml_dataset : Convert FBA result DataFrame into feature-encoded ML dataset.
save_ml_dataset  : Save the ML dataset to disk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phaopt.io import save_dataframe

logger = logging.getLogger(__name__)


def build_ml_dataset(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw FBA simulation results into a binary-encoded ML dataset.

    The output has:
      - One binary column per candidate reaction (``ko_<rxn_id>`` and
        ``up_<rxn_id>``), indicating whether that reaction is knocked out
        or upregulated in a given design.
      - Condition indicator columns (one-hot encoded).
      - ``biomass_fraction_required`` as a numeric feature.
      - Target columns: ``pha_flux``, ``biomass_flux``.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Raw FBA results from ``run_all_simulations``.

    Returns
    -------
    pd.DataFrame
        Feature-encoded ML dataset.
    """
    df = sim_df.copy()

    # --- Parse knockouts/upregulations into sets ---------------------------
    def _parse_pipe(val):
        if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
            return set()
        return set(str(val).split("|"))

    df["_ko_set"] = df["knockouts"].apply(_parse_pipe)
    df["_up_set"] = df["upregulations"].apply(_parse_pipe)

    # Collect all unique reaction IDs used across designs
    all_ko_rxns = sorted({r for s in df["_ko_set"] for r in s if r})
    all_up_rxns = sorted({r for s in df["_up_set"] for r in s if r})

    # --- Binary-encode KOs and UPs -----------------------------------------
    for rid in all_ko_rxns:
        df[f"ko_{rid}"] = df["_ko_set"].apply(lambda s, r=rid: int(r in s))
    for rid in all_up_rxns:
        df[f"up_{rid}"] = df["_up_set"].apply(lambda s, r=rid: int(r in s))

    # --- One-hot encode conditions -----------------------------------------
    if "condition" in df.columns:
        cond_dummies = pd.get_dummies(df["condition"], prefix="cond")
        df = pd.concat([df, cond_dummies], axis=1)

    # --- Select final columns ----------------------------------------------
    # Features: ko_*, up_*, cond_*, biomass_fraction_required, n_knockouts, n_upregulations
    feature_cols = sorted([c for c in df.columns if c.startswith(("ko_", "up_", "cond_"))])
    meta_cols = []
    for mc in ["biomass_fraction_required", "n_knockouts", "n_upregulations"]:
        if mc in df.columns:
            meta_cols.append(mc)

    target_cols = []
    for tc in ["pha_flux", "biomass_flux"]:
        if tc in df.columns:
            target_cols.append(tc)

    # Also keep raw text columns for downstream grouping
    keep_cols = []
    for kc in ["knockouts", "upregulations", "condition"]:
        if kc in df.columns:
            keep_cols.append(kc)

    final_cols = keep_cols + feature_cols + meta_cols + target_cols
    ml_df = df[final_cols].copy()

    # Convert boolean columns to int
    for c in ml_df.columns:
        if ml_df[c].dtype == bool:
            ml_df[c] = ml_df[c].astype(int)

    logger.info(
        "ML dataset built: %d rows x %d cols (%d features, %d targets)",
        len(ml_df),
        len(ml_df.columns),
        len(feature_cols) + len(meta_cols),
        len(target_cols),
    )
    return ml_df


def save_ml_dataset(
    df: pd.DataFrame,
    base_path: str = "data/processed/ml_dataset",
) -> None:
    """
    Save the ML dataset to parquet and CSV.

    Parameters
    ----------
    df : pd.DataFrame
    base_path : str
        Output base path (without extension).
    """
    save_dataframe(df, base_path)
    logger.info("Saved ML dataset: %d rows to %s", len(df), base_path)
