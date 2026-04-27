"""
phaopt.io — File I/O helpers for SBML models, DataFrames, and JSON.

Functions
---------
load_sbml_model  : Read an SBML model via COBRApy
load_dataframe   : Read a parquet or CSV into a DataFrame
save_dataframe   : Write a DataFrame to both parquet and CSV
save_json        : Serialise a dict to a JSON file
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from phaopt.utils import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SBML model loading
# ---------------------------------------------------------------------------

def load_sbml_model(path: str):
    """
    Load a COBRA SBML model.

    Parameters
    ----------
    path : str
        Path to the ``.xml`` file (relative to project root or absolute).

    Returns
    -------
    cobra.Model
        The loaded COBRA model object.
    """
    import cobra

    resolved = resolve_path(path)
    logger.info("Loading SBML model: %s", resolved)
    model = cobra.io.read_sbml_model(str(resolved))
    logger.info(
        "Model loaded: %d reactions, %d metabolites, %d genes",
        len(model.reactions),
        len(model.metabolites),
        len(model.genes),
    )
    return model


# ---------------------------------------------------------------------------
# DataFrame I/O
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from *path* (parquet preferred, CSV fallback).

    Parameters
    ----------
    path : str
        File path (relative to project root or absolute).  If the exact file
        is not found, ``.parquet`` and ``.csv`` extensions are tried in turn.

    Returns
    -------
    pd.DataFrame
    """
    resolved = resolve_path(path)

    # Try the path as given, then with extensions
    candidates = [resolved]
    if not resolved.suffix:
        candidates += [resolved.with_suffix(".parquet"), resolved.with_suffix(".csv")]
    elif resolved.suffix == ".parquet":
        candidates.append(resolved.with_suffix(".csv"))
    elif resolved.suffix == ".csv":
        candidates.insert(0, resolved.with_suffix(".parquet"))

    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            logger.info("Loaded DataFrame from %s (%d rows x %d cols)", p, *df.shape)
            return df

    raise FileNotFoundError(
        f"Could not find dataframe at any of: {[str(c) for c in candidates]}"
    )


def save_dataframe(df: pd.DataFrame, base_path: str) -> None:
    """
    Save *df* as both ``<base_path>.parquet`` and ``<base_path>.csv``.

    Parameters
    ----------
    df : pd.DataFrame
    base_path : str
        Path **without** extension, relative to the project root or absolute.
    """
    resolved = resolve_path(base_path)
    # Strip any extension that might have been passed
    if resolved.suffix in (".parquet", ".csv"):
        resolved = resolved.with_suffix("")

    resolved.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = resolved.with_suffix(".parquet")
    csv_path = resolved.with_suffix(".csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info("Saved DataFrame: %s (.parquet + .csv)", resolved)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Union[Dict[str, Any], list], path: str) -> None:
    """
    Serialise *data* to a JSON file at *path*.

    Parameters
    ----------
    data : dict or list
    path : str
        File path (relative to project root or absolute).
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON: %s", resolved)
