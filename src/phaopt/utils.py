"""
phaopt.utils — Configuration loading and logging utilities.

Functions
---------
load_model_config   : Load configs/model_config.yaml
load_al_config      : Load configs/active_learning.yaml
load_candidate_reactions : Load configs/candidate_reactions.yaml
load_conditions     : Load configs/conditions.yaml
setup_logging       : Return a named logger with console + file handlers
resolve_path        : Resolve a relative path against the project root
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains configs/)."""
    # src/phaopt/utils.py -> src/phaopt -> src -> PROJECT_ROOT
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "configs").is_dir():
        return candidate
    # Fallback: cwd
    cwd = Path.cwd()
    while cwd != cwd.parent:
        if (cwd / "configs").is_dir():
            return cwd
        cwd = cwd.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def resolve_path(rel: str) -> Path:
    """Resolve a path relative to the project root and return an absolute Path."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger *name* with a stream handler.

    Also creates a file handler under ``results/logs/<name>.log`` when the
    results/logs directory exists (or can be created).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = PROJECT_ROOT / "results" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(".", "_")
        fh = logging.FileHandler(log_dir / f"{safe_name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    return logger


# ---------------------------------------------------------------------------
# YAML config loaders
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / "configs" / filename
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_model_config() -> Dict[str, Any]:
    """Load ``configs/model_config.yaml`` and return its contents as a dict."""
    return _load_yaml("model_config.yaml")


def load_al_config() -> Dict[str, Any]:
    """Load ``configs/active_learning.yaml`` and return its contents as a dict."""
    return _load_yaml("active_learning.yaml")


def load_candidate_reactions() -> Dict[str, Any]:
    """Load ``configs/candidate_reactions.yaml`` and return its contents as a dict."""
    return _load_yaml("candidate_reactions.yaml")


def load_conditions() -> Dict[str, Any]:
    """Load ``configs/conditions.yaml`` and return its contents as a dict."""
    return _load_yaml("conditions.yaml")
