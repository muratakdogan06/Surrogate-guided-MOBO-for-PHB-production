"""
Human-readable labels and colour tiers for SHAP / ML feature columns.

Uses ``results/tables/reactions_by_group.csv`` for KO/UP reaction names when available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from phaopt.utils import resolve_path

Tier = Literal["constraint", "condition", "knockout", "upregulation", "meta"]

# Match manuscript / production-envelope C1–C7 codes
_COND_SUFFIX_TO_LABEL: dict[str, str] = {
    "base": "C1 — Glc, aerobic (ref.)",
    "low_carbon": "C2 — Glc, carbon-lim.",
    "low_oxygen": "C3 — Glc, micro-aer.",
    "glycerol_aerobic": "C4 — Gly, aerobic",
    "acetate_aerobic": "C5 — Ac, aerobic",
    "glycerol_low_oxygen": "C6 — Gly, micro-aer.",
    "mixed_glucose_glycerol": "C7 — Glc + Gly mix",
}

# Short gene / enzyme tag (reaction id appended separately as (bmXXXX))
_BM_OVERRIDE: dict[str, str] = {
    "bm00403": "phaC",
    "bm00387": "phaB",
    "bm00377": "acpH",
    "bm00286": "icd",
    "bm00472": "fabH",
    "bm00473": "mdcH",
    "bm00283": "gltA",
    "bm00500": "atoB",
    "bm00401": "hbd",
    "bm00292": "sucCS",
    "bm00293": "sdh",
    "bm00499": "ech",
}

_TIER_COLORS: dict[Tier, str] = {
    "constraint": "#5d6d7e",
    "condition": "#1f77b4",
    "knockout": "#c0392b",
    "upregulation": "#d35400",
    "meta": "#7f8c8d",
}


def tier_color(tier: Tier) -> str:
    return _TIER_COLORS[tier]


def feature_tier(name: str) -> Tier:
    if name == "biomass_fraction_required":
        return "constraint"
    if name.startswith("cond_"):
        return "condition"
    if name.startswith("ko_"):
        return "knockout"
    if name.startswith("up_"):
        return "upregulation"
    return "meta"


def _short_gene_symbol(reaction_name: str) -> str:
    """
    Prefer a trailing parenthetical gene symbol (fabH, mdcH, gltA, …).
    If none, use a very short stub — never the full reaction sentence.
    """
    s = reaction_name.strip()
    # Last (Symbol) in the string (gene / locus style)
    found = list(re.finditer(r"\(([A-Za-z][A-Za-z0-9_.-]{1,12})\)", s))
    if found:
        sym = found[-1].group(1).strip()
        if len(sym) <= 14 and sym.replace(".", "").isalnum():
            return sym
    # First alnum token, capped (fallback when CSV has no gene in parens)
    tok = re.split(r"[\s\[,/:]+", s)
    for t in tok:
        t = t.strip("-")
        if len(t) >= 3 and re.match(r"^[A-Za-z]", t):
            return t[:10]
    return "rxn"


class ReactionLabelLookup:
    """Lazy load reaction_id → reaction_name from reactions_by_group.csv."""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._path = Path(
            resolve_path(csv_path or "results/tables/reactions_by_group.csv")
        )
        self._by_id: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._by_id is not None:
            return self._by_id
        if not self._path.exists():
            self._by_id = {}
            return self._by_id
        df = pd.read_csv(self._path, usecols=["reaction_id", "reaction_name"])
        self._by_id = {
            str(r.reaction_id): str(r.reaction_name)
            for r in df.itertuples(index=False)
        }
        return self._by_id

    def describe_bm(self, bm_id: str) -> str:
        """One short token only (gene / symbol); id is added in format_feature_label."""
        if bm_id in _BM_OVERRIDE:
            return _BM_OVERRIDE[bm_id]
        names = self._load()
        rn = names.get(bm_id, "")
        if not rn:
            return "?"
        tag = _short_gene_symbol(rn)
        return tag[:14] if tag else "?"


_LOOKUP: ReactionLabelLookup | None = None


def _lookup() -> ReactionLabelLookup:
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = ReactionLabelLookup()
    return _LOOKUP


def format_feature_label(name: str) -> str:
    """Readable axis / legend label for one ML feature column."""
    if name == "biomass_fraction_required":
        return "ε (biomass threshold)"
    if name == "n_knockouts":
        return "Total knockout count"
    if name == "n_upregulations":
        return "Total upregulation count"

    if name.startswith("cond_"):
        suf = name[len("cond_") :]
        return _COND_SUFFIX_TO_LABEL.get(suf, suf.replace("_", " ").title())

    if name.startswith("ko_"):
        rid = name[3:]  # ko_bm00403 → bm00403
        body = _lookup().describe_bm(rid)
        return f"KO: {body} ({rid})"

    if name.startswith("up_"):
        rid = name[3:]
        body = _lookup().describe_bm(rid)
        return f"UP: {body} ({rid})"

    return name.replace("_", " ")
