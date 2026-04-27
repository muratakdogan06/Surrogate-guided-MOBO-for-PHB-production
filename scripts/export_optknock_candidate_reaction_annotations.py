#!/usr/bin/env python3
"""
Export SBML/COBRA annotations for every reaction in ``optknock_single_ko_results.csv``.

Reads the single-KO table (defines the 59-reaction order), loads the extended GEM
from ``configs/model_config.yaml``, and writes a wide CSV suitable for Table SX:
reaction name, subsystem, EC / PubMed from ``reaction.annotation``, GPR, model gene
IDs, plus gene *symbols* parsed from the reaction *name* when the curator encoded
them as ``(...)` (e.g. ``(fadA)``, ``(atoB)``).

Model gene IDs are ``BMD_xxxx`` style; RefSeq ``bm`` locus tags are not stored in
this SBML — use ``symbols_from_reaction_name`` and external mapping when needed.

Output: ``results/tables/optknock_candidate_reaction_annotations.csv``
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cobra
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.utils import load_model_config, resolve_path  # noqa: E402

DEFAULT_KO = PROJECT_ROOT / "optknock_single_ko_results.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "tables" / "optknock_candidate_reaction_annotations.csv"

# Gene-like tokens in parentheses inside the reaction *name* field.
_SYMBOL_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9_]*)\)")


def _flatten_annotation_value(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple, set)):
        parts = [str(x).strip() for x in val if str(x).strip()]
        return ";".join(parts)
    return str(val).strip()


def _ec_field(ann: dict) -> str:
    if not ann:
        return ""
    raw = ann.get("ec-code") or ann.get("EC Number") or ann.get("ec_number")
    return _flatten_annotation_value(raw)


def _pubmed_field(ann: dict) -> str:
    if not ann:
        return ""
    raw = ann.get("pubmed") or ann.get("pmid")
    return _flatten_annotation_value(raw)


def _symbols_from_reaction_name(name: str) -> str:
    if not name:
        return ""
    found = _SYMBOL_RE.findall(name)
    # preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return ";".join(out)


def _genes_verbose(rxn: cobra.Reaction) -> str:
    parts: list[str] = []
    for g in sorted(rxn.genes, key=lambda x: x.id):
        nm = (g.name or "").replace("\n", " ").strip()
        parts.append(f"{g.id}:{nm}" if nm else g.id)
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--ko-csv",
        type=Path,
        default=DEFAULT_KO,
        help="Single-KO results CSV (must contain column 'reaction').",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV path.",
    )
    args = ap.parse_args()

    mcfg = load_model_config()
    model_path = resolve_path(mcfg["extended_model_path"])
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    ko_df = pd.read_csv(args.ko_csv)
    if "reaction" not in ko_df.columns:
        raise SystemExit(f"Column 'reaction' missing in {args.ko_csv}")

    model = cobra.io.read_sbml_model(str(model_path))
    rxn_ids = {r.id for r in model.reactions}

    rows: list[dict[str, object]] = []
    for rank, row in enumerate(ko_df.itertuples(index=False), start=1):
        rid = getattr(row, "reaction")
        if rid not in rxn_ids:
            rows.append(
                {
                    "rank": rank,
                    "reaction_id": rid,
                    "in_model": False,
                    "reaction_name": "",
                    "subsystem": "",
                    "ec_code": "",
                    "pubmed_ids": "",
                    "gene_reaction_rule": "",
                    "gene_ids_sorted": "",
                    "genes_id_name": "",
                    "symbols_from_reaction_name": "",
                    "phb_flux": getattr(row, "phb_flux", ""),
                    "biomass": getattr(row, "biomass", ""),
                    "delta_phb": getattr(row, "delta_phb", ""),
                    "growth_coupled": getattr(row, "growth_coupled", ""),
                    "lethal": getattr(row, "lethal", ""),
                }
            )
            continue

        rxn = model.reactions.get_by_id(rid)
        ann = dict(rxn.annotation) if rxn.annotation else {}
        grr = (rxn.gene_reaction_rule or "").replace("\n", " ").strip()
        rname = (rxn.name or "").replace("\n", " ").strip()
        subs = (rxn.subsystem or "").replace("\n", " ").strip()
        gids = sorted({g.id for g in rxn.genes})

        base = {c: getattr(row, c, "") for c in ko_df.columns}
        rows.append(
            {
                "rank": rank,
                "reaction_id": rid,
                "in_model": True,
                "reaction_name": rname,
                "subsystem": subs,
                "ec_code": _ec_field(ann),
                "pubmed_ids": _pubmed_field(ann),
                "gene_reaction_rule": grr,
                "gene_ids_sorted": ";".join(gids),
                "genes_id_name": _genes_verbose(rxn),
                "symbols_from_reaction_name": _symbols_from_reaction_name(rname),
                **base,
            }
        )

    out_df = pd.DataFrame(rows)
    # Stable column order: annotations first, then original KO columns (drop duplicate ``reaction``)
    first = [
        "rank",
        "reaction_id",
        "in_model",
        "reaction_name",
        "subsystem",
        "ec_code",
        "pubmed_ids",
        "gene_reaction_rule",
        "gene_ids_sorted",
        "genes_id_name",
        "symbols_from_reaction_name",
    ]
    rest = [c for c in out_df.columns if c not in first and c != "reaction"]
    out_df = out_df[first + rest]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
