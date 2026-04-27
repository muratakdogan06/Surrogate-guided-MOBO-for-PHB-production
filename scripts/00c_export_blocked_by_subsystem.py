#!/usr/bin/env python3
"""
00c_export_blocked_by_subsystem.py — Table 2 verisi için COBRA çıktısı.

1) find_blocked_reactions (FVA tabanlı) ile bloklu reaksiyon kimlikleri
2) Her reaksiyon için reaction.subsystem ile gruplama
3) CSV yazımı + isteğe bağlı üst-seviye (manuscript) rollup

Çıktılar (varsayılan):
  results/tables/blocked_reactions_detail.csv
  results/tables/blocked_by_subsystem.csv
    — subsystem, n_reactions_in_model, n_blocked, pct_blocked_within_subsystem, ...
  results/tables/blocked_by_subsystem_rollup.csv
    — rollup_category, n_reactions_in_model, n_blocked, pct_blocked_within_rollup, ...

Not: open_exchanges=True/False Table 2 tanımını değiştirir; makale ile aynı seçin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cobra  # noqa: E402
from cobra.flux_analysis import find_blocked_reactions  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Manuscript tarzı üst kategorilere yaklaştırmak için sırayla ilk eşleşen kural kazanır.
# İhtiyaç halinde configs/blocked_subsystem_rollup.yaml ile genişletilebilir (bk. _load_yaml_rules).
ROLLUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Exchange reactions",
        ("exchange", "demand", "sink"),
    ),
    (
        "Transport",
        ("transport", "transporter", "diffusion", "channel"),
    ),
    (
        "Cofactor and vitamin metabolism",
        (
            "cofactor",
            "vitamin",
            "folate",
            "porphyrin",
            "quinone",
            "iron-sulfur",
            "one carbon",
        ),
    ),
    (
        "Amino acid metabolism",
        (
            "amino acid",
            "protein",
            "tRNA",
            "peptide",
            "nitrogen",
        ),
    ),
    (
        "Carbohydrate metabolism",
        (
            "carbohydrate",
            "glycolysis",
            "gluconeogenesis",
            "pentose",
            "starch",
            "sucrose",
            "fructose",
            "galactose",
            "glycogen",
        ),
    ),
]


def _load_yaml_rules() -> list[tuple[str, tuple[str, ...]]] | None:
    p = PROJECT_ROOT / "configs" / "blocked_subsystem_rollup.yaml"
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text()) or {}
    rules = data.get("rollup_contains") or []
    out: list[tuple[str, tuple[str, ...]]] = []
    for row in rules:
        if not isinstance(row, dict):
            continue
        cat = row.get("category")
        keys = row.get("substrings") or row.get("contains")
        if not cat or not keys:
            continue
        if isinstance(keys, str):
            keys = [keys]
        out.append((str(cat), tuple(str(k).lower() for k in keys)))
    return out or None


def _rollup_bucket(subsystem: str, boundary: bool, rules: list[tuple[str, tuple[str, ...]]]) -> str:
    s = (subsystem or "").strip().lower()
    if boundary:
        return "Exchange reactions"
    for bucket, needles in rules:
        if any(n in s for n in needles):
            return bucket
    return "Other / unassigned"


def _model_reactions_by_subsystem_and_rollup(
    model: cobra.Model, rules: list[tuple[str, tuple[str, ...]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tüm model reaksiyonları için subsystem ve rollup kategorisi.

    Dönüş
    -------
    by_subsystem : columns subsystem, n_reactions_in_model
    by_rollup    : columns rollup_category, n_reactions_in_model
    """
    rows = []
    for r in model.reactions:
        sub = (r.subsystem or "").strip() or "Unassigned"
        roll = _rollup_bucket(sub, bool(r.boundary), rules)
        rows.append({"subsystem": sub, "rollup_category": roll})
    df = pd.DataFrame(rows)
    by_sub = (
        df.groupby("subsystem", dropna=False)
        .size()
        .reset_index(name="n_reactions_in_model")
    )
    by_roll = (
        df.groupby("rollup_category", dropna=False)
        .size()
        .reset_index(name="n_reactions_in_model")
    )
    return by_sub, by_roll


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blocked reactions grouped by COBRA subsystem.")
    parser.add_argument(
        "--open-exchanges",
        action="store_true",
        help="Pass open_exchanges=True to find_blocked_reactions (daha gevşek ortam).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="SBML yolu (yoksa configs/model_config.yaml extended_model_path).",
    )
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        with open(PROJECT_ROOT / "configs" / "model_config.yaml") as f:
            cfg = yaml.safe_load(f)
        model_path = PROJECT_ROOT / cfg["extended_model_path"]

    model = cobra.io.read_sbml_model(str(model_path))
    print(f"Model: {model_path}")
    print(f"Reactions: {len(model.reactions)}, metabolites: {len(model.metabolites)}")

    open_ex = bool(args.open_exchanges)
    print(f"find_blocked_reactions(open_exchanges={open_ex}) ...")
    blocked = find_blocked_reactions(model, open_exchanges=open_ex)
    blocked_set = set(blocked)
    print(f"Blocked count: {len(blocked_set)}")

    rows = []
    for rid in sorted(blocked_set):
        r = model.reactions.get_by_id(rid)
        sub = (r.subsystem or "").strip() or "Unassigned"
        rows.append(
            {
                "reaction_id": rid,
                "reaction_name": r.name,
                "subsystem": sub,
                "boundary": r.boundary,
                "lower_bound": r.lower_bound,
                "upper_bound": r.upper_bound,
            }
        )

    detail = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = TABLES_DIR / "blocked_reactions_detail.csv"
    detail.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path.relative_to(PROJECT_ROOT)}")

    rules = _load_yaml_rules() or ROLLUP_RULES
    totals_sub, totals_roll = _model_reactions_by_subsystem_and_rollup(model, rules)

    blocked_counts = (
        detail.groupby("subsystem", dropna=False).size().reset_index(name="n_blocked")
        if len(detail)
        else pd.DataFrame(columns=["subsystem", "n_blocked"])
    )
    # Tüm subsystem satırları (bloklu = 0 olanlar dahil)
    by_sub = totals_sub.merge(blocked_counts, on="subsystem", how="left")
    by_sub["n_blocked"] = by_sub["n_blocked"].fillna(0).astype(int)
    by_sub = by_sub.sort_values(["n_blocked", "n_reactions_in_model"], ascending=[False, False])

    n_total = len(model.reactions)
    n_blocked = len(blocked_set)
    by_sub["pct_blocked_within_subsystem"] = (
        100.0 * by_sub["n_blocked"] / by_sub["n_reactions_in_model"].replace(0, pd.NA)
    ).round(2)
    by_sub["pct_of_blocked"] = (100.0 * by_sub["n_blocked"] / max(n_blocked, 1)).round(2)
    by_sub["pct_of_all_reactions"] = (100.0 * by_sub["n_blocked"] / max(n_total, 1)).round(2)
    # Kolon sırası: önce toplam, sonra bloklu
    col_order = [
        "subsystem",
        "n_reactions_in_model",
        "n_blocked",
        "pct_blocked_within_subsystem",
        "pct_of_blocked",
        "pct_of_all_reactions",
    ]
    by_sub = by_sub[col_order]
    sub_path = TABLES_DIR / "blocked_by_subsystem.csv"
    by_sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path.relative_to(PROJECT_ROOT)}")

    detail["rollup_category"] = [
        _rollup_bucket(sub, bool(boundary), rules) for sub, boundary in zip(detail["subsystem"], detail["boundary"])
    ]
    roll_blocked = (
        detail.groupby("rollup_category", dropna=False).size().reset_index(name="n_blocked")
        if len(detail)
        else pd.DataFrame(columns=["rollup_category", "n_blocked"])
    )
    rollup = totals_roll.merge(roll_blocked, on="rollup_category", how="left")
    rollup["n_blocked"] = rollup["n_blocked"].fillna(0).astype(int)
    rollup = rollup.sort_values(["n_blocked", "n_reactions_in_model"], ascending=[False, False])
    rollup["pct_blocked_within_rollup"] = (
        100.0 * rollup["n_blocked"] / rollup["n_reactions_in_model"].replace(0, pd.NA)
    ).round(2)
    rollup["pct_of_blocked"] = (100.0 * rollup["n_blocked"] / max(n_blocked, 1)).round(2)
    rollup = rollup[
        [
            "rollup_category",
            "n_reactions_in_model",
            "n_blocked",
            "pct_blocked_within_rollup",
            "pct_of_blocked",
        ]
    ]
    roll_path = TABLES_DIR / "blocked_by_subsystem_rollup.csv"
    rollup.to_csv(roll_path, index=False)
    print(f"Wrote {roll_path.relative_to(PROJECT_ROOT)}")
    print("\nRollup preview:")
    print(rollup.to_string(index=False))


if __name__ == "__main__":
    main()
