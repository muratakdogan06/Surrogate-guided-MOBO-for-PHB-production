#!/usr/bin/env python3
"""
Export supplementary cross-target SHAP ranking tables (XGBoost).

Definitions
-----------
For each feature f:
    cross_target_score(f) = mean_abs_shap_phb(f) + mean_abs_shap_biomass(f)

Outputs
-------
1) Feature-level table (all model features):
   results/tables/supp_table_cross_target_feature_ranking_xgboost.csv

2) Reaction-level table (59 candidate reactions; KO+UP collapsed per reaction):
   results/tables/supp_table_cross_target_reaction_ranking_xgboost.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.shap_feature_display import format_feature_label  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
P_PHB = TABLES_DIR / "shap_importance_xgboost_pha_flux.csv"
P_BIO = TABLES_DIR / "shap_importance_xgboost_biomass_flux.csv"

OUT_FEATURE = TABLES_DIR / "supp_table_cross_target_feature_ranking_xgboost.csv"
OUT_REACTION = TABLES_DIR / "supp_table_cross_target_reaction_ranking_xgboost.csv"

_BM_RE = re.compile(r"^(ko|up)_(bm\d+)$")


def _feature_type(name: str) -> str:
    if name.startswith("ko_") or name.startswith("up_"):
        return "reaction_perturbation"
    if name.startswith("cond_"):
        return "environment"
    if name == "biomass_fraction_required":
        return "constraint"
    return "meta"


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return np.inf
    return float(num / den)


def main() -> None:
    if not P_PHB.exists() or not P_BIO.exists():
        raise SystemExit(
            "Missing SHAP importance inputs. Run scripts/08_shap_analysis_publication_grade.py --model-family xgboost"
        )

    df_phb = pd.read_csv(P_PHB).rename(columns={"mean_abs_shap": "mean_abs_shap_phb"})
    df_bio = pd.read_csv(P_BIO).rename(columns={"mean_abs_shap": "mean_abs_shap_biomass"})

    merged = df_phb.merge(df_bio, on="feature", how="outer").fillna(0.0)
    merged["cross_target_score"] = merged["mean_abs_shap_phb"] + merged["mean_abs_shap_biomass"]
    merged["specificity_ratio_phb_to_biomass"] = merged.apply(
        lambda r: _safe_ratio(float(r["mean_abs_shap_phb"]), float(r["mean_abs_shap_biomass"])),
        axis=1,
    )
    merged["feature_type"] = merged["feature"].map(_feature_type)
    merged["display_label"] = merged["feature"].map(format_feature_label)

    pert = merged["feature"].str.extract(_BM_RE)
    merged["perturbation"] = pert[0]
    merged["reaction_id"] = pert[1]

    merged = merged.sort_values("cross_target_score", ascending=False).reset_index(drop=True)
    merged["cross_target_rank_overall"] = np.arange(1, len(merged) + 1)

    reaction_mask = merged["feature_type"] == "reaction_perturbation"
    merged["cross_target_rank_within_reaction_features"] = np.nan
    merged.loc[reaction_mask, "cross_target_rank_within_reaction_features"] = (
        merged.loc[reaction_mask, "cross_target_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    feat_cols = [
        "cross_target_rank_overall",
        "feature",
        "display_label",
        "feature_type",
        "reaction_id",
        "perturbation",
        "mean_abs_shap_phb",
        "mean_abs_shap_biomass",
        "cross_target_score",
        "specificity_ratio_phb_to_biomass",
        "cross_target_rank_within_reaction_features",
    ]
    feature_out = merged[feat_cols].copy()
    feature_out.to_csv(OUT_FEATURE, index=False)

    # Reaction-level collapsed table (KO + UP summed per reaction).
    rx = merged[reaction_mask].copy()
    rx_agg = (
        rx.groupby("reaction_id", as_index=False)
        .agg(
            ko_mean_abs_shap_phb=("mean_abs_shap_phb", lambda s: float(s[rx.loc[s.index, "perturbation"] == "ko"].sum())),
            up_mean_abs_shap_phb=("mean_abs_shap_phb", lambda s: float(s[rx.loc[s.index, "perturbation"] == "up"].sum())),
            ko_mean_abs_shap_biomass=(
                "mean_abs_shap_biomass",
                lambda s: float(s[rx.loc[s.index, "perturbation"] == "ko"].sum()),
            ),
            up_mean_abs_shap_biomass=(
                "mean_abs_shap_biomass",
                lambda s: float(s[rx.loc[s.index, "perturbation"] == "up"].sum()),
            ),
            display_label=("reaction_id", lambda s: format_feature_label(f"ko_{s.iloc[0]}")),
        )
    )
    rx_agg["mean_abs_shap_phb_total"] = rx_agg["ko_mean_abs_shap_phb"] + rx_agg["up_mean_abs_shap_phb"]
    rx_agg["mean_abs_shap_biomass_total"] = (
        rx_agg["ko_mean_abs_shap_biomass"] + rx_agg["up_mean_abs_shap_biomass"]
    )
    rx_agg["cross_target_score"] = rx_agg["mean_abs_shap_phb_total"] + rx_agg["mean_abs_shap_biomass_total"]
    rx_agg["specificity_ratio_phb_to_biomass"] = rx_agg.apply(
        lambda r: _safe_ratio(float(r["mean_abs_shap_phb_total"]), float(r["mean_abs_shap_biomass_total"])),
        axis=1,
    )
    rx_agg = rx_agg.sort_values("cross_target_score", ascending=False).reset_index(drop=True)
    rx_agg["cross_target_rank_reaction"] = np.arange(1, len(rx_agg) + 1)

    rx_cols = [
        "cross_target_rank_reaction",
        "reaction_id",
        "display_label",
        "ko_mean_abs_shap_phb",
        "up_mean_abs_shap_phb",
        "mean_abs_shap_phb_total",
        "ko_mean_abs_shap_biomass",
        "up_mean_abs_shap_biomass",
        "mean_abs_shap_biomass_total",
        "cross_target_score",
        "specificity_ratio_phb_to_biomass",
    ]
    reaction_out = rx_agg[rx_cols].copy()
    reaction_out.to_csv(OUT_REACTION, index=False)

    # Consistency checks
    if reaction_out.shape[0] != 59:
        raise SystemExit(f"Expected 59 reactions; got {reaction_out.shape[0]}")
    if not np.allclose(
        feature_out["cross_target_score"].to_numpy(),
        feature_out["mean_abs_shap_phb"].to_numpy() + feature_out["mean_abs_shap_biomass"].to_numpy(),
        rtol=0,
        atol=1e-15,
    ):
        raise SystemExit("Feature-level cross_target_score consistency check failed.")

    print(f"Wrote feature-level table  -> {OUT_FEATURE}")
    print(f"Wrote reaction-level table -> {OUT_REACTION}")
    print("Top 5 reaction-level ranks:")
    print(reaction_out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

