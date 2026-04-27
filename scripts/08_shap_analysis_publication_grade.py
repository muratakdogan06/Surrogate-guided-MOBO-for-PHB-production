#!/usr/bin/env python3
# ============================================================================
# 08_shap_analysis_publication_grade.py
# Publication-grade SHAP analysis on held-out grouped test split
# ============================================================================
"""
Runs publication-grade SHAP analysis for surrogate models.

Key improvements
----------------
1. Uses the same leakage-safe grouped split logic as surrogate training
2. Runs SHAP on held-out test designs only
3. Supports both pha_flux and biomass_flux
4. Reproducible subsampling with fixed random seed
5. Target/model-specific output filenames
6. Cross-target comparison table

Example
-------
python scripts/08_shap_analysis_publication_grade.py \
    --model-family gradient_boosting \
    --targets pha_flux biomass_flux \
    --sample-size 1000 \
    --random-seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaopt.io import load_dataframe, save_dataframe  # noqa: E402
from phaopt.shap_analysis import run_shap_analysis  # noqa: E402
from phaopt.train import split_features_targets  # noqa: E402
from phaopt.utils import setup_logging  # noqa: E402

logger = setup_logging("pipeline.08.shap")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-grade SHAP analysis on held-out grouped test split."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/ml_dataset.parquet",
        help="Path to processed ML dataset parquet.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="auto",
        choices=["auto", "gradient_boosting", "xgboost", "random_forest"],
        help=(
            "Which surrogate model family to explain. "
            "With 'auto', the first existing file is used: XGBoost, then gradient boosting, then random forest."
        ),
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["pha_flux", "biomass_flux"],
        choices=["pha_flux", "biomass_flux"],
        help="Targets to analyze.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of grouped held-out signatures for test split.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Maximum number of held-out rows used for SHAP.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for grouped split and SHAP subsampling.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Max features in bar / importance table; dual bar panels use this.",
    )
    parser.add_argument(
        "--beeswarm-max-display",
        type=int,
        default=12,
        help="Max features in beeswarm only (default 12; reduces crowding near zero).",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="results/figures",
        help="Directory for SHAP figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=str,
        default="results/tables",
        help="Directory for SHAP tables.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_intervention_signature(df: pd.DataFrame) -> pd.Series:
    """
    Build signature from knockouts/upregulations if available.
    Falls back to encoded ko_/up_ columns if raw columns are unavailable.
    """
    if {"knockouts", "upregulations"}.issubset(df.columns):
        def normalize(val):
            if isinstance(val, list):
                return tuple(sorted(map(str, val)))
            if pd.isna(val):
                return tuple()
            txt = str(val).strip()
            if txt in ("", "[]", "nan", "None"):
                return tuple()
            if "|" in txt:
                return tuple(sorted([x for x in txt.split("|") if x]))
            if "," in txt:
                return tuple(sorted([x.strip() for x in txt.split(",") if x.strip()]))
            return (txt,)

        ko = df["knockouts"].apply(normalize)
        up = df["upregulations"].apply(normalize)
        return ko.astype(str) + "||" + up.astype(str)

    ko_cols = sorted([c for c in df.columns if c.startswith("ko_")])
    up_cols = sorted([c for c in df.columns if c.startswith("up_")])

    if not ko_cols and not up_cols:
        raise ValueError(
            "Cannot build intervention signatures. "
            "Neither raw knockouts/upregulations nor ko_/up_ encoded columns were found."
        )

    def active_from_row(row: pd.Series, cols: List[str]) -> Tuple[str, ...]:
        return tuple(sorted([c for c in cols if int(row[c]) == 1]))

    ko = df[ko_cols].apply(lambda r: active_from_row(r, ko_cols), axis=1)
    up = df[up_cols].apply(lambda r: active_from_row(r, up_cols), axis=1)
    return ko.astype(str) + "||" + up.astype(str)


def make_grouped_test_subset(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> pd.DataFrame:
    groups = make_intervention_signature(df).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    idx_train, idx_test = next(splitter.split(df, groups=groups))
    test_df = df.iloc[idx_test].copy().reset_index(drop=True)

    logger.info(
        "Grouped held-out split created: total rows=%d | test rows=%d | unique test signatures=%d",
        len(df),
        len(test_df),
        pd.Series(groups[idx_test]).nunique(),
    )
    return test_df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_model_path(model_family: str, target: str) -> Path:
    candidates = []
    if model_family == "auto":
        # Prefer XGBoost for publication figures (beeswarm / bar filenames use this family).
        candidates = [
            PROJECT_ROOT / f"models/surrogate_xgboost__{target}.joblib",
            PROJECT_ROOT / f"models/surrogate_gradient_boosting__{target}.joblib",
            PROJECT_ROOT / f"models/surrogate_random_forest__{target}.joblib",
        ]
    else:
        candidates = [
            PROJECT_ROOT / f"models/surrogate_{model_family}__{target}.joblib",
        ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No trained model found for target='{target}' and model_family='{model_family}'. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def infer_model_name_from_path(path: Path) -> str:
    name = path.stem
    # surrogate_gradient_boosting__pha_flux -> gradient_boosting
    if name.startswith("surrogate_") and "__" in name:
        return name[len("surrogate_"):].split("__")[0]
    return name


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    importance_by_target: Dict[str, pd.DataFrame],
    model_name: str,
    out_dir: Path,
) -> Path:
    merged = None

    for target, df in importance_by_target.items():
        cur = df.copy()
        cur = cur.rename(columns={"mean_abs_shap": f"mean_abs_shap__{target}"})
        cur = cur[["feature", f"mean_abs_shap__{target}"]]

        if merged is None:
            merged = cur
        else:
            merged = merged.merge(cur, on="feature", how="outer")

    if merged is None:
        raise ValueError("No SHAP importance tables available to compare.")

    merged = merged.fillna(0.0)

    shap_cols = [c for c in merged.columns if c.startswith("mean_abs_shap__")]
    merged["combined_score"] = merged[shap_cols].sum(axis=1)
    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)

    out_path = out_dir / f"shap_top_features_comparison_{model_name}.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Saved comparison table: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    fig_dir = PROJECT_ROOT / args.fig_dir
    table_dir = PROJECT_ROOT / args.table_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("STEP 8 — Publication-grade SHAP analysis")
    logger.info("=" * 72)
    logger.info("Dataset      : %s", args.dataset)
    logger.info("Model family : %s", args.model_family)
    logger.info("Targets      : %s", args.targets)
    logger.info("Sample size  : %d", args.sample_size)
    logger.info("Random seed  : %d", args.random_seed)

    df = load_dataframe(args.dataset)
    logger.info("Loaded dataset: %d rows × %d cols", *df.shape)

    # Build held-out grouped test subset first
    test_df = make_grouped_test_subset(
        df=df,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    # Convert test subset into leakage-safe features/targets
    X_test, Y_test = split_features_targets(test_df)
    logger.info("Held-out feature matrix: %d rows × %d cols", *X_test.shape)

    importance_by_target: Dict[str, pd.DataFrame] = {}
    shap_bundle_by_target: Dict[str, Dict] = {}
    used_model_name = None

    dual_bar_panels = (
        len(args.targets) >= 2
        and "pha_flux" in args.targets
        and "biomass_flux" in args.targets
    )

    for target in args.targets:
        model_path = resolve_model_path(args.model_family, target)
        model_name = infer_model_name_from_path(model_path)
        used_model_name = model_name if used_model_name is None else used_model_name

        logger.info("Loading model for %s: %s", target, model_path)
        model = joblib.load(model_path)

        results = run_shap_analysis(
            model=model,
            X=X_test,
            feature_names=list(X_test.columns),
            max_display=args.max_display,
            out_dir=str(fig_dir),
            table_out_dir=str(table_dir),
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            model_name=model_name,
            target_name=target,
            plot_bar=not dual_bar_panels,
            plot_beeswarm=not dual_bar_panels,
            beeswarm_max_display=args.beeswarm_max_display,
        )

        imp = results["importance_df"].copy()
        importance_by_target[target] = imp
        shap_bundle_by_target[target] = results

        # Save the sampled held-out rows used in SHAP, for reproducibility
        sample_idx = results.get("sample_idx", None)
        if sample_idx is not None:
            sampled_rows = test_df.iloc[sample_idx].copy().reset_index(drop=True)
            sampled_out = table_dir / f"shap_sampled_rows_{model_name}_{target}.parquet"
            save_dataframe(sampled_rows, str(sampled_out).replace(".parquet", ""))
            logger.info("Saved SHAP sampled rows: %s", sampled_out)

        logger.info(
            "Finished SHAP for target=%s | model=%s | top feature=%s",
            target,
            model_name,
            imp.iloc[0]["feature"] if not imp.empty else "NA",
        )

    # Build cross-target comparison if >1 target analyzed
    if len(importance_by_target) >= 2 and used_model_name is not None:
        build_comparison_table(
            importance_by_target=importance_by_target,
            model_name=used_model_name,
            out_dir=table_dir,
        )

    if (
        dual_bar_panels
        and used_model_name is not None
        and "pha_flux" in importance_by_target
        and "biomass_flux" in importance_by_target
    ):
        from phaopt.shap_pub_plots import (
            plot_dual_mean_abs_shap_bar_panels,
            plot_dual_shap_beeswarm_panels,
        )

        plot_dual_mean_abs_shap_bar_panels(
            importance_by_target["pha_flux"],
            importance_by_target["biomass_flux"],
            fig_dir / f"shap_bar_{used_model_name}_pha_biomass_panels.png",
            top_k=args.max_display,
        )

        rp = shap_bundle_by_target["pha_flux"]
        rb = shap_bundle_by_target["biomass_flux"]
        plot_dual_shap_beeswarm_panels(
            shap_pha=rp["shap_values"],
            X_display_pha=rp["X_display"],
            imp_pha=importance_by_target["pha_flux"],
            shap_bio=rb["shap_values"],
            X_display_bio=rb["X_display"],
            imp_bio=importance_by_target["biomass_flux"],
            feature_names=list(X_test.columns),
            outpath=fig_dir / f"shap_beeswarm_{used_model_name}_pha_biomass_panels.png",
            top_k=args.beeswarm_max_display,
            base_pha=rp.get("expected_value"),
            base_bio=rb.get("expected_value"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()