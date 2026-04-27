"""
phaopt.train — Train and compare surrogate ML models for PHA and biomass flux prediction.

Functions
---------
split_features_targets   : Split an ML DataFrame into X (features) and Y (targets).
build_model              : Factory that returns a configured sklearn regressor by name.
train_surrogates         : Train a single model family for each target.
compare_surrogates       : Train and evaluate multiple model families with cross-validation.
save_train_results       : Persist trained models and metrics.
plot_model_comparison    : Publication-grade bar chart comparing all model families.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

from phaopt.io import save_dataframe
from phaopt.utils import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature / target splitting
# ---------------------------------------------------------------------------

_NON_FEATURE_COLS = {
    "pha_flux",
    "biomass_flux",
    "knockouts",
    "upregulations",
    "condition",
    "status",
    "max_biomass_unconstrained",
    "design_id",
}


def split_features_targets(
    df: pd.DataFrame,
    target_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an ML dataset DataFrame into features (X) and targets (Y).

    Parameters
    ----------
    df : pd.DataFrame
        ML dataset as produced by ``build_ml_dataset``.
    target_cols : list[str], optional
        Target column names.  Defaults to ``["pha_flux", "biomass_flux"]``.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature columns (ko_*, up_*, cond_*, biomass_fraction_required, ...).
    Y : pd.DataFrame
        Target columns.
    """
    if target_cols is None:
        target_cols = [c for c in ["pha_flux", "biomass_flux"] if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURE_COLS
        and c not in target_cols
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.uint8, np.bool_)
    ]

    X = df[feature_cols].copy()
    Y = df[target_cols].copy() if target_cols else pd.DataFrame()
    return X, Y


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

# Registry of supported model families and their default hyperparameters.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gradient_boosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "default_params": {"n_estimators": 200, "max_depth": 8},
    },
    "random_forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "default_params": {"n_estimators": 200, "max_depth": 12, "n_jobs": -1},
    },
    "xgboost": {
        "class": "xgboost.XGBRegressor",
        "default_params": {
            "n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
            "tree_method": "hist", "n_jobs": -1, "verbosity": 0,
        },
    },
    "elastic_net": {
        "class": "sklearn.linear_model.ElasticNet",
        "default_params": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 2000},
    },
}


def build_model(
    family: str,
    seed: int = 42,
    override_params: Optional[Dict[str, Any]] = None,
):
    """
    Instantiate a scikit-learn–compatible regressor by family name.

    Parameters
    ----------
    family : str
        Key in ``MODEL_REGISTRY`` (e.g. ``"random_forest"``, ``"xgboost"``).
    seed : int
        Random seed (injected as ``random_state`` where supported).
    override_params : dict, optional
        Override any default hyperparameters.

    Returns
    -------
    sklearn-compatible estimator
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[family]
    params = dict(entry["default_params"])
    if override_params:
        params.update(override_params)

    # Inject random_state where the estimator accepts it
    module_path, cls_name = entry["class"].rsplit(".", 1)

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)

    import inspect
    sig = inspect.signature(cls)
    if "random_state" in sig.parameters:
        params.setdefault("random_state", seed)

    return cls(**params)


# ---------------------------------------------------------------------------
# Grouped split helper
# ---------------------------------------------------------------------------

def _build_groups(X: pd.DataFrame) -> np.ndarray:
    """Build group keys from ko_/up_ columns (vectorised)."""
    ko_cols = sorted([c for c in X.columns if c.startswith("ko_")])
    up_cols = sorted([c for c in X.columns if c.startswith("up_")])

    if ko_cols or up_cols:
        parts = []
        if ko_cols:
            ko_arr = X[ko_cols].values.astype(np.int8)
            parts.append(np.packbits(ko_arr, axis=1))
        if up_cols:
            up_arr = X[up_cols].values.astype(np.int8)
            parts.append(np.packbits(up_arr, axis=1))
        combined = np.hstack(parts)
        _, inverse = np.unique(combined, axis=0, return_inverse=True)
        return inverse
    return np.arange(len(X))


# ---------------------------------------------------------------------------
# Single-family training (original API preserved)
# ---------------------------------------------------------------------------

def train_surrogates(
    df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 8,
    seed: int = 42,
    test_size: float = 0.20,
    model_family: str = "gradient_boosting",
) -> Dict[str, Any]:
    """
    Train surrogate regression models for ``pha_flux`` and ``biomass_flux``.

    Uses a grouped train/test split so that the same design signature never
    appears in both train and test sets (prevents data leakage).

    Parameters
    ----------
    df : pd.DataFrame
    n_estimators : int
    max_depth : int
    seed : int
    test_size : float
    model_family : str

    Returns
    -------
    dict
        Keys:
        - ``models``: dict mapping ``"<family>__<target>"`` to fitted model.
        - ``metrics``: DataFrame with R2, MAE, RMSE per target.
        - ``feature_names``: list of feature column names.
    """
    X, Y = split_features_targets(df)
    target_cols = list(Y.columns)

    logger.info(
        "Training surrogates (%s): %d samples, %d features, targets=%s",
        model_family, len(X), X.shape[1], target_cols,
    )

    groups = _build_groups(X)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    models: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []

    for target in target_cols:
        y_train = Y_train[target].values
        y_test = Y_test[target].values

        model = build_model(
            model_family, seed=seed,
            override_params={"n_estimators": n_estimators, "max_depth": max_depth}
            if model_family in ("gradient_boosting", "random_forest", "xgboost", "extra_trees")
            else None,
        )
        model.fit(X_train.values, y_train)

        y_pred = model.predict(X_test.values)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        key = f"{model_family}__{target}"
        models[key] = model

        metric_rows.append({
            "model_family": model_family,
            "target": target,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "R2": round(r2, 6),
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
        })

        logger.info("  %s — R2=%.4f  MAE=%.6f  RMSE=%.6f", target, r2, mae, rmse)

    return {
        "models": models,
        "metrics": pd.DataFrame(metric_rows),
        "feature_names": list(X.columns),
    }


# ---------------------------------------------------------------------------
# Multi-family comparison
# ---------------------------------------------------------------------------

def compare_surrogates(
    df: pd.DataFrame,
    families: Optional[List[str]] = None,
    seed: int = 42,
    test_size: float = 0.20,
    n_cv_splits: int = 5,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate multiple ML model families on the same data split.

    Parameters
    ----------
    df : pd.DataFrame
        ML dataset (from ``build_ml_dataset``).
    families : list[str], optional
        Model families to compare. Defaults to all available in ``MODEL_REGISTRY``.
    seed : int
    test_size : float
        Fraction held out for the final test set.
    n_cv_splits : int
        Number of cross-validation folds on the training set.
    scale_features : bool
        If True, standardise features (important for SVR, MLP, ElasticNet, KNN).

    Returns
    -------
    dict
        Keys:
        - ``comparison``: DataFrame with R2, MAE, RMSE per (family, target).
        - ``cv_results``: DataFrame with per-fold CV scores.
        - ``best_family``: dict with best family per target (by R2).
        - ``best_models``: dict mapping ``"<family>__<target>"`` to fitted model.
        - ``feature_names``: list of feature column names.
    """
    from sklearn.preprocessing import StandardScaler

    if families is None:
        families = _available_families()

    X, Y = split_features_targets(df)
    target_cols = list(Y.columns)

    logger.info(
        "Comparing %d model families: %s", len(families), families,
    )
    logger.info("Data: %d samples, %d features, targets=%s", len(X), X.shape[1], target_cols)

    # --- Single grouped split ------------------------------------------------
    groups = _build_groups(X)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, groups=groups))

    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    # --- Optional scaling ----------------------------------------------------
    if scale_features:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train_raw.values)
        X_test_arr = scaler.transform(X_test_raw.values)
    else:
        X_train_arr = X_train_raw.values
        X_test_arr = X_test_raw.values

    logger.info("Train: %d | Test: %d", len(X_train_arr), len(X_test_arr))

    comparison_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    all_models: Dict[str, Any] = {}
    best_models: Dict[str, Any] = {}
    best_r2: Dict[str, float] = {}
    best_family_per_target: Dict[str, str] = {}

    for family in families:
        logger.info("--- %s ---", family)
        t0 = time.time()

        for target in target_cols:
            y_train = Y_train[target].values
            y_test = Y_test[target].values

            try:
                model = build_model(family, seed=seed)

                # Fit on full training set
                model.fit(X_train_arr, y_train)
                y_pred = model.predict(X_test_arr)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                train_time = time.time() - t0

                # Cross-validation on training set
                try:
                    groups_train = groups[train_idx]
                    cv_splitter = GroupKFold(n_splits=n_cv_splits)
                    cv_model = build_model(family, seed=seed)
                    y_cv_pred = cross_val_predict(
                        cv_model, X_train_arr, y_train,
                        groups=groups_train, cv=cv_splitter,
                    )
                    cv_r2 = r2_score(y_train, y_cv_pred)
                    cv_mae = mean_absolute_error(y_train, y_cv_pred)
                    cv_rmse = float(np.sqrt(mean_squared_error(y_train, y_cv_pred)))
                except Exception as e:
                    logger.warning("  CV failed for %s/%s: %s", family, target, e)
                    cv_r2, cv_mae, cv_rmse = np.nan, np.nan, np.nan

            except Exception as e:
                logger.warning("  FAILED %s/%s: %s", family, target, e)
                r2, mae, rmse = np.nan, np.nan, np.nan
                cv_r2, cv_mae, cv_rmse = np.nan, np.nan, np.nan
                train_time = time.time() - t0

            comparison_rows.append({
                "model_family": family,
                "target": target,
                "n_train": len(X_train_arr),
                "n_test": len(X_test_arr),
                "n_features": X_train_arr.shape[1],
                "R2_test": round(r2, 6) if not np.isnan(r2) else np.nan,
                "MAE_test": round(mae, 6) if not np.isnan(mae) else np.nan,
                "RMSE_test": round(rmse, 6) if not np.isnan(rmse) else np.nan,
                "R2_cv": round(cv_r2, 6) if not np.isnan(cv_r2) else np.nan,
                "MAE_cv": round(cv_mae, 6) if not np.isnan(cv_mae) else np.nan,
                "RMSE_cv": round(cv_rmse, 6) if not np.isnan(cv_rmse) else np.nan,
                "train_time_s": round(train_time, 3),
            })

            logger.info(
                "  %s/%s — R2_test=%.4f  R2_cv=%.4f  MAE=%.6f  (%.1fs)",
                family, target,
                r2 if not np.isnan(r2) else -999,
                cv_r2 if not np.isnan(cv_r2) else -999,
                mae if not np.isnan(mae) else -999,
                train_time,
            )

            # Track all and best
            key = f"{family}__{target}"
            if not np.isnan(r2):
                all_models[key] = model
                if target not in best_r2 or r2 > best_r2[target]:
                    best_r2[target] = r2
                    best_family_per_target[target] = family
                    best_models[key] = model

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "comparison": comparison_df,
        "best_family": best_family_per_target,
        "best_models": best_models,
        "all_models": all_models,
        "feature_names": list(X.columns),
        # Same indices must be used for parity plots as for saved joblib models.
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }


def _available_families() -> List[str]:
    """Return model families that have their dependencies installed."""
    available = []
    for family in MODEL_REGISTRY:
        module_path = MODEL_REGISTRY[family]["class"].rsplit(".", 1)[0]
        try:
            import importlib
            importlib.import_module(module_path)
            available.append(family)
        except ImportError:
            logger.debug("Skipping %s — module %s not installed", family, module_path)
    return available


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Publication-grade grouped bar chart of R2, MAE, RMSE across model families.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output ``comparison`` from ``compare_surrogates()``.
    save_path : str, optional
        If given, save to this path (PNG, 300 dpi).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = comparison_df["target"].unique()
    families = comparison_df["model_family"].unique()

    fig, axes = plt.subplots(len(targets), 3, figsize=(16, 5 * len(targets)),
                             constrained_layout=True)
    if len(targets) == 1:
        axes = axes.reshape(1, -1)

    metrics = [("R2_test", "R²  (test)"), ("MAE_test", "MAE (test)"), ("RMSE_test", "RMSE (test)")]
    colors = plt.cm.Set2.colors

    for row, target in enumerate(targets):
        subset = comparison_df[comparison_df["target"] == target].set_index("model_family")

        for col, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            vals = [subset.loc[f, metric_col] if f in subset.index else 0.0 for f in families]
            x = np.arange(len(families))
            bars = ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(families))],
                          edgecolor="black", alpha=0.85)

            # Highlight best
            if metric_col == "R2_test":
                best_idx = int(np.nanargmax(vals))
            else:
                valid_vals = [v if not np.isnan(v) and v > 0 else np.inf for v in vals]
                best_idx = int(np.argmin(valid_vals))
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2.5)

            ax.set_xticks(x)
            ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{target} — {metric_label}")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Surrogate Model Comparison", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved model comparison figure: %s", save_path)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_parity(
    comparison_df: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 4,
    seed: int = 42,
    test_size: float = 0.20,
    save_path: Optional[str] = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    tables_dir: str = "results/tables",
) -> None:
    """
    Parity plots (predicted vs observed) for surrogate models.

    **Important:** Fitted models expect the *same* grouped train/test split and
    scaler as ``compare_surrogates``. Pass ``train_idx`` and ``test_idx`` from
    that run's return dict, or ensure ``surrogate_train_test_split.npz`` exists
    (written by ``save_train_results``). If indices are recomputed here, they
    may differ (e.g. different sklearn version), producing misleading parity plots.

    Publication-oriented layout:
    - Same model in each column for all targets (fixed order, not per-target R² sort).
    - Identical axis limits within each target row for direct visual comparison.
    - Observed ≈ 0 (infeasible / zero-flux designs) vs feasible points colour-coded.
    - Panel labels (a)–…; axis labels with units; R² (four decimals) and MAE on every panel.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import string

    X, Y = split_features_targets(df)
    targets = list(Y.columns)
    groups = _build_groups(X)

    if train_idx is None or test_idx is None:
        split_path = resolve_path(tables_dir) / "surrogate_train_test_split.npz"
        if split_path.exists():
            z = np.load(split_path)
            train_idx = np.asarray(z["train_idx"], dtype=np.int64)
            test_idx = np.asarray(z["test_idx"], dtype=np.int64)
            logger.info("Loaded train/test indices for parity from %s", split_path)
        else:
            logger.warning(
                "No surrogate_train_test_split.npz and no train_idx/test_idx passed; "
                "recomputing split — parity may NOT match saved models if sklearn/data differ. "
                "Re-run compare_surrogates + save_train_results, or pass indices from compare.",
            )
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, groups=groups))
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X.iloc[train_idx].values)
    X_test_arr = scaler.transform(X.iloc[test_idx].values)
    Y_test = Y.iloc[test_idx]

    fig, axes = plt.subplots(
        len(targets),
        top_n,
        figsize=(4.2 * top_n, 4.55 * len(targets)),
        constrained_layout=True,
    )
    if len(targets) == 1:
        axes = axes.reshape(1, -1)
    # Extra vertical gap between target rows (e.g. panel (d) vs (e) titles).
    fig.set_constrained_layout_pads(h_pad=0.03, w_pad=0.02, hspace=0.22, wspace=0.06)

    # Fixed column order: same model in each column for every target row.
    canonical_order = ["xgboost", "gradient_boosting", "random_forest", "elastic_net"]
    present = set(comparison_df["model_family"].astype(str).unique())
    fixed_families = [f for f in canonical_order if f in present][:top_n]

    family_labels = {
        "xgboost": "XGBoost",
        "gradient_boosting": "Gradient boosting",
        "random_forest": "Random Forest",
        "elastic_net": "Elastic Net",
    }

    # Mathtext axis labels (Bioresource Technology style).
    axis_label_xy = {
        "pha_flux": (
            r"Observed PHB flux (mmol gDW$^{-1}$ h$^{-1}$)",
            r"Predicted PHB flux (mmol gDW$^{-1}$ h$^{-1}$)",
        ),
        "biomass_flux": (
            r"Observed biomass flux (h$^{-1}$)",
            r"Predicted biomass flux (h$^{-1}$)",
        ),
    }

    panel_labels = list(string.ascii_lowercase)
    label_idx = 0
    models_dir = resolve_path("models")

    zero_thr = 1e-12

    for row, target in enumerate(targets):
        y_test = Y_test[target].values

        all_preds: list[np.ndarray] = []
        models_data: list[tuple[np.ndarray, float, float, str]] = []
        for family in fixed_families:
            model_path = models_dir / f"surrogate_{family}__{target}.joblib"
            ref = comparison_df[
                (comparison_df["model_family"] == family) & (comparison_df["target"] == target)
            ]
            r2_tab = float(ref["R2_test"].iloc[0]) if not ref.empty and not np.isnan(ref["R2_test"].iloc[0]) else None

            y_train_t = Y.iloc[train_idx][target].values
            y_pred: Optional[np.ndarray] = None

            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    y_try = model.predict(X_test_arr)
                    r2_try = r2_score(y_test, y_try)
                    if r2_tab is None or abs(r2_try - r2_tab) <= 0.02:
                        y_pred = y_try
                    else:
                        logger.warning(
                            "Loaded %s disagrees with metrics table (R² %.4f vs CSV %.4f); "
                            "refitting on the same split for the parity figure.",
                            model_path.name,
                            r2_try,
                            r2_tab,
                        )
                except Exception as exc:
                    logger.warning("Could not load %s (%s); refitting.", model_path, exc)

            if y_pred is None:
                model = build_model(family, seed=seed)
                model.fit(X_train_arr, y_train_t)
                y_pred = model.predict(X_test_arr)

            all_preds.append(y_pred)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            models_data.append((y_pred, r2, mae, family))

            if r2_tab is not None and abs(r2 - r2_tab) > 0.02:
                logger.warning(
                    "Parity R² (%.4f) differs from surrogate_comparison.csv (%.4f) for %s/%s — "
                    "regenerate CSV with the same train_idx/test_idx (run compare_surrogates).",
                    r2,
                    r2_tab,
                    family,
                    target,
                )

        global_min = float(min(y_test.min(), min(preds.min() for preds in all_preds)))
        global_max = float(max(y_test.max(), max(preds.max() for preds in all_preds)))
        span = global_max - global_min
        padding = span * 0.05 if span > 0 else 1e-6
        global_min -= padding
        global_max += padding

        for col, (y_pred, r2, mae, family) in enumerate(models_data):
            ax = axes[row, col]

            infeasible_mask = np.abs(y_test) <= zero_thr
            feasible_mask = ~infeasible_mask

            ax.scatter(
                y_test[infeasible_mask],
                y_pred[infeasible_mask],
                alpha=0.45,
                s=6,
                c="#C62828",
                edgecolors="none",
                label="Infeasible (observed ≈ 0)",
                rasterized=True,
                zorder=2,
            )
            ax.scatter(
                y_test[feasible_mask],
                y_pred[feasible_mask],
                alpha=0.28,
                s=6,
                c="#1565C0",
                edgecolors="none",
                label="Feasible",
                rasterized=True,
                zorder=2,
            )

            lims = [global_min, global_max]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            # Identity reference (y = x); grey so it does not clash with infeasible (red) points.
            ax.plot(
                lims,
                lims,
                color="#616161",
                linestyle="--",
                linewidth=1.35,
                alpha=0.95,
                zorder=3,
                label="_nolegend_",
            )

            x_label, y_label = axis_label_xy.get(
                target,
                ("Observed", "Predicted"),
            )
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)

            fam_name = family_labels.get(family, family.replace("_", " ").title())
            letter = panel_labels[label_idx]
            label_idx += 1
            metrics_line = rf"$R^2$ = {r2:.4f}  |  MAE = {mae:.4f}"
            ax.set_title(
                rf"({letter}) {fam_name}" + "\n" + metrics_line,
                fontsize=10,
                loc="left",
                pad=10 if row == 0 else 12,
            )

            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.35, linestyle="-", linewidth=0.4)

            if col == 0:
                ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92)

    if save_path:
        out = resolve_path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved parity plots: %s", out)
        plt.close(fig)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_train_results(
    results: Dict[str, Any],
    models_dir: str = "models",
    tables_dir: str = "results/tables",
) -> None:
    """
    Save trained model artefacts and metrics.

    Parameters
    ----------
    results : dict
        Output of ``train_surrogates()`` or ``compare_surrogates()``.
    models_dir : str
        Directory for joblib model files.
    tables_dir : str
        Directory for metrics CSV/parquet.
    """
    mdir = resolve_path(models_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    # Save models (from train_surrogates)
    for key, model in results.get("models", {}).items():
        out = mdir / f"surrogate_{key}.joblib"
        joblib.dump(model, out)
        logger.info("Saved model: %s", out)

    # Save ALL models (from compare_surrogates) so every family is available for SHAP
    for key, model in results.get("all_models", {}).items():
        out = mdir / f"surrogate_{key}.joblib"
        joblib.dump(model, out)
        logger.info("Saved model: %s", out)

    # Save best models (from compare_surrogates) — overwrites duplicates, no harm
    for key, model in results.get("best_models", {}).items():
        out = mdir / f"surrogate_{key}.joblib"
        joblib.dump(model, out)
        logger.info("Saved best model: %s", out)

    # Save metrics table
    if "metrics" in results:
        save_dataframe(results["metrics"], f"{tables_dir}/surrogate_metrics")
    if "comparison" in results:
        save_dataframe(results["comparison"], f"{tables_dir}/surrogate_comparison")

    tdir = resolve_path(tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    if "train_idx" in results and "test_idx" in results:
        split_npz = tdir / "surrogate_train_test_split.npz"
        np.savez_compressed(
            split_npz,
            train_idx=results["train_idx"],
            test_idx=results["test_idx"],
        )
        logger.info("Saved grouped train/test index split: %s", split_npz)

    logger.info("Saved surrogate results.")
