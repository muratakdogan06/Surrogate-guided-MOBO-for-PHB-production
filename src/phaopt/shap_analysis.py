"""
phaopt.shap_analysis — SHAP-based feature importance analysis for surrogates.

Functions
---------
run_shap_analysis : Compute SHAP values, generate plots, and return importance table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)

# Copy into manuscript caption / methods (beeswarm interpretation).
BEESWARM_CAPTION_GUIDANCE_EN = (
    "Beeswarm rows are ordered by mean |SHAP|. Continuous inputs (e.g. ε) show a spread of "
    "points; one-hot condition flags and binary knockout indicators appear as two dense "
    "vertical bands (feature off vs on), which is expected. SHAP values are contributions "
    "relative to the model baseline for that row, not raw PHB or biomass fluxes—e.g. a "
    "positive SHAP for an acetate condition flag does not by itself imply higher PHB than "
    "glucose reference; it shifts the surrogate prediction conditional on all other features. "
    "Sparse far-right points on biomass panels indicate rare design–context combinations where "
    "that knockout strongly affects predicted biomass."
)

BEESWARM_CAPTION_GUIDANCE_TR = (
    "Beeswarm satırları ortalama |SHAP| sırasındadır. ε gibi sürekli girdiler nokta yayılımı gösterir; "
    "koşul one-hot ve binary KO göstergeleri ise (kapalı/açık) iki dikey bant oluşturur — beklenen davranıştır. "
    "SHAP değerleri o satırdaki diğer özelliklere koşullu model tabanına göre katkıdır; ham PHB veya biyokütle "
    "akışı değildir (ör. asetat koşul bayrağı için pozitif SHAP, mutlaka glukoza göre daha yüksek PHB demek değildir). "
    "Biyokütle beeswarm'ında sağa sıçrayan seyrek noktalar, o KO'nun yalnızca belirli tasarım–bağlam kombinasyonlarında "
    "tahmini biyokütleyi güçlü etkilediğini gösterebilir."
)


def run_shap_analysis(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    max_display: int = 20,
    out_dir: str = "results/figures",
    table_out_dir: str = "results/tables",
    sample_size: int = 7680,
    random_seed: int = 42,
    model_name: str = "gradient_boosting",
    target_name: str = "pha_flux",
    plot_bar: bool = True,
    plot_beeswarm: bool = True,
    beeswarm_max_display: int | None = None,
) -> Dict[str, Any]:
    """
    Run SHAP analysis on a trained surrogate model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X : pd.DataFrame
        Feature matrix (held-out test rows).
    feature_names : list[str]
    max_display : int
        Max features to show in SHAP plots.
    out_dir : str
        Directory for SHAP figure outputs.
    table_out_dir : str
        Directory for SHAP importance tables.
    sample_size : int
        Maximum rows to use for SHAP computation (subsampled if needed).
    random_seed : int
    model_name : str
        Name of the model family (used in filenames).
    target_name : str
        Name of the target variable (used in filenames).
    plot_bar : bool
        If True, write ``shap_bar_*.png`` (disable when a dual-panel bar figure is built elsewhere).
    plot_beeswarm : bool
        If True, write ``shap_beeswarm_*.png``.
    beeswarm_max_display : int | None
        Max rows in the beeswarm figure only (defaults to ``min(max_display, 12)`` so lower
        tiers do not collapse on the x-axis).

    Returns
    -------
    dict
        Keys:
        - ``importance_df``: DataFrame with columns ``feature``, ``mean_abs_shap``,
          sorted descending.
        - ``shap_values``: np.ndarray of SHAP values.
        - ``sample_idx``: np.ndarray of row indices used (relative to input X).
        - ``X_display``: feature matrix used for SHAP (display column labels), aligned
          with ``shap_values`` rows.
        - ``expected_value``: scalar model baseline from the explainer (for
          ``shap.Explanation``); may be NaN if unavailable.
    """
    import shap

    fig_dir = Path(resolve_path(out_dir))
    tbl_dir = Path(resolve_path(table_out_dir))
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # --- Subsample if needed -----------------------------------------------
    rng = np.random.default_rng(random_seed)
    n = len(X)
    if n > sample_size:
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_idx.sort()
        X_sample = X.iloc[sample_idx].copy()
    else:
        sample_idx = np.arange(n)
        X_sample = X.copy()

    logger.info(
        "SHAP analysis: model=%s target=%s samples=%d features=%d",
        model_name, target_name, len(X_sample), len(feature_names),
    )

    # --- Compute SHAP values -----------------------------------------------
    explainer: Any = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample.values)
    except Exception:
        # Fallback for non-tree models
        explainer = shap.KernelExplainer(
            model.predict,
            shap.sample(X_sample.values, min(100, len(X_sample))),
        )
        shap_values = explainer.shap_values(X_sample.values)

    expected_value = float("nan")
    if explainer is not None:
        try:
            ev_raw = explainer.expected_value
            expected_value = float(np.ravel(np.asarray(ev_raw, dtype=float))[0])
        except Exception:
            expected_value = float("nan")

    # --- Importance table ---------------------------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # --- Bootstrap ranking stability (top-20) --------------------------------
    from scipy.stats import spearmanr

    n_bootstrap = 50
    n_top = min(20, len(feature_names))
    boot_rng = np.random.default_rng(random_seed)
    n_shap = len(shap_values)

    ref_ranking = np.argsort(-mean_abs)[:n_top]
    spearman_rhos = []
    for _ in range(n_bootstrap):
        idx = boot_rng.choice(n_shap, size=n_shap, replace=True)
        boot_mean = np.abs(shap_values[idx]).mean(axis=0)
        boot_ranking = np.argsort(-boot_mean)[:n_top]
        rho, _ = spearmanr(ref_ranking, boot_ranking)
        spearman_rhos.append(rho)

    mean_rho = float(np.mean(spearman_rhos))
    std_rho = float(np.std(spearman_rhos))
    logger.info(
        "Bootstrap ranking stability (top-%d, %d resamples): "
        "Spearman rho = %.3f +/- %.3f",
        n_top, n_bootstrap, mean_rho, std_rho,
    )
    importance_df.attrs["bootstrap_spearman_rho_mean"] = mean_rho
    importance_df.attrs["bootstrap_spearman_rho_std"] = std_rho

    # Save table
    tbl_path = tbl_dir / f"shap_importance_{model_name}_{target_name}.csv"
    importance_df.to_csv(tbl_path, index=False)
    logger.info("Saved SHAP importance table: %s", tbl_path)

    # --- Generate SHAP plots -----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phaopt.shap_feature_display import format_feature_label

    display_names = [format_feature_label(str(f)) for f in feature_names]
    X_display = pd.DataFrame(
        X_sample.values,
        columns=display_names,
        index=X_sample.index,
    )
    bee_k = int(beeswarm_max_display) if beeswarm_max_display is not None else min(int(max_display), 12)

    # Summary bar plot
    if plot_bar:
        try:
            plt.figure(figsize=(10, max(6, max_display * 0.3)))
            shap.summary_plot(
                shap_values,
                X_display,
                max_display=max_display,
                plot_type="bar",
                show=False,
            )
            bar_path = fig_dir / f"shap_bar_{model_name}_{target_name}.png"
            plt.savefig(bar_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved SHAP bar plot: %s", bar_path)
        except Exception as exc:
            logger.warning("Could not generate SHAP bar plot: %s", exc)

    # Summary beeswarm plot (fewer rows than bar: readability)
    if plot_beeswarm:
        try:
            plt.figure(figsize=(10.5, max(5.5, bee_k * 0.42)))
            shap.summary_plot(
                shap_values,
                X_display,
                max_display=bee_k,
                show=False,
            )
            bee_path = fig_dir / f"shap_beeswarm_{model_name}_{target_name}.png"
            plt.savefig(bee_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(
                "Saved SHAP beeswarm plot: %s (max_display=%d; caption notes in "
                "phaopt.shap_analysis.BEESWARM_CAPTION_GUIDANCE_EN)",
                bee_path,
                bee_k,
            )
        except Exception as exc:
            logger.warning("Could not generate SHAP beeswarm plot: %s", exc)

    return {
        "importance_df": importance_df,
        "shap_values": shap_values,
        "sample_idx": sample_idx,
        "X_display": X_display,
        "expected_value": expected_value,
    }
