"""
Publication SHAP figures (custom matplotlib, not shap.summary_plot bar).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phaopt.shap_feature_display import feature_tier, format_feature_label, tier_color
from phaopt.utils import resolve_path

logger = logging.getLogger(__name__)


def _three_tier_bar_color(feature: str) -> str:
    """Map features to three legend tiers: ε, environment, genetic (KO/UP/meta)."""
    t = feature_tier(feature)
    if t == "constraint":
        return tier_color("constraint")
    if t == "condition":
        return tier_color("condition")
    return tier_color("knockout")


def _union_top_features(
    imp_pha: pd.DataFrame,
    imp_bio: pd.DataFrame,
    k: int,
) -> list[str]:
    """Top-k features by sum of mean |SHAP| across PHA and biomass panels."""
    m = pd.merge(
        imp_pha[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "pha"}),
        imp_bio[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "bio"}),
        on="feature",
        how="outer",
    ).fillna(0.0)
    m["score"] = m["pha"] + m["bio"]
    m = m.sort_values("score", ascending=False)
    return m.head(k)["feature"].tolist()


def plot_dual_mean_abs_shap_bar_panels(
    imp_pha: pd.DataFrame,
    imp_bio: pd.DataFrame,
    outpath: str | Path,
    *,
    top_k: int = 20,
) -> Path:
    """
    One figure, two horizontal bar panels: (a) PHB (pha) surrogate, (b) biomass surrogate.

    Same feature set and order in both panels; bar colours by three tiers
    (ε / environment / genetic perturbations: KO, UP, and aggregate design features).
    X-axis label: ``mean(|SHAP value|)``.
    Panel (b) typically has a smaller horizontal scale than (a); state that in the caption.
    """
    out = Path(resolve_path(outpath))
    out.parent.mkdir(parents=True, exist_ok=True)

    feats = _union_top_features(imp_pha, imp_bio, top_k)
    pha_map = imp_pha.set_index("feature")["mean_abs_shap"].to_dict()
    bio_map = imp_bio.set_index("feature")["mean_abs_shap"].to_dict()
    vals_pha = np.array([float(pha_map.get(f, 0.0)) for f in feats])
    vals_bio = np.array([float(bio_map.get(f, 0.0)) for f in feats])
    labels = [format_feature_label(f) for f in feats]
    colors = [_three_tier_bar_color(f) for f in feats]

    n = len(feats)
    y = np.arange(n)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13.2, max(6.0, 0.32 * n + 1.6)),
        sharey=True,
    )
    h = 0.72

    ax_a.barh(y, vals_pha, height=h, color=colors, edgecolor="white", linewidth=0.35, align="center")
    ax_b.barh(y, vals_bio, height=h, color=colors, edgecolor="white", linewidth=0.35, align="center")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8.1)
    ax_a.invert_yaxis()
    ax_b.tick_params(axis="y", labelleft=False)

    ax_a.set_xlabel("mean(|SHAP value|)", fontsize=11)
    ax_b.set_xlabel("mean(|SHAP value|)", fontsize=11)
    ax_a.set_title("(a) PHB flux surrogate", fontsize=11, fontweight="bold", loc="left")
    ax_b.set_title("(b) Biomass flux surrogate", fontsize=11, fontweight="bold", loc="left")

    ax_a.grid(True, axis="x", ls=":", alpha=0.28)
    ax_b.grid(True, axis="x", ls=":", alpha=0.28)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    tier_handles = [
        mlines.Line2D([0], [0], color=tier_color("constraint"), lw=6, solid_capstyle="butt", label="ε (constraint)"),
        mlines.Line2D([0], [0], color=tier_color("condition"), lw=6, solid_capstyle="butt", label="Environment"),
        mlines.Line2D([0], [0], color=tier_color("knockout"), lw=6, solid_capstyle="butt", label="Knockout"),
    ]
    fig.legend(
        handles=tier_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8.0,
        frameon=False,
    )

    fig.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.16, wspace=0.12)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved dual SHAP bar panels: %s", out)
    return out


def plot_dual_shap_beeswarm_panels(
    shap_pha: np.ndarray,
    X_display_pha: pd.DataFrame,
    imp_pha: pd.DataFrame,
    shap_bio: np.ndarray,
    X_display_bio: pd.DataFrame,
    imp_bio: pd.DataFrame,
    feature_names: list[str],
    outpath: str | Path,
    *,
    top_k: int = 12,
    base_pha: float | None = None,
    base_bio: float | None = None,
) -> Path:
    """
    One figure, two SHAP beeswarm panels: (a) PHB (pha) surrogate, (b) biomass surrogate.

    Uses the same union top-``top_k`` feature ordering as
    :func:`plot_dual_mean_abs_shap_bar_panels` (sum of mean |SHAP| across targets).
    Requires ``shap`` ≥ 0.41 (``shap.plots.beeswarm`` with ``ax=``).
    """
    import shap

    out = Path(resolve_path(outpath))
    out.parent.mkdir(parents=True, exist_ok=True)

    feats = _union_top_features(imp_pha, imp_bio, top_k)
    if not feats:
        raise ValueError("No features for dual beeswarm (empty importance tables).")

    idxs = [feature_names.index(f) for f in feats]
    display_cols = [format_feature_label(str(f)) for f in feats]

    sv_p = np.asarray(shap_pha)[:, idxs]
    sv_b = np.asarray(shap_bio)[:, idxs]
    d_p = X_display_pha.iloc[:, idxs].to_numpy()
    d_b = X_display_bio.iloc[:, idxs].to_numpy()

    fn = np.array(display_cols, dtype=object)
    bp = float(base_pha) if base_pha is not None and np.isfinite(base_pha) else 0.0
    bb = float(base_bio) if base_bio is not None and np.isfinite(base_bio) else 0.0

    exp_p = shap.Explanation(values=sv_p, base_values=bp, data=d_p, feature_names=fn)
    exp_b = shap.Explanation(values=sv_b, base_values=bb, data=d_b, feature_names=fn)

    k = len(feats)
    order = np.arange(k, dtype=int)

    n = k
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13.8, max(5.8, 0.38 * n + 1.85)),
        sharey=True,
    )

    shap.plots.beeswarm(
        exp_p,
        max_display=k,
        order=order,
        ax=ax_a,
        show=False,
        color_bar=False,
        plot_size=None,
        group_remaining_features=False,
    )
    shap.plots.beeswarm(
        exp_b,
        max_display=k,
        order=order,
        ax=ax_b,
        show=False,
        color_bar=True,
        plot_size=None,
        group_remaining_features=False,
    )

    ax_a.set_title("(a) PHB flux surrogate", fontsize=11, fontweight="bold", loc="left")
    ax_b.set_title("(b) Biomass flux surrogate", fontsize=11, fontweight="bold", loc="left")
    ax_b.tick_params(axis="y", labelleft=False)

    # Shared symmetric x-axis for side-by-side comparison
    try:
        vmax = float(
            np.nanpercentile(
                np.abs(np.concatenate([sv_p.ravel(), sv_b.ravel()])),
                99.9,
            )
        )
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1e-6
        ax_a.set_xlim(-vmax, vmax)
        ax_b.set_xlim(-vmax, vmax)
    except Exception:
        pass

    fig.subplots_adjust(left=0.26, right=0.91, top=0.92, bottom=0.10, wspace=0.10)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved dual SHAP beeswarm panels: %s", out)
    return out
