#!/usr/bin/env python3
"""
Strip chart: conventional single/pairwise KO screens vs MC-EHVI designs.

Reads
  - optknock_single_ko_results.csv, optknock_double_ko_results.csv (repo root)
  - results/tables/epsilon_constraint_matrix.csv (WT μ_max, ε-floor biomass, MC-EHVI μ_max)

Single-KO floor cluster uses biomass < 0.15. High-biomass (~0.238 h⁻¹) knockouts with
negligible PHB flux are plotted separately as non-PHB-producing optima.

Suggested figure caption (title removed from figure; use in manuscript)
------------------------------------------------------------------------
[See previous versions — add:] Two single knockouts reach high biomass without
meaningful PHB formation (non-PHB-producing optimum). Legend below panels.

Output: results/figures/knockout_vs_mcehvi_biomass_strip.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EPS_FOCUS = 0.30
RNG = np.random.default_rng(42)

# bm00377 = acetyl phosphate phosphohydrolase (gene acpH); not β-oxidation fadA (e.g. bm00477).
HIGHLIGHT_KO_RXN_ID = "bm00377"
HIGHLIGHT_GENE_LABEL = "acpH"
SINGLE_KO_BIOMASS_MAX = 0.15
NON_PHB_PHB_THRESHOLD = 0.05


def _load_single_ko_split() -> tuple[np.ndarray, float | None, np.ndarray, int]:
    """
    Returns (y_floor_bulk, y_highlight | None, y_non_phb, n_single_total).

    Floor bulk: non-lethal, biomass < cutoff, excluding the highlighted KO (plotted separately).
    non_phb: non-lethal, biomass >= cutoff and PHB flux below threshold.
    """
    path = PROJECT_ROOT / "optknock_single_ko_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run optknock_comparison.py first.")
    df = pd.read_csv(path)
    sub = df[~df["lethal"].astype(bool)].copy()
    sub["biomass"] = sub["biomass"].astype(float)
    sub["phb_flux"] = sub["phb_flux"].astype(float)

    non_phb = sub[
        (sub["biomass"] >= SINGLE_KO_BIOMASS_MAX)
        & (sub["phb_flux"] < NON_PHB_PHB_THRESHOLD)
    ]
    floor_sub = sub[sub["biomass"] < SINGLE_KO_BIOMASS_MAX]

    hi_rows = floor_sub[floor_sub["reaction"].astype(str) == HIGHLIGHT_KO_RXN_ID]
    bulk = floor_sub[floor_sub["reaction"].astype(str) != HIGHLIGHT_KO_RXN_ID]

    y_bulk = bulk["biomass"].to_numpy(dtype=float)
    y_highlight = float(hi_rows["biomass"].iloc[0]) if len(hi_rows) else None
    y_non = non_phb["biomass"].to_numpy(dtype=float)

    n_total = len(floor_sub) + len(non_phb)
    return y_bulk, y_highlight, y_non, n_total


def _load_pairwise_biomass() -> np.ndarray:
    path = PROJECT_ROOT / "optknock_double_ko_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run optknock_comparison.py first.")
    df = pd.read_csv(path)
    return df["biomass"].astype(float).to_numpy()


def _load_eps30_refs() -> tuple[np.ndarray, float, float]:
    path = TABLES_DIR / "epsilon_constraint_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    eps = pd.read_csv(path)
    sub = eps[
        (eps["condition"] == "base")
        & (np.abs(eps["epsilon"].astype(float) - EPS_FOCUS) < 0.01)
    ]
    wt = sub[sub["design"] == "wild_type"]
    if wt.empty:
        raise ValueError("No wild_type row at ε≈0.30 (base).")
    w0 = wt.iloc[0]
    wt_mu_max = float(w0["max_biomass"])
    wt_floor_bio = float(w0["biomass_flux"])

    des = sub[sub["design"] != "wild_type"].copy()
    if des.empty:
        raise ValueError("No design rows at ε≈0.30 (base).")
    vals = des.groupby("design", as_index=False)["max_biomass"].first()["max_biomass"].astype(float)
    return vals.to_numpy(), wt_mu_max, wt_floor_bio


def _strip_x(base: float, n: int) -> np.ndarray:
    if n == 0:
        return np.array([])
    return base + RNG.uniform(-0.14, 0.14, size=n)


def main() -> None:
    y_bulk, y_highlight, y_nonphb, n_single = _load_single_ko_split()
    y_pair = _load_pairwise_biomass()
    y_mce, wt_ref, eps_floor = _load_eps30_refs()

    n_p, n_m = len(y_pair), len(y_mce)
    n_np = len(y_nonphb)

    fig, ax = plt.subplots(figsize=(7.2, 4.85))

    x0, x1, x2 = 0.0, 1.0, 2.0

    ax.scatter(
        _strip_x(x0, len(y_bulk)),
        y_bulk,
        s=26,
        alpha=0.55,
        c="#5c636a",
        edgecolors="white",
        linewidth=0.35,
        zorder=3,
        label=f"Single KO — PHB optimum (n={len(y_bulk)})",
    )

    if n_np > 0:
        xs_np = _strip_x(x0, n_np)
        ax.scatter(
            xs_np,
            y_nonphb,
            s=120,
            alpha=0.9,
            facecolors="#9b59b6",
            edgecolors="white",
            linewidth=0.8,
            zorder=5,
            marker="X",
            label=f"Single KO — non-PHB-producing (n={n_np})",
        )
        mx, my = float(np.mean(xs_np)), float(np.mean(y_nonphb))
        ax.annotate(
            "non-PHB-producing\noptimum",
            xy=(mx, my),
            xytext=(0, 28),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="#6c3483",
            zorder=6,
            arrowprops={
                "arrowstyle": "-",
                "color":       "#884ea0",
                "lw":          0.65,
                "shrinkA":     3,
                "shrinkB":     4,
            },
        )

    if y_highlight is not None:
        x_f = x0 + RNG.uniform(-0.14, 0.14)
        ax.scatter(
            [x_f],
            [y_highlight],
            s=95,
            alpha=0.95,
            c="#e67e22",
            edgecolors="white",
            linewidth=0.7,
            zorder=5,
            marker="o",
            label=rf"Single KO: {HIGHLIGHT_GENE_LABEL} ({HIGHLIGHT_KO_RXN_ID})",
        )
        ax.annotate(
            rf"{HIGHLIGHT_GENE_LABEL} ({HIGHLIGHT_KO_RXN_ID})",
            xy=(x_f, y_highlight),
            xytext=(22, 18),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#a04000",
            zorder=6,
            arrowprops={
                "arrowstyle": "-",
                "color":       "#c0392b",
                "lw":          0.65,
                "shrinkA":     2,
                "shrinkB":     3,
            },
        )

    ax.scatter(
        _strip_x(x1, n_p),
        y_pair,
        s=26,
        alpha=0.55,
        c="#2c7fb8",
        edgecolors="white",
        linewidth=0.35,
        zorder=3,
        label=f"Pairwise KO (n={n_p})",
    )
    ax.scatter(
        _strip_x(x2, n_m),
        y_mce,
        s=42,
        alpha=0.9,
        c="#1a9850",
        edgecolors="white",
        linewidth=0.6,
        zorder=4,
        label=f"MC-EHVI (n={n_m})",
    )

    ax.axhline(
        wt_ref,
        color="0.25",
        linestyle="--",
        linewidth=1.35,
        zorder=2,
        label=rf"WT $\mu_{{\mathrm{{max}}}}$ = {wt_ref:.4f} h$^{{-1}}$",
    )
    ax.axhline(
        eps_floor,
        color="0.45",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
        label=rf"WT biomass at $\varepsilon$ floor (PHB opt.) = {eps_floor:.4f} h$^{{-1}}$",
    )

    ax.set_xticks([x0, x1, x2])
    ax.set_xticklabels(
        [
            f"Single KO\n(n={n_single})",
            f"Pairwise KO\n(n={n_p})",
            f"MC-EHVI\n(n={n_m})",
        ],
        fontsize=10,
    )
    ax.set_ylabel("Biomass flux (h⁻¹)", fontsize=11)
    ax.set_xlim(-0.55, 2.55)
    y_hi = max(
        float(np.max(y_bulk)) if len(y_bulk) else 0.0,
        float(np.max(y_pair)),
        float(np.max(y_mce)),
        float(np.max(y_nonphb)) if n_np else 0.0,
        wt_ref,
        y_highlight or 0.0,
    ) * 1.06
    ax.set_ylim(0.0, y_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45, linewidth=0.8)

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=7.5,
        framealpha=0.97,
        edgecolor="0.85",
        columnspacing=1.0,
    )
    plt.subplots_adjust(bottom=0.30)

    out = FIGS_DIR / "knockout_vs_mcehvi_biomass_strip.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
