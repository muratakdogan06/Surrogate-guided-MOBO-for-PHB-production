#!/usr/bin/env python3
"""
Section 3.2 — Two-panel figure from ε-constraint FBA ensemble (fba_results.parquet).

(a) PHB (pha_flux) distribution for PHB > 0; inset only for zero vs non-zero counts.
(b) Biomass flux distribution; reference vertical lines at ε·μ_WT (legend only, no text box).

Output: results/figures/figure_section32_phb_biomass_histograms.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT_ROOT / "data" / "processed" / "fba_results.parquet"
OUT_FIG = PROJECT_ROOT / "results" / "figures" / "figure_section32_phb_biomass_histograms.png"

PHA_ZERO_THR = 1e-9
N_BINS_PHA = 50
N_BINS_BIO = 50
# Extra shared y-headroom so panel (b) legend clears histogram peaks
Y_HEADROOM = 1.16

# (epsilon, line colour, dash pattern) — visually distinct in figure and legend
EPS_STYLES: tuple[tuple[float, str, tuple], ...] = (
    (0.10, "#c0392b", (0, (5, 2))),
    (0.30, "#d35400", (0, (4, 1, 1, 1))),
    (0.50, "#2980b9", (0, (1, 1))),
    (0.70, "#6c3483", (0, (6, 2, 2, 2))),
)


def _report_biomass_hist_vs_floors(
    bio: np.ndarray,
    floor_specs: list[tuple[float, float, str, tuple]],
    n_bins: int,
) -> None:
    """
    Matplotlib ``hist(..., bins=n)`` matches ``np.histogram`` default edges (data min–max).
    Print bin index / edges for each ε·μ_WT so apparent line-vs-peak shift can be checked
    against bin width (visual illusion) vs a numeric bug.
    """
    counts, edges = np.histogram(bio, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    peak_i = int(np.argmax(counts))
    print(
        "  biomass tallest bin:",
        f"i={peak_i}, interval=[{edges[peak_i]:.8f}, {edges[peak_i + 1]:.8f}],",
        f"center={centers[peak_i]:.8f}, count={int(counts[peak_i])}",
    )
    for eps, xv, _, _ in floor_specs:
        # bin j where edges[j] <= xv < edges[j+1] (numpy/matplotlib convention)
        j = int(np.searchsorted(edges, xv, side="right") - 1)
        j = max(0, min(j, n_bins - 1))
        lo, hi = float(edges[j]), float(edges[j + 1])
        cj = float(centers[j])
        half_w = (hi - lo) / 2.0
        on_edge = (
            "left edge"
            if abs(xv - lo) <= 1e-12 * max(1.0, abs(lo))
            else ("right edge" if abs(xv - hi) <= 1e-12 * max(1.0, abs(hi)) else "interior")
        )
        print(
            f"  eps={eps:.2f}  xv={xv:.10f}  -> bin {j} [{lo:.8f}, {hi:.8f}]",
            f"center={cj:.10f}  (xv - bin_center)={xv - cj:+.3e}  half_width={half_w:.3e}  [{on_edge}]",
        )
    xv_03 = next(xv for eps, xv, _, _ in floor_specs if abs(eps - 0.30) < 1e-9)
    j03 = int(np.searchsorted(edges, xv_03, side="right") - 1)
    j03 = max(0, min(j03, n_bins - 1))
    print(
        "  note: vline at exact ε·μ_WT; bar 'center' is mid-bin — if xv sits on a bin edge,",
        "the line looks offset from the bar centroid (not a flux miscalculation).",
        f"Global max count is bin {peak_i}, not necessarily the bin at ε=0.30 (bin {j03}, n={int(counts[j03])}).",
    )


def _wt_max_biomass(df: pd.DataFrame) -> float:
    """
    μ_WT: wild-type unconstrained maximum biomass flux (FBA), base condition, no knock-outs.
    """
    ko = df["knockouts"].fillna("").astype(str)
    m = (ko == "") & (df["condition"] == "base")
    if not m.any():
        return float(df["max_biomass_unconstrained"].median())
    return float(df.loc[m, "max_biomass_unconstrained"].iloc[0])


def main() -> None:
    if not PARQUET.exists():
        print(f"Missing dataset: {PARQUET}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(
        PARQUET,
        columns=[
            "pha_flux",
            "biomass_flux",
            "knockouts",
            "condition",
            "max_biomass_unconstrained",
        ],
    )

    pha = df["pha_flux"].to_numpy(dtype=float)
    zero_mask = np.abs(pha) <= PHA_ZERO_THR
    n_zero = int(zero_mask.sum())
    n_pos = int((~zero_mask).sum())
    n_total = len(df)

    pha_pos = pha[~zero_mask]
    mu_wt = _wt_max_biomass(df)
    # Exact ε·μ positions (no rounding before multiply)
    floor_specs: list[tuple[float, float, str, tuple]] = []
    for eps, color, dashes in EPS_STYLES:
        xv = float(eps * mu_wt)
        floor_specs.append((eps, xv, color, dashes))

    bio = df["biomass_flux"].to_numpy(dtype=float)
    _report_biomass_hist_vs_floors(bio, floor_specs, N_BINS_BIO)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.35),
        constrained_layout=True,
    )

    # --- (a) PHB: histogram excludes PHB = 0; x-axis from 0 for clarity ---
    ax_a.hist(
        pha_pos,
        bins=N_BINS_PHA,
        color="#2980b9",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
    )
    pha_max = float(np.max(pha_pos)) if len(pha_pos) else 1.0
    ax_a.set_xlim(0.0, pha_max * 1.02)
    ax_a.set_xlabel(r"PHB flux (mmol gDW$^{-1}$ h$^{-1}$)", fontsize=10)
    ax_a.set_ylabel("Number of simulations", fontsize=10)
    ax_a.set_title("(a) PHB flux distribution", fontsize=11, fontweight="bold", loc="left")

    # Inset only (counts + log scale); upper-right to avoid main-histogram bars
    ax_in = ax_a.inset_axes((0.66, 0.62, 0.30, 0.32))
    ax_in.bar(
        [0, 1],
        [max(n_zero, 1), max(n_pos, 1)],
        color=["#c0392b", "#27ae60"],
        width=0.52,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_in.set_xticks([0, 1])
    ax_in.set_xticklabels(["PHB = 0", "PHB > 0"], fontsize=7, rotation=0)
    ax_in.set_ylabel("Count (log)", fontsize=7)
    ax_in.set_yscale("log")
    ax_in.tick_params(axis="both", labelsize=6.5)
    for xi, nv in zip([0, 1], [n_zero, n_pos]):
        ax_in.text(xi, nv * 1.12, f"{nv:,}", ha="center", va="bottom", fontsize=6.8)
    ax_in.set_title("Zero vs non-zero", fontsize=7.5, pad=6)

    # --- (b) Biomass ---
    _, _, patches_b = ax_b.hist(
        bio,
        bins=N_BINS_BIO,
        color="#16a085",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=1,
    )
    for _p in patches_b:
        _p.set_zorder(6)
    ax_b.set_xlabel(r"Biomass flux (h$^{-1}$)", fontsize=10)
    ax_b.set_ylabel("Number of simulations", fontsize=10)
    ax_b.set_title("(b) Biomass flux distribution", fontsize=11, fontweight="bold", loc="left")

    leg_handles = []
    for eps, xv, color, dashes in floor_specs:
        ax_b.axvline(
            xv,
            color=color,
            linestyle=dashes,
            linewidth=1.15,
            alpha=0.88,
            zorder=4,
        )
        leg_handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=color,
                linestyle=dashes,
                linewidth=1.35,
                label=rf"$\epsilon$={eps:.2f}·$\mu_{{\mathrm{{WT}}}}$ = {xv:.5f} h$^{{-1}}$",
            )
        )

    ax_b.legend(
        handles=leg_handles,
        loc="upper right",
        ncol=2,
        columnspacing=0.9,
        fontsize=6.5,
        framealpha=0.94,
        title=rf"$\mu_{{\mathrm{{WT}}}}$ = {mu_wt:.6f} h$^{{-1}}$",
        title_fontsize=6.5,
    )

    y1 = ax_a.get_ylim()[1]
    y2 = ax_b.get_ylim()[1]
    ymax = max(y1, y2) * Y_HEADROOM
    ax_a.set_ylim(0, ymax)
    ax_b.set_ylim(0, ymax)

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_FIG.relative_to(PROJECT_ROOT)}")
    print(f"  n_total={n_total}, n_pha_zero={n_zero}, n_pha_pos={n_pos}, mu_wt={mu_wt:.8f}")
    print("  epsilon floors (h-1):", [f"{xv:.8f}" for _, xv, _, _ in floor_specs])


if __name__ == "__main__":
    main()
