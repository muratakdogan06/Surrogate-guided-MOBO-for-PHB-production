#!/usr/bin/env python3
"""
Figure 1 — Metabolic pathway schematic (β-oxidation vs PHB, branch-point emphasis).

Outputs:
  results/figures/figure1_metabolic_pathway_schematic.png
  results/figures/figure1_metabolic_pathway_schematic.pdf

Reaction IDs follow configs/candidate_reactions.yaml (P. megaterium iJA1121 extension).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "figures"


def _bbox(ax, xy, w, h, text, facecolor="#F5F5F5", edgecolor="#333333", lw=1.2, fontsize=8, zorder=2):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=zorder + 1,
        wrap=True,
    )
    return p


def _enzyme_chip(ax, xy, w, h, text, priority: str):
    """priority: 'high' | 'medium' | 'neutral'"""
    ec = "#B71C1C" if priority == "high" else "#E65100" if priority == "medium" else "#424242"
    lw = 2.2 if priority in ("high", "medium") else 1.0
    fc = "#FFEBEE" if priority == "high" else "#FFF3E0" if priority == "medium" else "#ECEFF1"
    return _bbox(ax, xy, w, h, text, facecolor=fc, edgecolor=ec, lw=lw, fontsize=7)


def _arrow(ax, xy_start, xy_end, color="#212121", style="solid", lw=1.8, mutation_scale=14, label=""):
    arr = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle="-|>",
        color=color,
        linewidth=lw,
        linestyle=style,
        mutation_scale=mutation_scale,
        shrinkA=4,
        shrinkB=4,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = (xy_start[0] + xy_end[0]) / 2
        my = (xy_start[1] + xy_end[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=6.5, color=color)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 9), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- Title ----
    ax.text(
        8,
        9.55,
        "Figure 1. β-Oxidation — PHB branch point and engineering targets",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        8,
        9.15,
        "fadN1/2/3 (3-hydroxyacyl-CoA dehydrogenases) compete with PhaB for the 3-OH-CoA pool",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="#37474F",
    )
    ax.text(
        8,
        8.82,
        "Thiolase in β-oxidation cycle: bm00477 (fadA). Pareto designs commonly include bm00377 (overflow / acetate routing).",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#546E7A",
    )

    # ---- Section labels ----
    ax.text(3.3, 8.55, "β-Oxidation (one turn)", ha="center", fontsize=10, fontweight="bold", color="#1565C0")
    ax.text(12.5, 8.55, "PHB biosynthesis", ha="center", fontsize=10, fontweight="bold", color="#2E7D32")

    # ---- LEFT: metabolite stack (biochemical order) ----
    _bbox(ax, (1.0, 7.35), 2.6, 0.55, "Acyl-CoA\n(long-chain)", facecolor="#E3F2FD")
    _arrow(ax, (2.3, 7.35), (2.3, 6.95), label="acyl-CoA dehydrogenase\n(e.g. bm00490 C4)")
    _bbox(ax, (1.0, 6.35), 2.6, 0.55, "trans-Δ²-Enoyl-CoA", facecolor="#E8EAF6")
    _arrow(ax, (2.3, 6.35), (2.3, 5.95), label="enoyl-CoA hydratase\n(e.g. bm00491)")
    _bbox(ax, (1.0, 5.35), 2.6, 0.55, "(S)-3-Hydroxyacyl-CoA", facecolor="#FFF9C4")
    _arrow(ax, (2.3, 5.35), (2.3, 4.95), label="3-OH-acyl-CoA DH")
    _bbox(ax, (1.0, 4.35), 2.6, 0.55, "3-Ketoacyl-CoA", facecolor="#F3E5F5")
    _arrow(ax, (2.3, 4.35), (2.3, 3.95), label="β-ox thiolase (fadA)\nbm00477")
    _bbox(ax, (0.55, 3.2), 3.5, 0.65, "Acetyl-CoA  +  Acyl-CoA (n−2)", facecolor="#E8F5E9")

    # Spiral return (conceptual)
    spiral = FancyArrowPatch(
        (1.2, 3.45),
        (1.2, 7.9),
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.35",
        color="#5C6BC0",
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        mutation_scale=12,
        shrinkA=2,
        shrinkB=2,
        zorder=0,
    )
    ax.add_patch(spiral)
    ax.text(0.35, 5.6, "β-oxidation\nspiral", ha="center", va="center", fontsize=7, color="#5C6BC0")

    # ---- HIGH / MEDIUM 3-OH-CoA DH chips ----
    ax.text(4.35, 5.05, "3-OH-CoA DH\n(engineering)", ha="center", fontsize=8, fontweight="bold")
    _enzyme_chip(
        ax,
        (3.55, 4.35),
        1.65,
        0.52,
        "HIGH\nbm00476 (fadN1)\nC16 3-OH-CoA DH",
        "high",
    )
    _enzyme_chip(ax, (3.55, 3.72), 1.65, 0.52, "HIGH\nbm00480 (fadN2)\nC12 3-OH-CoA DH", "high")
    _enzyme_chip(ax, (3.55, 3.09), 1.65, 0.52, "HIGH\nbm00492 (fadN3)\nC4 3-OH-CoA DH", "high")
    _enzyme_chip(ax, (5.35, 4.35), 1.55, 0.52, "MED\nbm00484\nC8 3-OH-CoA DH", "medium")
    _enzyme_chip(ax, (5.35, 3.72), 1.55, 0.52, "MED\nbm00488\nC6 3-OH-CoA DH", "medium")
    _enzyme_chip(ax, (5.35, 3.09), 1.55, 0.52, "MED\nbm00496\nshort-chain", "medium")

    # Arrows from (S)-3-OH pool to chips and to center
    ax.add_patch(
        FancyArrowPatch(
            (3.6, 5.6),
            (7.15, 5.35),
            arrowstyle="-|>",
            color="#C62828",
            linewidth=1.4,
            linestyle=(0, (2, 2)),
            mutation_scale=11,
            shrinkA=6,
            shrinkB=8,
            zorder=1,
        )
    )
    ax.text(5.1, 5.72, "competes", fontsize=6.5, color="#C62828", fontweight="bold")

    # ---- CENTER: branch pool ----
    pool = Ellipse((8.0, 5.35), 3.0, 1.55, facecolor="#FFFDE7", edgecolor="#F9A825", linewidth=3.0, zorder=3)
    ax.add_patch(pool)
    ax.text(
        8.0,
        5.55,
        "Shared 3-OH-CoA pool",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        8.0,
        5.18,
        "(R)-3-Hydroxybutyryl-CoA /\n3-hydroxyacyl-CoA",
        ha="center",
        va="center",
        fontsize=7.5,
        zorder=4,
    )
    ax.text(
        8.0,
        4.78,
        "← fadN drain   |   PhaB pull →",
        ha="center",
        va="center",
        fontsize=6.8,
        zorder=4,
        color="#5D4037",
    )

    # ---- RIGHT: PHB linear pathway (top → bottom) ----
    cx = 12.35
    _bbox(ax, (10.95, 6.88), 2.8, 0.48, "Acetyl-CoA", facecolor="#E8F5E9", edgecolor="#2E7D32")
    _arrow(ax, (cx, 6.88), (cx, 6.62))
    _enzyme_chip(ax, (10.85, 6.08), 3.0, 0.5, "PhaA / AtoB\nbm00500", "neutral")
    _arrow(ax, (cx, 6.08), (cx, 5.82))
    _bbox(ax, (10.95, 5.12), 2.8, 0.48, "Acetoacetyl-CoA", facecolor="#E8F5E9", edgecolor="#2E7D32")
    _arrow(ax, (cx, 5.12), (cx, 4.86))
    _enzyme_chip(ax, (10.85, 4.32), 3.0, 0.5, "PhaB\nbm00387", "neutral")
    _arrow(ax, (cx, 4.32), (cx, 4.06))
    _bbox(
        ax,
        (10.95, 3.36),
        2.8,
        0.5,
        "(R)-3-Hydroxybutyryl-CoA",
        facecolor="#FFF9C4",
        edgecolor="#F9A825",
        lw=2.0,
    )
    _arrow(ax, (cx, 3.36), (cx, 3.1))
    _enzyme_chip(ax, (10.85, 2.56), 3.0, 0.5, "PhaC\nbm00403", "neutral")
    _arrow(ax, (cx, 2.56), (cx, 2.3))
    _bbox(ax, (10.95, 1.62), 2.8, 0.55, "PHB\n(polymer)", facecolor="#C8E6C9", edgecolor="#1B5E20", lw=2.0)

    # Pool ↔ PhaB substrate
    ax.add_patch(
        FancyArrowPatch(
            (9.55, 5.35),
            (11.0, 3.6),
            arrowstyle="-|>",
            color="#2E7D32",
            linewidth=2.0,
            mutation_scale=14,
            shrinkA=10,
            shrinkB=10,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (11.0, 3.6),
            (9.55, 5.35),
            arrowstyle="-|>",
            color="#9E9E9E",
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            mutation_scale=10,
            shrinkA=10,
            shrinkB=10,
            zorder=2,
        )
    )
    ax.text(10.35, 4.65, "PhaB pull", fontsize=7, color="#2E7D32", fontweight="bold")

    # Acetyl-CoA link left stack to right column
    ax.add_patch(
        FancyArrowPatch(
            (4.05, 3.52),
            (12.35, 7.36),
            arrowstyle="-|>",
            color="#212121",
            linewidth=1.5,
            mutation_scale=12,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )
    ax.text(7.9, 6.85, "Acetyl-CoA feed", fontsize=6.5, color="#424242")

    # ---- Conserved KO in Pareto designs (bm00377) + thiolase (bm00477) ----
    ax.text(2.85, 4.12, "X", fontsize=14, color="#C62828", ha="center", va="center", fontweight="bold")
    ax.text(2.85, 3.72, "Common KO:\nbm00377\n(overflow)", fontsize=6, ha="center", color="#C62828")
    ax.text(4.55, 2.55, "β-ox thiolase:\nbm00477", fontsize=6, ha="left", color="#424242")

    # ---- Optional branches (bottom) ----
    ax.text(8, 1.95, "Additional context", ha="center", fontsize=9, fontweight="bold", color="#455A64")
    _bbox(
        ax,
        (0.6, 0.85),
        3.4,
        0.95,
        "Overflow / acetate\npta bm00375 → ackA bm00376\n(dashed = carbon loss)",
        facecolor="#FBE9E7",
        edgecolor="#BF360C",
        fontsize=7,
    )
    ax.add_patch(
        FancyArrowPatch(
            (2.3, 3.2),
            (2.2, 1.85),
            arrowstyle="-|>",
            color="#BF360C",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            mutation_scale=10,
            shrinkA=4,
            shrinkB=2,
            zorder=1,
        )
    )

    _bbox(
        ax,
        (4.6, 0.85),
        3.6,
        0.95,
        "TCA drain from Acetyl-CoA\ncitrate synthase bm00283\n(+ NADPH link icd bm00286)",
        facecolor="#E8EAF6",
        edgecolor="#283593",
        fontsize=7,
    )
    ax.add_patch(
        FancyArrowPatch(
            (12.35, 7.1),
            (6.4, 1.75),
            arrowstyle="-|>",
            color="#283593",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            mutation_scale=10,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )

    # ---- Legend ----
    lx, ly = 11.95, 8.05
    ax.text(lx + 1.0, ly + 0.55, "Legend", fontsize=8.5, fontweight="bold")
    ax.plot([lx, lx + 0.45], [ly + 0.35, ly + 0.35], color="#212121", linewidth=2)
    ax.text(lx + 0.55, ly + 0.35, "Wild-type flux", fontsize=7, va="center")
    ax.plot([lx, lx + 0.45], [ly + 0.08, ly + 0.08], color="#1565C0", linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(lx + 0.55, ly + 0.08, "Spiral / alternate route", fontsize=7, va="center")
    ax.plot([lx, lx + 0.45], [ly - 0.18, ly - 0.18], color="#BF360C", linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(lx + 0.55, ly - 0.18, "Post-KO / overflow (conceptual)", fontsize=7, va="center")
    rect = mpatches.Rectangle((lx, ly - 0.58), 0.35, 0.22, linewidth=2, edgecolor="#B71C1C", facecolor="#FFEBEE")
    ax.add_patch(rect)
    ax.text(lx + 0.55, ly - 0.47, "High-priority KO target", fontsize=7, va="center")
    rect2 = mpatches.Rectangle((lx, ly - 0.95), 0.35, 0.22, linewidth=2, edgecolor="#E65100", facecolor="#FFF3E0")
    ax.add_patch(rect2)
    ax.text(lx + 0.55, ly - 0.84, "Medium-priority KO target", fontsize=7, va="center")

    fig.tight_layout()
    png = OUT_DIR / "figure1_metabolic_pathway_schematic.png"
    pdf = OUT_DIR / "figure1_metabolic_pathway_schematic.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {pdf.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
    sys.exit(0)
