#!/usr/bin/env python3
"""
11_phase_plane_analysis.py — Phenotypic Phase Plane & Production Envelope
==========================================================================
Quality Improvement Step 8 — Bioresource Technology submission

Uses actual COBRA FBA with epsilon-constraint sweeps to demonstrate
'growth-compatible' PHA strategies — the core claim of the paper title.

Outputs
-------
  results/tables/production_envelope_wt.csv
  results/tables/production_envelope_top_designs.csv
  results/tables/pareto_front_summary.csv
  results/tables/epsilon_constraint_matrix.csv
  results/figures/production_envelope_wt.png
  results/figures/production_envelope_top5.png
  results/figures/pareto_front_comparison.png
  results/figures/epsilon_constraint_heatmap.png

References
----------
  Edwards et al. (2002) Nat Biotechnol 20:545-548  (PhPP analysis)
  Burgard et al. (2003) Biotechnol Bioeng 84:647-657 (OptKnock)
  Thiele & Palsson (2010) Nat Protoc 5:93-121
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOGS_DIR   = PROJECT_ROOT / "results" / "logs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGS_DIR   = PROJECT_ROOT / "results" / "figures"
for d in [LOGS_DIR, TABLES_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "11_phase_plane.log"),
    ],
)
logger = logging.getLogger("pipeline.11")


# ── CONFIG LOADING ───────────────────────────────────────────────────────────

def load_configs() -> tuple[dict, dict, dict]:
    cfg_path  = PROJECT_ROOT / "configs" / "model_config.yaml"
    al_path   = PROJECT_ROOT / "configs" / "active_learning.yaml"
    cond_path = PROJECT_ROOT / "configs" / "conditions.yaml"
    with open(cfg_path)  as f: cfg  = yaml.safe_load(f)
    with open(al_path)   as f: al   = yaml.safe_load(f)
    with open(cond_path) as f: cond = yaml.safe_load(f)
    return cfg, al, cond


# ── LOAD TOP DESIGNS FROM AL RESULTS ─────────────────────────────────────────

DESIGN_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#3498db", "#9b59b6",
                 "#1abc9c", "#d35400", "#8e44ad", "#2980b9", "#c0392b"]


def _load_top_designs_from_al(n: int = 5) -> list[dict]:
    """
    Load top-performing designs from AL evaluated CSVs.
    Returns the *n* unique designs with highest PHA flux (averaged
    across seeds) that also maintain positive biomass.
    """
    al_files = sorted(TABLES_DIR.glob("al_evaluated_seed*.csv"))
    if not al_files:
        raise FileNotFoundError(
            "No AL results found in results/tables/al_evaluated_seed*.csv. "
            "Run scripts 06-07 first."
        )

    frames = [pd.read_csv(f) for f in al_files]
    combined = pd.concat(frames, ignore_index=True)

    grouped = combined.groupby(["knockouts", "upregulations"]).agg(
        pha_mean=("pha_flux", "mean"),
        bio_mean=("biomass_flux", "mean"),
        n_seeds=("pha_flux", "count"),
    ).reset_index()

    viable = grouped[grouped["bio_mean"] > 0.01]
    top = viable.nlargest(n, "pha_mean")

    designs: list[dict] = []
    for i, (_, row) in enumerate(top.iterrows()):
        kos = [r for r in str(row["knockouts"]).split("|") if r and r != "nan"]
        ups = [r for r in str(row["upregulations"]).split("|") if r and r != "nan"]
        ko_short = ",".join(kos[:2]) + ("…" if len(kos) > 2 else "")
        up_short = ",".join(ups[:2]) + ("…" if len(ups) > 2 else "")
        designs.append({
            "knockouts": kos,
            "upregulations": ups,
            "label": f"Design {i+1}: KO({ko_short}) UP({up_short})",
            "color": DESIGN_COLORS[i % len(DESIGN_COLORS)],
            "expected_pha": float(row["pha_mean"]),
            "expected_bio": float(row["bio_mean"]),
        })

    logger.info("Loaded %d top designs from %d AL seed files.", len(designs), len(al_files))
    return designs


# ── COBRA FBA PRODUCTION ENVELOPE ────────────────────────────────────────────

def _run_epsilon_sweep(
    model,
    cfg: dict,
    candidates_cfg: dict,
    condition_name: str,
    overrides: dict,
    design: dict | None = None,
    n_points: int = 25,
) -> pd.DataFrame:
    """
    Sweep epsilon from 0 to 0.95 using real COBRA epsilon-constraint FBA.
    Reuses the same FBA logic as the main pipeline (simulate_single).
    """
    from phaopt.simulation import simulate_single

    eps_values = np.linspace(0.0, 0.95, n_points)
    rows: list[dict] = []

    design_dict = design or {"knockouts": [], "upregulations": []}
    design_label = "wild_type" if design is None else design.get("label", "engineered")

    for eps in eps_values:
        result = simulate_single(
            model, design_dict, condition_name, overrides,
            biomass_fraction=float(eps),
            cfg=cfg,
            candidates_cfg=candidates_cfg,
        )
        rows.append({
            "condition":                condition_name,
            "design":                   design_label,
            "epsilon_level":            float(eps),
            "biomass_flux":             result["biomass_flux"],
            "max_biomass_unconstrained": result["max_biomass_unconstrained"],
            "pha_flux":                 result["pha_flux"],
            "status":                   result["status"],
        })

    return pd.DataFrame(rows)


def compute_wt_envelopes(
    model, cfg: dict, candidates_cfg: dict, conditions_cfg: dict, n_points: int = 25
) -> pd.DataFrame:
    """Wild-type production envelope across all conditions."""
    conditions = conditions_cfg.get("conditions", {})
    frames: list[pd.DataFrame] = []
    for cond_name, cond_data in conditions.items():
        overrides = cond_data.get("overrides") or {}
        logger.info("  WT envelope — condition: %s (%d points)", cond_name, n_points)
        df = _run_epsilon_sweep(model, cfg, candidates_cfg, cond_name, overrides,
                                design=None, n_points=n_points)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_design_envelopes(
    model,
    cfg: dict,
    candidates_cfg: dict,
    condition_name: str,
    overrides: dict,
    designs: list[dict],
    n_points: int = 25,
) -> dict[int, pd.DataFrame]:
    """Production envelopes for top designs under base condition."""
    result: dict[int, pd.DataFrame] = {}
    for i, design in enumerate(designs):
        logger.info("  Design %d/%d envelope — %s", i + 1, len(designs), design["label"])
        df = _run_epsilon_sweep(model, cfg, candidates_cfg, condition_name, overrides,
                                design=design, n_points=n_points)
        result[i] = df
    return result


def compute_epsilon_constraint_matrix(
    model,
    cfg: dict,
    candidates_cfg: dict,
    conditions_cfg: dict,
    designs: list[dict],
) -> pd.DataFrame:
    """
    Compute PHA flux at each epsilon level × condition for WT and top designs.
    Uses the same four epsilon levels as the main FBA dataset.
    """
    from phaopt.simulation import simulate_single

    epsilon_levels = cfg.get("biomass_fraction_grid", [0.10, 0.30, 0.50, 0.70])
    conditions = conditions_cfg.get("conditions", {})
    rows: list[dict] = []

    all_designs = [{"knockouts": [], "upregulations": [], "label": "wild_type"}] + designs
    total = len(all_designs) * len(conditions) * len(epsilon_levels)
    done = 0

    for cond_name, cond_data in conditions.items():
        overrides = cond_data.get("overrides") or {}
        wt_pha_at_eps: dict[float, float] = {}

        for design in all_designs:
            for eps in epsilon_levels:
                result = simulate_single(
                    model, design, cond_name, overrides,
                    biomass_fraction=float(eps),
                    cfg=cfg,
                    candidates_cfg=candidates_cfg,
                )
                pha = result["pha_flux"]
                label = design.get("label", "wild_type")
                is_wt = label == "wild_type"

                if is_wt:
                    wt_pha_at_eps[eps] = pha

                wt_ref = wt_pha_at_eps.get(eps, 0.001)
                improvement = 100 * (pha - wt_ref) / max(wt_ref, 1e-6) if not is_wt else 0.0

                rows.append({
                    "condition":         cond_name,
                    "design":            label[:40],
                    "epsilon":           eps,
                    "pha_flux":          round(pha, 8),
                    "biomass_flux":      round(result["biomass_flux"], 8),
                    "biomass_fraction":  eps,
                    "max_biomass":       round(result["max_biomass_unconstrained"], 8),
                    "vs_wt_improvement": round(improvement, 1),
                })
                done += 1
                if done % 20 == 0:
                    logger.info("    ε-matrix progress: %d / %d", done, total)

    return pd.DataFrame(rows)


# ── FIGURES ──────────────────────────────────────────────────────────────────

# C1–C7: maximally distinct hues (tab10-style permutation)
COND_COLORS = {
    "base":                    "#1f77b4",
    "low_carbon":              "#ff7f0e",
    "low_oxygen":              "#2ca02c",
    "glycerol_aerobic":        "#d62728",
    "acetate_aerobic":         "#9467bd",
    "glycerol_low_oxygen":     "#8c564b",
    "mixed_glucose_glycerol":  "#e377c2",
}

# Plot order C1→C7; short legend text (full names in figure caption)
WT_COND_ORDER: list[tuple[str, str]] = [
    ("base",                    "C1 — Glc, aerobic (ref.)"),
    ("low_carbon",              "C2 — Glc, carbon-lim."),
    ("low_oxygen",              "C3 — Glc, micro-aer."),
    ("glycerol_aerobic",        "C4 — Gly, aerobic"),
    ("acetate_aerobic",         "C5 — Ac, aerobic"),
    ("glycerol_low_oxygen",     "C6 — Gly, micro-aer."),
    ("mixed_glucose_glycerol",  "C7 — Glc + Gly mix"),
]

COND_LINESTYLE: dict[str, str | tuple] = {
    "base":                    (0, (6, 2)),
    "low_oxygen":              "-",
    "low_carbon":              (0, (4, 1.5)),
    "glycerol_aerobic":        "-",
    "acetate_aerobic":         (0, (8, 2)),
    "glycerol_low_oxygen":     "-",
    "mixed_glucose_glycerol":  (0, (1, 1.2)),
}

EPS_GRID = (0.1, 0.3, 0.5, 0.7)
EPS_MARKERS = "os^D"  # one char per EPS_GRID index (avoid float-key dict quirks)


def _load_wt_epsilon_points(eps_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Wild-type rows from epsilon_constraint_matrix at EPS_GRID."""
    if eps_df is None or eps_df.empty:
        return None
    m = eps_df["design"].astype(str) == "wild_type"
    sub = eps_df.loc[m].copy()
    if sub.empty:
        return None
    keep = np.zeros(len(sub), dtype=bool)
    for e in EPS_GRID:
        keep |= np.isclose(sub["epsilon"].to_numpy(dtype=float), float(e), rtol=0, atol=0.02)
    return sub.loc[keep]


def plot_wt_production_envelope(
    wt_df: pd.DataFrame,
    outpath: Path,
    eps_df: pd.DataFrame | None = None,
) -> None:
    """
    Dense WT envelope curves plus discrete ε markers from ``epsilon_constraint_matrix``
    (same grid as manuscript tables). C1–C7 order; legends below axes; caption carries footnotes.
    """
    import matplotlib.lines as mlines

    if eps_df is None:
        p = TABLES_DIR / "epsilon_constraint_matrix.csv"
        if p.exists():
            eps_df = pd.read_csv(p)

    wt_eps = _load_wt_epsilon_points(eps_df)

    fig, ax = plt.subplots(figsize=(10.5, 6.9))

    line_handles: list[mlines.Line2D] = []
    line_labels: list[str] = []

    for zi, (cond, leg_short) in enumerate(WT_COND_ORDER):
        sub = wt_df[wt_df["condition"] == cond].sort_values("biomass_flux")
        if sub.empty:
            logger.warning("WT envelope: no rows for condition %s", cond)
            continue
        color = COND_COLORS.get(cond, "#7f8c8d")
        ls = COND_LINESTYLE.get(cond, "-")
        lw = 2.85 if cond == "base" else 2.05
        (ln,) = ax.plot(
            sub["biomass_flux"],
            sub["pha_flux"],
            color=color,
            linestyle=ls,
            lw=lw,
            zorder=3 + zi,
            alpha=0.95,
        )
        line_handles.append(ln)
        line_labels.append(leg_short)

    if wt_eps is not None:
        for cond, _ in WT_COND_ORDER:
            block = wt_eps[wt_eps["condition"] == cond]
            if block.empty:
                continue
            color = COND_COLORS.get(cond, "#7f8c8d")
            for j, e in enumerate(EPS_GRID):
                row = block[np.isclose(block["epsilon"].to_numpy(dtype=float), float(e), rtol=0, atol=0.02)]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.scatter(
                    float(r["biomass_flux"]),
                    float(r["pha_flux"]),
                    s=52,
                    marker=EPS_MARKERS[j],
                    facecolors=color,
                    edgecolors=color,
                    linewidths=0.55,
                    zorder=20,
                )

    ax.set_xlabel(r"Biomass flux (h$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"PHB flux (mmol gDW$^{-1}$ h$^{-1}$)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="both", ls=":", alpha=0.22)

    eps_handles = [
        mlines.Line2D(
            [0],
            [0],
            color="#333333",
            marker=EPS_MARKERS[j],
            linestyle="None",
            markersize=7,
            label=rf"$\varepsilon$ = {e:.2f}",
        )
        for j, e in enumerate(EPS_GRID)
    ]

    leg_cond = fig.legend(
        line_handles,
        line_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        ncol=4,
        fontsize=8.4,
        title="Condition",
        title_fontsize=9,
        framealpha=0.97,
        borderaxespad=0.5,
    )
    fig.add_artist(leg_cond)

    leg_eps = fig.legend(
        handles=eps_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.14),
        bbox_transform=fig.transFigure,
        ncol=4,
        fontsize=8.2,
        title="Discrete ε (markers)",
        title_fontsize=8.8,
        framealpha=0.97,
        borderaxespad=0.5,
    )

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.28, top=0.96)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved WT production envelope: %s", outpath)


def plot_design_envelopes(
    wt_df: pd.DataFrame,
    design_dfs: dict[int, pd.DataFrame],
    designs: list[dict],
    outpath: Path,
) -> None:
    """Overlay top design envelopes vs wild-type (base condition)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    wt_base = wt_df[wt_df.condition == "base"].sort_values("biomass_flux")
    ax.plot(wt_base["biomass_flux"], wt_base["pha_flux"],
            color="gray", lw=2.5, linestyle="--", marker="s", markersize=3,
            label="Wild-type", zorder=2)

    for idx, df in design_dfs.items():
        sub = df.sort_values("biomass_flux")
        d = designs[idx]
        ax.plot(sub["biomass_flux"], sub["pha_flux"],
                color=d["color"], lw=2.2, marker="o", markersize=3,
                label=d["label"], zorder=5)
        max_row = sub.loc[sub["pha_flux"].idxmax()]
        ax.scatter(max_row["biomass_flux"], max_row["pha_flux"],
                   color=d["color"], s=80, zorder=10, edgecolors="white", lw=1.5)

    ax.set_xlabel("Biomass Flux (h⁻¹)", fontsize=12)
    ax.set_ylabel("PHA Flux — DM_POLHYBU_c (mmol gDW⁻¹ h⁻¹)", fontsize=12)
    ax.set_title("Production Envelope: Top AL-Prioritised Designs vs Wild-Type\n"
                 "P. megaterium | Aerobic glucose | ε-Constraint FBA",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", title="Design", title_fontsize=9,
              framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved design envelopes: %s", outpath)


def plot_epsilon_heatmap(eps_df: pd.DataFrame, outpath: Path) -> None:
    """Heatmap of PHA improvement (%) vs condition × design at ε=0.30."""
    eps_focus = 0.30
    sub = eps_df[np.abs(eps_df["epsilon"] - eps_focus) < 0.01].copy()
    sub_designs = sub[sub["design"] != "wild_type"]

    if sub_designs.empty:
        logger.warning("No design data for heatmap at ε=%.2f", eps_focus)
        return

    pivot = sub_designs.pivot_table(
        index="design", columns="condition", values="vs_wt_improvement"
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.8 + 2)))
    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax * 0.3, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n").title() for c in pivot.columns],
                       fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Design {i+1}" for i in range(len(pivot.index))], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            tc = "black" if abs(val) < vmax * 0.6 else "white"
            txt = f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=tc)

    plt.colorbar(im, ax=ax, label="PHA Yield Improvement vs WT (%)")
    ax.set_title(f"PHA Yield Improvement Across Designs × Conditions\n"
                 f"(ε = {eps_focus} | {int(eps_focus*100)}% biomass preservation)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved epsilon constraint heatmap: %s", outpath)


def _wt_plus_merged_design_points(
    wt_x: float,
    wt_y: float,
    design_xy: list[tuple[int, float, float]],
    designs: list[dict],
    *,
    x_ndigits: int = 6,
    y_ndigits: int = 6,
) -> list[dict]:
    """
    WT as one marker; designs with identical (x, y) after rounding merge to one marker.

    ``design_xy`` entries are (design_idx, x, y). Legend/annotation text uses
    comma-separated design ids, e.g. "D1, D2".
    """
    from collections import defaultdict

    buckets: dict[tuple[float, float], list[tuple[int, float, float]]] = defaultdict(list)
    for idx, x, y in design_xy:
        key = (round(float(x), x_ndigits), round(float(y), y_ndigits))
        buckets[key].append((idx, float(x), float(y)))

    group_markers = ["o", "s", "^", "v", "P"]
    out: list[dict] = []
    for bi, (_key, members) in enumerate(sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
        idxs = sorted(t[0] for t in members)
        xs_m = [t[1] for t in members]
        ys_m = [t[2] for t in members]
        mx = float(np.mean(xs_m))
        my = float(np.mean(ys_m))
        short = ", ".join(f"D{i + 1}" for i in idxs)
        first_idx = idxs[0]
        out.append({
            "x":          mx,
            "y":          my,
            "short":      short,
            "legend":     short,
            "color":      designs[first_idx]["color"],
            "marker":     group_markers[bi % len(group_markers)],
        })

    out.append({
        "x":      float(wt_x),
        "y":      float(wt_y),
        "short":  "WT",
        "legend": "WT",
        "color":  "gray",
        "marker": "D",
    })
    out.sort(key=lambda p: p["x"])
    return out


def _build_eps_panel_points(
    eps_df: pd.DataFrame,
    designs: list[dict],
    design_dfs: dict[int, pd.DataFrame],
    *,
    epsilon_focus: float,
    condition: str = "base",
) -> list[dict] | None:
    """FBA (biomass_flux, pha_flux) at fixed ε; merged to same 3-point logic as top panel."""
    sub = eps_df[
        (eps_df["condition"] == condition)
        & (np.abs(eps_df["epsilon"].astype(float) - epsilon_focus) < 0.01)
    ]
    if sub.empty:
        logger.warning("No rows in eps_df for condition=%r, epsilon=%.2f", condition, epsilon_focus)
        return None

    wt = sub[sub["design"] == "wild_type"]
    if wt.empty:
        logger.warning("No wild_type row at epsilon=%.2f in eps_df", epsilon_focus)
        return None
    r0 = wt.iloc[0]
    wt_x = float(r0["biomass_flux"])
    wt_y = float(r0["pha_flux"])

    design_xy: list[tuple[int, float, float]] = []
    for idx in sorted(design_dfs.keys()):
        d = designs[idx]
        pref = d["label"][:40]
        dr = sub[sub["design"] == pref]
        if dr.empty:
            logger.warning("No ε=%.2f row for design key %r", epsilon_focus, pref)
            continue
        r = dr.iloc[0]
        design_xy.append((idx, float(r["biomass_flux"]), float(r["pha_flux"])))

    if not design_xy:
        return None
    return _wt_plus_merged_design_points(wt_x, wt_y, design_xy, designs)


def _pareto_point_annotate_with_leader(
    ax,
    pt: dict,
    *,
    panel: str,
) -> None:
    """Place text away from neighbours; thin line from label toward marker."""
    short = str(pt["short"])
    x, y = float(pt["x"]), float(pt["y"])
    c = pt["color"]
    ap = {
        "arrowstyle": "-",
        "color":      c,
        "lw":         0.65,
        "alpha":      0.88,
        "shrinkA":    1,
        "shrinkB":    2,
    }
    # Panel (B) uses a tighter vertical scale — slightly smaller offsets.
    if panel == "B":
        if short == "WT":
            xytext, ha, va = (0, 22), "center", "bottom"
        elif short == "D1, D2":
            xytext, ha, va = (-50, -20), "right", "top"
        elif short == "D3, D4, D5":
            xytext, ha, va = (54, 20), "left", "bottom"
        else:
            xytext, ha, va = (0, -18), "center", "top"
    else:
        if short == "WT":
            xytext, ha, va = (0, 30), "center", "bottom"
        elif short == "D1, D2":
            xytext, ha, va = (-62, -26), "right", "top"
        elif short == "D3, D4, D5":
            xytext, ha, va = (68, 30), "left", "bottom"
        else:
            xytext, ha, va = (0, -22), "center", "top"

    fs = 7 if len(short) > 9 else 8
    ax.annotate(
        short,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fs,
        fontweight="bold",
        color=c,
        zorder=11,
        arrowprops=ap,
    )


def plot_pareto_front(
    wt_df: pd.DataFrame,
    design_dfs: dict[int, pd.DataFrame],
    designs: list[dict],
    outpath: Path,
    eps_df: pd.DataFrame | None = None,
    *,
    epsilon_focus: float = 0.30,
) -> None:
    """Single-panel Pareto comparison on unconstrained envelope (panel A only)."""
    wt_base = wt_df[wt_df.condition == "base"]
    wt_x = float(wt_base["max_biomass_unconstrained"].max())
    wt_y = float(wt_base["pha_flux"].max())
    design_xy: list[tuple[int, float, float]] = []
    for idx, df in sorted(design_dfs.items()):
        design_xy.append((
            idx,
            float(df["max_biomass_unconstrained"].max()),
            float(df["pha_flux"].max()),
        ))
    points = _wt_plus_merged_design_points(wt_x, wt_y, design_xy, designs)

    xs = np.array([p["x"] for p in points], dtype=float)
    ys = np.array([p["y"] for p in points], dtype=float)
    x_span = float(np.ptp(xs))
    y_span = float(np.ptp(ys))
    pad_x = max(x_span * 0.30, 2.5e-4) if x_span > 0 else 2e-3
    pad_y = max(y_span * 0.26, 0.04) if y_span > 0 else 0.07

    # Publication update: keep only unconstrained-envelope panel.
    fig, ax0 = plt.subplots(figsize=(10, 7.2))
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("#2f2f2f")
    fig.patch.set_linewidth(1.2)

    for pt in points:
        ax0.scatter(
            pt["x"],
            pt["y"],
            color=pt["color"],
            marker=pt["marker"],
            s=210,
            zorder=10,
            edgecolors="white",
            linewidth=2.0,
            alpha=0.95,
            label=pt["legend"],
        )
        _pareto_point_annotate_with_leader(ax0, pt, panel="A")

    ax0.set_xlim(float(xs.min()) - pad_x, float(xs.max()) + pad_x)
    ax0.set_ylim(float(ys.min()) - pad_y, float(ys.max()) + pad_y)
    ax0.set_xlabel(
        r"$\mu_{\mathrm{max}}$ (h$^{-1}$) — max biomass on unconstrained envelope",
        fontsize=11,
    )
    ax0.set_ylabel("Max PHB flux (mmol gDW⁻¹ h⁻¹)", fontsize=11)
    # Keep full axis frame for publication panel styling.
    for side in ("top", "right", "left", "bottom"):
        ax0.spines[side].set_visible(True)
        ax0.spines[side].set_linewidth(1.0)
        ax0.spines[side].set_color("#2f2f2f")


    # Put legend panel inside the figure area.
    ax0.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.18),
        ncol=1,
        fontsize=9,
        framealpha=0.97,
        edgecolor="0.85",
    )
    plt.tight_layout()

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Pareto front: %s", outpath)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STEP 11 — Phenotypic Phase Plane Analysis (COBRA FBA)")
    logger.info("=" * 70)

    from phaopt.io import load_sbml_model
    from phaopt.utils import load_candidate_reactions, load_conditions, load_model_config

    cfg = load_model_config()
    cond_cfg = load_conditions()
    cand_cfg = load_candidate_reactions()

    logger.info("Loading COBRA model: %s", cfg["extended_model_path"])
    model = load_sbml_model(cfg["extended_model_path"])

    conditions = cond_cfg.get("conditions", {})
    n_env_points = 25

    # ── 1. Wild-type production envelopes ────────────────────────────────
    logger.info("Computing wild-type production envelopes (%d conditions × %d ε-points)…",
                len(conditions), n_env_points)
    wt_df = compute_wt_envelopes(model, cfg, cand_cfg, cond_cfg, n_points=n_env_points)
    wt_df.to_csv(TABLES_DIR / "production_envelope_wt.csv", index=False)
    logger.info("Saved WT envelope: %d rows", len(wt_df))

    # ── 2. Top designs from AL ───────────────────────────────────────────
    try:
        top_designs = _load_top_designs_from_al(n=5)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error("Cannot proceed without AL results. Exiting.")
        return

    base_overrides = conditions.get("base", {}).get("overrides") or {}

    logger.info("Computing design production envelopes (base condition, %d designs)…",
                len(top_designs))
    design_dfs = compute_design_envelopes(
        model, cfg, cand_cfg, "base", base_overrides, top_designs, n_points=n_env_points
    )

    design_all = pd.concat(
        [df.assign(design_idx=idx) for idx, df in design_dfs.items()],
        ignore_index=True,
    )
    design_all.to_csv(TABLES_DIR / "production_envelope_top_designs.csv", index=False)

    # ── 3. Epsilon-constraint matrix (all designs × all conditions) ──────
    logger.info("Computing epsilon-constraint matrix (%d designs × %d conditions × %d ε-levels)…",
                len(top_designs) + 1, len(conditions),
                len(cfg.get("biomass_fraction_grid", [0.10, 0.30, 0.50, 0.70])))
    eps_df = compute_epsilon_constraint_matrix(model, cfg, cand_cfg, cond_cfg, top_designs)
    eps_df.to_csv(TABLES_DIR / "epsilon_constraint_matrix.csv", index=False)
    logger.info("Saved epsilon matrix: %d rows", len(eps_df))

    # ── 4. Pareto-front summary table ────────────────────────────────────
    wt_base = wt_df[wt_df.condition == "base"]
    wt_max_bio = float(wt_base["max_biomass_unconstrained"].max())
    wt_max_pha = float(wt_base["pha_flux"].max())

    pareto_rows: list[dict] = []
    for idx, df in design_dfs.items():
        d = top_designs[idx]
        eps30_row = eps_df[
            (eps_df["design"] == d["label"][:40]) &
            (eps_df["condition"] == "base") &
            (np.abs(eps_df["epsilon"] - 0.30) < 0.01)
        ]
        pha_at_30 = float(eps30_row["pha_flux"].iloc[0]) if len(eps30_row) else 0.0

        max_bio_design = float(df["max_biomass_unconstrained"].max())
        max_pha_design = float(df["pha_flux"].max())

        pareto_rows.append({
            "design_id":         d["label"][:40],
            "label":             d["label"][:60],
            "knockouts":         "|".join(d["knockouts"]),
            "upregulations":     "|".join(d["upregulations"]),
            "max_biomass":       round(max_bio_design, 6),
            "max_pha":           round(max_pha_design, 6),
            "max_pha_at_eps30":  round(pha_at_30, 6),
            "pha_improvement_pct": round(100 * (max_pha_design - wt_max_pha) / max(wt_max_pha, 1e-6), 1),
            "biomass_cost_pct":  round(100 * (wt_max_bio - max_bio_design) / max(wt_max_bio, 1e-6), 1),
        })

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.to_csv(TABLES_DIR / "pareto_front_summary.csv", index=False)

    # ── 5. Figures ───────────────────────────────────────────────────────
    logger.info("Generating figures…")
    plot_wt_production_envelope(wt_df, FIGS_DIR / "production_envelope_wt.png", eps_df)
    plot_design_envelopes(wt_df, design_dfs, top_designs,
                          FIGS_DIR / "production_envelope_top5.png")
    plot_epsilon_heatmap(eps_df, FIGS_DIR / "epsilon_constraint_heatmap.png")
    plot_pareto_front(
        wt_df[wt_df.condition == "base"],
        design_dfs,
        top_designs,
        FIGS_DIR / "pareto_front_comparison.png",
        eps_df,
        epsilon_focus=0.30,
    )

    elapsed = time.time() - t0
    logger.info("Done in %.1f s", elapsed)

    print("\n" + "=" * 70)
    print("PARETO FRONT SUMMARY — Top Designs vs Wild-Type (COBRA FBA)")
    print("=" * 70)
    print(pareto_df[["label", "max_biomass", "max_pha_at_eps30",
                     "pha_improvement_pct", "biomass_cost_pct"]].to_string(index=False))
    print("=" * 70)

    # Print WT reference
    print(f"\nWild-type reference (base):  max biomass = {wt_max_bio:.6f} h⁻¹,  "
          f"max PHA = {wt_max_pha:.6f} mmol/gDW/h")

    wt_eps30 = eps_df[
        (eps_df["design"] == "wild_type") &
        (eps_df["condition"] == "base") &
        (np.abs(eps_df["epsilon"] - 0.30) < 0.01)
    ]
    if len(wt_eps30):
        print(f"WT PHA at ε=0.30 (base):     {float(wt_eps30['pha_flux'].iloc[0]):.6f} mmol/gDW/h")


if __name__ == "__main__":
    main()
