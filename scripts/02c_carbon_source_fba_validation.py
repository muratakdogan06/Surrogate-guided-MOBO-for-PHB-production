#!/usr/bin/env python3
"""
02c_carbon_source_fba_validation.py
Real cobra FBA carbon-source phenotyping for iJA1121.

Experimental growth/no-growth data:
  Positive controls: Aminian-Dehkordi et al. (2019) Sci Rep 9:18762
                     Biedendieck et al. (2021) Appl Microbiol Biotechnol 105:5665
                     Cal et al. (2025) PLoS ONE e0322838
  Negative controls: Biedendieck et al. (2021) — substrates not utilised by DSM319

Outputs
-------
  results/tables/carbon_source_fba_phenotyping.csv
  results/tables/fba_validation_report.json
  results/figures/carbon_source_confusion_matrix_fba.png
"""

import json
import warnings
from pathlib import Path

import cobra
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"
FIGS_DIR     = PROJECT_ROOT / "results" / "figures"

GLUCOSE_EXCHANGE = "Re_bm01273"
GROWTH_THRESHOLD = 1e-6


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            _cfg = yaml.safe_load(f)
        model_rel = _cfg.get("extended_model_path") or "data/raw/bmeg_original.xml"
    else:
        model_rel = "data/raw/bmeg_original.xml"
    model_path = PROJECT_ROOT / model_rel
    print(f"Loading iJA1121 via cobra from {model_path} ...")
    model = cobra.io.read_sbml_model(str(model_path))
    print(f"  Reactions  : {len(model.reactions)}")
    print(f"  Metabolites: {len(model.metabolites)}")
    print(f"  Genes      : {len(model.genes)}")

    default_glucose_lb = model.reactions.get_by_id(GLUCOSE_EXCHANGE).lower_bound
    print(f"  Default glucose lb: {default_glucose_lb} mmol/gDW/h")

    with model:
        sol = model.optimize()
        wt_growth = sol.objective_value if sol.status == "optimal" else 0.0
    print(f"  WT growth (default glucose bounds): {wt_growth:.4f} h-1\n")

    lb = default_glucose_lb

    carbon_sources = {
        "glucose":    ("Re_bm01273", lb, True,  "Aminian-Dehkordi 2019"),
        "fructose":   ("Re_bm01296", lb, True,  "Aminian-Dehkordi 2019"),
        "arabinose":  ("Re_bm01274", lb, True,  "Aminian-Dehkordi 2019"),
        "xylose":     ("Re_bm01327", lb, True,  "Aminian-Dehkordi 2019"),
        "galactose":  ("Re_bm01325", lb, True,  "Aminian-Dehkordi 2019"),
        # D-mannose proton symport (Rt_bm01037) — present in iJA1121 / extended GEM
        "mannose":    ("Rt_bm01037", lb, True,  "Aminian-Dehkordi 2019"),
        "mannitol":   ("Re_bm01297", lb, True,  "Aminian-Dehkordi 2019"),
        "lactose":    ("Re_bm01275", lb, True,  "Biedendieck 2021"),
        "glycerol":   ("Re_bm01324", lb, True,  "Cal 2025"),
        "acetate":    ("Re_bm01320", lb, True,  "Aminian-Dehkordi 2019"),
        "pyruvate":   ("Re_bm01344", lb, True,  "Aminian-Dehkordi 2019"),
        "succinate":  ("Re_bm01347", lb, True,  "Aminian-Dehkordi 2019"),
        "malate":     ("Re_bm01346", lb, True,  "Aminian-Dehkordi 2019"),
        "citrate":    ("Re_bm01345", lb, True,  "Aminian-Dehkordi 2019"),
        "cellulose":  (None,         lb, False, "Biedendieck 2021"),
        "rhamnose":   (None,         lb, False, "Biedendieck 2021"),
        "sorbitol":   (None,         lb, False, "Biedendieck 2021"),
        "inositol":   (None,         lb, False, "Biedendieck 2021"),
    }

    print(f"{'Substrate':<12} {'ExRxn':<15} {'Pred':>6} {'Rate(h-1)':>10} "
          f"{'Exp':>6} {'Match':>6}")
    print("-" * 62)

    rows = []
    for substrate, (ex_rxn, _lb, exp_growth, ref) in carbon_sources.items():
        if ex_rxn is None:
            if not exp_growth:
                pred_growth = False
                rate        = 0.0
                correct     = True
                note        = "No exchange reaction → correctly predicts no growth"
            else:
                # Positive experiment but no explicit uptake reaction in this GEM
                # representation → FBA cannot predict growth; count as predicted
                # no-growth (FN) for confusion-matrix honesty (not omitted).
                pred_growth = False
                rate        = 0.0
                correct     = False
                note        = (
                    "No exchange reaction in iJA1121 (e.g. mannose/PTS); "
                    "counts as predicted no growth vs experimental growth (FN)"
                )
            match = "✓" if correct else ("✗" if correct is False else "?")
            rate_str = f"{rate:.4f}" if rate is not None else "?"
            print(f"{substrate:<12} {'NO_EXCHANGE':<15} "
                  f"{str(pred_growth):>6} {rate_str:>10} "
                  f"{str(exp_growth):>6} {match:>6}")
            rows.append({
                "substrate":             substrate,
                "exchange_reaction":     "NO_EXCHANGE",
                "experimental_growth":   exp_growth,
                "predicted_growth":      pred_growth,
                "predicted_growth_rate": rate,
                "correct":               correct,
                "reference":             ref,
                "note":                  note,
            })
            continue

        with model:
            model.reactions.get_by_id(GLUCOSE_EXCHANGE).lower_bound = 0.0
            model.reactions.get_by_id(ex_rxn).lower_bound = _lb
            sol  = model.optimize()
            rate = sol.objective_value if sol.status == "optimal" else 0.0

        pred_growth = rate > GROWTH_THRESHOLD
        correct     = (pred_growth == exp_growth)
        match       = "✓" if correct else "✗"

        print(f"{substrate:<12} {ex_rxn:<15} {str(pred_growth):>6} {rate:>10.4f} "
              f"{str(exp_growth):>6} {match:>6}")
        rows.append({
            "substrate":             substrate,
            "exchange_reaction":     ex_rxn,
            "experimental_growth":   exp_growth,
            "predicted_growth":      pred_growth,
            "predicted_growth_rate": round(rate, 6),
            "correct":               correct,
            "reference":             ref,
            "note":                  "cobra FBA",
        })

    df = pd.DataFrame(rows)

    completed = df[df["correct"].notna()].copy()
    completed["predicted_growth"]    = completed["predicted_growth"].astype(bool)
    completed["experimental_growth"] = completed["experimental_growth"].astype(bool)

    tp = int(( completed["predicted_growth"] &  completed["experimental_growth"]).sum())
    tn = int((~completed["predicted_growth"] & ~completed["experimental_growth"]).sum())
    fp = int(( completed["predicted_growth"] & ~completed["experimental_growth"]).sum())
    fn = int((~completed["predicted_growth"] &  completed["experimental_growth"]).sum())

    sensitivity = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp)       if (tn + fp) > 0 else 0.0
    accuracy    = (tp + tn) / len(completed) if len(completed) > 0 else 0.0
    mcc_den     = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc         = (tp*tn - fp*fn) / mcc_den if mcc_den > 0 else 0.0

    print(f"\n{'='*62}")
    print(f"FBA Phenotyping — iJA1121 ({len(completed)}/{len(df)} evaluated)")
    print(f"{'='*62}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Sensitivity : {sensitivity:.3f}")
    print(f"  Specificity : {specificity:.3f}")
    print(f"  Accuracy    : {accuracy:.3f}")
    print(f"  MCC         : {mcc:.3f}")
    print(f"  WT growth   : {wt_growth:.4f} h-1 (lb={default_glucose_lb})")

    csv_path = TABLES_DIR / "carbon_source_fba_phenotyping.csv"
    df.to_csv(csv_path, index=False)

    report = {
        "method":               "cobra FBA",
        "model_file":           str(model_path.relative_to(PROJECT_ROOT)),
        "model_id":             "iJA1121",
        "organism":             "Priestia megaterium DSM319",
        "mannose_uptake_rxn":   "Rt_bm01037",
        "n_substrates_total":   len(df),
        "n_evaluated":          len(completed),
        "n_skipped":            len(df) - len(completed),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensitivity":          round(sensitivity, 4),
        "specificity":          round(specificity, 4),
        "accuracy":             round(accuracy, 4),
        "MCC":                  round(mcc, 4),
        "wt_growth_glucose":    round(wt_growth, 6),
        "default_glucose_lb":   default_glucose_lb,
        "growth_threshold":     GROWTH_THRESHOLD,
        "reference_exp_data": [
            "Aminian-Dehkordi et al. (2019) Sci Rep 9:18762",
            "Biedendieck et al. (2021) Appl Microbiol Biotechnol 105:5665",
            "Cal et al. (2025) PLoS ONE e0322838",
        ],
    }
    json_path = TABLES_DIR / "fba_validation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)

    ax = axes[0]
    cm_vals   = [[tp, fn], [fp, tn]]
    cm_labels = [["TP", "FN"], ["FP", "TN"]]
    cm_colors = [["#27ae60", "#e74c3c"], ["#e67e22", "#2980b9"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle([j, 1-i], 1, 1,
                         facecolor=cm_colors[i][j], alpha=0.80,
                         edgecolor="white", lw=2.5))
            ax.text(j+0.5, 1.5-i,
                    f"{cm_labels[i][j]}\nn = {cm_vals[i][j]}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nGrowth", "Predicted\nNo growth"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Experimental\nNo growth", "Experimental\nGrowth"], fontsize=10)
    ax.set_title("(a) Confusion matrix", fontsize=11, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    ax2 = axes[1]
    plot_df = completed.sort_values("predicted_growth_rate", ascending=True).reset_index(drop=True)
    n_bars = len(plot_df)
    y_pos = np.arange(n_bars)
    rates = plot_df["predicted_growth_rate"].values.astype(float)
    xmax = max(float(rates.max()), GROWTH_THRESHOLD) * 1.18
    if xmax <= 0:
        xmax = 0.05

    MANNOSE_RED = "#C62828"
    for i, row in plot_df.iterrows():
        rate = float(row["predicted_growth_rate"])
        ok = bool(row["correct"])
        sub = str(row["substrate"])
        is_mannose = sub == "mannose"
        # Mannose: always highlighted in red (FN / model–literature mismatch).
        if is_mannose:
            face = MANNOSE_RED
        elif ok:
            face = "#27ae60"
        else:
            face = "#e74c3c"
        # Zero predicted rate: no visible bar — use small stub width for mannose only.
        bar_w = rate
        if is_mannose and bar_w <= GROWTH_THRESHOLD:
            bar_w = 0.022 * xmax
        ax2.barh(i, bar_w, color=face, alpha=0.88, edgecolor="white", linewidth=0.5, height=0.72)
        x_text = rate + 0.008 * xmax if rate > GROWTH_THRESHOLD else 0.012 * xmax
        ax2.text(
            x_text,
            i,
            f"{rate:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=(MANNOSE_RED if is_mannose else "#212121"),
        )

    ax2.set_yticks(y_pos)
    ytl = ax2.set_yticklabels([str(s) for s in plot_df["substrate"]], fontsize=9)
    for tick, lab in zip(ytl, plot_df["substrate"].astype(str)):
        if lab == "mannose":
            tick.set_color(MANNOSE_RED)
            tick.set_fontweight("bold")
    ax2.set_xlim(0, xmax)
    ax2.axvline(x=GROWTH_THRESHOLD, color="black", ls="--",
                lw=1.2, alpha=0.6, label="Growth threshold")
    ax2.set_xlabel("Predicted growth rate (h⁻¹)", fontsize=10)
    ax2.set_title("(b) Per-substrate FBA growth rates", fontsize=11, fontweight="bold", pad=10)
    p1 = mpatches.Patch(color="#27ae60", alpha=0.88, label="Correct")
    p2 = mpatches.Patch(color="#C62828", alpha=0.88, label="Incorrect")
    ax2.legend(handles=[p1, p2], fontsize=9, loc="lower right")
    fig_path = FIGS_DIR / "carbon_source_confusion_matrix_fba.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()