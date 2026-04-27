#!/usr/bin/env python3
"""
WT vs mean(5 AL designs) PHB flux at ε = 0.30, per environmental condition.

Reads: results/tables/epsilon_constraint_matrix.csv

Output: results/tables/condition_phb_wt_vs_design_mean_eps030.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPS_PATH = PROJECT_ROOT / "results" / "tables" / "epsilon_constraint_matrix.csv"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "condition_phb_wt_vs_design_mean_eps030.csv"

EPS = 0.30
BIO_TOL = 1e-6

ROW_ORDER: list[tuple[str, str, str]] = [
    ("base", "C1", "Glc, aerobic (ref.)"),
    ("low_carbon", "C2", "Glc, carbon-lim."),
    ("low_oxygen", "C3", "Glc, micro-aer."),
    ("glycerol_aerobic", "C4", "Gly, aerobic"),
    ("acetate_aerobic", "C5", "Ac, aerobic"),
    ("glycerol_low_oxygen", "C6", "Gly, micro-aer."),
    ("mixed_glucose_glycerol", "C7", "Glc + Gly mix"),
]


def main() -> None:
    df = pd.read_csv(EPS_PATH)
    sub = df[np.isclose(df["epsilon"].astype(float), EPS, atol=0.01)].copy()

    rows: list[dict] = []
    for code, cid, short in ROW_ORDER:
        wt = sub[(sub["design"] == "wild_type") & (sub["condition"] == code)]
        des = sub[(sub["design"] != "wild_type") & (sub["condition"] == code)]
        if wt.empty or des.empty:
            raise SystemExit(f"Missing rows for condition={code!r}")
        n = des["design"].nunique()
        if n != 5:
            print(f"Warning: {code} has {n} designs (expected 5).", flush=True)

        w_pha = float(wt["pha_flux"].iloc[0])
        d_mean = float(des["pha_flux"].mean())
        d_std = float(des["pha_flux"].std(ddof=1)) if len(des) > 1 else 0.0
        d_bio_mean = float(des["biomass_flux"].mean())

        if abs(w_pha) > 1e-12:
            delta_pct = 100.0 * (d_mean - w_pha) / w_pha
        else:
            delta_pct = float("nan")

        viable = (d_mean > BIO_TOL) and (d_bio_mean > BIO_TOL)

        rows.append({
            "ID": cid,
            "Condition": f"{cid} — {short}",
            "condition_key": code,
            "n_designs": int(n),
            "wt_phb_flux": round(w_pha, 6),
            "design_phb_mean": round(d_mean, 6),
            "design_phb_std": round(d_std, 6),
            "delta_pct_vs_wt": round(delta_pct, 4) if delta_pct == delta_pct else None,
            "design_biomass_flux_mean": round(d_bio_mean, 6),
            "viable_mean_phb_positive": viable,
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
