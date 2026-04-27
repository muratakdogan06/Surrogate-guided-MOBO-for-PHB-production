#!/usr/bin/env python3
"""
Table S2 — Wild-type PHB flux (mmol gDW⁻¹ h⁻¹) vs condition × ε.

Rows follow manuscript order C1–C7; values from ``epsilon_constraint_matrix.csv``
(``design == wild_type``), same FBA setup as Step 11.

Output: results/tables/table_s2_wt_phb_flux_condition_epsilon.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPS_PATH = PROJECT_ROOT / "results" / "tables" / "epsilon_constraint_matrix.csv"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "table_s2_wt_phb_flux_condition_epsilon.csv"

# (code_key, label_C, short description)
ROW_ORDER: list[tuple[str, str, str]] = [
    ("base", "C1", "Aerobic glucose (reference)"),
    ("low_carbon", "C2", "Carbon-limited glucose"),
    ("low_oxygen", "C3", "Micro-aerobic glucose"),
    ("glycerol_aerobic", "C4", "Aerobic glycerol"),
    ("acetate_aerobic", "C5", "Aerobic acetate"),
    ("glycerol_low_oxygen", "C6", "Micro-aerobic glycerol"),
    ("mixed_glucose_glycerol", "C7", "Mixed glucose–glycerol"),
]

EPS_COLS = [0.1, 0.3, 0.5, 0.7]


def main() -> None:
    df = pd.read_csv(EPS_PATH)
    wt = df[df["design"].astype(str) == "wild_type"].copy()
    wt["epsilon"] = wt["epsilon"].astype(float)

    rows: list[dict] = []
    for code, cid, desc in ROW_ORDER:
        sub = wt[wt["condition"].astype(str) == code]
        if sub.empty:
            raise SystemExit(f"Missing wild_type rows for condition={code!r}")
        row: dict[str, str | float] = {
            "ID": cid,
            "Condition": f"{cid} — {desc}",
            "condition_key": code,
        }
        for eps in EPS_COLS:
            m = sub[np.isclose(sub["epsilon"].to_numpy(), eps, rtol=0, atol=0.02)]
            if m.empty:
                raise SystemExit(f"No row for {code!r} at epsilon={eps}")
            val = float(m["pha_flux"].iloc[0])
            row[f"epsilon_{eps:g}"] = round(val, 6)
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(out)} rows)")
    # Pretty preview (3 decimals)
    disp = out.copy()
    for c in [f"epsilon_{e:g}" for e in EPS_COLS]:
        disp[c] = disp[c].map(lambda x: f"{x:.3f}")
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
