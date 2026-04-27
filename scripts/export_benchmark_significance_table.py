#!/usr/bin/env python3
"""
Export a single manuscript-style table: MC-EHVI (``al``) vs each baseline on three
AUCC metrics from ``results/tables/benchmark_auc_summary.csv``.

**Direction (matches manuscript §3.5.2):** AUCC here is the trapezoidal integral of the
reported convergence curve; **lower AUCC means faster / more efficient convergence**
to comparable outcomes, so “MC-EHVI superior” is tested as **median(AL − baseline) < 0**
(SciPy: ``alternative='less'`` on the paired differences). Using ``greater`` would
invert conclusions versus this CSV.

Columns include Wilcoxon (paired), Benjamini–Hochberg adjusted p-values across
all nine tests, rank-biserial r (Kerby formula when exact Wilcoxon has no z), and
paired Cohen's d on the AUCC differences.

Note
----
* ``wilcoxon_statistic`` is SciPy's signed-rank statistic for the chosen
  ``alternative`` (not Mann–Whitney ``W``).
* Default ``wilcox_method='auto'`` uses exact small-``n`` p-values where available.
* AUCC values are exactly as produced by ``08b_summarize_al_benchmarks.py``
  (trapezoid over each method's own evaluation grid). They may differ from the
  sparse-grid hypervolume AUC used in ``10_statistical_significance.py``.

Output
------
  results/tables/benchmark_significance_multimetric.csv
  results/tables/benchmark_significance_summary_table.csv
    Reviewer-friendly table: Comparison, Metric, W, p_adj, r, Cohen's d, post-hoc power.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_INPUT = TABLE_DIR / "benchmark_auc_summary.csv"
DEFAULT_OUTPUT = TABLE_DIR / "benchmark_significance_multimetric.csv"
DEFAULT_SUMMARY_OUTPUT = TABLE_DIR / "benchmark_significance_summary_table.csv"


def approximate_power(n: int, effect_size: float, alpha: float = 0.05) -> float:
    """
    Approximate Wilcoxon signed-rank power (same form as ``10_statistical_significance``).
    ``effect_size`` = |Cohen's d| on paired AUCC differences.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    ncp = abs(float(effect_size)) * np.sqrt(n / 1.5)
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    return float(np.clip(power, 0, 1))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR correction (same logic as ``10_statistical_significance``)."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.minimum(1.0, sorted_p * n / (np.arange(n) + 1))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


def cohens_d_paired(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(d) / sd)


def sd_paired_diff(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float("nan")
    return float(np.std(d, ddof=1))


def rank_biserial_kerby_w(statistic: float, n_pairs: int) -> float:
    """
    Rank-biserial correlation from the Wilcoxon signed-rank ``statistic`` (SciPy's
    returned value for the chosen ``alternative``) and n_pairs after zero-drop.

    r = 1 - 2W / (n(n+1)/2). Matches manuscript §3.5.2 when exact Wilcoxon is used.
    """
    denom = n_pairs * (n_pairs + 1) / 2.0
    if denom <= 0:
        return float("nan")
    return float(1.0 - 2.0 * float(statistic) / denom)


def wilcoxon_alternative(metric_col: str, baseline: str, *, default: str) -> str:
    """Manuscript: Pareto vs greedy uses two-sided; all other rows use ``default``."""
    if metric_col == "pareto_discovered_auc" and baseline == "greedy":
        return "two-sided"
    return default


def run_wilcoxon_row(
    al: np.ndarray,
    bl: np.ndarray,
    *,
    alternative: str,
    wilcox_method: str,
) -> dict:
    d = np.asarray(al, dtype=float) - np.asarray(bl, dtype=float)
    n_pairs = int(len(d))
    n_nonzero = int(np.sum(d != 0.0))

    res = stats.wilcoxon(
        d,
        alternative=alternative,
        zero_method="wilcox",
        method=wilcox_method,
    )
    z = getattr(res, "zstatistic", None)
    if z is None or (isinstance(z, float) and np.isnan(z)):
        z = float("nan")
    else:
        z = float(z)

    r_z = float(z / np.sqrt(n_pairs)) if np.isfinite(z) and n_pairs > 0 else float("nan")
    n_ranked = n_nonzero if n_nonzero > 0 else n_pairs
    r_kerby = rank_biserial_kerby_w(float(res.statistic), n_ranked)
    # Use Kerby r when exact test has no z (matches manuscript r = 1, 0.93, …)
    r_rb = r_z if np.isfinite(r_z) else r_kerby

    return {
        "n_pairs": n_pairs,
        "n_nonzero_diff": n_nonzero,
        "wilcoxon_statistic": float(res.statistic),
        "z_statistic": z,
        "p_value": float(res.pvalue),
        "rank_biserial_r": r_rb,
        "rank_biserial_r_kerby": r_kerby,
        "mean_diff_AL_minus_baseline": float(np.mean(d)),
        "sd_paired_diff": sd_paired_diff(d),
        "cohens_d_paired": cohens_d_paired(d),
    }


def build_table(
    df: pd.DataFrame,
    *,
    alternative: str,
    wilcox_method: str,
    alpha: float,
) -> pd.DataFrame:
    required = {"method", "seed", "hypervolume_auc", "best_viable_pha_auc", "pareto_discovered_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"benchmark_auc_summary missing columns: {sorted(missing)}")

    metrics = [
        ("hypervolume_auc", "Hypervolume AUCC"),
        ("best_viable_pha_auc", "Best viable PHB flux AUCC"),
        ("pareto_discovered_auc", "Pareto discovery AUCC"),
    ]
    baselines = ["random", "greedy", "exploitation"]

    rows: list[dict] = []
    keys_for_bh: list[tuple[str, str]] = []
    pvals: list[float] = []

    for metric_col, metric_label in metrics:
        wide = df.pivot(index="seed", columns="method", values=metric_col)
        if "al" not in wide.columns:
            raise ValueError("Column 'al' (MC-EHVI) missing after pivot.")
        al = wide["al"].sort_index().to_numpy(dtype=float)

        for bl_name in baselines:
            if bl_name not in wide.columns:
                raise ValueError(f"Baseline '{bl_name}' missing after pivot.")
            bl = wide[bl_name].sort_index().to_numpy(dtype=float)

            alt = wilcoxon_alternative(metric_col, bl_name, default=alternative)
            wrow = run_wilcoxon_row(al, bl, alternative=alt, wilcox_method=wilcox_method)
            row = {
                "comparison": f"MC-EHVI (al) vs {bl_name}",
                "metric_key": metric_col,
                "metric_label": metric_label,
                "baseline": bl_name,
                "AL_mean_auc": float(np.mean(al)),
                "baseline_mean_auc": float(np.mean(bl)),
                **wrow,
            }
            rows.append(row)
            keys_for_bh.append((metric_col, bl_name))
            pvals.append(wrow["p_value"])

    p_adj = benjamini_hochberg(pvals, alpha=alpha)
    for row, p_bh, (mk, bln) in zip(rows, p_adj, keys_for_bh, strict=True):
        row["p_value_BH"] = float(p_bh)
        row["significant_alpha_0.05"] = bool(p_bh < alpha)
        row["hypothesis_alternative"] = wilcoxon_alternative(mk, bln, default=alternative)
        row["wilcoxon_method"] = wilcox_method

    out = pd.DataFrame(rows)
    # Stable row order
    cat_metric = pd.Categorical(out["metric_key"], [m[0] for m in metrics])
    cat_bl = pd.Categorical(out["baseline"], baselines)
    out = out.assign(_m=cat_metric, _b=cat_bl).sort_values(["_m", "_b"]).drop(columns=["_m", "_b"])
    return out.reset_index(drop=True)


def build_reviewer_summary_table(
    detail: pd.DataFrame,
    *,
    n_seeds: int,
    alpha: float,
) -> pd.DataFrame:
    """
    Single table: Comparison × metric with W, BH-adjusted p, rank-biserial r,
    paired Cohen's d, and post-hoc normal-approximation power (§3.5.2 style).
    """
    baseline_label = {
        "random": "Random",
        "greedy": "Greedy",
        "exploitation": "Exploitation",
    }
    metric_short = {
        "hypervolume_auc": "HV AUCC",
        "best_viable_pha_auc": "PHB flux AUCC",
        "pareto_discovered_auc": "Pareto AUCC",
    }
    baseline_order = ["random", "greedy", "exploitation"]
    metric_order = [
        "hypervolume_auc",
        "best_viable_pha_auc",
        "pareto_discovered_auc",
    ]

    rows: list[dict] = []
    for bl in baseline_order:
        sub_bl = detail[detail["baseline"] == bl]
        for mk in metric_order:
            r0 = sub_bl[sub_bl["metric_key"] == mk]
            if r0.empty:
                continue
            row = r0.iloc[0]
            d = row["cohens_d_paired"]
            d_abs = float(abs(d)) if np.isfinite(d) else float("nan")
            if np.isfinite(d_abs):
                pow_ = approximate_power(n_seeds, d_abs, alpha=alpha)
            else:
                pow_ = float("nan")
            r_val = row["rank_biserial_r"]
            rows.append(
                {
                    "comparison": f"MC-EHVI vs {baseline_label.get(bl, bl.title())}",
                    "metric": metric_short.get(mk, mk),
                    "W": int(round(float(row["wilcoxon_statistic"]))),
                    "p_adj": float(row["p_value_BH"]),
                    "r": float(r_val) if np.isfinite(r_val) else float("nan"),
                    "cohens_d": float(d) if np.isfinite(d) else float("nan"),
                    "power_posthoc": float(pow_) if np.isfinite(pow_) else float("nan"),
                    "significant_BH_0.05": bool(row["significant_alpha_0.05"]),
                    "wilcox_alternative": row["hypothesis_alternative"],
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("Output", 1)[0].strip())
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to benchmark_auc_summary.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the multimetric significance CSV.",
    )
    p.add_argument(
        "--alternative",
        type=str,
        default="less",
        choices=["greater", "less", "two-sided"],
        help=(
            "Default SciPy ``alternative`` for paired Wilcoxon on (AL - baseline). "
            "``less`` = lower AUCC is better (§3.5.2). "
            "``pareto_discovered_auc`` vs ``greedy`` is always ``two-sided`` (manuscript)."
        ),
    )
    p.add_argument(
        "--wilcox-method",
        type=str,
        default="auto",
        choices=["approx", "auto", "exact"],
        help="SciPy ``method`` for Wilcoxon (``auto`` = exact small n, matches manuscript p).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for BH significance flag.",
    )
    p.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Reviewer summary table (Comparison, Metric, W, p_adj, r, d, power).",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the summary table CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    if not inp.is_file():
        print(f"[ERROR] Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(inp)
    out_df = build_table(
        df,
        alternative=args.alternative,
        wilcox_method=args.wilcox_method,
        alpha=args.alpha,
    )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outp, index=False)
    print(f"[INFO] Wrote {outp} ({len(out_df)} rows)")

    if not args.no_summary:
        n_seeds = int(out_df["n_pairs"].iloc[0]) if not out_df.empty else 10
        summ = build_reviewer_summary_table(out_df, n_seeds=n_seeds, alpha=args.alpha)
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(sp, index=False)
        print(f"[INFO] Wrote {sp} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
