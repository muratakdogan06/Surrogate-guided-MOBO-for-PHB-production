#!/usr/bin/env python3
"""
RUN_ALL_IMPROVEMENTS.py — Master Pipeline Runner
=================================================
Bioresource Technology Quality Improvement Suite
Runs all improvement steps in order and generates a comprehensive report.

Quality Improvement Steps Implemented:
  Step 1  → 02b_validate_gem_extended.py   (GEM validation + C-source phenotyping)
  Step 2  → conditions.yaml                 (Glycerol/acetate conditions — config only)
  Step 4  → 09_hyperparameter_sensitivity.py (OAT sensitivity analysis)
  Step 5  → 10_statistical_significance.py  (Wilcoxon tests, BH correction)
  Step 6  → active_learning.yaml            (10 seeds — config only)
  Step 7  → candidate_reactions.yaml        (Beta-ox annotation — config only)
  Step 8  → 11_phase_plane_analysis.py      (Production envelope / PhPP)
  Step 12 → 12_dataset_statistics_and_learning_curves.py (Table 1, learning curves)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "results/logs/master_pipeline.log"),
    ],
)
logger = logging.getLogger("master")

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

IMPROVEMENT_STEPS = [
    {
        "step":   "Step 1 & 7",
        "script": "02b_validate_gem_extended.py",
        "title":  "Extended GEM Validation & Beta-Oxidation Annotation",
        "description": (
            "Carbon-source utilization phenotyping with confusion matrix, "
            "gene essentiality analysis, and individual beta-oxidation reaction "
            "annotation with engineering priority classification."
        ),
    },
    {
        "step":   "Step 4",
        "script": "09_hyperparameter_sensitivity.py",
        "title":  "Hyperparameter Sensitivity Analysis (OAT)",
        "description": (
            "One-at-a-time sensitivity analysis for UCB kappa, diversity lambda, "
            "biomass penalty, and biomass viability threshold. Tornado chart "
            "and sensitivity curves generated."
        ),
    },
    {
        "step":   "Step 5 & 6",
        "script": "10_statistical_significance.py",
        "title":  "Statistical Significance Testing (n=10 seeds)",
        "description": (
            "Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR "
            "correction comparing AL-UCB vs all baselines. Power analysis "
            "table for transparent reporting."
        ),
    },
    {
        "step":   "Step 8",
        "script": "11_phase_plane_analysis.py",
        "title":  "Production Envelope & Phase Plane Analysis",
        "description": (
            "Phenotypic Phase Plane analysis demonstrating 'growth-compatible' "
            "strategies across all conditions. Top-5 design envelopes vs WT. "
            "Epsilon-constraint heatmap across 6 conditions."
        ),
    },
    {
        "step":   "Step 12",
        "script": "12_dataset_statistics_and_learning_curves.py",
        "title":  "Dataset Statistics & Surrogate Learning Curves",
        "description": (
            "Table 1 generation with complete dataset and model statistics. "
            "Surrogate learning curves (R² vs training size) to assess "
            "whether the model is data-starved or has plateaued."
        ),
    },
]


def run_script(script_name: str) -> tuple[bool, float, str]:
    """Run a pipeline script and return (success, elapsed_time, output_summary)."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return False, 0.0, f"Script not found: {script_path}"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            # Extract last 10 lines of stdout as summary
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            summary = "\n".join(lines[-10:]) if lines else "(no output)"
            return True, elapsed, summary
        else:
            error = result.stderr[-500:] if result.stderr else "(no stderr)"
            return False, elapsed, f"ERROR:\n{error}"
    except subprocess.TimeoutExpired:
        return False, 600.0, "TIMEOUT: Script exceeded 600 seconds"
    except Exception as e:
        return False, time.time() - t0, f"EXCEPTION: {e}"


def generate_final_report(results: list[dict]) -> None:
    """Generate a comprehensive improvement report."""
    report = {
        "pipeline":    "Bioresource Technology Quality Improvement Suite",
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_steps": len(results),
        "passed":      sum(1 for r in results if r["success"]),
        "failed":      sum(1 for r in results if not r["success"]),
        "total_time":  round(sum(r["elapsed"] for r in results), 1),
        "steps":       results,
        "outputs": {
            "tables": [str(f.name) for f in (PROJECT_ROOT / "results/tables").glob("*.csv")],
            "figures": [str(f.name) for f in (PROJECT_ROOT / "results/figures").glob("*.png")],
        },
        "quality_improvements_summary": {
            "Step 1":  "Extended GEM validation with C-source phenotyping (confusion matrix, MCC)",
            "Step 2":  "Added glycerol_aerobic, acetate_aerobic, glycerol_low_oxygen, mixed conditions",
            "Step 4":  "OAT sensitivity for kappa, diversity_lambda, biomass_penalty, bio_threshold",
            "Step 5":  "Paired Wilcoxon + BH FDR correction for all AL-vs-baseline comparisons",
            "Step 6":  "Increased seeds from 5 to 10 for 80% power at medium effect sizes",
            "Step 7":  "All 25 beta-oxidation reactions individually annotated with chain length, enzyme type, priority",
            "Step 8":  "PhPP production envelopes for WT and top-5 designs across 7 conditions",
            "Step 12": "Table 1 with exact dataset stats; surrogate learning curves to assess data sufficiency",
        },
        "manuscript_additions_needed": [
            "Mention carbon-source confusion matrix in Section 2.2 (Materials & Methods)",
            "Report R², MAE from Table 1 in Section 3.4 (Surrogate Model Performance)",
            "Add BH-corrected p-values to Table in Section 3.5 (Statistical Benchmarking)",
            "Add 'Computational Limitations' subsection to Discussion (FBA steady-state caveat)",
            "Update Introduction to mention glycerol/acetate as simulated sustainable feedstocks",
            "Reference power analysis table in Supplementary when reporting non-significant results",
        ],
        "references": {
            "Aminian-Dehkordi 2019":  "doi:10.1038/s41598-019-55041-w",
            "Thiele & Palsson 2010":  "doi:10.1038/nprot.2009.203",
            "Ebrahim 2013 COBRApy":   "doi:10.1186/1752-0509-7-74",
            "Heckmann 2023":          "doi:10.1021/acssynbio.3c00186",
            "Lundberg & Lee 2017":    "doi:10.48550/arXiv.1705.07874",
            "Burgard 2003 OptKnock":  "doi:10.1002/bit.10803",
            "Cal 2025 PHA glycerol":  "doi:10.1371/journal.pone.0322838",
            "Biedendieck 2021":       "doi:10.1007/s00253-021-11424-6",
        },
    }

    report_path = PROJECT_ROOT / "results" / "IMPROVEMENT_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved final report: %s", report_path)
    return report


def main() -> None:
    t_total = time.time()

    print("\n" + "=" * 75)
    print("  BIORESOURCE TECHNOLOGY — QUALITY IMPROVEMENT PIPELINE")
    print("  P. megaterium PHA Production — Computational Study Enhancement")
    print("=" * 75 + "\n")

    # Create required directories
    for d in ["results/tables", "results/figures", "results/logs"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    step_results = []
    for step_info in IMPROVEMENT_STEPS:
        print(f"\n{'─'*70}")
        print(f"  Running: {step_info['step']} — {step_info['title']}")
        print(f"  {step_info['description'][:80]}...")
        print(f"{'─'*70}")

        success, elapsed, summary = run_script(step_info["script"])

        status  = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n  Status: {status} | Time: {elapsed:.1f}s")
        if summary:
            for line in summary.split("\n")[-5:]:
                if line.strip():
                    print(f"  {line}")

        step_results.append({
            "step":        step_info["step"],
            "script":      step_info["script"],
            "title":       step_info["title"],
            "success":     success,
            "elapsed":     round(elapsed, 2),
            "summary":     summary[-300:] if summary else "",
        })

    # Generate final report
    report = generate_final_report(step_results)

    # Print summary
    passed = sum(1 for r in step_results if r["success"])
    failed = len(step_results) - passed
    total_t = time.time() - t_total

    print("\n" + "=" * 75)
    print(f"  PIPELINE COMPLETE — {passed}/{len(step_results)} steps passed | {total_t:.0f}s total")
    print("=" * 75)

    # Output inventory
    tables  = list((PROJECT_ROOT / "results/tables").glob("*.csv"))
    figs    = list((PROJECT_ROOT / "results/figures").glob("*.png"))
    jsons   = list((PROJECT_ROOT / "results/tables").glob("*.json"))
    print(f"\n  📊 Tables generated   : {len(tables)}")
    print(f"  🖼  Figures generated  : {len(figs)}")
    print(f"  📋 JSON reports       : {len(jsons)}")
    print(f"\n  📁 All outputs in: {PROJECT_ROOT / 'results'}")
    print(f"  📋 Full report:     {PROJECT_ROOT / 'results/IMPROVEMENT_REPORT.json'}")
    print("\n  KEY OUTPUTS FOR MANUSCRIPT:")
    print("  ─────────────────────────────────────────────────────────────")
    key_outputs = [
        "results/tables/table1_dataset_statistics.csv        → Table 1",
        "results/tables/statistical_significance_tests.csv   → Table 2 (stats)",
        "results/tables/pareto_front_summary.csv             → Table 3 (designs)",
        "results/figures/validation_confusion_matrix.png     → Figure 1C",
        "results/figures/production_envelope_top5.png        → Figure 3C",
        "results/figures/hyperparameter_tornado_chart.png    → Supplementary S3",
        "results/figures/significance_boxplots.png           → Figure 5A",
        "results/figures/convergence_with_significance.png   → Figure 5B",
        "results/figures/surrogate_learning_curves.png       → Supplementary S4",
        "results/figures/power_analysis_heatmap.png          → Supplementary S5",
    ]
    for o in key_outputs:
        print(f"  • {o}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
