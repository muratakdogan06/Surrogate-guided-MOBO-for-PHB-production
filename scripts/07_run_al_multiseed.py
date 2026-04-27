#!/usr/bin/env python3
"""
07_run_al_multiseed.py — Run active learning across all seeds for statistical benchmarking.

Reads seed list from configs/active_learning.yaml instead of hardcoding.
"""

import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AL_CONFIG = PROJECT_ROOT / "configs" / "active_learning.yaml"
SCRIPT = PROJECT_ROOT / "scripts" / "06_run_active_learning_option_c.py"

with open(AL_CONFIG) as f:
    al_cfg = yaml.safe_load(f)

SEEDS = al_cfg.get("seeds", list(range(al_cfg.get("n_seeds", 10))))

print(f"Seeds ({len(SEEDS)}): {SEEDS}")
print(f"Surrogate: {al_cfg.get('surrogate_model', 'xgboost')}")
print(f"Acquisition: {al_cfg.get('acquisition', 'mc_ehvi')}")

for seed in SEEDS:
    print(f"\n{'='*65}")
    print(f"  Running AL with seed {seed}")
    print(f"{'='*65}")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", str(seed)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  [WARN] seed {seed} exited with code {result.returncode}")

print("\nAll runs finished.")
