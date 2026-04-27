import subprocess
import pandas as pd
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]

script = "scripts/06_run_active_learning_option_c.py"

for seed in SEEDS:

    print(f"\n=== Running AL with seed {seed} ===")

    subprocess.run([
        "python",
        script,
        "--seed",
        str(seed)
    ])

print("\nAll runs finished.")