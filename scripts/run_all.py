#!/usr/bin/env python
"""Run the complete pipeline (steps 01-13) in order. Each step is resumable and logs to experiments/logs."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["01_get_data.py", "02_eda.py", "03_make_splits.py", "04_tune.py", "05_train.py", "06_evaluate_test.py",
         "07_explain.py", "08_multiplicity.py", "09_robustness.py", "10_fairness.py", "11_selective.py",
         "12_ablations.py", "13_make_report.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--from-step", type=int, default=1, help="start at this step number")
    ap.add_argument("--to-step", type=int, default=13)
    args = ap.parse_args()
    for s in STEPS:
        k = int(s[:2])
        if k < args.from_step or k > args.to_step:
            continue
        t0 = time.time()
        print(f"\n===== {s} =====", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s), "--config", args.config], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Step {s} failed (exit code {r.returncode}). Fix the error and rerun with --from-step {k}.")
        print(f"----- {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
