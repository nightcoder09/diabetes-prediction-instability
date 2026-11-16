#!/usr/bin/env python
"""Step 09 - test-time stress tests: random and structured missingness, measurement noise."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.experiments import run_robustness  # noqa: E402


def run(cfg):
    agg = run_robustness(cfg)
    print(agg[agg.scenario == "mcar"][["level", "model", "auroc", "delta_auroc"]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
