#!/usr/bin/env python
"""Step 06 - single evaluation of the frozen models on the held-out test set (metrics, CIs, DeLong, calibration, DCA)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.pipeline import evaluate_test  # noqa: E402


def run(cfg):
    res = evaluate_test(cfg)
    print(res["main"][["auroc", "auprc", "brier", "cal_slope", "sensitivity", "specificity"]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
