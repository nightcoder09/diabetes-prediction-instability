#!/usr/bin/env python
"""Step 05 - train all models on the development set: OOF predictions, stacking, calibration, thresholds; freeze."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.pipeline import train_development  # noqa: E402


def run(cfg):
    fr = train_development(cfg)
    print(f"OK: best single = {fr['best_single']}; stack weights = {fr['stack_weights']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
