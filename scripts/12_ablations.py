#!/usr/bin/env python
"""Step 12 - ablations (weighting, member removal, calibration, homogeneous bag) and sensitivity analyses."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.experiments import run_ablations, run_error_analysis  # noqa: E402


def run(cfg):
    abl = run_ablations(cfg)
    print(abl.round(4).to_string())
    prof = run_error_analysis(cfg)
    print(prof.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
