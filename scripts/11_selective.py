#!/usr/bin/env python
"""Step 11 - selective prediction: risk-coverage analysis of deferral signals."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.experiments import run_selective  # noqa: E402


def run(cfg):
    df = run_selective(cfg)
    print(df.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
