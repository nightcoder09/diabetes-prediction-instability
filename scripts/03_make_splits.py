#!/usr/bin/env python
"""Step 03 - stratified held-out test split and K-fold assignment of the development set."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.data import make_splits  # noqa: E402


def run(cfg):
    info = make_splits(cfg)
    print(f"OK: dev={info['n_dev']:,} (prev {info['prev_dev']:.4f}), test={info['n_test']:,} "
          f"(prev {info['prev_test']:.4f}), folds {info['fold_sizes']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
