#!/usr/bin/env python
"""Step 07 - exact SHAP explanations of the frozen models: global, local, cross-model agreement, perturbation sensitivity."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.experiments import run_explanations  # noqa: E402


def run(cfg):
    run_explanations(cfg)
    print("OK: explanations written")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
