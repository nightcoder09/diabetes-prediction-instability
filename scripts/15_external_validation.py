#!/usr/bin/env python
"""Step 15 - temporal external validation of the frozen 2015 models on BRFSS 2021 and 2023.

Requires data/external/brfss{year}_diabetes.csv from scripts/14_build_external.py. The models,
calibrators and thresholds are applied unchanged; see src/ehai/external.py for details.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ehai.config import load_config, resolve  # noqa: E402
from ehai.external import (analyse_external_multiplicity, run_external,  # noqa: E402
                           unavailable_item_sensitivity)


def item_sensitivity(cfg):
    s = unavailable_item_sensitivity(cfg)
    if len(s):
        print("Items not asked in some year, filled with development medians:")
        print(s.round(4).to_string())


def run(cfg):
    summary = run_external(cfg)
    res = resolve(cfg, "results_dir")
    m = pd.read_csv(res / "external_metrics.csv")
    print(m[m.model.isin(["stack", "catboost", "logreg"])][["year", "model", "n", "prevalence", "auroc", "auroc_lo",
                                                              "auroc_hi", "brier", "cal_slope", "oe_ratio",
                                                              "sensitivity", "specificity"]].round(4).to_string())
    print(summary)
    item_sensitivity(cfg)
    ext = analyse_external_multiplicity(cfg)
    if ext is not None:
        print(ext.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sensitivity-only", action="store_true",
                    help="only re-run the check of items a survey year did not ask")
    args = ap.parse_args()
    cfg = load_config(args.config)
    item_sensitivity(cfg) if args.sensitivity_only else run(cfg)


if __name__ == "__main__":
    main()
