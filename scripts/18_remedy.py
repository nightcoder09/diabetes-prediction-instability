#!/usr/bin/env python
"""Step 18 (post hoc) - a remedy for arbitrary individual decisions.

  select     choose, among tuning-equivalent booster settings, the one whose decisions vary least across
             re-training (development data only)                                   ~30 min on 2 cores
  frozen     train the stable stack (chosen settings, 5 seeds per booster) on the full development
             set and read the test set once                                         ~30 min
  replicates re-train it on the same subsamples/folds as main runs 0-9 (paired)     ~20 min per run
  analyse    summaries and paired per-person comparisons -> results/remedy_*.csv/json

Examples
--------
python scripts/18_remedy.py --config configs/kaggle_local.yaml --step all
python scripts/18_remedy.py --config configs/kaggle_local.yaml --step replicates --variant stable --max 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ehai.config import load_config  # noqa: E402
from ehai.remedy import (VARIANTS, _selected, analyse_remedy, evaluate_stable_test,  # noqa: E402
                         run_remedy_replicates, select_stable_settings, train_stable)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--step", choices=["select", "frozen", "replicates", "analyse", "all"], default="all")
    ap.add_argument("--variant", choices=list(VARIANTS) + ["both"], default="both",
                    help="re-training variant(s); 'stable' is skipped automatically if selection kept the tuned settings")
    ap.add_argument("--max", type=int, default=None, help="at most this many new re-trainings in this call")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pd.set_option("display.width", 200)

    if args.step in ("select", "all"):
        df = select_stable_settings(cfg)
        print(df.drop(columns="params").round(4).to_string(index=False))
    if args.step in ("frozen", "all"):
        train_stable(cfg)
        print(evaluate_stable_test(cfg).set_index("model")[["auroc", "brier", "cal_slope", "sensitivity",
                                                           "specificity"]].round(4).to_string())
    if args.step in ("replicates", "all"):
        chosen = _selected(cfg)["chosen"]
        all_tuned = all(c["is_tuned_setting"] for c in chosen.values())
        variants = ["stable_seedavg", "stable"] if args.variant == "both" else [args.variant]
        for v in variants:
            if v == "stable" and all_tuned:
                print("Selection kept every tuned setting: the 'stable' (single-seed) variant equals the main audit.")
                continue
            run_remedy_replicates(cfg, v, max_replicates=args.max)
    if args.step in ("analyse", "all"):
        res = analyse_remedy(cfg)
        if res is None:
            print("No complete remedy re-training set yet.")
        else:
            print(res["summary"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
