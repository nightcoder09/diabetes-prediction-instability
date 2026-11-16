#!/usr/bin/env python
"""Step 08 - re-train the whole pipeline on independent subsamples; quantify arbitrariness of predictions and explanations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ehai.config import load_config  # noqa: E402
from ehai.experiments import (analyse_multiplicity, analyse_seed_only,  # noqa: E402
                             compare_seed_only_with_subsampling, run_multiplicity,
                             seed_only_mechanism)
from ehai.external import analyse_external_multiplicity  # noqa: E402


def run(cfg, seed_only=False):
    if seed_only:
        run_multiplicity(cfg, seed_only=True)
        print(analyse_seed_only(cfg).round(4).to_string())
        cmp_ = compare_seed_only_with_subsampling(cfg)
        if cmp_ is not None:
            print(cmp_.round(4).to_string())
        mech = seed_only_mechanism(cfg)
        if mech is not None:
            print(pd.DataFrame(mech["deviation_corr"]).round(3).to_string())
        return
    run_multiplicity(cfg)
    res = analyse_multiplicity(cfg)
    print(res["summary"].round(4).to_string())
    ext = analyse_external_multiplicity(cfg)
    if ext is not None:
        print(ext.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed-only", action="store_true", help="re-train on the full development set changing only seeds")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.seed_only)


if __name__ == "__main__":
    main()
