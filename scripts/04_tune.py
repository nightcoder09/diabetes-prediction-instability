#!/usr/bin/env python
"""Step 04 - randomised hyper-parameter search on the development set only (AUROC, stratified K-fold)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.config import derive_seed, n_jobs  # noqa: E402
from ehai.data import load_split  # noqa: E402
from ehai.pipeline import exp_dir  # noqa: E402
from ehai.tuning import random_search  # noqa: E402
from ehai.utils import get_logger, load_json, save_json, timer  # noqa: E402


def run(cfg, only=None):
    logger = get_logger("tune", exp_dir(cfg, "logs"))
    X, y, _ = load_split(cfg, "dev")
    out = exp_dir(cfg, "tuning")
    best_path = out / "best_params.json"
    best = load_json(best_path) if best_path.exists() else {}
    for name in cfg["models"]["baselines"]:
        if only and name not in only:
            continue
        if name in best and not only:
            logger.info(f"{name}: already tuned, skipping (delete {best_path} to re-tune)")
            continue
        with timer(logger, f"tuning {name}"):
            params, res = random_search(name, X, y, cfg["tuning"]["n_iter"][name], cfg["tuning"]["n_splits"],
                                        derive_seed(cfg["project"]["seed"], "tune", name), n_jobs(cfg), logger)
        res.to_csv(out / f"{name}_search.csv", index=False)
        best[name] = params
        save_json(best, best_path)
        logger.info(f"{name}: best {params} (CV AUROC {res.iloc[0]['auroc_mean']:.4f})")
    print(f"OK: tuned parameters in {best_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="*", help="re-tune only these models")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.only)


if __name__ == "__main__":
    main()
