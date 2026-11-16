#!/usr/bin/env python
"""Step 01 - download (or read) the dataset, validate its schema, canonicalise and save it."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
from ehai.data import prepare  # noqa: E402
from ehai.utils import environment_info, save_json  # noqa: E402
from ehai.config import resolve  # noqa: E402


def run(cfg):
    meta = prepare(cfg)
    save_json(environment_info(), resolve(cfg, "results_dir") / "environment.json")
    print(f"OK: {meta['n_rows']:,} rows | prevalence {meta['prevalence']:.4f} | "
          f"duplicates {meta['duplicates']} | sha256 {meta['content_sha256'][:16]}...")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
