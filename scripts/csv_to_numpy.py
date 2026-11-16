#!/usr/bin/env python
"""Convert the study's CSV data into NumPy arrays (.npy / .npz).

What it writes (default folder data/numpy/):
  brfss2015_all.npz       X (253,680 x 21), y, feature_names        - the whole 2015 dataset
  brfss2015_dev.npz       X (202,944 x 21), y, folds, feature_names - the paper's development set
  brfss2015_test.npz      X ( 50,736 x 21), y, feature_names        - the paper's held-out test set
  brfss2021.npz, brfss2023.npz                                      - later survey years (if built)
  X_dev.npy, y_dev.npy, X_test.npy, y_test.npy, feature_names.txt   - the same split as plain .npy files

X is float32 (one row per person, one column per question, in the order of feature_names);
y is int8 (1 = diagnosed diabetes, 0 = no diabetes). In 2023 the fruit and vegetable questions
were not asked, so those two columns are NaN there.

Examples
--------
python scripts/csv_to_numpy.py                                   # paper run (Linux / cloud copy)
python scripts/csv_to_numpy.py --config configs/repro_mac.yaml   # on the Mac (uses the Mac's own run)

Load them back:
    import numpy as np
    d = np.load("data/numpy/brfss2015_dev.npz")
    X, y, names = d["X"], d["y"], d["feature_names"]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ehai.config import load_config, resolve  # noqa: E402
from ehai.data import FEATURES, TARGET  # noqa: E402


def to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy().astype(np.int8)
    return X, y


def check(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, name: str) -> None:
    """The arrays must hold exactly the CSV values (NaN where a question was not asked)."""
    ref = df[FEATURES].to_numpy(dtype=float)
    same = np.array_equal(np.nan_to_num(X.astype(float), nan=-1), np.nan_to_num(ref, nan=-1))
    if not same or not np.array_equal(y, df[TARGET].to_numpy()):
        raise SystemExit(f"{name}: arrays differ from the CSV - nothing trustworthy was written")
    print(f"  {name:<16} X {str(X.shape):<12} y {str(y.shape):<10} diabetes {y.mean():6.2%}   "
          f"missing values {int(np.isnan(X).sum()):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/numpy", help="output folder")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = np.array(FEATURES)

    csv = resolve(cfg, "processed_dir") / "brfss2015_diabetes.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found - run  python scripts/01_get_data.py --config {args.config}  first")
    df = pd.read_csv(csv)
    print(f"Read {csv.relative_to(ROOT)} ({len(df):,} rows, {df.shape[1]} columns)")
    X, y = to_arrays(df)
    check(X, y, df, "2015 (all)")
    np.savez_compressed(out / "brfss2015_all.npz", X=X, y=y, feature_names=names)

    split = resolve(cfg, "experiments_dir") / "splits" / "splits.npz"
    if split.exists():
        sp = np.load(split)
        for part in ("dev", "test"):
            idx = sp[f"{part}_idx"]
            Xp, yp = X[idx], y[idx]
            check(Xp, yp, df.iloc[idx], f"2015 {part}")
            extra = {"folds": sp["folds"]} if part == "dev" else {}
            np.savez_compressed(out / f"brfss2015_{part}.npz", X=Xp, y=yp, feature_names=names, **extra)
            np.save(out / f"X_{part}.npy", Xp)
            np.save(out / f"y_{part}.npy", yp)
    else:
        print(f"  (no split file at {split.relative_to(ROOT)} - run scripts/03_make_splits.py for dev/test arrays)")

    ext_dir = ROOT / cfg.get("external", {}).get("dir", "data/external")
    for year in cfg.get("external", {}).get("years", []):
        f = ext_dir / f"brfss{year}_diabetes.csv"
        if f.exists():
            d = pd.read_csv(f)
            Xe, ye = to_arrays(d)
            check(Xe, ye, d, f"{year}")
            np.savez_compressed(out / f"brfss{year}.npz", X=Xe, y=ye, feature_names=names)

    (out / "feature_names.txt").write_text("\n".join(FEATURES) + "\n")
    print(f"Wrote NumPy files to {out.relative_to(ROOT)}/:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:<22} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
