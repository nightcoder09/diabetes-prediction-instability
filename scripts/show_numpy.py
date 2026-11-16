#!/usr/bin/env python
"""Show what is inside the NumPy files made by scripts/csv_to_numpy.py.

    python scripts/show_numpy.py            # sizes, counts and the first 5 people
    python scripts/show_numpy.py --export   # also writes ALL people to one spreadsheet (CSV) file
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1] / "data" / "numpy"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

if not FOLDER.exists():
    raise SystemExit("No data/numpy folder yet - run:  python scripts/csv_to_numpy.py --config configs/repro_mac.yaml")

names = (FOLDER / "feature_names.txt").read_text().split()
for part, label in (("dev", "TRAINING (development) data"), ("test", "TEST data")):
    X = np.load(FOLDER / f"X_{part}.npy")
    y = np.load(FOLDER / f"y_{part}.npy")
    print("=" * 100)
    print(f"{label}:  X_{part}.npy has shape {X.shape}  ->  {X.shape[0]:,} people x {X.shape[1]} questions")
    print(f"           y_{part}.npy has shape {y.shape}  ->  {int(y.sum()):,} with diabetes (1), "
          f"{int((y == 0).sum()):,} without (0)")
    df = pd.DataFrame(X, columns=names).astype("Int64")
    df["Diabetes"] = y
    print("\nFirst 5 people (one row = one person):")
    print(df.head().to_string())
    print()

for year in (2021, 2023):
    f = FOLDER / f"brfss{year}.npz"
    if f.exists():
        d = np.load(f)
        print(f"{f.name}: X {d['X'].shape}, y {d['y'].shape}, diabetes {d['y'].mean():.1%}")

if "--export" in sys.argv:
    parts = []
    for part, label in (("dev", "training"), ("test", "test")):
        X = np.load(FOLDER / f"X_{part}.npy")
        y = np.load(FOLDER / f"y_{part}.npy")
        df = pd.DataFrame(X, columns=names).astype("Int64")
        df.insert(0, "part", label)
        df["Diabetes"] = y
        parts.append(df)
    out = FOLDER / "numpy_data_all_2015_people.csv"
    full = pd.concat(parts, ignore_index=True)
    full.insert(0, "person", range(1, len(full) + 1))
    full.to_csv(out, index=False)
    print(f"\nWrote all {len(full):,} people to {out.relative_to(FOLDER.parents[1])} "
          f"({out.stat().st_size / 1e6:.1f} MB) - double-click it to open in Numbers or Excel.")
