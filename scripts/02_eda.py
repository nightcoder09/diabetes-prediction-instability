#!/usr/bin/env python
"""Step 02 - dataset summary (distributions, class balance, duplicates) and EDA figure."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehai.config import load_config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ehai.config import resolve  # noqa: E402
from ehai.data import BINARY, CONTINUOUS, DESCRIPTIONS, FEATURES, ORDINAL, TARGET, load_processed  # noqa: E402
from ehai.fairness import group_labels  # noqa: E402
from ehai.plotting import fig_prevalence  # noqa: E402
from ehai.utils import load_json, save_json  # noqa: E402


def run(cfg):
    df = load_processed(cfg)
    res = resolve(cfg, "results_dir")
    meta = load_json(resolve(cfg, "processed_dir") / "metadata.json")
    rows = []
    for f in FEATURES:
        x = df[f]
        rows.append({"feature": f, "type": "binary" if f in BINARY else "ordinal" if f in ORDINAL else "count/continuous",
                     "description": DESCRIPTIONS[f], "min": x.min(), "max": x.max(), "mean": x.mean(), "sd": x.std(),
                     "n_unique": x.nunique(), "mean_no_diabetes": x[df[TARGET] == 0].mean(),
                     "mean_diabetes": x[df[TARGET] == 1].mean(), "missing": int(x.isna().sum())})
    pd.DataFrame(rows).to_csv(res / "dataset_variables.csv", index=False)
    g = group_labels(df)
    sub = []
    for a in g.columns:
        for grp, s in df.groupby(g[a]):
            sub.append({"attribute": a, "group": grp, "n": len(s), "share": len(s) / len(df),
                        "prevalence": s[TARGET].mean()})
    pd.DataFrame(sub).to_csv(res / "dataset_subgroups.csv", index=False)
    summary = {"n_rows": len(df), "n_features": len(FEATURES), "n_positive": int(df[TARGET].sum()),
               "prevalence": float(df[TARGET].mean()), "missing_cells": int(df[FEATURES + [TARGET]].isna().sum().sum()),
               "duplicates": meta["duplicates"], "content_sha256": meta["content_sha256"],
               "label_mapping": meta.get("label_mapping"), "validation": meta["validation"]}
    if "prediabetes" in df.columns:
        summary["n_prediabetes"] = int(df["prediabetes"].sum())
    save_json(summary, res / "dataset_summary.json")
    fig_prevalence(df, TARGET, resolve(cfg, "figures_dir"))
    print(f"OK: summary written to {res}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
