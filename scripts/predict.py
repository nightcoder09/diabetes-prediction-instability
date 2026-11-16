#!/usr/bin/env python
"""Score new records with the frozen stacked ensemble (research use only).

Examples
--------
python scripts/predict.py --input data/example_input.csv --output results/example_predictions.csv
python scripts/predict.py --input data/example_input.csv --stable     # post hoc stable stack
python scripts/predict.py --json '{"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":31,"Smoker":0,"Stroke":0,
  "HeartDiseaseorAttack":0,"PhysActivity":1,"Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"AnyHealthcare":1,
  "NoDocbcCost":0,"GenHlth":3,"MentHlth":0,"PhysHlth":2,"DiffWalk":0,"Sex":1,"Age":9,"Education":5,"Income":6}'
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ehai.config import load_config  # noqa: E402
from ehai.predict import DISCLAIMER, InputError, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="CSV with the 21 predictor columns")
    g.add_argument("--json", help="one record as a JSON object")
    ap.add_argument("--output", help="where to write predictions (CSV)")
    ap.add_argument("--stable", action="store_true",
                    help="use the post hoc stable stack (scripts/18_remedy.py) instead of the pre-specified one")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stable:
        cfg.setdefault("inference", {})["models_dir"] = "models_stable"
    df = pd.read_csv(args.input) if args.input else pd.DataFrame([json.loads(args.json)])
    try:
        out = predict(cfg, df)
    except InputError as exc:
        sys.exit(f"Input error: {exc}")
    print(DISCLAIMER)
    print(out.to_string())
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
