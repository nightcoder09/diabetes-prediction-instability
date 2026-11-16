"""End-to-end integration test on a small SYNTHETIC dataset (runs every pipeline step)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import ROOT, synthetic_brfss


@pytest.fixture(scope="module")
def tmp_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline")
    csv = d / "synthetic.csv"
    synthetic_brfss(6000, seed=11).to_csv(csv, index=False)
    cfg = {
        "inherit": str(ROOT / "configs" / "smoke.yaml"),
        "paths": {k: str(d / k) for k in ("raw_dir", "processed_dir", "experiments_dir", "results_dir", "figures_dir")},
        "data": {"source": "local", "local_binary_csv": str(csv), "local_012_csv": None, "subsample_rows": None,
                 "expected_rows": 6000},
        "evaluation": {"n_bootstrap": 20}, "fairness": {"n_bootstrap": 10, "min_group_size": 100},
        "selective": {"n_bootstrap": 10}, "multiplicity": {"n_replicates": 2, "n_explain": 20},
        "explain": {"n_background": 20, "n_explain": 40},
        "tuning": {"n_iter": {k: 1 for k in ("logreg", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost")}},
    }
    path = d / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, d


def test_full_pipeline_runs(tmp_cfg):
    path, d = tmp_cfg
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--config", str(path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = d / "results_dir"
    for f in ["test_main_metrics.csv", "test_comparisons.csv", "multiplicity_summary.csv", "explanation_stability.csv",
              "robustness_summary.csv", "fairness_subgroups.csv", "selective_summary.csv", "ablations.csv",
              "ALL_TABLES.md"]:
        assert (res / f).exists(), f
    main = pd.read_csv(res / "test_main_metrics.csv")
    assert main.auroc.between(0.5, 1).all()
    assert len(list((d / "figures_dir").glob("*.png"))) >= 10
    # inference on new records
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--config", str(path),
                          "--input", str(d / "synthetic.csv")], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Research use only" in out.stdout
