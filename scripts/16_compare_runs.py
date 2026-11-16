#!/usr/bin/env python
"""Step 16 - compare two complete runs of the study (e.g. the paper run and an independent re-run on a
second machine) and report how closely every headline number and every hypothesis decision agrees.

    python scripts/16_compare_runs.py --a configs/default.yaml --b configs/repro_mac.yaml
    python scripts/16_compare_runs.py --a-dir results --b-dir results_repro_mac      # result folders directly

Writes comparison_with_paper_run.json and comparison_with_paper_run.md into run B's results folder
(or --out).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ehai.config import load_config, resolve  # noqa: E402
from ehai.utils import load_json, save_json  # noqa: E402

MODELS = ["stack", "catboost", "xgboost", "lightgbm", "random_forest", "logreg", "decision_tree", "bag"]


def _csv(d: Path, f: str, **kw):
    p = d / f
    return pd.read_csv(p, **kw) if p.exists() else None


def _hyp(res: Path) -> dict:
    """Re-derive the pre-specified hypothesis decisions from one run's result files."""
    out = {}
    fr = load_json(res / "frozen_summary.json")
    best = fr["best_single"]
    cmp_ = pd.read_csv(res / "test_comparisons.csv")
    row = cmp_[cmp_.is_best_single].iloc[0]
    out["H1_ci_excludes_0"] = bool(row.delong_lo > 0)
    out["H1_margin_0.005_reached"] = bool(row.delong_diff >= 0.005)
    ms = pd.read_csv(res / "multiplicity_summary.csv").set_index("model")
    out["H2_stack_less_arbitrary_than_best"] = bool(ms.loc["stack", "pred_sd_mean"] < ms.loc[best, "pred_sd_mean"]
                                                    and ms.loc["stack", "ambiguity"] < ms.loc[best, "ambiguity"])
    out["H2_stack_less_arbitrary_than_bag"] = bool(ms.loc["stack", "pred_sd_mean"] < ms.loc["bag", "pred_sd_mean"]
                                                   and ms.loc["stack", "ambiguity"] < ms.loc["bag", "ambiguity"])
    out["least_arbitrary_model"] = str(ms.pred_sd_mean.idxmin())
    out["most_arbitrary_model"] = str(ms.pred_sd_mean.idxmax())
    sl = pd.read_csv(res / "selective_summary.csv").set_index("signal")
    col = "aurc_cost_50_100"
    out["H4_best_deferral_signal"] = str(sl[col].drop("random", errors="ignore").idxmin())
    out["H4_disagreement_helps"] = bool(sl.loc["margin+disagreement", col] < sl.loc["margin", col])
    rb = pd.read_csv(res / "robustness_summary.csv")
    r20 = rb[(rb.scenario == "mcar") & (rb.level.round(2) == 0.2)]
    out["H6_max_auroc_loss_at_20pct"] = float(-r20.delta_auroc.min())
    fd = pd.read_csv(res / "fairness_disparities.csv")
    age = fd[fd.attribute == "Age band"].set_index("model").auroc_gap
    out["H7_age_auroc_gap_range_across_models"] = float(age.max() - age.min())
    return out


def compare(A: Path, B: Path) -> dict:
    out = {"run_a": str(A), "run_b": str(B)}
    da, db = load_json(A / "dataset_summary.json"), load_json(B / "dataset_summary.json")
    out["data_identical"] = da["content_sha256"] == db["content_sha256"]
    out["content_sha256"] = [da["content_sha256"], db["content_sha256"]]
    out["environments"] = [load_json(A / "environment.json"), load_json(B / "environment.json")]
    fa, fb = load_json(A / "frozen_summary.json"), load_json(B / "frozen_summary.json")
    out["best_single"] = [fa["best_single"], fb["best_single"]]
    out["stack_weights"] = [fa["stack_weights"], fb["stack_weights"]]
    out["thresholds"] = [fa["thresholds"], fb["thresholds"]]

    ma = pd.read_csv(A / "test_main_metrics.csv", index_col=0)
    mb = pd.read_csv(B / "test_main_metrics.csv", index_col=0)
    cols = [c for c in ["auroc", "auprc", "brier", "cal_slope", "oe_ratio", "ece", "sensitivity", "specificity"]
            if c in ma.columns]
    common = [m for m in ma.index if m in mb.index]
    diff = (mb.loc[common, cols] - ma.loc[common, cols])
    out["test_metrics_a"] = ma.loc[common, cols].round(5).to_dict(orient="index")
    out["test_metrics_b"] = mb.loc[common, cols].round(5).to_dict(orient="index")
    out["test_max_abs_diff_by_metric"] = diff.abs().max().round(5).to_dict()
    out["test_max_abs_diff_by_model"] = diff.abs().max(axis=1).round(5).to_dict()

    ca, cb = pd.read_csv(A / "test_comparisons.csv"), pd.read_csv(B / "test_comparisons.csv")
    out["stack_minus_best_single_auroc"] = [float(ca[ca.is_best_single].delong_diff.iloc[0]),
                                            float(cb[cb.is_best_single].delong_diff.iloc[0])]

    blocks = {"multiplicity_summary.csv": ["auroc_mean", "pred_sd_mean", "pred_sd_p95", "risk_range_gt_0.05",
                                           "ambiguity", "pairwise_flip_rate"],
              "explanation_stability.csv": ["spearman_mean", "top3_jaccard_mean", "top1_agreement",
                                            "kendall_tau_mean"],
              "shap_perturbation_sensitivity.csv": None}
    for f, c in blocks.items():
        xa, xb = _csv(A, f), _csv(B, f)
        if xa is None or xb is None or "model" not in xa.columns:
            continue
        xa, xb = xa.set_index("model"), xb.set_index("model")
        c = c or [k for k in xa.select_dtypes("number").columns][:4]
        c = [k for k in c if k in xa.columns and k in xb.columns]
        idx = [m for m in xa.index if m in xb.index]
        out[f] = {"a": xa.loc[idx, c].round(5).to_dict(orient="index"),
                  "b": xb.loc[idx, c].round(5).to_dict(orient="index")}

    sa, sb = _csv(A, "selective_summary.csv"), _csv(B, "selective_summary.csv")
    if sa is not None and sb is not None:
        out["deferral_aurc_cost"] = {"a": sa.set_index("signal").aurc_cost_50_100.round(3).to_dict(),
                                     "b": sb.set_index("signal").aurc_cost_50_100.round(3).to_dict()}
    fa_, fb_ = _csv(A, "fairness_disparities.csv"), _csv(B, "fairness_disparities.csv")
    if fa_ is not None and fb_ is not None:
        k = ["model", "attribute"]
        j = fa_.merge(fb_, on=k, suffixes=("_a", "_b"))
        out["fairness_max_abs_diff"] = {c: float((j[f"{c}_a"] - j[f"{c}_b"]).abs().max())
                                        for c in ["auroc_gap", "tpr_gap", "fpr_gap"]}
    ra, rb_ = _csv(A, "robustness_summary.csv"), _csv(B, "robustness_summary.csv")
    if ra is not None and rb_ is not None:
        j = ra.merge(rb_, on=["scenario", "level", "model"], suffixes=("_a", "_b"))
        out["robustness_max_abs_diff_delta_auroc"] = float((j.delta_auroc_a - j.delta_auroc_b).abs().max())
    out["hypotheses"] = {"a": _hyp(A), "b": _hyp(B)}
    out["hypothesis_decisions_identical"] = {k: out["hypotheses"]["a"][k] == out["hypotheses"]["b"][k]
                                             for k in out["hypotheses"]["a"] if not isinstance(
                                                 out["hypotheses"]["a"][k], float)}
    return out


def to_markdown(res: dict) -> str:
    L = ["# Comparison of two complete runs", "",
         f"- Run A: `{res['run_a']}`", f"- Run B: `{res['run_b']}`",
         f"- Identical input data (SHA-256): **{res['data_identical']}**",
         f"- Environments: A = {res['environments'][0].get('platform')}, Python {res['environments'][0].get('python')}; "
         f"B = {res['environments'][1].get('platform')}, Python {res['environments'][1].get('python')}",
         f"- Strongest single model: A = {res['best_single'][0]}, B = {res['best_single'][1]}",
         f"- Stack minus strongest single model (test AUROC): A = {res['stack_minus_best_single_auroc'][0]:.5f}, "
         f"B = {res['stack_minus_best_single_auroc'][1]:.5f}", "",
         "## Test-set metrics (A / B)", "", "| Model | AUROC | Brier | Cal. slope | Sensitivity |", "|---|---|---|---|---|"]
    for m, a in res["test_metrics_a"].items():
        b = res["test_metrics_b"][m]
        L.append(f"| {m} | {a['auroc']:.4f} / {b['auroc']:.4f} | {a['brier']:.4f} / {b['brier']:.4f} | "
                 f"{a['cal_slope']:.3f} / {b['cal_slope']:.3f} | {a['sensitivity']:.3f} / {b['sensitivity']:.3f} |")
    if "multiplicity_summary.csv" in res:
        L += ["", "## Arbitrariness across re-trainings (A / B)", "",
              "| Model | Mean SD of risk (pp) | Decision ambiguity |", "|---|---|---|"]
        for m, a in res["multiplicity_summary.csv"]["a"].items():
            b = res["multiplicity_summary.csv"]["b"][m]
            L.append(f"| {m} | {100 * a['pred_sd_mean']:.2f} / {100 * b['pred_sd_mean']:.2f} | "
                     f"{100 * a['ambiguity']:.1f}% / {100 * b['ambiguity']:.1f}% |")
    L += ["", "## Hypothesis decisions", "", "| Item | Run A | Run B |", "|---|---|---|"]
    for k, a in res["hypotheses"]["a"].items():
        b = res["hypotheses"]["b"][k]
        fa = f"{a:.4f}" if isinstance(a, float) else str(a)
        fb = f"{b:.4f}" if isinstance(b, float) else str(b)
        L.append(f"| {k} | {fa} | {fb} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="configs/default.yaml", help="config of run A (the paper run)")
    ap.add_argument("--b", default="configs/repro_mac.yaml", help="config of run B (the re-run)")
    ap.add_argument("--a-dir", default=None, help="results folder of run A (overrides --a)")
    ap.add_argument("--b-dir", default=None, help="results folder of run B (overrides --b)")
    ap.add_argument("--out", default=None, help="output folder (default: run B's results folder)")
    args = ap.parse_args()
    A = Path(args.a_dir) if args.a_dir else resolve(load_config(args.a), "results_dir")
    B = Path(args.b_dir) if args.b_dir else resolve(load_config(args.b), "results_dir")
    res = compare(A, B)
    out = Path(args.out) if args.out else B
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / "comparison_with_paper_run.json")
    (out / "comparison_with_paper_run.md").write_text(to_markdown(res))
    print(f"data identical: {res['data_identical']}; best single: {res['best_single']}")
    print("max |difference| in test metrics:", res["test_max_abs_diff_by_metric"])
    print("hypothesis decisions identical:", res["hypothesis_decisions_identical"])
    print(f"Saved {out / 'comparison_with_paper_run.json'} and .md")


if __name__ == "__main__":
    main()
