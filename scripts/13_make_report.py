#!/usr/bin/env python
"""Step 13 - regenerate every paper table (Markdown + LaTeX) and figure from saved experiment outputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ehai import plotting as P  # noqa: E402
from ehai.config import load_config, resolve  # noqa: E402
from ehai.data import FEATURES, TARGET, load_processed, load_split  # noqa: E402
from ehai.models import DISPLAY  # noqa: E402
from ehai.pipeline import exp_dir, load_bundle  # noqa: E402
from ehai.utils import load_json  # noqa: E402


def fmt_ci(est, lo, hi, d=3):
    return f"{est:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


def uniq(*names):
    return list(dict.fromkeys(names))


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(v) -> str:
    return "".join(_TEX_ESC.get(ch, ch) for ch in ("" if pd.isna(v) else str(v)))


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Self-contained booktabs table (no Jinja2 dependency, unlike pandas.DataFrame.to_latex)."""
    lines = [r"\begin{table}", r"\centering", rf"\caption{{{_tex(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
             " & ".join(_tex(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(df: pd.DataFrame, out: Path, name: str, caption: str):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    md = f"**{caption}**\n\n" + to_markdown(df) + "\n"
    (out / f"{name}.md").write_text(md)
    (out / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}"))
    return md


def run(cfg):
    res = resolve(cfg, "results_dir")
    figs = resolve(cfg, "figures_dir")
    tabs = res / "tables"
    md_all = []
    b = load_bundle(cfg)
    fr = b["frozen"]
    best = fr["best_single"]
    order = ["stack"] + fr["baselines"]

    # ---------------- Table 1: dataset
    var = pd.read_csv(res / "dataset_variables.csv")
    t1 = var[["feature", "type", "description", "mean_no_diabetes", "mean_diabetes"]].copy()
    t1.columns = ["Feature", "Type", "Definition", "Mean (no diabetes)", "Mean (diabetes)"]
    t1[["Mean (no diabetes)", "Mean (diabetes)"]] = t1[["Mean (no diabetes)", "Mean (diabetes)"]].round(3)
    md_all.append(write_table(t1, tabs, "table1_variables", "Table 1. Predictors (BRFSS 2015) and class-conditional means"))

    # ---------------- Table 2: main test metrics
    m = pd.read_csv(res / "test_main_metrics.csv").set_index("model")
    t2 = pd.DataFrame({
        "Model": [DISPLAY[n] for n in order],
        "AUROC (95% CI)": [fmt_ci(m.loc[n, "auroc"], m.loc[n, "auroc_lo"], m.loc[n, "auroc_hi"]) for n in order],
        "AUPRC (95% CI)": [fmt_ci(m.loc[n, "auprc"], m.loc[n, "auprc_lo"], m.loc[n, "auprc_hi"]) for n in order],
        "Brier (95% CI)": [fmt_ci(m.loc[n, "brier"], m.loc[n, "brier_lo"], m.loc[n, "brier_hi"], 4) for n in order],
        "Cal. slope (95% CI)": [fmt_ci(m.loc[n, "cal_slope"], m.loc[n, "cal_slope_lo"], m.loc[n, "cal_slope_hi"], 2) for n in order],
        "O/E": [f"{m.loc[n, 'oe_ratio']:.3f}" for n in order],
        "Sens.": [f"{m.loc[n, 'sensitivity']:.3f}" for n in order],
        "Spec.": [f"{m.loc[n, 'specificity']:.3f}" for n in order],
        "PPV": [f"{m.loc[n, 'ppv']:.3f}" for n in order],
    })
    md_all.append(write_table(t2, tabs, "table2_test_performance",
                              "Table 2. Held-out test performance (n = {:,}); threshold fixed on development OOF predictions "
                              "for 80% sensitivity".format(int(m.loc["stack", "tp"] + m.loc["stack", "fp"] +
                                                               m.loc["stack", "tn"] + m.loc["stack", "fn"]))))

    c = pd.read_csv(res / "test_comparisons.csv")
    t3 = pd.DataFrame({
        "Comparison": ["Stacked ensemble - " + DISPLAY[s.split(" - ")[1]] + (" (best single, pre-specified)" if bs else "")
                       for s, bs in zip(c.comparison, c.is_best_single)],
        "Delta AUROC (DeLong 95% CI)": [f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})" for d, lo, hi in zip(c.delong_diff, c.delong_lo, c.delong_hi)],
        "p (Holm)": [f"{p:.2g}" for p in c.delong_p_holm],
        "Delta AUPRC (95% CI)": [f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})" for d, lo, hi in zip(c.d_auprc, c.d_auprc_lo, c.d_auprc_hi)],
        "Delta Brier (95% CI)": [f"{d:+.5f} ({lo:+.5f} to {hi:+.5f})" for d, lo, hi in zip(c.d_brier, c.d_brier_lo, c.d_brier_hi)],
    })
    md_all.append(write_table(t3, tabs, "table3_comparisons", "Table 3. Paired comparisons on the test set (2,000 stratified bootstrap resamples; DeLong for AUROC)"))

    cv = pd.read_csv(res / "test_calibration_variants.csv")
    cv["model"] = cv.model.map(DISPLAY)
    cv = cv.round({"auroc": 4, "brier": 5, "log_loss": 4, "cal_intercept": 3, "cal_slope": 3, "oe_ratio": 3, "ece": 4})
    md_all.append(write_table(cv, tabs, "table4_calibration_variants", "Table 4. Calibration before and after OOF-fitted post-hoc calibration (test set)"))

    # ---------------- multiplicity / stability
    ms = pd.read_csv(res / "multiplicity_summary.csv")
    ms_t = ms[["model", "auroc_mean", "auroc_sd", "pred_sd_mean", "pred_sd_p95", "risk_range_gt_0.05", "ambiguity", "pairwise_flip_rate"]].copy()
    ms_t["model"] = ms_t.model.map(lambda n: DISPLAY.get(n, n))
    ms_t.columns = ["Model", "AUROC mean", "AUROC SD", "Mean SD of risk", "95th pct SD of risk",
                    "Share with risk range > 5 pp", "Decision ambiguity", "Pairwise flip rate"]
    md_all.append(write_table(ms_t.round(4), tabs, "table5_multiplicity", f"Table 5. Arbitrariness across {int(ms.n_replicates[0])} independent re-trainings (test set)"))
    es = pd.read_csv(res / "explanation_stability.csv")
    es["model"] = es.model.map(lambda n: DISPLAY.get(n, n))
    md_all.append(write_table(es.round(4), tabs, "table6_explanation_stability", "Table 6. Stability of SHAP explanations across re-trainings"))
    ps = pd.read_csv(res / "shap_perturbation_sensitivity.csv")
    ps["model"] = ps.model.map(lambda n: DISPLAY.get(n, n))
    md_all.append(write_table(ps.round(4), tabs, "table7_explanation_perturbation", "Table 7. Sensitivity of explanations to clinically negligible input changes (BMI +/-1, health-days +/-1)"))

    # ---------------- robustness
    rb = pd.read_csv(res / "robustness_summary.csv")
    rb = rb[rb.model.isin(["stack", best, "logreg", "random_forest"])]
    rb_t = rb.pivot_table(index=["scenario", "level"], columns="model", values="delta_auroc", sort=False).reset_index()
    rb_t.columns = [DISPLAY.get(c, c) for c in rb_t.columns]
    md_all.append(write_table(rb_t.round(4), tabs, "table8_robustness", "Table 8. Change in test AUROC under missingness and noise (median imputation)"))

    # ---------------- fairness
    fd = pd.read_csv(res / "fairness_disparities.csv")
    fd["model"] = fd.model.map(lambda n: DISPLAY.get(n, n))
    md_all.append(write_table(fd.round(4), tabs, "table9_fairness_gaps", "Table 9. Largest subgroup gaps (max - min across groups)"))
    fs = pd.read_csv(res / "fairness_subgroups.csv")
    fs_t = fs[fs.model == "stack"][["attribute", "group", "n", "prevalence", "auroc", "auroc_lo", "auroc_hi", "tpr", "fpr", "ppv", "oe_ratio", "oe_ratio_lo", "oe_ratio_hi"]]
    md_all.append(write_table(fs_t.round(3), tabs, "table10_subgroups_stack", "Table 10. Subgroup performance of the stacked ensemble (bootstrap 95% CIs)"))

    # ---------------- selective & ablations
    sl = pd.read_csv(res / "selective_summary.csv")
    md_all.append(write_table(sl.round(4), tabs, "table11_selective", "Table 11. Selective prediction: error among automatically decided cases"))
    ab = pd.read_csv(res / "ablations.csv")
    md_all.append(write_table(ab.round(4), tabs, "table12_ablations", "Table 12. Ablations and design variants (test set)"))
    tp = load_json(exp_dir(cfg, "tuning") / "best_params.json") if (exp_dir(cfg, "tuning") / "best_params.json").exists() else fr["params"]
    hp = pd.DataFrame([{"Model": DISPLAY[k], "Selected hyper-parameters": ", ".join(f"{a}={(round(v, 4) if isinstance(v, float) else v)}" for a, v in p.items())} for k, p in tp.items()])
    md_all.append(write_table(hp, tabs, "tableS1_hyperparameters", "Table S1. Hyper-parameters selected by development-set cross-validation"))
    # member diversity on development OOF predictions (why stacking can or cannot help)
    oofm = np.load(exp_dir(cfg, "predictions") / "dev_oof_margins.npz")
    dev = pd.read_csv(exp_dir(cfg, "predictions") / "dev_oof_predictions.csv.gz")
    names = [n for n in fr["baselines"] if n in oofm.files]
    resid = pd.DataFrame({DISPLAY[n]: dev.y.to_numpy() - 1 / (1 + np.exp(-oofm[n])) for n in names}).corr()
    marg = pd.DataFrame({DISPLAY[n]: oofm[n] for n in names}).corr()
    resid.round(4).to_csv(res / "member_diversity_residual_corr.csv")
    marg.round(4).to_csv(res / "member_diversity_margin_corr.csv")
    md_all.append(write_table(resid.round(3).reset_index().rename(columns={"index": "Model"}), tabs,
                              "tableS7_member_diversity",
                              "Table S7. Correlation of development OOF residuals between models (member diversity)"))
    (res / "ALL_TABLES.md").write_text("\n\n".join(md_all))

    # ---------------- figures
    preds = pd.read_csv(exp_dir(cfg, "predictions") / "test_predictions.csv.gz")
    y = preds.y.to_numpy()
    P.fig_roc_pr(y, {n: preds[f"final_{n}"].to_numpy() for n in order}, figs)
    variants = {n: {"none": preds[f"none_{n}"].to_numpy(), "platt": preds[f"platt_{n}"].to_numpy()} for n in fr["baselines"]}
    variants["stack"] = {"none": preds["none_stack"].to_numpy()}
    P.fig_reliability(y, variants, figs, uniq("stack", best, "logreg", "random_forest"))
    P.fig_dca(pd.read_csv(res / "test_decision_curve.csv"), figs, uniq("stack", best, "logreg"))
    sh = np.load(exp_dir(cfg, "explain") / "shap_frozen.npz")
    Xe = sh["X"]
    P.fig_beeswarm(sh["phi_stack"], Xe, figs, "fig05_shap_beeswarm_stack", "Stacked ensemble: SHAP values (test sample)")
    imp = pd.read_csv(res / "shap_global_importance.csv", index_col=0)
    P.fig_importance_compare(imp, figs, ["stack", "logreg", "xgboost", "lightgbm", "catboost"])
    P.fig_dependence({n: sh[f"phi_{n}"] for n in ["stack", "logreg", best if best in ("xgboost", "lightgbm", "catboost") else "lightgbm"]},
                     Xe, figs, ["BMI", "Age", "GenHlth", "Income"])
    ps_ = 1 / (1 + np.exp(-(sh["base_stack"] + sh["phi_stack"].sum(1))))
    ex_idx = [int(np.argmin(np.abs(ps_ - q))) for q in (0.05, 0.25, 0.60)]
    P.fig_waterfalls(sh["phi_stack"], float(sh["base_stack"]), Xe, ex_idx, figs)
    pi = np.load(exp_dir(cfg, "multiplicity") / "per_individual.npz")
    mm = [n for n in ["stack", "bag", fr["bag_family"], "logreg"] if f"sd_{n}" in pi]
    mm += [n for n in ["xgboost", "lightgbm", "catboost"] if n not in mm and f"sd_{n}" in pi]
    fig, ax = P.fig_multiplicity({n: pi[f"sd_{n}"] for n in mm}, mm, figs)
    P.fig_multiplicity_finish(fig, ax, ms, mm, figs)
    # local explanation stability per individual (recomputed from replicate SHAP arrays)
    from ehai.explain import local_stability
    files = sorted(exp_dir(cfg, "multiplicity").glob("replicate_[0-9][0-9].npz"))
    reps = [np.load(f) for f in files]
    names = [str(n) for n in reps[0]["names"]]
    local = {n: local_stability([r["shap"][j] for r in reps])["per_instance_spearman"] for j, n in enumerate(names)}
    P.fig_explanation_stability(es, local, [n for n in ["stack", "bag", "logreg", "xgboost", "lightgbm", "catboost"] if n in local], figs)
    P.fig_robustness(pd.read_csv(res / "robustness_summary.csv"), uniq("stack", best, "logreg", "random_forest", "decision_tree"), figs)
    P.fig_fairness(fs, uniq("stack", best, "logreg"), figs)
    P.fig_risk_coverage(joblib.load(exp_dir(cfg, "selective") / "risk_coverage_curves.joblib"), figs)
    # ---------------- temporal external validation (optional: only when step 15 has been run)
    if (res / "external_metrics.csv").exists():
        from ehai.external import load_external
        from ehai.metrics import reliability_curve
        from ehai.pipeline import final_proba

        em = pd.read_csv(res / "external_metrics.csv")
        curves = {2015: reliability_curve(y, preds["final_stack"].to_numpy(), 15)[:2]}
        for yr, (Xx, yx, _) in load_external(cfg).items():
            curves[yr] = reliability_curve(yx, final_proba(b, Xx)["stack"], 15)[:2]
        P.fig_external(em, curves, figs, uniq("stack", best, "logreg", "random_forest"),
                       pd.read_csv(res / "external_recalibration.csv"))
        t = em[em.model.isin(order)].copy()
        t["model"] = t.model.map(DISPLAY)
        t = t[["year", "model", "n", "prevalence", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier", "cal_intercept",
               "cal_slope", "oe_ratio", "ece", "sensitivity", "specificity", "ppv", "flagged_fraction"]].round(4)
        md_all.append(write_table(t, tabs, "table13_temporal_validation",
                                  "Table 13. Frozen 2015 models on the 2015 test set and on BRFSS 2021 and 2023"))
        rc = pd.read_csv(res / "external_recalibration.csv")
        rc["model"] = rc.model.map(DISPLAY)
        md_all.append(write_table(rc.round(4), tabs, "table14_recalibration",
                                  "Table 14. Recalibration fitted on half of each external year, evaluated on the other half"))
        if (res / "external_multiplicity.csv").exists():
            xm = pd.read_csv(res / "external_multiplicity.csv")
            order_x = xm[xm.year == xm.year.min()].sort_values("pred_sd_mean").model.tolist()  # least arbitrary first
            P.fig_external_arbitrariness(xm, [m for m in order_x if m in set(xm.model)][:6], figs)
            md_all.append(write_table(xm.round(4), tabs, "table15_arbitrariness_by_year",
                                      "Table 15. Arbitrariness of the re-trained pipelines by survey year"))
    # ---------------- participant flow (needs the build report of scripts/14_build_external.py)
    br = Path(cfg["_root"]) / cfg.get("external", {}).get("dir", "data/external") / "build_report.json"
    if br.exists():
        _, yd_, _ = load_split(cfg, "dev")
        _, yt_, _ = load_split(cfg, "test")
        P.fig_flow(load_json(br), len(yd_), len(yt_), int(yd_.sum()), int(yt_.sum()), figs)
    # ---------------- seed-only re-trainings (optional: scripts/08_multiplicity.py --seed-only)
    if (res / "multiplicity_seed_only_summary.csv").exists():
        so = pd.read_csv(res / "multiplicity_seed_only_summary.csv")
        so["model"] = so.model.map(lambda n: DISPLAY.get(n, n))
        md_all.append(write_table(so.round(4), tabs, "table16_seed_only",
                                  "Table 16. Arbitrariness when only random seeds change (full development set)"))
    if (res / "multiplicity_seed_vs_sample.csv").exists():
        sv = pd.read_csv(res / "multiplicity_seed_vs_sample.csv")
        sv["model"] = sv.model.map(lambda n: DISPLAY.get(n, n))
        md_all.append(write_table(sv.round(4), tabs, "table17_seed_vs_sample",
                                  "Table 17. Seeds only versus new training samples (same number of re-trainings)"))
    (res / "ALL_TABLES.md").write_text("\n\n".join(md_all))
    print(f"OK: tables in {tabs}, figures in {figs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
