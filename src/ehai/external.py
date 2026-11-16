"""Temporal external validation of the frozen 2015 models on later BRFSS survey years.

Nothing here refits the models: the frozen ensemble, calibrators and thresholds are applied as-is.
The only fitted quantities are (i) the optional recalibration, fitted on a random half of each
external year and evaluated on the other half, and (ii) a 'membership' classifier that measures
how separable the external sample is from the development sample (a covariate-shift index).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import derive_seed, resolve
from .data import FEATURES, TARGET, load_split
from .explain import global_importance
from .fairness import group_labels, subgroup_table
from .metrics import (_logistic_fit, all_metrics, auroc, net_benefit, net_benefit_treat_all, threshold_metrics)
from .pipeline import exp_dir, final_proba, load_bundle
from .stats import bootstrap, delong_ci, delong_test, holm
from .utils import get_logger, logit, save_json, sigmoid, timer


def external_paths(cfg) -> dict[int, Path]:
    ecfg = cfg.get("external", {})
    d = Path(cfg["_root"]) / ecfg.get("dir", "data/external")
    out = {}
    for y in ecfg.get("years", [2021, 2023]):
        p = d / f"brfss{y}_diabetes.csv"
        if p.exists():
            out[int(y)] = p
    return out


def load_external(cfg) -> dict[int, tuple[pd.DataFrame, np.ndarray, pd.DataFrame]]:
    """External years; items a survey year did not ask (all-missing columns) are filled with the
    development-set medians - the same rule used in the robustness experiments."""
    from .robustness import MedianImputer

    out, imputer = {}, None
    for y, p in external_paths(cfg).items():
        df = pd.read_csv(p)
        X = df[FEATURES].astype(float)
        if X.isna().any().any():
            if imputer is None:
                Xd, _, _ = load_split(cfg, "dev")
                imputer = MedianImputer().fit(Xd)
            X = imputer.transform(X)
        out[y] = (X, df[TARGET].to_numpy().astype(int), df.drop(columns=FEATURES + [TARGET]))
    return out


def recalibrate(p_fit, y_fit, p_eval):
    """Intercept-only update and logistic recalibration (a + b logit p) fitted on a separate half."""
    a0, _ = _logistic_fit(np.zeros(len(p_fit)), y_fit, offset=logit(p_fit), fit_slope=False)
    a, b = _logistic_fit(logit(p_fit), y_fit)
    return {"intercept_update": sigmoid(logit(p_eval) + a0), "logistic": sigmoid(a + b * logit(p_eval)),
            "params": {"delta_intercept": a0, "a": a, "b": b}}


def smd(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def unavailable_items(cfg) -> set[str]:
    """Items absent from at least one external survey year (filled with development medians)."""
    out = set()
    for _, p in external_paths(cfg).items():
        raw = pd.read_csv(p, nrows=5000)
        out |= {f for f in FEATURES if raw[f].isna().all()}
    return out


def membership_auc(X0: pd.DataFrame, X1: pd.DataFrame, seed: int, n: int = 40_000) -> float:
    """Cross-validated AUROC of a classifier separating two samples (0.5 = indistinguishable).
    Pass only items observed in both samples, otherwise an imputed constant makes them trivially separable."""
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(seed)
    a = X0.iloc[rng.choice(len(X0), min(n, len(X0)), replace=False)]
    b = X1.iloc[rng.choice(len(X1), min(n, len(X1)), replace=False)]
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    scores = np.zeros(len(y))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        scores[va] = m.predict_proba(X.iloc[va])[:, 1]
    return auroc(y, scores)


def run_external(cfg) -> dict:
    logger = get_logger("external", exp_dir(cfg, "logs"))
    ext = load_external(cfg)
    if not ext:
        raise FileNotFoundError("No external datasets found. Run scripts/14_build_external.py first.")
    ecfg = cfg.get("external", {})
    seed = int(cfg["project"]["seed"])
    b = load_bundle(cfg)
    fr = b["frozen"]
    order = ["stack"] + fr["baselines"]
    best = fr["best_single"]
    key = list(dict.fromkeys(["stack", best, "logreg"]))
    Xd, _, _ = load_split(cfg, "dev")
    Xt, yt, _ = load_split(cfg, "test")
    res = resolve(cfg, "results_dir")
    rows, comps, recal_rows, sub_tabs, drift_rows, shap_rows, dca_rows = [], [], [], [], [], [], []
    summary = {}
    # reference: the 2015 held-out test set
    sets = {2015: (Xt, yt)}
    sets.update({y: (X, yy) for y, (X, yy, _) in ext.items()})
    imp15 = None
    unavail = unavailable_items(cfg)
    from .experiments import _background

    bg = _background(cfg, Xd)
    for year, (X, y) in sets.items():
        with timer(logger, f"external evaluation {year} (n={len(y):,})"):
            P = final_proba(b, X)
            B = int(ecfg.get("n_bootstrap", 1000))
            boot = bootstrap(y, {n: P[n] for n in key}, ["brier", "cal_slope", "cal_intercept", "oe_ratio"],
                             B=B, seed=derive_seed(seed, "external_boot", year))
            for n in order:
                m = all_metrics(y, P[n], fr["thresholds"][n], cfg["evaluation"]["ece_bins"])
                _, lo, hi = delong_ci(y, P[n])
                row = {"year": year, "model": n, "n": int(len(y)), "prevalence": float(y.mean()),
                       **m, "auroc_lo": lo, "auroc_hi": hi}
                if n in key:
                    for met in boot:
                        row[f"{met}_lo"] = boot[met]["per_model"][n]["ci_low"]
                        row[f"{met}_hi"] = boot[met]["per_model"][n]["ci_high"]
                rows.append(row)
            dl = {n: delong_test(y, P["stack"], P[n]) for n in fr["baselines"]}
            adj = holm({n: d["p_value"] for n, d in dl.items()})
            for n in fr["baselines"]:
                comps.append({"year": year, "comparison": f"stack - {n}", "diff": dl[n]["diff"],
                              "lo": dl[n]["ci_low"], "hi": dl[n]["ci_high"], "p": dl[n]["p_value"], "p_holm": adj[n]})
            ts = np.round(np.arange(0.05, 0.501, 0.01), 2)
            dca = pd.DataFrame({"year": year, "threshold": ts, "treat_all": net_benefit_treat_all(y, ts),
                                **{n: net_benefit(y, P[n], ts) for n in key}})
            dca_rows.append(dca)
            if year != 2015:
                # recalibration on a random half, evaluated on the other half
                idx = np.arange(len(y))
                fit_i, ev_i = train_test_split(idx, test_size=1 - float(ecfg.get("recal_fraction", 0.5)),
                                               stratify=y, random_state=derive_seed(seed, "recal", year))
                for n in key:
                    rc = recalibrate(P[n][fit_i], y[fit_i], P[n][ev_i])
                    for label, p in [("frozen", P[n][ev_i]), ("intercept_update", rc["intercept_update"]),
                                     ("logistic", rc["logistic"])]:
                        mm = all_metrics(y[ev_i], p, fr["thresholds"][n], cfg["evaluation"]["ece_bins"])
                        recal_rows.append({"year": year, "model": n, "recalibration": label,
                                           **{k: mm[k] for k in ("auroc", "brier", "cal_intercept", "cal_slope",
                                                                 "oe_ratio", "ece", "sensitivity", "specificity",
                                                                 "ppv", "flagged_fraction")},
                                           **(rc["params"] if label != "frozen" else {})})
                # subgroups
                g = group_labels(X)
                for n in key:
                    t = subgroup_table(y, P[n], fr["thresholds"][n], g, B=int(ecfg.get("subgroup_bootstrap", 200)),
                                       seed=derive_seed(seed, "ext_fair", year, n),
                                       min_size=cfg["fairness"]["min_group_size"])
                    t.insert(0, "model", n)
                    t.insert(0, "year", year)
                    sub_tabs.append(t)
                # covariate drift vs the development sample (items observed in every year only)
                for f in FEATURES:
                    na = f in unavail
                    drift_rows.append({"year": year, "feature": f, "mean_2015_dev": float(Xd[f].mean()),
                                       "mean_external": np.nan if na else float(X[f].mean()),
                                       "smd": np.nan if na else smd(Xd[f], X[f]), "imputed_in_some_year": na})
                common = [f for f in FEATURES if f not in unavail]
                summary[str(year)] = {"n": int(len(y)), "prevalence": float(y.mean()),
                                      "items_compared_for_drift": len(common),
                                      "membership_auroc_vs_2015": membership_auc(
                                          Xd[common], X[common], derive_seed(seed, "membership", year))}
            # explanation drift: global SHAP importance of the frozen stack
            rng = np.random.default_rng(derive_seed(seed, "ext_shap", year))
            ix = rng.choice(len(y), size=min(int(ecfg.get("n_explain", 1000)), len(y)), replace=False)
            phi, _ = b["stack"].shap_logit(X.iloc[ix], bg)
            imp = global_importance(phi)
            if year == 2015:
                imp15 = imp
            for f, v in zip(FEATURES, imp):
                shap_rows.append({"year": year, "feature": f, "mean_abs_shap": float(v), "share": float(v / imp.sum())})
            if imp15 is not None and year != 2015:
                summary[str(year)]["shap_kendall_tau_vs_2015"] = float(kendalltau(imp15, imp).statistic)
                summary[str(year)]["shap_top5_same"] = (set(np.argsort(-imp15)[:5]) == set(np.argsort(-imp)[:5]))
        logger.info(f"{year}: stack AUROC {auroc(y, P['stack']):.4f}, {best} {auroc(y, P[best]):.4f}, "
                    f"logreg {auroc(y, P['logreg']):.4f}; O/E stack {np.sum(y) / np.sum(P['stack']):.3f}")
    pd.DataFrame(rows).to_csv(res / "external_metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(res / "external_comparisons.csv", index=False)
    pd.DataFrame(recal_rows).to_csv(res / "external_recalibration.csv", index=False)
    if sub_tabs:
        pd.concat(sub_tabs, ignore_index=True).to_csv(res / "external_subgroups.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(res / "external_drift.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(res / "external_shap_importance.csv", index=False)
    pd.concat(dca_rows, ignore_index=True).to_csv(res / "external_decision_curve.csv", index=False)
    save_json(summary, res / "external_summary.json")
    return summary


def analyse_external_multiplicity(cfg) -> pd.DataFrame | None:
    """Arbitrariness of the 20 re-trained pipelines on each external year (needs proba_ext_* arrays)."""
    from itertools import combinations

    files = sorted(exp_dir(cfg, "multiplicity").glob("replicate_[0-9][0-9].npz"))
    if not files:
        return None
    reps = [np.load(f) for f in files]
    reps = [r for r in reps if any(k.startswith("proba_ext_") for k in r.files)]   # replicates that scored them
    if len(reps) < 2:
        return None
    years = sorted(int(k.split("_")[-1]) for k in reps[0].files if k.startswith("proba_ext_"))
    names = [str(n) for n in reps[0]["names"]]
    T = np.stack([r["thresholds"] for r in reps])
    ext = load_external(cfg)
    _, yt, _ = load_split(cfg, "test")
    rows = []
    for y in [2015] + years:
        key = "proba" if y == 2015 else f"proba_ext_{y}"
        P = np.stack([r[key] for r in reps]).astype(float)   # (R, M, n)
        D = P >= T[:, :, None]
        yy = yt if y == 2015 else (ext[y][1] if y in ext else None)
        R = len(reps)
        for j, n in enumerate(names):
            sd = P[:, j].std(axis=0, ddof=1)
            rng_ = P[:, j].max(0) - P[:, j].min(0)
            amb = D[:, j].any(0) & ~D[:, j].all(0)
            row = {"year": y, "model": n, "n_replicates": R, "pred_sd_mean": sd.mean(),
                   "pred_sd_p95": np.quantile(sd, 0.95), "risk_range_gt_0.05": float(np.mean(rng_ > 0.05)),
                   "ambiguity": float(amb.mean()), "n_ambiguous": int(amb.sum()),
                   "pairwise_flip_rate": float(np.mean([np.mean(D[a, j] != D[c, j])
                                                        for a, c in combinations(range(R), 2)]))}
            if yy is not None:
                aucs = [auroc(yy, P[r, j]) for r in range(R)]
                row.update({"auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs, ddof=1))})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg, "results_dir") / "external_multiplicity.csv", index=False)
    return df


def unavailable_item_sensitivity(cfg) -> pd.DataFrame:
    """How much does filling the items a later year did not ask (Fruits, Veggies in 2023) with
    development medians cost by itself? Re-score the 2015 test set and every year in which the items
    *were* asked with those items set to the medians, and compare with the unmodified data."""
    from .robustness import MedianImputer

    items = sorted(unavailable_items(cfg))
    if not items:
        return pd.DataFrame()
    b = load_bundle(cfg)
    Xd, _, _ = load_split(cfg, "dev")
    med = MedianImputer().fit(Xd)
    Xt, yt, _ = load_split(cfg, "test")
    data = {2015: (Xt.astype(float), yt)}
    for y, (Xx, yx, _) in load_external(cfg).items():
        data[y] = (Xx, yx)
    rows = []
    for y, (X, yy) in data.items():
        asked = not all(X[f].nunique() == 1 for f in items)
        Xm = X.copy()
        Xm[items] = np.nan
        Xm = med.transform(Xm)
        p0 = final_proba(b, X) if asked else None
        p1 = final_proba(b, Xm)
        for n in p1:
            row = {"year": y, "model": n, "items_asked": asked, "items": "+".join(items),
                   "auroc_items_filled": auroc(yy, p1[n]), "oe_items_filled": float(yy.mean() / p1[n].mean())}
            if asked:
                row.update({"auroc_observed": auroc(yy, p0[n]), "oe_observed": float(yy.mean() / p0[n].mean())})
                row["delta_auroc"] = row["auroc_items_filled"] - row["auroc_observed"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg, "results_dir") / "external_item_sensitivity.csv", index=False)
    return df
