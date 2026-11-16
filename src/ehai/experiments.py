"""Post-freeze analyses: explanations, retraining multiplicity (arbitrariness), robustness,
subgroup equity, selective prediction, and ablations. None of these alter the frozen models."""
from __future__ import annotations

from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .calibration import IsotonicCalibrator, PlattCalibrator
from .config import derive_seed, resolve
from .data import FEATURES, load_split
from .ensemble import LogitBag, LogitStackingEnsemble, fit_meta, oof_margins_for
from .explain import (cross_model_agreement, global_importance, importance_frame, local_stability,
                      pairwise_global_stability, perturb_small, perturbation_sensitivity)
from .fairness import disparity_summary, group_labels, subgroup_table
from .metrics import all_metrics, auroc, brier, threshold_for_sensitivity, threshold_metrics
from .pipeline import exp_dir, factory, final_proba, load_bundle, load_params
from .robustness import MedianImputer, mask_columns, mask_mcar, noise_bmi, noise_days, noise_genhlth
from .selective import (aurc, combined_uncertainty, disagreement_uncertainty, margin_uncertainty,
                        retained_mask, risk_coverage)
from .stats import bootstrap_mean_diff
from .utils import get_logger, load_json, save_json, sigmoid, timer

SHAP_MODELS = ["logreg", "xgboost", "lightgbm", "catboost", "stack"]


def _explain_rows(cfg, n_test: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "explain_rows"))
    return np.sort(rng.choice(n_test, size=min(n, n_test), replace=False))


def _background(cfg, Xd: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "shap_background"))
    return Xd.iloc[rng.choice(len(Xd), size=cfg["explain"]["n_background"], replace=False)]


# =============================================================================================
# 1. Explanations of the frozen models
# =============================================================================================
def run_explanations(cfg) -> dict:
    logger = get_logger("explain", exp_dir(cfg, "logs"))
    b = load_bundle(cfg)
    Xd, _, _ = load_split(cfg, "dev")
    Xt, yt, _ = load_split(cfg, "test")
    bg = _background(cfg, Xd)
    bg.to_csv(exp_dir(cfg, "models") / "shap_background.csv", index=False)  # used by scripts/predict.py
    rows = _explain_rows(cfg, len(Xt), cfg["explain"]["n_explain"])
    Xe = Xt.iloc[rows].reset_index(drop=True)
    models = {n: b["models"][n] for n in SHAP_MODELS if n != "stack"}
    models["stack"] = b["stack"]
    phi, base = {}, {}
    for n, m in models.items():
        with timer(logger, f"SHAP {n} on {len(Xe)} instances"):
            phi[n], base[n] = m.shap_logit(Xe, bg)
        err = np.abs(base[n] + phi[n].sum(1) - m.margin(Xe)).max()
        logger.info(f"{n}: local-accuracy max error {err:.2e}")
    out = exp_dir(cfg, "explain")
    np.savez(out / "shap_frozen.npz", rows=rows, X=Xe.to_numpy(), **{f"phi_{n}": v for n, v in phi.items()},
             **{f"base_{n}": np.array(v) for n, v in base.items()})
    imp = {n: global_importance(v) for n, v in phi.items()}
    res_dir = resolve(cfg, "results_dir")
    importance_frame(imp).to_csv(res_dir / "shap_global_importance.csv")
    cross_model_agreement(imp).to_csv(res_dir / "shap_cross_model_agreement.csv", index=False)

    # sensitivity of explanations to clinically negligible input changes
    deltas = cfg["explain"]["perturb"]
    Xp = perturb_small(Xe, deltas, derive_seed(cfg["project"]["seed"], "perturb"))
    sens_rows = []
    for n, m in models.items():
        phi_p, _ = m.shap_logit(Xp, bg)
        r = perturbation_sensitivity(phi[n], phi_p, m.margin(Xe), m.margin(Xp))
        sens_rows.append({"model": n, **r})
    pd.DataFrame(sens_rows).to_csv(res_dir / "shap_perturbation_sensitivity.csv", index=False)
    # contribution of each member to the ensemble attribution mass
    w = b["stack"].weight_table()
    save_json({"stack_weights": w, "local_accuracy_checked": True}, out / "explain_meta.json")
    return {"importance": imp}


# =============================================================================================
# 2. Retraining multiplicity: arbitrariness of individual predictions and explanations
# =============================================================================================
def _bag_oof(fac, Xr, yr, folds, size, seed):
    out = np.zeros(len(yr))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        out[va] = LogitBag(fac, size, seed + 101 * (k + 1)).fit(Xr.iloc[tr], yr[tr]).margin(Xr.iloc[va])
    return out


def run_replicate(cfg, r: int, Xd, yd, Xt, Xe, bg, frozen, params, ext: dict | None = None,
                  seed_only: bool = False) -> dict:
    """Re-run the whole pipeline (subsample -> OOF -> stack -> calibration -> threshold) once.

    seed_only=True keeps the full development set and changes only the random seeds (fold
    assignment, model randomness), isolating algorithmic from sampling variability.
    ext: optional {year: X} external datasets that are scored by every re-trained pipeline.
    """
    seed = derive_seed(cfg["project"]["seed"], "seed_only" if seed_only else "replicate", r)
    rng = np.random.default_rng(seed)
    if seed_only:
        idx = np.arange(len(yd))
    else:
        n_sub = int(round(cfg["multiplicity"]["subsample"] * len(yd)))
        idx = np.sort(rng.choice(len(yd), size=n_sub, replace=False))
    Xr, yr = Xd.iloc[idx].reset_index(drop=True), yd[idx]
    folds = np.empty(len(yr), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True,
                                                random_state=seed).split(np.zeros(len(yr)), yr)):
        folds[va] = k
    members = frozen["members"]
    bagf = frozen["bag_family"]
    target = cfg["operating_point"]["target_sensitivity"]
    oof, full = {}, {}
    for j, m in enumerate(members):
        fac = factory(cfg, m, params)
        oof[m] = oof_margins_for(fac, Xr, yr, folds, seed + 10 * j)
        full[m] = fac(seed + 10 * j + 5).fit(Xr, yr)
    ens = LogitStackingEnsemble({m: factory(cfg, m, params) for m in members}, cfg["cv"]["n_splits"], seed,
                                cfg["ensemble"]["nonnegative"], cfg["ensemble"]["meta_l2"])
    ens.fit_from_oof(full, np.column_stack([oof[m] for m in members]), yr, folds)
    size = cfg["multiplicity"]["bag_size"]
    bag_fac = factory(cfg, bagf, params)
    bag_oof = _bag_oof(bag_fac, Xr, yr, folds, size, seed + 777)
    bag = LogitBag(bag_fac, size, seed + 999).fit(Xr, yr)

    proba, thr, shap = {}, {}, {}
    for m in members:
        cal = PlattCalibrator().fit(sigmoid(oof[m]), yr)
        thr[m] = threshold_for_sensitivity(yr, cal.transform(sigmoid(oof[m])), target)
        proba[m] = cal.transform(full[m].predict_proba(Xt))
        shap[m] = full[m].shap_logit(Xe, bg)[0]
    thr["stack"] = threshold_for_sensitivity(yr, ens.oof_proba_, target)
    proba["stack"] = ens.predict_proba(Xt)
    shap["stack"] = sum(w * shap[m] for w, m in zip(ens.weights_, members))
    calb = PlattCalibrator().fit(sigmoid(bag_oof), yr)
    thr["bag"] = threshold_for_sensitivity(yr, calb.transform(sigmoid(bag_oof)), target)
    proba["bag"] = calb.transform(bag.predict_proba(Xt))
    shap["bag"] = bag.shap_logit(Xe, bg)[0]
    names = members + ["stack", "bag"]
    out = {"names": np.array(names), "proba": np.stack([proba[n] for n in names]).astype(np.float32),
           "thresholds": np.array([thr[n] for n in names]),
           "shap": np.stack([shap[n] for n in names]).astype(np.float32),
           "weights": np.r_[ens.intercept_, ens.weights_], "subsample_idx": idx}
    for year, Xx in (ext or {}).items():
        px = {}
        for m in members:
            cal = PlattCalibrator().fit(sigmoid(oof[m]), yr)
            px[m] = cal.transform(full[m].predict_proba(Xx))
        px["stack"] = ens.predict_proba(Xx)
        px["bag"] = calb.transform(bag.predict_proba(Xx))
        out[f"proba_ext_{year}"] = np.stack([px[n] for n in names]).astype(np.float32)
    return out


def run_multiplicity(cfg, seed_only: bool = False) -> None:
    logger = get_logger("multiplicity", exp_dir(cfg, "logs"))
    frozen = load_json(exp_dir(cfg, "models") / "frozen.json")
    params = load_params(cfg)
    Xd, yd, _ = load_split(cfg, "dev")
    Xt, yt, _ = load_split(cfg, "test")
    bg = _background(cfg, Xd)
    rows = _explain_rows(cfg, len(Xt), cfg["multiplicity"]["n_explain"])
    Xe = Xt.iloc[rows].reset_index(drop=True)
    out = exp_dir(cfg, "multiplicity")
    np.save(out / "explain_rows.npy", rows)
    from .external import load_external

    ext_all = {y: X for y, (X, _, _) in load_external(cfg).items()} if not seed_only else {}
    n_scored = int(cfg.get("external", {}).get("n_replicates_scored", 10**6))
    if ext_all:
        logger.info(f"external datasets {sorted(ext_all)} scored by the first {n_scored} replicates")
    n_rep = int(cfg["multiplicity"]["seed_only_replicates" if seed_only else "n_replicates"])
    prefix = "seedonly" if seed_only else "replicate"
    for r in range(n_rep):
        ext = ext_all if r < n_scored else {}
        f = out / f"{prefix}_{r:02d}.npz"
        if f.exists():
            try:
                with np.load(f) as z:
                    z["proba"].shape
                    missing_ext = [y for y in ext if f"proba_ext_{y}" not in z.files]
                if not missing_ext:
                    logger.info(f"{prefix} {r}: exists, skipping")
                    continue
                logger.info(f"{prefix} {r}: recomputing to add external predictions {missing_ext}")
            except Exception:  # corrupt/incomplete file from an interrupted run
                logger.warning(f"replicate {r}: unreadable file, recomputing")
                f.unlink()
        with timer(logger, f"{prefix} {r}"):
            res = run_replicate(cfg, r, Xd, yd, Xt, Xe, bg, frozen, params, ext=ext, seed_only=seed_only)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(f)  # atomic: an interrupted run never leaves a half-written replicate
        aucs = {str(n): round(auroc(yt, p), 4) for n, p in zip(res["names"], res["proba"])}
        logger.info(f"{prefix} {r}: test AUROC {aucs}")


def analyse_seed_only(cfg) -> pd.DataFrame | None:
    """Arbitrariness when only random seeds change (same development data)."""
    files = sorted(exp_dir(cfg, "multiplicity").glob("seedonly_[0-9][0-9].npz"))
    if len(files) < 2:
        return None
    reps = [np.load(f) for f in files]
    names = [str(n) for n in reps[0]["names"]]
    _, yt, _ = load_split(cfg, "test")
    P = np.stack([r["proba"] for r in reps]).astype(float)
    T = np.stack([r["thresholds"] for r in reps])
    S = np.stack([r["shap"] for r in reps]).astype(float)
    D = P >= T[:, :, None]
    R = len(reps)
    rows = []
    for j, n in enumerate(names):
        sd = P[:, j].std(axis=0, ddof=1)
        amb = D[:, j].any(0) & ~D[:, j].all(0)
        ls = local_stability([S[r, j] for r in range(R)], k=3)
        rows.append({"model": n, "n_replicates": R, "auroc_mean": float(np.mean([auroc(yt, P[r, j]) for r in range(R)])),
                     "pred_sd_mean": sd.mean(), "pred_sd_p95": np.quantile(sd, 0.95),
                     "risk_range_gt_0.05": float(np.mean((P[:, j].max(0) - P[:, j].min(0)) > 0.05)),
                     "ambiguity": float(amb.mean()), "n_ambiguous": int(amb.sum()),
                     "pairwise_flip_rate": float(np.mean([np.mean(D[a, j] != D[c, j])
                                                          for a, c in combinations(range(R), 2)])),
                     "threshold_sd": float(T[:, j].std(ddof=1)), "shap_spearman_mean": ls["spearman_mean"]})
    W = np.stack([r["weights"] for r in reps])
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg, "results_dir") / "multiplicity_seed_only_summary.csv", index=False)
    save_json({"weights_mean": W.mean(0), "weights_sd": W.std(0, ddof=1), "weights_min": W.min(0),
               "weights_max": W.max(0), "order": ["intercept"] + names[:W.shape[1] - 1]},
              resolve(cfg, "results_dir") / "multiplicity_seed_only_weights.json")
    return df


def analyse_multiplicity(cfg) -> dict:
    """Summaries of per-individual arbitrariness and explanation stability across replicates."""
    out = exp_dir(cfg, "multiplicity")
    files = sorted(out.glob("replicate_[0-9][0-9].npz"))
    if len(files) < 2:
        raise RuntimeError("Need at least 2 replicates; run scripts/08_multiplicity.py")
    reps = [np.load(f, allow_pickle=False) for f in files]
    names = [str(n) for n in reps[0]["names"]]
    Xt, yt, _ = load_split(cfg, "test")
    P = np.stack([r["proba"] for r in reps]).astype(float)        # (R, M, n)
    T = np.stack([r["thresholds"] for r in reps])                 # (R, M)
    S = np.stack([r["shap"] for r in reps]).astype(float)         # (R, M, n_e, F)
    D = P >= T[:, :, None]                                        # decisions
    R = len(reps)
    res_dir = resolve(cfg, "results_dir")
    per_ind = {}
    rows = []
    for j, n in enumerate(names):
        sd = P[:, j, :].std(axis=0, ddof=1)
        rng_ = P[:, j, :].max(axis=0) - P[:, j, :].min(axis=0)
        flip_any = D[:, j, :].any(axis=0) & ~D[:, j, :].all(axis=0)
        pair_flip = np.mean([np.mean(D[a, j] != D[b, j]) for a, b in combinations(range(R), 2)])
        aucs = [auroc(yt, P[r, j]) for r in range(R)]
        per_ind[n] = {"sd": sd, "range": rng_, "ambiguous": flip_any}
        rows.append({"model": n, "n_replicates": R, "auroc_mean": np.mean(aucs), "auroc_sd": np.std(aucs, ddof=1),
                     "auroc_min": np.min(aucs), "auroc_max": np.max(aucs),
                     "pred_sd_mean": sd.mean(), "pred_sd_median": np.median(sd), "pred_sd_p95": np.quantile(sd, 0.95),
                     "risk_range_mean": rng_.mean(), "risk_range_gt_0.05": np.mean(rng_ > 0.05),
                     "risk_range_gt_0.10": np.mean(rng_ > 0.10), "ambiguity": flip_any.mean(),
                     "pairwise_flip_rate": pair_flip, "threshold_sd": T[:, j].std(ddof=1)})
    summ = pd.DataFrame(rows)
    summ.to_csv(res_dir / "multiplicity_summary.csv", index=False)
    np.savez_compressed(out / "per_individual.npz", **{f"{k}_{n}": v for n in per_ind for k, v in per_ind[n].items()})

    frozen = load_json(exp_dir(cfg, "models") / "frozen.json")
    bagf = frozen["bag_family"]
    B = int(cfg["evaluation"]["n_bootstrap"])
    comps = {}
    for a, b in [("stack", bagf), ("stack", "bag"), ("bag", bagf), ("stack", "logreg")]:
        if a in per_ind and b in per_ind:
            comps[f"{a} vs {b}: pred_sd"] = bootstrap_mean_diff(per_ind[a]["sd"], per_ind[b]["sd"], B=B,
                                                                seed=derive_seed(cfg["project"]["seed"], a, b, "sd"))
            comps[f"{a} vs {b}: ambiguity"] = bootstrap_mean_diff(per_ind[a]["ambiguous"].astype(float),
                                                                  per_ind[b]["ambiguous"].astype(float), B=B,
                                                                  seed=derive_seed(cfg["project"]["seed"], a, b, "amb"))
    save_json(comps, res_dir / "multiplicity_comparisons.json")

    # arbitrariness by subgroup
    g = group_labels(Xt)
    sub_rows = []
    for attr in g.columns:
        for grp in g[attr].dropna().unique():
            m = (g[attr] == grp).to_numpy()
            for n in names:
                sub_rows.append({"attribute": attr, "group": grp, "model": n, "n": int(m.sum()),
                                 "pred_sd_mean": per_ind[n]["sd"][m].mean(),
                                 "ambiguity": per_ind[n]["ambiguous"][m].mean()})
    pd.DataFrame(sub_rows).to_csv(res_dir / "multiplicity_by_subgroup.csv", index=False)

    # explanation stability across replicates
    st_rows = []
    for j, n in enumerate(names):
        imps = [global_importance(S[r, j]) for r in range(R)]
        gs = pairwise_global_stability(imps, k=5)
        ls = local_stability([S[r, j] for r in range(R)], k=3)
        st_rows.append({"model": n, **gs, **{k: v for k, v in ls.items() if not k.startswith("per_instance")}})
        per_ind[n]["local_spearman"] = ls["per_instance_spearman"]
    pd.DataFrame(st_rows).to_csv(res_dir / "explanation_stability.csv", index=False)
    ex = {}
    for a, b in [("stack", bagf), ("stack", "bag"), ("bag", bagf)]:
        ex[f"{a} vs {b}: local_spearman"] = bootstrap_mean_diff(per_ind[a]["local_spearman"],
                                                                per_ind[b]["local_spearman"], B=B,
                                                                seed=derive_seed(cfg["project"]["seed"], a, b, "ls"))
    save_json(ex, res_dir / "explanation_stability_comparisons.json")
    return {"summary": summ}


# =============================================================================================
# 3. Robustness to missing inputs and measurement noise
# =============================================================================================
def run_robustness(cfg) -> pd.DataFrame:
    logger = get_logger("robustness", exp_dir(cfg, "logs"))
    b = load_bundle(cfg)
    fr = b["frozen"]
    Xd, _, _ = load_split(cfg, "dev")
    Xt, yt, _ = load_split(cfg, "test")
    imp = MedianImputer().fit(Xd)
    clean = final_proba(b, Xt)
    rc = cfg["robustness"]
    seed = int(cfg["project"]["seed"])
    rows = []

    def record(scenario, level, rep, Xs):
        P = final_proba(b, imp.transform(Xs))
        for n, p in P.items():
            t = fr["thresholds"][n]
            rows.append({"scenario": scenario, "level": level, "rep": rep, "model": n, "auroc": auroc(yt, p),
                         "brier": brier(yt, p), "sensitivity": threshold_metrics(yt, p, t)["sensitivity"],
                         "specificity": threshold_metrics(yt, p, t)["specificity"],
                         "mean_abs_change": float(np.abs(p - clean[n]).mean()),
                         "decision_flip_rate": float(np.mean((p >= t) != (clean[n] >= t)))})

    with timer(logger, "MCAR masking"):
        for rate in rc["mcar_rates"]:
            reps = 1 if rate == 0 else rc["n_mask_seeds"]
            for s in range(reps):
                record("mcar", rate, s, mask_mcar(Xt, rate, derive_seed(seed, "mcar", rate, s),
                                                  rc["protected_from_masking"]))
    with timer(logger, "structured missingness"):
        for name, cols in rc["scenarios"].items():
            record(name, 1.0, 0, mask_columns(Xt, cols))
    with timer(logger, "noise"):
        for sd in rc["bmi_noise_sd"]:
            for s in range(rc["n_mask_seeds"]):
                record("bmi_noise", sd, s, noise_bmi(Xt, sd, derive_seed(seed, "bmi", sd, s)))
        record("bmi_underreport", rc["bmi_shift"], 0, noise_bmi(Xt, 0.0, 0, shift=rc["bmi_shift"]))
        for s in range(rc["n_mask_seeds"]):
            record("days_noise", rc["days_noise_max"], s, noise_days(Xt, rc["days_noise_max"],
                                                                     derive_seed(seed, "days", s)))
            record("genhlth_noise", rc["genhlth_flip_prob"], s,
                   noise_genhlth(Xt, rc["genhlth_flip_prob"], derive_seed(seed, "gh", s)))
    df = pd.DataFrame(rows)
    agg = df.groupby(["scenario", "level", "model"], sort=False).agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), brier=("brier", "mean"),
        sensitivity=("sensitivity", "mean"), specificity=("specificity", "mean"),
        mean_abs_change=("mean_abs_change", "mean"), decision_flip_rate=("decision_flip_rate", "mean"),
        n_reps=("rep", "count")).reset_index()
    base = agg[(agg.scenario == "mcar") & (agg.level == 0)].set_index("model")
    agg["delta_auroc"] = agg.apply(lambda r: r.auroc - base.loc[r.model, "auroc"], axis=1)
    agg["delta_brier"] = agg.apply(lambda r: r.brier - base.loc[r.model, "brier"], axis=1)
    res_dir = resolve(cfg, "results_dir")
    df.to_csv(exp_dir(cfg, "robustness") / "robustness_raw.csv", index=False)
    agg.to_csv(res_dir / "robustness_summary.csv", index=False)
    return agg


# =============================================================================================
# 4. Subgroup equity
# =============================================================================================
def run_fairness(cfg) -> pd.DataFrame:
    logger = get_logger("fairness", exp_dir(cfg, "logs"))
    b = load_bundle(cfg)
    fr = b["frozen"]
    Xt, yt, _ = load_split(cfg, "test")
    P = final_proba(b, Xt)
    g = group_labels(Xt)
    fc = cfg["fairness"]
    tabs, disp = [], []
    for n in dict.fromkeys(["stack", fr["best_single"], "logreg"] + list(fr["baselines"])):
        with timer(logger, f"subgroups {n}"):
            t = subgroup_table(yt, P[n], fr["thresholds"][n], g, B=fc["n_bootstrap"],
                               seed=derive_seed(cfg["project"]["seed"], "fair", n), min_size=fc["min_group_size"])
        t.insert(0, "model", n)
        tabs.append(t)
        d = disparity_summary(t)
        d.insert(0, "model", n)
        disp.append(d)
    tab = pd.concat(tabs, ignore_index=True)
    res_dir = resolve(cfg, "results_dir")
    tab.to_csv(res_dir / "fairness_subgroups.csv", index=False)
    pd.concat(disp, ignore_index=True).to_csv(res_dir / "fairness_disparities.csv", index=False)
    return tab


# =============================================================================================
# 5. Selective prediction (deferral)
# =============================================================================================
def run_selective(cfg) -> pd.DataFrame:
    b = load_bundle(cfg)
    fr = b["frozen"]
    Xt, yt, _ = load_split(cfg, "test")
    dev = pd.read_csv(exp_dir(cfg, "predictions") / "dev_oof_predictions.csv.gz")
    p = b["stack"].predict_proba(Xt)
    t = fr["thresholds"]["stack"]
    mp = b["stack"].member_proba(Xt)
    # reference distributions from development OOF predictions (no test information)
    dev_p = dev["oof_stack"].to_numpy()
    dev_mp = np.column_stack([dev[f"oofraw_{m}"].to_numpy() for m in fr["members"]])
    u = {"margin": margin_uncertainty(p, t), "disagreement": disagreement_uncertainty(mp)}
    u["margin+disagreement"] = combined_uncertainty(u["margin"], u["disagreement"],
                                                    margin_uncertainty(dev_p, t), disagreement_uncertainty(dev_mp))
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "random_deferral"))
    u["random"] = rng.random(len(yt))
    rep_file = exp_dir(cfg, "multiplicity") / "per_individual.npz"
    if rep_file.exists():
        u["retraining_sd"] = np.load(rep_file)["sd_stack"]
    covs = cfg["selective"]["coverages"]
    rows = []
    for name, uu in u.items():
        _, cost = risk_coverage(yt, p, t, uu, covs, kind="cost")
        _, err = risk_coverage(yt, p, t, uu, covs, kind="error")
        row = {"signal": name, "aurc_cost_50_100": aurc(yt, p, t, uu, kind="cost"),
               "aurc_error_50_100": aurc(yt, p, t, uu, kind="error")}
        for c, rc, re in zip(covs, cost, err):
            m = retained_mask(uu, c)
            tm = threshold_metrics(yt[m], p[m], t)
            row[f"cost_per_1000@{c:.2f}"] = rc
            row[f"error@{c:.2f}"] = re
            row[f"sens@{c:.2f}"] = tm["sensitivity"]
            row[f"spec@{c:.2f}"] = tm["specificity"]
            row[f"auroc@{c:.2f}"] = auroc(yt[m], p[m])
            row[f"prev_retained@{c:.2f}"] = float(yt[m].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    # paired bootstrap of AURC (threshold-consistent cost) differences
    B = int(cfg["selective"]["n_bootstrap"])
    rngb = np.random.default_rng(derive_seed(cfg["project"]["seed"], "aurc_boot"))
    pairs = [("margin+disagreement", "margin"), ("disagreement", "margin"), ("margin", "random"),
             ("disagreement", "random")]
    if "retraining_sd" in u:
        pairs.append(("retraining_sd", "margin"))
    used = sorted({a for pr in pairs for a in pr})
    diffs = {f"{a} - {b_}": [] for a, b_ in pairs}
    n = len(yt)
    for _ in range(B):
        ib = rngb.integers(0, n, n)
        a = {k: aurc(yt[ib], p[ib], t, u[k][ib], kind="cost") for k in used}
        for x, z in pairs:
            diffs[f"{x} - {z}"].append(a[x] - a[z])
    point = {k: aurc(yt, p, t, u[k], kind="cost") for k in used}
    comp = {k: {"estimate": point[k.split(" - ")[0]] - point[k.split(" - ")[1]],
                "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))}
            for k, v in diffs.items()}
    # who is deferred? subgroup composition at 80% coverage
    g = group_labels(Xt)
    comp_rows = []
    for name in ("margin", "margin+disagreement"):
        deferred = ~retained_mask(u[name], 0.80)
        for attr in g.columns:
            for grp in g[attr].dropna().unique():
                m = (g[attr] == grp).to_numpy()
                comp_rows.append({"signal": name, "attribute": attr, "group": grp,
                                  "deferral_rate": float(deferred[m].mean()), "overall_rate": float(deferred.mean())})
    res_dir = resolve(cfg, "results_dir")
    df.to_csv(res_dir / "selective_summary.csv", index=False)
    save_json(comp, res_dir / "selective_comparisons.json")
    pd.DataFrame(comp_rows).to_csv(res_dir / "selective_deferral_by_subgroup.csv", index=False)
    curves = {}
    for name, uu in u.items():
        cov, risk = risk_coverage(yt, p, t, uu, kind="cost")
        keep = np.linspace(0, len(cov) - 1, 400).astype(int)
        curves[name] = (cov[keep], risk[keep])
    joblib.dump(curves, exp_dir(cfg, "selective") / "risk_coverage_curves.joblib")
    return df


# =============================================================================================
# 6. Ablations
# =============================================================================================
def run_ablations(cfg) -> pd.DataFrame:
    logger = get_logger("ablations", exp_dir(cfg, "logs"))
    b = load_bundle(cfg)
    fr = b["frozen"]
    params = load_params(cfg)
    Xd, yd, extra_d = load_split(cfg, "dev")
    Xt, yt, extra_t = load_split(cfg, "test")
    folds = extra_d["fold"].to_numpy()
    oofm = np.load(exp_dir(cfg, "predictions") / "dev_oof_margins.npz")
    members = fr["members"]
    M = np.column_stack([oofm[m] for m in members])
    Mt = np.column_stack([b["models"][m].margin(Xt) for m in members])
    target = cfg["operating_point"]["target_sensitivity"]
    rows = []

    def add(label, p_test, p_dev_oof):
        t = threshold_for_sensitivity(yd, p_dev_oof, target)
        r = all_metrics(yt, p_test, t, cfg["evaluation"]["ece_bins"])
        rows.append({"variant": label, **{k: r[k] for k in ("auroc", "auprc", "brier", "cal_slope",
                                                             "cal_intercept", "ece", "sensitivity",
                                                             "specificity", "ppv")}})

    def crossfit(Mx, nonneg, equal=False):
        oof = np.zeros(len(yd))
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            e = LogitStackingEnsemble({}, nonnegative=nonneg, equal_weights=equal)
            bb, w = e._fit_meta(Mx[tr], yd[tr])
            oof[va] = bb + Mx[va] @ w
        e = LogitStackingEnsemble({}, nonnegative=nonneg, equal_weights=equal)
        bb, w = e._fit_meta(Mx, yd)
        return sigmoid(oof), bb, w

    p_oof, bb, w = crossfit(M, True)
    add("Stack (non-negative weights; proposed)", sigmoid(bb + Mt @ w), p_oof)
    p_oof, bb, w = crossfit(M, False)
    add("Stack, unconstrained weights", sigmoid(bb + Mt @ w), p_oof)
    p_oof, bb, w = crossfit(M, True, equal=True)
    add("Equal-weight logit average", sigmoid(bb + Mt @ w), p_oof)
    for j, m in enumerate(members):
        keep = [i for i in range(len(members)) if i != j]
        p_oof, bb, w = crossfit(M[:, keep], True)
        add(f"Stack without {m}", sigmoid(bb + Mt[:, keep] @ w), p_oof)
    iso = IsotonicCalibrator().fit(b["stack"].oof_proba_, yd)
    add("Stack + isotonic recalibration", iso.transform(b["stack"].predict_proba(Xt)), iso.transform(b["stack"].oof_proba_))
    best = fr["best_single"]
    for v, cal in (("uncalibrated", None), ("Platt", "platt"), ("isotonic", "isotonic")):
        pt = b["models"][best].predict_proba(Xt)
        po = sigmoid(oofm[best])
        if cal:
            pt = b["calibrators"][best][cal].transform(pt)
            po = b["calibrators"][best][cal].transform(po)
        add(f"Best single ({best}), {v}", pt, po)
    # homogeneous bag of the best additive single model (same member count as the stack)
    with timer(logger, "homogeneous bag reference fit"):
        bagf = fr["bag_family"]
        fac = factory(cfg, bagf, params)
        seed = derive_seed(cfg["project"]["seed"], "ablation_bag")
        bag_oof = _bag_oof(fac, Xd, yd, folds, cfg["multiplicity"]["bag_size"], seed)
        bag = LogitBag(fac, cfg["multiplicity"]["bag_size"], seed + 1).fit(Xd, yd)
        calb = PlattCalibrator().fit(sigmoid(bag_oof), yd)
        add(f"Homogeneous bag of {cfg['multiplicity']['bag_size']} x {bagf} (Platt)",
            calb.transform(bag.predict_proba(Xt)), calb.transform(sigmoid(bag_oof)))
    res_dir = resolve(cfg, "results_dir")
    abl = pd.DataFrame(rows)
    abl.to_csv(res_dir / "ablations.csv", index=False)

    # sensitivity analysis: exclude respondents with prediabetes (requires the 3-class release)
    if "prediabetes" in extra_d.columns:
        with timer(logger, "prediabetes-excluded sensitivity analysis"):
            keep_d = extra_d["prediabetes"].to_numpy() == 0
            keep_t = extra_t["prediabetes"].to_numpy() == 0
            Xd2, yd2, f2 = Xd[keep_d].reset_index(drop=True), yd[keep_d], folds[keep_d]
            seed = derive_seed(cfg["project"]["seed"], "no_prediabetes")
            oof2 = {m: oof_margins_for(factory(cfg, m, params), Xd2, yd2, f2, seed + j) for j, m in enumerate(members)}
            full2 = {m: factory(cfg, m, params)(seed + 50 + j).fit(Xd2, yd2) for j, m in enumerate(members)}
            ens2 = LogitStackingEnsemble({m: factory(cfg, m, params) for m in members}).fit_from_oof(
                full2, np.column_stack([oof2[m] for m in members]), yd2, f2)
            Xt2, yt2 = Xt[keep_t], yt[keep_t]
            main = final_proba(b, Xt2)
            sens = {"n_dev_excluded": int((~keep_d).sum()), "n_test_excluded": int((~keep_t).sum()),
                    "auroc_stack_retrained": auroc(yt2, ens2.predict_proba(Xt2)),
                    "auroc_stack_main_model": auroc(yt2, main["stack"]),
                    f"auroc_{best}_main_model": auroc(yt2, main[best]),
                    "auroc_stack_main_model_full_test": auroc(yt, final_proba(b, Xt)["stack"]),
                    "brier_stack_retrained": brier(yt2, ens2.predict_proba(Xt2)),
                    "brier_stack_main_model": brier(yt2, main["stack"]),
                    "weights_retrained": ens2.weight_table()}
            save_json(sens, res_dir / "sensitivity_prediabetes_excluded.json")
    else:
        logger.info("No prediabetes flag (3-class release not provided) - sensitivity analysis skipped")
    return abl


# =============================================================================================
# 7. Error analysis
# =============================================================================================
def run_error_analysis(cfg) -> pd.DataFrame:
    """Profile of true/false positives/negatives of the frozen stack at its screening threshold, and
    of individuals whose decision is ambiguous across re-trainings."""
    b = load_bundle(cfg)
    fr = b["frozen"]
    Xt, yt, _ = load_split(cfg, "test")
    p = b["stack"].predict_proba(Xt)
    t = fr["thresholds"]["stack"]
    yhat = p >= t
    cat = np.select([(yt == 1) & yhat, (yt == 1) & ~yhat, (yt == 0) & yhat], ["TP", "FN", "FP"], "TN")
    prof = Xt.assign(outcome=cat, score=p).groupby("outcome")[FEATURES + ["score"]].mean().T
    prof = prof[[c for c in ["TP", "FN", "FP", "TN"] if c in prof.columns]]
    counts = pd.Series(cat).value_counts()
    res_dir = resolve(cfg, "results_dir")
    prof.to_csv(res_dir / "error_profile.csv")
    out = {"counts": counts.to_dict(), "threshold": t}
    g = group_labels(Xt)
    rows = []
    for attr in g.columns:
        for grp in g[attr].dropna().unique():
            m = (g[attr] == grp).to_numpy()
            pos = m & (yt == 1)
            rows.append({"attribute": attr, "group": grp, "n_cases": int(pos.sum()),
                         "miss_rate": float((~yhat[pos]).mean()) if pos.any() else np.nan,
                         "false_alarm_rate": float(yhat[m & (yt == 0)].mean())})
    pd.DataFrame(rows).to_csv(res_dir / "error_by_subgroup.csv", index=False)
    pi_file = exp_dir(cfg, "multiplicity") / "per_individual.npz"
    if pi_file.exists():
        amb = np.load(pi_file)["ambiguous_stack"].astype(bool)
        ap = Xt.assign(ambiguous=amb, score=p, y=yt).groupby("ambiguous")[FEATURES + ["score", "y"]].mean().T
        ap.columns = ["stable" if not c else "ambiguous" for c in ap.columns]
        ap.to_csv(res_dir / "ambiguous_profile.csv")
        out["n_ambiguous"] = int(amb.sum())
        out["ambiguous_mean_abs_logit_gap"] = float(np.abs(np.log(p[amb] / (1 - p[amb])) - np.log(t / (1 - t))).mean())
        out["stable_mean_abs_logit_gap"] = float(np.abs(np.log(p[~amb] / (1 - p[~amb])) - np.log(t / (1 - t))).mean())
    save_json(out, res_dir / "error_summary.json")
    return prof


def compare_seed_only_with_subsampling(cfg, seed_glob: str = "seedonly_[0-9][0-9].npz",
                                       sub_glob: str = "replicate_[0-9][0-9].npz") -> pd.DataFrame | None:
    """Split re-training variability into an algorithmic part (seeds only, full development data) and the
    part added by drawing a new training sample (80% subsamples with new seeds).

    Ambiguity and flip rates grow with the number of re-trainings, so the subsampling replicates are cut
    into disjoint groups of the same size as the seed-only set and averaged over the groups. Variances are
    unbiased whatever the number of replicates, so the variance share uses all replicates."""
    out = exp_dir(cfg, "multiplicity")
    seed_files = sorted(out.glob(seed_glob))
    sub_files = sorted(out.glob(sub_glob))
    if len(seed_files) < 2 or len(sub_files) < len(seed_files):
        return None
    k = len(seed_files)

    def load(files):
        reps = [np.load(f) for f in files]
        return ([str(n) for n in reps[0]["names"]], np.stack([r["proba"] for r in reps]).astype(float),
                np.stack([r["thresholds"] for r in reps]))

    def stats(P, T, j):
        D = P[:, j] >= T[:, j][:, None]
        amb = D.any(0) & ~D.all(0)
        flips = np.mean([np.mean(D[a] != D[c]) for a, c in combinations(range(len(P)), 2)])
        return P[:, j].std(axis=0, ddof=1).mean(), amb.mean(), flips

    names_s, Ps, Ts = load(seed_files)
    names_r, Pr, Tr = load(sub_files)
    groups = [list(range(g, g + k)) for g in range(0, len(sub_files) - k + 1, k)]
    rows = []
    for j, n in enumerate(names_s):
        if n not in names_r:
            continue
        jr = names_r.index(n)
        sd_s, amb_s, fl_s = stats(Ps, Ts, j)
        g = np.array([stats(Pr[idx], Tr[idx], jr) for idx in groups])
        var_s = Ps[:, j].var(axis=0, ddof=1).mean()
        var_r = Pr[:, jr].var(axis=0, ddof=1).mean()
        rows.append({"model": n, "n_seed_only": k, "n_subsample_groups": len(groups),
                     "pred_sd_seed_only": sd_s, "pred_sd_subsample": g[:, 0].mean(),
                     "ambiguity_seed_only": amb_s, "ambiguity_subsample": g[:, 1].mean(),
                     "ambiguity_subsample_min": g[:, 1].min(), "ambiguity_subsample_max": g[:, 1].max(),
                     "flip_rate_seed_only": fl_s, "flip_rate_subsample": g[:, 2].mean(),
                     "variance_share_seed_only": var_s / var_r if var_r > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg, "results_dir") / "multiplicity_seed_vs_sample.csv", index=False)
    return df


def seed_only_mechanism(cfg, seed_glob: str = "seedonly_[0-9][0-9].npz",
                        sub_glob: str = "replicate_[0-9][0-9].npz") -> dict | None:
    """Paired comparisons of arbitrariness under seed-only re-training (stack vs the bag family, the bag
    and logistic regression), and the correlation between members of their re-training deviations
    (each member's prediction minus its own mean over re-trainings), under seeds and under subsampling.
    Uncorrelated deviations can be averaged away by an ensemble; shared ones cannot."""
    out = exp_dir(cfg, "multiplicity")
    seed_files, sub_files = sorted(out.glob(seed_glob)), sorted(out.glob(sub_glob))
    if len(seed_files) < 2 or len(sub_files) < 2:
        return None

    def load(files):
        reps = [np.load(f) for f in files]
        return ([str(n) for n in reps[0]["names"]], np.stack([r["proba"] for r in reps]).astype(float),
                np.stack([r["thresholds"] for r in reps]))

    names, P, T = load(seed_files)
    names_r, PR, _ = load(sub_files)
    D = P >= T[:, :, None]
    sd = {n: P[:, j].std(axis=0, ddof=1) for j, n in enumerate(names)}
    amb = {n: (D[:, j].any(0) & ~D[:, j].all(0)).astype(float) for j, n in enumerate(names)}
    frozen = load_json(exp_dir(cfg, "models") / "frozen.json")
    bagf = frozen["bag_family"]
    B = int(cfg["evaluation"]["n_bootstrap"])
    comps = {}
    for a, b in [("stack", bagf), ("stack", "bag"), ("stack", "xgboost"), (bagf, "logreg")]:
        if a in sd and b in sd:
            comps[f"{a} vs {b}: pred_sd"] = bootstrap_mean_diff(
                sd[a], sd[b], B=B, seed=derive_seed(cfg["project"]["seed"], "seedonly", a, b, "sd"))
            comps[f"{a} vs {b}: ambiguity"] = bootstrap_mean_diff(
                amb[a], amb[b], B=B, seed=derive_seed(cfg["project"]["seed"], "seedonly", a, b, "amb"))

    def devcorr(P_, nm, a, b):
        ja, jb = nm.index(a), nm.index(b)
        da, db = P_[:, ja] - P_[:, ja].mean(0), P_[:, jb] - P_[:, jb].mean(0)
        return float(np.corrcoef(da.ravel(), db.ravel())[0, 1])

    members = [m for m in frozen["members"] if m in names and m in names_r]
    rows = [{"member_a": a, "member_b": b, "corr_seed_only": devcorr(P, names, a, b),
             "corr_subsample": devcorr(PR, names_r, a, b)}
            for a, b in combinations(members, 2)]
    res = resolve(cfg, "results_dir")
    pd.DataFrame(rows).to_csv(res / "multiplicity_member_deviation_corr.csv", index=False)
    save_json(comps, res / "multiplicity_seed_only_comparisons.json")
    return {"comparisons": comps, "deviation_corr": rows}
