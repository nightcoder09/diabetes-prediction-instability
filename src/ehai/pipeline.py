"""Core pipeline: development-set training (OOF, calibration, thresholds, freezing) and the single
held-out test evaluation. Model selection is completed and frozen before the test set is read."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrator, PlattCalibrator
from .config import derive_seed, n_jobs, resolve
from .data import FEATURES, load_split
from .ensemble import LogitStackingEnsemble, oof_margins_for
from .metrics import all_metrics, auroc, net_benefit, net_benefit_treat_all, threshold_for_sensitivity
from .models import ADDITIVE_LOGIT, DEFAULT_PARAMS, make_model
from .stats import bootstrap, delong_ci, delong_test, holm
from .utils import get_logger, load_json, save_json, sigmoid, timer


# ----------------------------------------------------------------------------- helpers
def exp_dir(cfg, *parts) -> Path:
    p = resolve(cfg, "experiments_dir").joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_params(cfg, logger=None) -> dict:
    path = exp_dir(cfg, "tuning") / "best_params.json"
    if path.exists():
        return load_json(path)
    if logger:
        logger.warning("No tuned parameters found - using DEFAULT_PARAMS (run scripts/04_tune.py first)")
    return {k: dict(v) for k, v in DEFAULT_PARAMS.items()}


def _make(name, params, nj, seed):
    return make_model(name, params, seed=seed, n_jobs=nj)


def factory(cfg, name: str, params: dict):
    """Picklable callable seed -> unfitted model."""
    return partial(_make, name, params[name], n_jobs(cfg))


def bag_family(frozen: dict) -> str:
    """Homogeneous-bag family: the best single model if it has exact log-odds SHAP, else best additive."""
    if frozen["best_single"] in ADDITIVE_LOGIT:
        return frozen["best_single"]
    ranked = sorted(frozen["dev_oof_auroc"].items(), key=lambda kv: -kv[1])
    return next(n for n, _ in ranked if n in ADDITIVE_LOGIT)


def load_bundle(cfg) -> dict:
    mdir = exp_dir(cfg, "models")
    frozen = load_json(mdir / "frozen.json")
    models = {n: joblib.load(mdir / f"{n}.joblib") for n in frozen["baselines"]}
    return {"frozen": frozen, "models": models, "stack": joblib.load(mdir / "stack.joblib"),
            "calibrators": joblib.load(mdir / "calibrators.joblib")}


def final_proba(bundle: dict, X: pd.DataFrame) -> dict:
    """Final (pre-specified) outputs: single models -> Platt(OOF)-calibrated; stack -> native."""
    out = {}
    for n, m in bundle["models"].items():
        out[n] = bundle["calibrators"][n]["platt"].transform(m.predict_proba(X))
    out["stack"] = bundle["stack"].predict_proba(X)
    return out


# ----------------------------------------------------------------------------- training
def train_development(cfg) -> dict:
    logger = get_logger("train", exp_dir(cfg, "logs"))
    seed = int(cfg["project"]["seed"])
    X, y, extra = load_split(cfg, "dev")
    folds = extra["fold"].to_numpy()
    params = load_params(cfg, logger)
    baselines = cfg["models"]["baselines"]
    members = cfg["models"]["ensemble_members"]
    for m in members:
        if m not in ADDITIVE_LOGIT:
            raise ValueError(f"Ensemble member '{m}' lacks exact log-odds SHAP; allowed: {sorted(ADDITIVE_LOGIT)}")
    mdir = exp_dir(cfg, "models")
    oof, full = {}, {}
    for name in baselines:
        fac = factory(cfg, name, params)
        with timer(logger, f"{name}: {cfg['cv']['n_splits']}-fold OOF + full refit"):
            oof[name] = oof_margins_for(fac, X, y, folds, derive_seed(seed, name, "oof"))
            full[name] = fac(derive_seed(seed, name, "full")).fit(X, y)
        joblib.dump(full[name], mdir / f"{name}.joblib")
        logger.info(f"{name}: dev OOF AUROC = {auroc(y, oof[name]):.4f}")

    ens = LogitStackingEnsemble({m: factory(cfg, m, params) for m in members}, cfg["cv"]["n_splits"],
                                derive_seed(seed, "stack"), nonnegative=cfg["ensemble"]["nonnegative"],
                                l2=cfg["ensemble"]["meta_l2"])
    ens.fit_from_oof({m: full[m] for m in members}, np.column_stack([oof[m] for m in members]), y, folds)
    joblib.dump(ens, mdir / "stack.joblib")
    logger.info(f"Stack weights: {ens.weight_table()}")

    oof_p = {n: sigmoid(oof[n]) for n in baselines}
    oof_p["stack"] = ens.oof_proba_
    cal = {n: {"platt": PlattCalibrator().fit(p, y), "isotonic": IsotonicCalibrator().fit(p, y)}
           for n, p in oof_p.items()}
    joblib.dump(cal, mdir / "calibrators.joblib")
    final_oof = {n: cal[n]["platt"].transform(oof_p[n]) for n in baselines}
    final_oof["stack"] = oof_p["stack"]
    target = cfg["operating_point"]["target_sensitivity"]
    thresholds = {n: threshold_for_sensitivity(y, p, target) for n, p in final_oof.items()}
    dev_auc = {n: auroc(y, oof_p[n]) for n in baselines}
    best = max(dev_auc, key=dev_auc.get)
    dev_rows = []
    for n, p in final_oof.items():
        r = all_metrics(y, p, thresholds[n], cfg["evaluation"]["ece_bins"])
        dev_rows.append({"model": n, **r})
    pd.DataFrame(dev_rows).to_csv(exp_dir(cfg, "predictions") / "dev_oof_metrics.csv", index=False)
    pd.DataFrame({"y": y, "fold": folds, **{f"oof_{n}": p for n, p in final_oof.items()},
                  **{f"oofraw_{n}": p for n, p in oof_p.items()}}).to_csv(
        exp_dir(cfg, "predictions") / "dev_oof_predictions.csv.gz", index=False)
    np.savez(exp_dir(cfg, "predictions") / "dev_oof_margins.npz", **{n: oof[n] for n in baselines})
    frozen = {"baselines": baselines, "members": members, "best_single": best, "dev_oof_auroc": dev_auc,
              "dev_oof_auroc_stack": auroc(y, oof_p["stack"]), "thresholds": thresholds,
              "target_sensitivity": target, "stack_weights": ens.weight_table(), "params": params,
              "platt": {n: [cal[n]["platt"].a_, cal[n]["platt"].b_] for n in cal}, "seed": seed}
    frozen["bag_family"] = bag_family(frozen)
    save_json(frozen, mdir / "frozen.json")
    logger.info(f"Best single model on dev OOF AUROC: {best} ({dev_auc[best]:.4f}); "
                f"stack dev OOF AUROC {frozen['dev_oof_auroc_stack']:.4f}")
    return frozen


# ----------------------------------------------------------------------------- test evaluation
def evaluate_test(cfg) -> dict:
    logger = get_logger("evaluate", exp_dir(cfg, "logs"))
    seed = int(cfg["project"]["seed"])
    bundle = load_bundle(cfg)
    fr = bundle["frozen"]
    X, y, _ = load_split(cfg, "test")
    res_dir = resolve(cfg, "results_dir")
    raw = {n: m.predict_proba(X) for n, m in bundle["models"].items()}
    raw["stack"] = bundle["stack"].predict_proba(X)
    variants = {n: {"none": p, "platt": bundle["calibrators"][n]["platt"].transform(p),
                    "isotonic": bundle["calibrators"][n]["isotonic"].transform(p)} for n, p in raw.items()}
    final = final_proba(bundle, X)
    member_p = bundle["stack"].member_proba(X)
    pred_df = pd.DataFrame({"y": y, **{f"final_{n}": p for n, p in final.items()},
                            **{f"{v}_{n}": variants[n][v] for n in variants for v in ("none", "platt", "isotonic")},
                            **{f"member_{m}": member_p[:, j] for j, m in enumerate(fr["members"])}})
    pred_df.to_csv(exp_dir(cfg, "predictions") / "test_predictions.csv.gz", index=False)

    # ---- main performance table with bootstrap CIs
    order = ["stack"] + fr["baselines"]
    best = fr["best_single"]
    pairs = [("stack", n) for n in fr["baselines"]]
    B = int(cfg["evaluation"]["n_bootstrap"])
    with timer(logger, f"bootstrap B={B}"):
        boot = bootstrap(y, {n: final[n] for n in order},
                         ["auroc", "auprc", "brier", "log_loss", "cal_slope", "cal_intercept", "oe_ratio"],
                         B=B, seed=derive_seed(seed, "test_bootstrap"), pairs=pairs)
    rows = []
    for n in order:
        r = all_metrics(y, final[n], fr["thresholds"][n], cfg["evaluation"]["ece_bins"])
        a, lo, hi = delong_ci(y, final[n])
        r.update({"model": n, "auroc_delong_lo": lo, "auroc_delong_hi": hi})
        for met in boot:
            r[f"{met}_lo"] = boot[met]["per_model"][n]["ci_low"]
            r[f"{met}_hi"] = boot[met]["per_model"][n]["ci_high"]
        rows.append(r)
    main = pd.DataFrame(rows).set_index("model")
    main.to_csv(res_dir / "test_main_metrics.csv")

    comp = []
    dl = {n: delong_test(y, final["stack"], final[n]) for n in fr["baselines"]}
    adj = holm({n: d["p_value"] for n, d in dl.items()})
    for n in fr["baselines"]:
        row = {"comparison": f"stack - {n}", "is_best_single": n == best, "delong_diff": dl[n]["diff"],
               "delong_lo": dl[n]["ci_low"], "delong_hi": dl[n]["ci_high"], "delong_p": dl[n]["p_value"],
               "delong_p_holm": adj[n]}
        for met in boot:
            pr = boot[met]["pairs"][f"stack - {n}"]
            row.update({f"d_{met}": pr["estimate"], f"d_{met}_lo": pr["ci_low"], f"d_{met}_hi": pr["ci_high"]})
        comp.append(row)
    pd.DataFrame(comp).to_csv(res_dir / "test_comparisons.csv", index=False)

    # ---- calibration table across post-hoc variants
    cal_rows = []
    for n in order:
        for v in ("none", "platt", "isotonic"):
            p = variants[n][v]
            m = all_metrics(y, p, None, cfg["evaluation"]["ece_bins"])
            cal_rows.append({"model": n, "calibration": v, **{k: m[k] for k in
                             ("auroc", "brier", "log_loss", "cal_intercept", "cal_slope", "oe_ratio", "ece")}})
    pd.DataFrame(cal_rows).to_csv(res_dir / "test_calibration_variants.csv", index=False)

    # ---- decision curve analysis
    t0, t1, st = cfg["evaluation"]["dca_thresholds"]
    ts = np.round(np.arange(t0, t1 + 1e-9, st), 4)
    dca = pd.DataFrame({"threshold": ts, "treat_all": net_benefit_treat_all(y, ts), "treat_none": 0.0,
                        **{n: net_benefit(y, final[n], ts) for n in order}})
    dca.to_csv(res_dir / "test_decision_curve.csv", index=False)

    # ---- novel feature vectors (rows whose exact predictor vector never occurs in the dev set)
    Xd, _, _ = load_split(cfg, "dev")
    dev_keys = set(map(tuple, Xd[FEATURES].to_numpy().astype(int)))
    seen = np.array([tuple(r) in dev_keys for r in X[FEATURES].to_numpy().astype(int)])
    nov = {"n_test": int(len(y)), "n_seen_vector": int(seen.sum()), "n_novel_vector": int((~seen).sum())}
    def _safe_auc(mask, p):
        return auroc(y[mask], p[mask]) if mask.any() and 0 < y[mask].sum() < mask.sum() else None

    for n in dict.fromkeys(["stack", best, "logreg"]):
        nov[f"auroc_all_{n}"] = auroc(y, final[n])
        nov[f"auroc_novel_{n}"] = _safe_auc(~seen, final[n])
        nov[f"auroc_seen_{n}"] = _safe_auc(seen, final[n])
    save_json(nov, res_dir / "test_novel_vectors.json")
    save_json({"thresholds": fr["thresholds"], "best_single": best, "stack_weights": fr["stack_weights"]},
              res_dir / "frozen_summary.json")
    logger.info(f"Test AUROC stack={main.loc['stack', 'auroc']:.4f}, {best}={main.loc[best, 'auroc']:.4f}")
    return {"main": main, "comparisons": pd.DataFrame(comp)}
