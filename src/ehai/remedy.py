"""Post hoc remedy for the arbitrariness of individual screening decisions.

Added after the main analysis (an exploratory pilot on CatBoost suggested it); everything here is
reported as post hoc and none of it changes the pre-specified frozen models in experiments/models/.

Two changes, both decided on development data only:

1. Stability-aware choice of hyper-parameters. Tuning ends with several settings whose
   cross-validated AUROC differs by less than its own noise. Among the settings within
   ``remedy.delta_auroc`` of the best tuning AUROC, we keep the one whose screening decisions change
   least across re-training, measured on a fixed audit part of the development set
   (``remedy.audit_fraction``), provided its audit AUROC is within ``remedy.auroc_guard`` of the best.
2. Seed averaging. Each gradient-boosting member becomes the log-odds average of ``remedy.n_seeds``
   copies trained on the same rows with different seeds (``ensemble.SeedAverage``); SHAP stays exact.

The remedied ("stable") stack is then
  (a) frozen on the full development set and read once on the test set, and
  (b) re-trained on the same 80% subsamples and folds as the first ``remedy.n_replicates`` runs of the
      main re-training audit, so that per-person comparisons with the original stack are paired.
"""
from __future__ import annotations

import json
import shutil
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .calibration import PlattCalibrator
from .config import derive_seed, n_jobs, resolve
from .data import load_split
from .ensemble import LogitStackingEnsemble, oof_margins_for, seed_average_factory
from .explain import local_stability
from .metrics import all_metrics, auroc, brier, calibration_slope, threshold_for_sensitivity
from .models import make_model
from .pipeline import exp_dir, factory, load_params
from .stats import bootstrap, bootstrap_mean_diff, delong_test
from .utils import get_logger, load_json, save_json, sigmoid, timer

FAMILIES = ("xgboost", "lightgbm", "catboost")
VARIANTS = ("stable", "stable_seedavg")     # stability-chosen settings, without / with seed averaging
_META = {"candidate", "auroc_mean", "auroc_sd", "seconds"}
_INT = {"n_estimators", "max_depth", "num_leaves", "min_child_samples", "iterations", "depth"}


def _rc(cfg) -> dict:
    return cfg["remedy"]


def _same(p: dict, q: dict) -> bool:
    return set(p) == set(q) and all(np.isclose(float(p[k]), float(q[k]), rtol=1e-9, atol=1e-12) for k in p)


def candidate_settings(cfg, fam: str) -> list[tuple[int, dict, float]]:
    """Tuning-equivalent settings: tuning AUROC within delta_auroc of the best (from the saved search)."""
    df = pd.read_csv(exp_dir(cfg, "tuning") / f"{fam}_search.csv")
    best = df["auroc_mean"].max()
    keep = df[df["auroc_mean"] >= best - float(_rc(cfg)["delta_auroc"])].sort_values("auroc_mean", ascending=False)
    out = []
    for _, row in keep.iterrows():
        p = {k: (int(row[k]) if k in _INT else float(row[k])) for k in df.columns if k not in _META}
        out.append((int(row["candidate"]), p, float(row["auroc_mean"])))
    return out


def _decision_stats(P: np.ndarray, T: np.ndarray) -> dict:
    D = P >= T[:, None]
    R = len(P)
    amb = D.any(0) & ~D.all(0)
    return {"sd": P.std(0, ddof=1), "ambiguous": amb,
            "pairwise_flip": float(np.mean([np.mean(D[a] != D[b]) for a, b in combinations(range(R), 2)]))}


# ============================================================================== 1. selection
def select_stable_settings(cfg, force: bool = False) -> pd.DataFrame:
    logger = get_logger("remedy", exp_dir(cfg, "logs"))
    rc, master = _rc(cfg), int(cfg["project"]["seed"])
    target = cfg["operating_point"]["target_sensitivity"]
    out, res_dir = exp_dir(cfg, "remedy"), resolve(cfg, "results_dir")
    Xd, yd, _ = load_split(cfg, "dev")
    pool, audit = train_test_split(np.arange(len(yd)), test_size=float(rc["audit_fraction"]), stratify=yd,
                                   random_state=derive_seed(master, "remedy", "audit"))
    pool, audit = np.sort(pool), np.sort(audit)
    n_sub = int(round(cfg["multiplicity"]["subsample"] * len(pool)))
    subs = []
    for k in range(int(rc["n_subsamples"])):
        rng = np.random.default_rng(derive_seed(master, "remedy", "subsample", k))
        s = np.sort(rng.choice(pool, size=n_sub, replace=False))
        subs.append((s, np.setdiff1d(pool, s)))
    Xa, ya = Xd.iloc[audit], yd[audit]
    cache = out / "selection_rows.csv"
    rows = [] if (force or not cache.exists()) else pd.read_csv(cache).to_dict("records")
    tuned = load_params(cfg)
    logger.info(f"selection: audit n={len(audit)}, pool n={len(pool)}, {len(subs)} re-trainings per setting")
    for fam in FAMILIES:
        for cand, p, cv in candidate_settings(cfg, fam):
            if any(r["family"] == fam and int(r["candidate"]) == cand for r in rows):
                continue
            P, T = [], []
            with timer(logger, f"selection {fam} candidate {cand}"):
                for k, (s, h) in enumerate(subs):
                    m = make_model(fam, p, seed=derive_seed(master, "remedy", fam, cand, k),
                                   n_jobs=n_jobs(cfg)).fit(Xd.iloc[s], yd[s])
                    T.append(threshold_for_sensitivity(yd[h], m.predict_proba(Xd.iloc[h]), target))
                    P.append(m.predict_proba(Xa))
            P, T = np.array(P), np.array(T)
            st = _decision_stats(P, T)
            rows.append({"family": fam, "candidate": cand, "tuned": _same(p, tuned[fam]), "cv_auroc": cv,
                         "audit_auroc": float(np.mean([auroc(ya, x) for x in P])),
                         "audit_ambiguity": float(st["ambiguous"].mean()), "audit_pred_sd": float(st["sd"].mean()),
                         "audit_pairwise_flip": st["pairwise_flip"], "params": json.dumps(p)})
            pd.DataFrame(rows).to_csv(cache, index=False)
            logger.info(f"{fam} {cand}: audit AUROC {rows[-1]['audit_auroc']:.4f}, "
                        f"ambiguity {rows[-1]['audit_ambiguity']:.4f}")
    df = pd.DataFrame(rows)
    df["selected"] = False
    sel, chosen = {k: dict(v) for k, v in tuned.items()}, {}
    for fam in FAMILIES:
        d = df[df["family"] == fam]
        ok = d[d["audit_auroc"] >= d["audit_auroc"].max() - float(rc["auroc_guard"])]
        i = ok["audit_ambiguity"].idxmin()
        df.loc[i, "selected"] = True
        sel[fam] = json.loads(df.loc[i, "params"])
        chosen[fam] = {"candidate": int(df.loc[i, "candidate"]), "is_tuned_setting": bool(df.loc[i, "tuned"])}
    df.to_csv(res_dir / "remedy_selection.csv", index=False)
    save_json({"params": sel, "chosen": chosen, "audit_n": int(len(audit)), "pool_n": int(len(pool)),
               "n_subsamples": len(subs), "rule": {k: rc[k] for k in ("delta_auroc", "auroc_guard")}},
              out / "selected_params.json")
    logger.info(f"selected settings: {chosen}")
    return df


def _selected(cfg) -> dict:
    path = exp_dir(cfg, "remedy") / "selected_params.json"
    if not path.exists():
        raise FileNotFoundError("Run the selection step first: python scripts/18_remedy.py --step select")
    return load_json(path)


def _member_factories(cfg, params: dict, n_seeds: int) -> dict:
    return {m: (seed_average_factory(factory(cfg, m, params), n_seeds) if m in FAMILIES else factory(cfg, m, params))
            for m in cfg["models"]["ensemble_members"]}


# ============================================================================== 2. frozen stable stack
def train_stable(cfg) -> dict:
    logger = get_logger("remedy", exp_dir(cfg, "logs"))
    rc, seed = _rc(cfg), int(cfg["project"]["seed"])
    params, S = _selected(cfg)["params"], int(rc["n_seeds"])
    X, y, extra = load_split(cfg, "dev")
    folds = extra["fold"].to_numpy()
    facs = _member_factories(cfg, params, S)
    mdir = exp_dir(cfg, "models_stable")
    oof, full = {}, {}
    for m, fac in facs.items():
        with timer(logger, f"stable {m}: OOF + full refit ({S if m in FAMILIES else 1} seed(s))"):
            oof[m] = oof_margins_for(fac, X, y, folds, derive_seed(seed, m, "oof"))
            full[m] = fac(derive_seed(seed, m, "full")).fit(X, y)
    ens = LogitStackingEnsemble(facs, cfg["cv"]["n_splits"], derive_seed(seed, "stack"),
                                nonnegative=cfg["ensemble"]["nonnegative"], l2=cfg["ensemble"]["meta_l2"])
    ens.fit_from_oof(full, np.column_stack([oof[m] for m in facs]), y, folds)
    joblib.dump(ens, mdir / "stack.joblib")
    target = cfg["operating_point"]["target_sensitivity"]
    cal = {m: PlattCalibrator().fit(sigmoid(oof[m]), y) for m in facs}
    joblib.dump(cal, mdir / "calibrators.joblib")
    frozen = {"variant": "stable stack (post hoc remedy)", "members": list(facs), "n_seeds": S, "params": params,
              "thresholds": {"stack": threshold_for_sensitivity(y, ens.oof_proba_, target),
                             **{m: threshold_for_sensitivity(y, cal[m].transform(sigmoid(oof[m])), target)
                                for m in facs}},
              "target_sensitivity": target, "stack_weights": ens.weight_table(),
              "dev_oof_auroc": {m: auroc(y, oof[m]) for m in facs},
              "dev_oof_auroc_stack": auroc(y, ens.oof_proba_), "seed": seed}
    save_json(frozen, mdir / "frozen.json")
    bg = exp_dir(cfg, "models") / "shap_background.csv"
    if bg.exists():
        shutil.copy(bg, mdir / "shap_background.csv")
    logger.info(f"stable stack weights {ens.weight_table()}, dev OOF AUROC {frozen['dev_oof_auroc_stack']:.4f}")
    return frozen


def evaluate_stable_test(cfg) -> pd.DataFrame:
    """Single read of the test set for the frozen stable stack, compared with the frozen originals."""
    logger = get_logger("remedy", exp_dir(cfg, "logs"))
    seed = int(cfg["project"]["seed"])
    mdir = exp_dir(cfg, "models_stable")
    fr_s, fr = load_json(mdir / "frozen.json"), load_json(exp_dir(cfg, "models") / "frozen.json")
    ens, cal = joblib.load(mdir / "stack.joblib"), joblib.load(mdir / "calibrators.joblib")
    X, y, _ = load_split(cfg, "test")
    main = pd.read_csv(exp_dir(cfg, "predictions") / "test_predictions.csv.gz")
    best = fr["best_single"]
    preds = {"stable_stack": ens.predict_proba(X),
             f"stable_{best}": cal[best].transform(ens.members_[best].predict_proba(X)),
             "stack": main["final_stack"].to_numpy(), best: main[f"final_{best}"].to_numpy()}
    thr = {"stable_stack": fr_s["thresholds"]["stack"], f"stable_{best}": fr_s["thresholds"][best],
           "stack": fr["thresholds"]["stack"], best: fr["thresholds"][best]}
    np.savez_compressed(exp_dir(cfg, "remedy") / "test_predictions_stable.npz", **preds)
    pairs = [("stable_stack", "stack"), ("stable_stack", best), (f"stable_{best}", best)]
    boot = bootstrap(y, preds, ["auroc", "auprc", "brier", "cal_slope", "oe_ratio"],
                     B=int(cfg["evaluation"]["n_bootstrap"]), seed=derive_seed(seed, "remedy_test_bootstrap"),
                     pairs=pairs)
    rows = []
    for n, p in preds.items():
        r = {"model": n, **all_metrics(y, p, thr[n], cfg["evaluation"]["ece_bins"])}
        for met in boot:
            r[f"{met}_lo"], r[f"{met}_hi"] = boot[met]["per_model"][n]["ci_low"], boot[met]["per_model"][n]["ci_high"]
        rows.append(r)
    res_dir = resolve(cfg, "results_dir")
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / "remedy_test_metrics.csv", index=False)
    comp = []
    for a, b in pairs:
        d = delong_test(y, preds[a], preds[b])
        row = {"comparison": f"{a} - {b}", "delong_diff": d["diff"], "delong_lo": d["ci_low"],
               "delong_hi": d["ci_high"], "delong_p": d["p_value"]}
        for met in boot:
            pr = boot[met]["pairs"][f"{a} - {b}"]
            row.update({f"d_{met}": pr["estimate"], f"d_{met}_lo": pr["ci_low"], f"d_{met}_hi": pr["ci_high"]})
        comp.append(row)
    pd.DataFrame(comp).to_csv(res_dir / "remedy_test_comparisons.csv", index=False)
    logger.info(f"stable stack test AUROC {auroc(y, preds['stable_stack']):.4f} "
                f"(original {auroc(y, preds['stack']):.4f})")
    return tab


# ============================================================================== 3. paired re-training audit
def _remedy_replicate(cfg, r: int, Xd, yd, Xt, Xe, bg, params: dict, n_seeds: int, ext: dict) -> dict:
    """Same subsample, folds and seeds as main replicate r; only the booster settings / seed averaging differ."""
    ref = np.load(exp_dir(cfg, "multiplicity") / f"replicate_{r:02d}.npz")
    idx = ref["subsample_idx"]
    seed = derive_seed(cfg["project"]["seed"], "replicate", r)
    Xr, yr = Xd.iloc[idx].reset_index(drop=True), yd[idx]
    folds = np.empty(len(yr), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True,
                                                random_state=seed).split(np.zeros(len(yr)), yr)):
        folds[va] = k
    members = cfg["models"]["ensemble_members"]
    facs = _member_factories(cfg, params, n_seeds)
    oof, full = {}, {}
    for j, m in enumerate(members):
        oof[m] = oof_margins_for(facs[m], Xr, yr, folds, seed + 10 * j)
        full[m] = facs[m](seed + 10 * j + 5).fit(Xr, yr)
    ens = LogitStackingEnsemble(facs, cfg["cv"]["n_splits"], seed, cfg["ensemble"]["nonnegative"],
                                cfg["ensemble"]["meta_l2"])
    ens.fit_from_oof(full, np.column_stack([oof[m] for m in members]), yr, folds)
    target = cfg["operating_point"]["target_sensitivity"]
    cal = {m: PlattCalibrator().fit(sigmoid(oof[m]), yr) for m in members}
    thr = {m: threshold_for_sensitivity(yr, cal[m].transform(sigmoid(oof[m])), target) for m in members}
    thr["stack"] = threshold_for_sensitivity(yr, ens.oof_proba_, target)
    proba = {m: cal[m].transform(full[m].predict_proba(Xt)) for m in members}
    proba["stack"] = ens.predict_proba(Xt)
    shap = {m: full[m].shap_logit(Xe, bg)[0] for m in members}
    shap["stack"] = sum(w * shap[m] for w, m in zip(ens.weights_, members))
    names = members + ["stack"]
    out = {"names": np.array(names), "proba": np.stack([proba[n] for n in names]).astype(np.float32),
           "thresholds": np.array([thr[n] for n in names]),
           "shap": np.stack([shap[n] for n in names]).astype(np.float32),
           "weights": np.r_[ens.intercept_, ens.weights_], "subsample_idx": idx}
    for year, Xx in ext.items():
        px = {m: cal[m].transform(full[m].predict_proba(Xx)) for m in members}
        px["stack"] = ens.predict_proba(Xx)
        out[f"proba_ext_{year}"] = np.stack([px[n] for n in names]).astype(np.float32)
    return out


def run_remedy_replicates(cfg, variant: str = "stable_seedavg", max_replicates: int | None = None) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    logger = get_logger("remedy", exp_dir(cfg, "logs"))
    rc = _rc(cfg)
    R = int(rc["n_replicates"])
    if R > int(cfg["multiplicity"]["n_replicates"]):
        raise ValueError("remedy.n_replicates cannot exceed multiplicity.n_replicates (paired design)")
    params = _selected(cfg)["params"]
    S = 1 if variant == "stable" else int(rc["n_seeds"])
    from .experiments import _background
    from .external import load_external

    Xd, yd, _ = load_split(cfg, "dev")
    Xt, _, _ = load_split(cfg, "test")
    bg = _background(cfg, Xd)
    rows = np.load(exp_dir(cfg, "multiplicity") / "explain_rows.npy")
    Xe = Xt.iloc[rows].reset_index(drop=True)
    ext_all = {y: X for y, (X, _, _) in load_external(cfg).items()}
    n_scored = int(cfg.get("external", {}).get("n_replicates_scored", 10**6))
    out = exp_dir(cfg, "remedy")
    done = 0
    for r in range(R):
        f = out / f"{variant}_{r:02d}.npz"
        if f.exists():
            try:
                with np.load(f) as z:
                    z["proba"].shape
                continue
            except Exception:
                f.unlink()
        if max_replicates is not None and done >= max_replicates:
            break
        with timer(logger, f"{variant} replicate {r} ({S} seed(s) per booster)"):
            res = _remedy_replicate(cfg, r, Xd, yd, Xt, Xe, bg, params, S,
                                    ext_all if r < n_scored else {})
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(f)
        done += 1


# ============================================================================== 4. analysis
def _load_runs(paths) -> dict:
    reps = [np.load(p) for p in paths]
    names = [str(n) for n in reps[0]["names"]]
    d = {"names": names, "P": np.stack([r["proba"] for r in reps]).astype(float),
         "T": np.stack([r["thresholds"] for r in reps]), "S": np.stack([r["shap"] for r in reps]).astype(float)}
    for key in reps[0].files:
        if key.startswith("proba_ext_") and all(key in r.files for r in reps):
            d[key] = np.stack([r[key] for r in reps]).astype(float)
    return d


def analyse_remedy(cfg) -> dict | None:
    rc = _rc(cfg)
    R = int(rc["n_replicates"])
    mult, rem = exp_dir(cfg, "multiplicity"), exp_dir(cfg, "remedy")
    runs = {"original": _load_runs([mult / f"replicate_{r:02d}.npz" for r in range(R)])}
    for v in VARIANTS:
        files = [rem / f"{v}_{r:02d}.npz" for r in range(R)]
        if all(f.exists() for f in files):
            runs[v] = _load_runs(files)
    if len(runs) == 1:
        return None
    _, yt, _ = load_split(cfg, "test")
    fr = load_json(exp_dir(cfg, "models") / "frozen.json")
    best = fr["best_single"]
    show = ["logreg", "xgboost", "lightgbm", "catboost", "stack"]
    rows, per = [], {}
    for run, d in runs.items():
        for n in show:
            if n not in d["names"]:
                continue
            j = d["names"].index(n)
            P, T = d["P"][:, j], d["T"][:, j]
            st = _decision_stats(P, T)
            ls = local_stability([d["S"][r, j] for r in range(len(P))], k=3)
            per[(run, n)] = {"sd": st["sd"], "ambiguous": st["ambiguous"].astype(float),
                             "spearman": ls["per_instance_spearman"]}
            rows.append({"run": run, "model": n, "n_replicates": len(P),
                         "auroc_mean": float(np.mean([auroc(yt, p) for p in P])),
                         "brier_mean": float(np.mean([brier(yt, p) for p in P])),
                         "cal_slope_mean": float(np.mean([calibration_slope(yt, p) for p in P])),
                         "pred_sd_mean": float(st["sd"].mean()), "ambiguity": float(st["ambiguous"].mean()),
                         "n_ambiguous": int(st["ambiguous"].sum()), "pairwise_flip_rate": st["pairwise_flip"],
                         "shap_spearman_mean": float(np.mean(ls["per_instance_spearman"]))})
            for key in d:
                if key.startswith("proba_ext_"):
                    se = _decision_stats(d[key][:, j], T)
                    rows[-1][f"ambiguity_{key[10:]}"] = float(se["ambiguous"].mean())
    summ = pd.DataFrame(rows)
    res_dir = resolve(cfg, "results_dir")
    summ.to_csv(res_dir / "remedy_summary.csv", index=False)
    B, master = int(cfg["evaluation"]["n_bootstrap"]), int(cfg["project"]["seed"])
    comps = {}
    for v in runs:
        if v == "original":
            continue
        for a, b in [((v, "stack"), ("original", "stack")), ((v, "stack"), ("original", best)),
                     ((v, best), ("original", best)), ((v, "stack"), (v, best)),
                     ((v, "stack"), ("original", "logreg"))]:
            if a not in per or b not in per:
                continue
            for k in ("sd", "ambiguous", "spearman"):
                comps[f"{a[0]} {a[1]} vs {b[0]} {b[1]}: {k}"] = bootstrap_mean_diff(
                    per[a][k], per[b][k], B=B, seed=derive_seed(master, "remedy", *a, *b, k))
    save_json(comps, res_dir / "remedy_comparisons.json")
    return {"summary": summ, "comparisons": comps}
