#!/usr/bin/env python
"""Step 17 - compact figures sized for an IEEE two-column paper (column 3.45 in, page 7.0 in).

Reads only saved results (no re-training):
    python scripts/17_ieee_figures.py --config configs/default.yaml --out paper/ieee/figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ehai.config import load_config, resolve  # noqa: E402
from ehai.data import FEATURES  # noqa: E402
from ehai.plotting import COLORS, GRID, INK, INK2, MARKERS, SURFACE, YEAR_COLORS  # noqa: E402

FULL, COL = 7.0, 3.45
SHORT = {"stack": "Stack", "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "logreg": "Logistic reg.", "bag": "Bag (4×CatBoost)", "random_forest": "Random forest",
         "decision_tree": "Decision tree"}
NICE = {"GenHlth": "General health (1-5)", "BMI": "Body-mass index", "Age": "Age band", "HighBP": "High blood pressure",
        "HighChol": "High cholesterol", "Income": "Income band", "Sex": "Sex (male)", "HeartDiseaseorAttack":
        "Heart disease / MI", "CholCheck": "Cholesterol check", "HvyAlcoholConsump": "Heavy alcohol use",
        "MentHlth": "Poor mental-health days", "DiffWalk": "Difficulty walking", "Education": "Education",
        "PhysHlth": "Poor physical-health days", "PhysActivity": "Physical activity", "Fruits": "Fruit daily",
        "Veggies": "Vegetables daily", "Smoker": "Smoker", "Stroke": "Stroke", "AnyHealthcare": "Health coverage",
        "NoDocbcCost": "Cost barrier to care"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.axisbelow": True, "savefig.dpi": 600, "pdf.fonttype": 42})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig_arbitrariness(cfg, out):
    res = resolve(cfg, "results_dir")
    mdir = resolve(cfg, "experiments_dir") / "multiplicity"
    ms = pd.read_csv(res / "multiplicity_summary.csv").set_index("model")
    order = ms.pred_sd_mean.sort_values().index.tolist()
    pi = np.load(mdir / "per_individual.npz")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.05, 1.1, 0.95]})
    # (a) per-person SD across 20 re-trainings
    for n in order:
        sd = np.sort(pi[f"sd_{n}"]) * 100
        a.plot(sd, np.linspace(0, 1, len(sd)), color=COLORS[n], lw=1.6 if n == "stack" else 1.0, label=SHORT[n])
    a.set(xlim=(0, 2.2), ylim=(0, 1.01), xlabel="SD of predicted risk across 20 re-trainings (pp)",
          ylabel="Cumulative share of test set")
    a.set_title("(a) Variability of individual risk", loc="left")
    a.legend(loc="lower right", handlelength=1.4)
    # (b) seeds only vs new samples, five re-trainings each
    sv = pd.read_csv(res / "multiplicity_seed_vs_sample.csv").set_index("model").loc[order]
    x = np.arange(len(order))
    b.bar(x - 0.19, sv.ambiguity_subsample * 100, 0.38, color="#184f95", label="New 80% sample + seeds")
    b.bar(x + 0.19, sv.ambiguity_seed_only * 100, 0.38, color="#86b6ef", label="Seeds only (same data)")
    b.set_xticks(x)
    b.set_xticklabels([SHORT[n].replace(" (4×CatBoost)", "") for n in order], rotation=30, ha="right")
    b.set_ylabel("Decisions flipping in ≥1 of 5\nre-trainings (%)")
    b.set_title("(b) Sources of arbitrariness", loc="left")
    b.set_ylim(0, max(sv.ambiguity_subsample) * 100 * 1.38)
    b.legend(loc="upper left")
    b.grid(axis="x", visible=False)
    # (c) by survey year, ten re-trainings
    xm = pd.read_csv(res / "external_multiplicity.csv")
    years = sorted(xm.year.unique())
    for n in order:
        s = xm[xm.model == n].set_index("year").loc[years]
        c.plot(range(len(years)), s.ambiguity * 100, marker=MARKERS[n], ms=3.5, color=COLORS[n], lw=1.0,
               label=SHORT[n])
    c.set_xticks(range(len(years)))
    c.set_xticklabels([f"{y}\n{'test' if y == 2015 else 'external'}" for y in years])
    c.set_xlim(-0.3, len(years) - 0.7)
    c.set_ylabel("Decisions flipping in ≥1 of 10\nre-trainings (%)")
    c.set_title("(c) Stability over survey years", loc="left")
    c.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_arbitrariness")


def fig_temporal(cfg, out):
    from ehai.external import load_external
    from ehai.metrics import reliability_curve
    from ehai.pipeline import exp_dir, final_proba, load_bundle

    res = resolve(cfg, "results_dir")
    em = pd.read_csv(res / "external_metrics.csv")
    rc = pd.read_csv(res / "external_recalibration.csv")
    b = load_bundle(cfg)
    preds = pd.read_csv(exp_dir(cfg, "predictions") / "test_predictions.csv.gz")
    curves = {2015: reliability_curve(preds.y.to_numpy(), preds["final_stack"].to_numpy(), 10)[:2]}
    for yr, (Xx, yx, _) in load_external(cfg).items():
        curves[yr] = reliability_curve(yx, final_proba(b, Xx)["stack"], 10)[:2]
    years = sorted(em.year.unique())
    ticks = [f"{y}\n{'test' if y == 2015 else 'external'}" for y in years]
    fig, (a, o, r) = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.1, 0.95, 0.95]})
    models = ["stack", "catboost", "logreg", "random_forest"]
    off = np.linspace(-0.24, 0.24, len(models))
    for k, n in enumerate(models):
        s = em[em.model == n].set_index("year").loc[years]
        xx = np.arange(len(years)) + off[k]
        a.errorbar(xx, s.auroc, yerr=[s.auroc - s.auroc_lo, s.auroc_hi - s.auroc], fmt=MARKERS[n], ms=3.5,
                   color=COLORS[n], elinewidth=0.8, capsize=0, label=SHORT[n])
    a.set_xticks(range(len(years)))
    a.set_xticklabels(ticks)
    a.set_ylabel("AUROC (95% CI)")
    a.set_title("(a) Discrimination", loc="left")
    a.legend(loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    a.grid(axis="x", visible=False)
    om = ["stack", "catboost", "logreg"]
    off2 = np.linspace(-0.18, 0.18, len(om))
    for k, n in enumerate(om):
        s = em[em.model == n].set_index("year").loc[years]
        xx = np.arange(len(years)) + off2[k]
        o.errorbar(xx, s.oe_ratio, yerr=[s.oe_ratio - s.oe_ratio_lo, s.oe_ratio_hi - s.oe_ratio], fmt=MARKERS[n],
                   ms=3.5, color=COLORS[n], elinewidth=0.8, capsize=0)
        u = rc[(rc.model == n) & (rc.recalibration == "intercept_update")].set_index("year")
        yy = [y for y in years if y in u.index]
        o.plot([years.index(y) + off2[k] for y in yy], u.loc[yy, "oe_ratio"], MARKERS[n], ms=4, mfc=SURFACE,
               mec=COLORS[n], mew=0.9, ls="none", label="after intercept update" if k == 0 else None)
    o.axhline(1, color=INK2, lw=0.6, ls="--")
    o.set_xticks(range(len(years)))
    o.set_xticklabels(ticks)
    o.set_ylabel("Observed / expected cases")
    o.set_title("(b) Calibration-in-the-large", loc="left")
    o.legend(loc="upper left", handletextpad=0.3)
    o.grid(axis="x", visible=False)
    lim = 0
    for y in years:
        mp, ob = curves[y]
        lim = max(lim, mp.max(), ob.max())
        r.plot(mp, ob, marker="o", ms=2.2, color=YEAR_COLORS.get(y, INK2), lw=1.0, label=str(y))
    lim = min(1.0, lim * 1.05)
    r.plot([0, lim], [0, lim], color=INK2, lw=0.6, ls="--")
    r.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk", ylabel="Observed proportion")
    r.set_title("(c) Reliability of the frozen stack", loc="left")
    r.legend(loc="upper left", title="Survey year", title_fontsize=6.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_temporal")


def fig_shap(cfg, out, k=10):
    sh = np.load(resolve(cfg, "experiments_dir") / "explain" / "shap_frozen.npz")
    phi, X = sh["phi_stack"], sh["X"]
    imp = np.abs(phi).mean(0)
    order = np.argsort(imp)[::-1][:k][::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        x = X[:, j]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        cc = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        ax.scatter(phi[:, j], row + rng.uniform(-0.3, 0.3, len(x)), c=cc, cmap=SEQ, s=2.2, lw=0, alpha=0.85,
                   rasterized=True)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NICE.get(FEATURES[j], FEATURES[j]) for j in order])
    ax.set_xlabel("Exact SHAP value of the stack (log-odds)")
    ax.grid(axis="y", visible=False)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ), ax=ax, pad=0.01, aspect=30)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Feature value", color=INK2)
    cb.outline.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, out, "fig_shap")


def fig_subgroups(cfg, out):
    res = resolve(cfg, "results_dir")
    f15 = pd.read_csv(res / "fairness_subgroups.csv")
    f15 = f15[f15.model == "stack"].assign(year=2015)
    fx = pd.read_csv(res / "external_subgroups.csv")
    fx = fx[fx.model == "stack"]
    d = pd.concat([f15, fx], ignore_index=True)
    d["group"] = d.group.astype(str).str.replace(">=", "≥")
    groups = [("Age band", "18-44"), ("Age band", "45-64"), ("Age band", "65+"),
              ("Income", "< $25k"), ("Income", "$25k-$50k"), ("Income", "≥ $50k")]
    names = ["Age 18-44", "Age 45-64", "Age 65+", "Income <$25k", "Income $25-50k", "Income ≥$50k"]
    years = [2015, 2021, 2023]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    for ax, col, ttl in [(axes[0], "auroc", "AUROC"), (axes[1], "tpr", "Sensitivity at threshold")]:
        for k, y in enumerate(years):
            vals = []
            for att, g in groups:
                row = d[(d.year == y) & (d.attribute == att) & (d.group == g)]
                vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
            ax.plot(vals, np.arange(len(groups)) + (k - 1) * 0.22, "o", ms=3, color=YEAR_COLORS[y],
                    label=str(y))
        ax.set_title(ttl, fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", handletextpad=0.2)
    fig.tight_layout(w_pad=0.5)
    save(fig, out, "fig_subgroups")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="paper/ieee/figures")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["_root"]) / args.out
    setup()
    jobs = {"arbitrariness": fig_arbitrariness, "temporal": fig_temporal, "shap": fig_shap, "subgroups": fig_subgroups}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        fn(cfg, out)
        print("wrote", out / f"fig_{name}.pdf")


if __name__ == "__main__":
    main()
