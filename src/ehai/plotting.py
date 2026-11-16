"""Publication figures generated directly from saved experiment outputs.

Colour follows the entity (each model keeps one colour in every figure); categorical slots come
from a CVD-validated palette in fixed order; magnitude uses a single-hue sequential ramp.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import FEATURES  # noqa: E402
from .models import DISPLAY  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
COLORS = {  # fixed entity -> colour mapping (validated categorical order)
    "stack": "#2a78d6", "lightgbm": "#eb6834", "xgboost": "#1baf7a", "catboost": "#eda100",
    "logreg": "#e87ba4", "random_forest": "#008300", "decision_tree": "#6250d6", "bag": "#e34948",
}
MARKERS = {"stack": "o", "lightgbm": "s", "xgboost": "^", "catboost": "D", "logreg": "v",
           "random_forest": "P", "decision_tree": "X", "bag": "*"}
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"  # diverging pair: red raises, blue lowers risk
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.axisbelow": True,
    })


def label(n: str) -> str:
    return DISPLAY.get(n, n)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- data overview
def fig_prevalence(df: pd.DataFrame, target: str, out: Path):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    bmi_cat = pd.cut(df["BMI"], [0, 18.5, 25, 30, 35, 40, 100], right=False,
                     labels=["<18.5", "18.5-25", "25-30", "30-35", "35-40", ">=40"])
    specs = [("Age group (code)", df["Age"]), ("Self-rated health\n(1 excellent - 5 poor)", df["GenHlth"]),
             ("BMI (kg/m$^2$)", bmi_cat), ("Income (code)", df["Income"])]
    for ax, (ttl, col) in zip(axes, specs):
        prev = df.groupby(col, observed=True)[target].mean()
        ax.bar(range(len(prev)), prev.values * 100, color=COLORS["stack"], width=0.8, edgecolor=SURFACE, lw=1)
        ax.set_xticks(range(len(prev)))
        ax.set_xticklabels([str(i) for i in prev.index], rotation=0 if len(prev) < 9 else 90, fontsize=7)
        ax.set_title(ttl, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Diabetes prevalence (%)")
    fig.tight_layout()
    save(fig, out, "fig01_prevalence_by_predictor")


# ----------------------------------------------------------------------------- discrimination
def fig_roc_pr(y, preds: dict, out: Path, highlight=("stack",)):
    from sklearn.metrics import precision_recall_curve, roc_curve

    setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for n, p in preds.items():
        lw = 2.0 if n in highlight else 1.0
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=COLORS[n], lw=lw, label=label(n))
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=COLORS[n], lw=lw, label=label(n))
    a1.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
    a2.axhline(np.mean(y), ls="--", color=INK2, lw=0.8)
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves (test set)")
    a2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall curves (test set)")
    a2.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig02_roc_pr")


def fig_reliability(y, variants: dict, out: Path, models: list[str], n_bins: int = 15):
    from .metrics import reliability_curve

    setup()
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, models):
        lim = 0
        for v, ls in (("none", ":"), ("platt", "-")):
            if v not in variants[n]:
                continue
            mp, ob, _ = reliability_curve(y, variants[n][v], n_bins)
            lim = max(lim, mp.max(), ob.max())
            ax.plot(mp, ob, ls=ls, marker=MARKERS[n], ms=3.5, color=COLORS[n],
                    label=("native (stacked logistic layer)" if n == "stack" else "uncalibrated") if v == "none"
                    else "Platt (OOF)")
        lim = min(1.0, lim * 1.05)
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        ax.set(xlim=(0, lim), ylim=(0, lim), title=label(n), xlabel="Mean predicted probability")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Observed proportion")
    fig.tight_layout()
    save(fig, out, "fig03_reliability")


def fig_dca(dca: pd.DataFrame, out: Path, models: list[str]):
    setup()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(dca.threshold, dca.treat_all, color=INK2, lw=1, ls="--", label="Treat all")
    ax.plot(dca.threshold, dca.treat_none, color=INK2, lw=1, ls=":", label="Treat none")
    for n in models:
        ax.plot(dca.threshold, dca[n], color=COLORS[n], lw=2.0 if n == "stack" else 1.2, label=label(n))
    ax.set_ylim(-0.01, max(0.15, dca[models].to_numpy().max() * 1.1))
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision-curve analysis (test set)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig04_decision_curve")


# ----------------------------------------------------------------------------- explanations
def fig_beeswarm(phi: np.ndarray, X: np.ndarray, out: Path, name: str, title: str, max_features: int = 15):
    setup()
    imp = np.abs(phi).mean(0)
    order = np.argsort(imp)[::-1][:max_features][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        x = X[:, j]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        c = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.uniform(-0.32, 0.32, len(x))
        ax.scatter(phi[:, j], row + jitter, c=c, cmap=SEQ, s=5, lw=0, alpha=0.8, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEATURES[j] for j in order])
    ax.set_xlabel("SHAP value (contribution to log-odds of diabetes)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=SEQ)
    cb = fig.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Feature value", color=INK2)
    fig.tight_layout()
    save(fig, out, name)


def fig_importance_compare(imp: pd.DataFrame, out: Path, models: list[str]):
    setup()
    order = imp["stack"].sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y = np.arange(len(order))
    for n in models:
        v = imp.loc[order, n] / imp[n].sum()
        ax.plot(v, y, ls="", marker=MARKERS[n], ms=5 if n == "stack" else 4, color=COLORS[n], label=label(n),
                mec=SURFACE, mew=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Share of total mean |SHAP| (log-odds)")
    ax.set_title("Global feature importance by model (test sample)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig06_importance_by_model")


def fig_dependence(phis: dict, X: np.ndarray, out: Path, features: list[str]):
    setup()
    fig, axes = plt.subplots(1, len(features), figsize=(3.0 * len(features), 2.9))
    rng = np.random.default_rng(1)
    for ax, f in zip(np.atleast_1d(axes), features):
        j = FEATURES.index(f)
        x = X[:, j]
        jit = rng.uniform(-0.25, 0.25, len(x)) if len(np.unique(x)) < 15 else 0
        for n, ph in phis.items():
            ax.scatter(x + jit, ph[:, j], s=4, color=COLORS[n], alpha=0.35, lw=0, label=label(n), rasterized=True)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set(xlabel=f, title=f)
    np.atleast_1d(axes)[0].set_ylabel("SHAP value (log-odds)")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(phis), fontsize=7.5, markerscale=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "fig07_dependence")


def fig_waterfalls(phi: np.ndarray, base: float, X: np.ndarray, idx: list[int], out: Path, top: int = 8):
    from .utils import sigmoid

    setup()
    fig, axes = plt.subplots(1, len(idx), figsize=(3.6 * len(idx), 3.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ph = phi[i]
        order = np.argsort(np.abs(ph))[::-1]
        keep, rest = order[:top], order[top:]
        vals = list(ph[keep]) + [ph[rest].sum()]
        names = [f"{FEATURES[j]} = {X[i, j]:g}" for j in keep] + [f"{len(rest)} other features"]
        ypos = np.arange(len(vals))[::-1]
        ax.barh(ypos, vals, color=[RISK_UP if v > 0 else RISK_DOWN for v in vals],
                height=0.7, edgecolor=SURFACE)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color=INK2, lw=0.8)
        tot = base + ph.sum()
        ax.set_title(f"Predicted risk {sigmoid(tot) * 100:.1f}% (base {sigmoid(base) * 100:.1f}%)", fontsize=8.5)
        ax.set_xlabel("SHAP (log-odds): red raises, blue lowers risk")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig08_individual_explanations")


# ----------------------------------------------------------------------------- multiplicity
def fig_multiplicity(per_ind: dict, models: list[str], out: Path):
    setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for n in models:
        sd = np.sort(per_ind[n]) * 100
        a1.plot(sd, np.linspace(0, 1, len(sd)), color=COLORS[n], lw=2.0 if n == "stack" else 1.3, label=label(n))
    a1.set(xlabel="SD of predicted risk across re-trainings (percentage points)",
           ylabel="Cumulative share of test individuals", title="Individual-level arbitrariness")
    a1.set_xlim(0, np.percentile(np.concatenate([per_ind[n] for n in models]) * 100, 99.5))
    a1.legend(fontsize=7.5, loc="lower right")
    return fig, a2


def fig_multiplicity_finish(fig, ax, summary: pd.DataFrame, models: list[str], out: Path):
    s = summary.set_index("model").loc[models]
    x = np.arange(len(models))
    ax.bar(x, s["ambiguity"] * 100, color=[COLORS[m] for m in models], width=0.65, edgecolor=SURFACE)
    for xi, v in zip(x, s["ambiguity"] * 100):
        ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in models], rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Individuals whose screening decision flips (%)")
    ax.set_title("Decision ambiguity across re-trainings")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig09_multiplicity")


def fig_explanation_stability(stab: pd.DataFrame, local: dict, models: list[str], out: Path):
    setup()
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    data = [local[m] for m in models]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2},
                    capprops={"color": INK2})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK2)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels([label(m) for m in models])
    ax.set_xlabel("Per-individual Spearman correlation of SHAP vectors\nbetween independent re-trainings")
    ax.set_title("Stability of individual explanations")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig10_explanation_stability")


# ----------------------------------------------------------------------------- robustness
def fig_robustness(agg: pd.DataFrame, models: list[str], out: Path):
    setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
    mc = agg[agg.scenario == "mcar"]
    for n in models:
        s = mc[mc.model == n].sort_values("level")
        a1.plot(s.level * 100, s.auroc, marker=MARKERS[n], ms=4, color=COLORS[n],
                lw=2.0 if n == "stack" else 1.2, label=label(n))
    a1.set(xlabel="Share of predictor values missing at test time (%)", ylabel="AUROC",
           title="Random (MCAR) missingness, median-imputed")
    a1.legend(fontsize=7)
    scen = ["income_missing", "ses_missing", "lifestyle_missing", "history_missing", "bmi_missing",
            "genhlth_missing", "bmi_noise", "bmi_underreport", "days_noise", "genhlth_noise"]
    nice = {"income_missing": "Income missing", "ses_missing": "Income + education missing",
            "lifestyle_missing": "Lifestyle items missing", "history_missing": "Clinical history missing",
            "bmi_missing": "BMI missing", "genhlth_missing": "Self-rated health missing",
            "bmi_noise": "BMI noise (largest SD)", "bmi_underreport": "BMI under-reported by 1",
            "days_noise": "Health-days noise", "genhlth_noise": "Self-rated health +/-1"}
    sub = agg[agg.scenario.isin(scen)]
    sub = sub[(sub.scenario != "bmi_noise") | (sub.level == sub[sub.scenario == "bmi_noise"].level.max())]
    w = 0.8 / len(models)
    for k, n in enumerate(models):
        v = [sub[(sub.scenario == s) & (sub.model == n)].delta_auroc.mean() for s in scen]
        a2.barh(np.arange(len(scen)) + k * w - 0.4 + w / 2, v, height=w, color=COLORS[n], label=label(n))
    a2.set_yticks(np.arange(len(scen)))
    a2.set_yticklabels([nice[s] for s in scen], fontsize=7.5)
    a2.invert_yaxis()
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set(xlabel="Change in AUROC vs. clean test data", title="Structured missingness and measurement noise")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig11_robustness")


# ----------------------------------------------------------------------------- fairness
def fig_fairness(tab: pd.DataFrame, models: list[str], out: Path):
    setup()
    metrics = [("auroc", "AUROC"), ("tpr", "Sensitivity at screening threshold"), ("oe_ratio", "Observed / expected")]
    groups = tab[tab.model == models[0]][["attribute", "group"]].drop_duplicates().to_numpy().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.33 * len(groups) * len(models) / 2 + 1.6), sharey=True)
    ylabels = [f"{a}: {g}".replace("$", r"\$") for a, g in groups]  # literal dollar signs, not mathtext
    off = np.linspace(-0.25, 0.25, len(models))
    for ax, (met, ttl) in zip(axes, metrics):
        for k, n in enumerate(models):
            s = tab[tab.model == n].set_index(["attribute", "group"])
            for i, (a, g) in enumerate(groups):
                r = s.loc[(a, g)]
                ax.errorbar(r[met], i + off[k], xerr=[[r[met] - r[f"{met}_lo"]], [r[f"{met}_hi"] - r[met]]],
                            fmt=MARKERS[n], ms=4, color=COLORS[n], ecolor=COLORS[n], elinewidth=1, capsize=0,
                            label=label(n) if i == 0 else None)
        if met == "oe_ratio":
            ax.axvline(1, color=INK2, lw=0.8, ls="--")
        ax.set_title(ttl, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(ylabels, fontsize=7.5)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig12_subgroups")


def fig_risk_coverage(curves: dict, out: Path):
    setup()
    sty = {"margin": (COLORS["stack"], "-"), "disagreement": (COLORS["lightgbm"], "-"),
           "margin+disagreement": (COLORS["xgboost"], "-"), "retraining_sd": (COLORS["catboost"], "-"),
           "random": (INK2, "--")}
    nice = {"margin": "Distance to threshold", "disagreement": "Member disagreement",
            "margin+disagreement": "Distance + disagreement", "retraining_sd": "Re-training SD",
            "random": "Random deferral"}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k, (cov, risk) in curves.items():
        c, ls = sty.get(k, (INK, "-"))
        m = cov >= 0.5
        ax.plot(cov[m] * 100, risk[m], color=c, ls=ls, lw=1.6, label=nice.get(k, k))
    ax.invert_xaxis()
    ax.set(xlabel="Coverage: share of individuals decided automatically (%)",
           ylabel="Misclassification cost per 1,000 decided", title="Risk-coverage curves (stacked ensemble)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig13_risk_coverage")


# ----------------------------------------------------------------------------- temporal validation
YEAR_COLORS = {2015: "#86b6ef", 2021: "#2a78d6", 2023: "#0d366b"}   # one-hue ordinal ramp (older = lighter)


def fig_external(metrics: pd.DataFrame, curves: dict, out: Path, models: list[str],
                 recal: pd.DataFrame | None = None):
    """(a) AUROC with 95% CI by survey year; (b) observed/expected ratio of the frozen models, and after an
    intercept update fitted on half of each external year; (c) reliability of the frozen stack by year."""
    setup()
    fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"width_ratios": [1.15, 0.85, 1]})
    years = sorted(metrics.year.unique())
    ticks = [f"{y}\n(development test)" if y == 2015 else f"{y}\n(external)" for y in years]
    off = np.linspace(-0.27, 0.27, len(models))
    for k, n in enumerate(models):
        s = metrics[metrics.model == n].set_index("year").loc[years]
        x = np.arange(len(years)) + off[k]
        a1.errorbar(x, s.auroc, yerr=[s.auroc - s.auroc_lo, s.auroc_hi - s.auroc], fmt=MARKERS[n], ms=5,
                    color=COLORS[n], ecolor=COLORS[n], elinewidth=1, capsize=0, label=label(n))
    a1.set_xticks(range(len(years)))
    a1.set_xticklabels(ticks)
    a1.set_ylabel("AUROC (95% CI)")
    a1.set_title("(a) Discrimination by survey year")
    a1.legend(fontsize=7.5, loc="lower left")
    a1.grid(axis="x", visible=False)
    # (b) calibration-in-the-large
    oe_models = [n for n in models if metrics[metrics.model == n].oe_ratio_lo.notna().all()]
    off2 = np.linspace(-0.2, 0.2, max(len(oe_models), 1))
    for k, n in enumerate(oe_models):
        s = metrics[metrics.model == n].set_index("year").loc[years]
        x = np.arange(len(years)) + off2[k]
        a3.errorbar(x, s.oe_ratio, yerr=[s.oe_ratio - s.oe_ratio_lo, s.oe_ratio_hi - s.oe_ratio], fmt=MARKERS[n],
                    ms=5, color=COLORS[n], ecolor=COLORS[n], elinewidth=1, capsize=0,
                    label=label(n))
        if recal is not None:
            r = recal[(recal.model == n) & (recal.recalibration == "intercept_update")].set_index("year")
            yy = [y for y in years if y in r.index]
            if yy:
                a3.plot([years.index(y) + off2[k] for y in yy], r.loc[yy, "oe_ratio"], MARKERS[n], ms=6,
                        mfc=SURFACE, mec=COLORS[n], mew=1.3, ls="none",
                        label="Open markers: after intercept update" if k == 0 else None)
    a3.axhline(1.0, color=INK2, lw=0.8, ls="--")
    lo_, hi_ = a3.get_ylim()
    a3.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    a3.set_xticks(range(len(years)))
    a3.set_xticklabels(ticks)
    a3.set_ylabel("Observed / expected cases (95% CI)")
    a3.set_title("(b) Calibration-in-the-large")
    a3.legend(fontsize=7, loc="upper left")
    a3.grid(axis="x", visible=False)
    # (c) reliability of the frozen stack
    lim = 0
    for y in years:
        mp, ob = curves[y]
        lim = max(lim, mp.max(), ob.max())
        a2.plot(mp, ob, marker="o", ms=3.5, color=YEAR_COLORS.get(y, INK2), lw=1.6, label=str(y))
    lim = min(1.0, lim * 1.05)
    a2.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
    a2.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability", ylabel="Observed proportion",
           title="(c) Reliability of the frozen stacked ensemble")
    a2.legend(fontsize=7.5, title="Survey year", title_fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig14_temporal_validation")


def fig_external_arbitrariness(df: pd.DataFrame, models: list[str], out: Path):
    """Decision ambiguity and mean SD of predicted risk across re-trainings, by survey year."""
    setup()
    years = sorted(df.year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    w = 0.8 / len(years)
    for ax, col, scale, ylab, ttl in [
            (axes[0], "ambiguity", 100, "Decisions that flip in at least\none re-training (%)", "(a) Decision ambiguity"),
            (axes[1], "pred_sd_mean", 100, "Mean SD of predicted risk\nacross re-trainings (pp)", "(b) Variability of individual risk")]:
        for k, y in enumerate(years):
            s = df[df.year == y].set_index("model").loc[models]
            x = np.arange(len(models)) + k * w - 0.4 + w / 2
            ax.bar(x, s[col] * scale, width=w, color=YEAR_COLORS.get(y, INK2), edgecolor=SURFACE, lw=1,
                   label=f"{y}" + (" (development test)" if y == 2015 else " (external)"))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([label(m) for m in models], fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"Arbitrariness of the re-trained pipelines by survey year ({int(df.n_replicates.iloc[0])} re-trainings)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig15_arbitrariness_by_year")


def fig_flow(build: dict, n_dev: int, n_test: int, pos_dev: int, pos_test: int, out: Path):
    """Participant flow from the raw CDC files to the analysed samples (TRIPOD+AI item 20a).
    ``build`` is data/external/build_report.json; per year it needs raw_rows, clean_rows, prevalence."""
    from matplotlib.patches import FancyBboxPatch

    setup()
    years = sorted(int(y) for y in build)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 3 * len(years))
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc=SURFACE, ec=INK2, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    for k, y in enumerate(years):
        r = build[str(y)]
        cx = 1.5 + 3 * k
        raw, clean = int(r["raw_rows"]), int(r["clean_rows"])
        pos = int(round(r["prevalence"] * clean))
        role = "development and internal test" if y == 2015 else "temporal validation"
        box(cx, 9.2, 2.6, 0.9, f"BRFSS {y}\n{role}", fc=YEAR_COLORS.get(y, GRID) + "33", bold=True)
        box(cx, 7.6, 2.6, 1.0, f"Raw CDC records\nn = {raw:,}")
        arrow(cx, 8.75, cx, 8.1)
        unasked = r.get("unavailable_items") or []
        why = "missing, don't know or refused\non any of the 22 items"
        if unasked:
            names = {"Fruits": "fruit", "Veggies": "vegetable"}
            items = " and ".join(names.get(u, u) for u in unasked)
            why = f"missing, don't know or refused\non any asked item ({items}\nintake not asked; imputed later)"
        box(cx, 5.75, 2.6, 1.3, f"Excluded n = {raw - clean:,}\n{why}", fc="#f0efec")
        arrow(cx, 7.1, cx, 6.4)
        txt = f"Analysed n = {clean:,}\n{pos:,} with diabetes ({100 * r['prevalence']:.1f}%)"
        if y == 2015 and r.get("validation_vs_published", {}).get("identical"):
            txt += "\nidentical to the published extract"
        box(cx, 3.95, 2.6, 1.2, txt)
        arrow(cx, 5.1, cx, 4.55)
        if y == 2015:
            box(cx - 0.72, 1.6, 1.3, 1.25, f"Development\n(80%)\nn = {n_dev:,}\n{pos_dev:,} cases", fc=GRID)
            box(cx + 0.72, 1.6, 1.3, 1.25, f"Test (20%)\nn = {n_test:,}\n{pos_test:,} cases", fc=GRID)
            arrow(cx, 3.35, cx - 0.72, 2.25)
            arrow(cx, 3.35, cx + 0.72, 2.25)
        else:
            box(cx, 1.6, 2.6, 1.0, "Frozen 2015 models applied\nunchanged (all respondents)", fc=GRID)
            arrow(cx, 3.35, cx, 2.1)
    fig.tight_layout()
    save(fig, out, "figS2_participant_flow")
