"""Aggregate 5-seed results, run significance tests, and produce paper tables/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / __import__("os").environ.get("MEMTGF_PAPER", "paper")
RES, FIG, TAB = ROOT / "results", PAPER_DIR / "figures", PAPER_DIR / "tables"
NAME = __import__("os").environ.get("MEMTGF_NAME", "MemTGF")  # display name of the proposed model
TASKS = ["binary", "4class", "16class"]
PANEL = {"binary": "(a) ", "4class": "(b) ", "16class": "(c) "}
TASK_LABEL = {"binary": "Binary", "4class": "4-class (category)", "16class": "16-class (family)"}
# reference palette, categorical slots 1-3 + neutral ink
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
FAMILY_COLOR = {"proposed": BLUE, "graph": ORANGE, "tabular": GRAY}
MAIN_MODELS = ["LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "MLP",
               "Soft-vote ensemble (LGBM+XGB+RF)", "kNN-Graph", "SGC (SSG)", "FeatureGCN (FIG)", "GraphSAGE (SSG)", "MemTGF"]
SOTA_MODELS = ["Farfoura2026-HAGEN", "Zakaria2025-SoftVote", "AlGhanem2025-MADANET", "Hussain2024-GNBiLSTM", "Mahdi2025-1DCNN", "Hadjila2024-ANOVADNN", "Cevallos2024-DNN"]
SOTA_REF = {"Farfoura2026-HAGEN": "farfoura2026", "Zakaria2025-SoftVote": "zakaria2025", "AlGhanem2025-MADANET": "alghanem2025", "Hussain2024-GNBiLSTM": "hussain2024",
            "Mahdi2025-1DCNN": "mahdi2025", "Hadjila2024-ANOVADNN": "hadjila2024", "Cevallos2024-DNN": "cevallos2024"}
DISPLAY = {
    "LogisticRegression": "Logistic regression", "DecisionTree": "Decision tree", "RandomForest": "Random forest", "ExtraTrees": "Extra trees",
    "Soft-vote ensemble (LGBM+XGB+RF)": "Soft-vote ensemble", "kNN-Graph": "$k$NN (SSG)", "SGC (SSG)": "SGC (SSG)",
    "FeatureGCN (FIG)": "FeatureGCN (FIG)", "GraphSAGE (SSG)": "GraphSAGE (SSG)", "MemTGF": NAME,
    "Farfoura2026-HAGEN": "HAGEN", "Zakaria2025-SoftVote": "GA soft voting", "AlGhanem2025-MADANET": "MAD-ANET", "Hussain2024-GNBiLSTM": "GN-BiLSTM",
    "Mahdi2025-1DCNN": "Pearson + 1D-CNN", "Hadjila2024-ANOVADNN": "ANOVA-DNN", "Cevallos2024-DNN": "SMOTE-DNN",
    "w/o LHG (task-level experts)": "w/o LHG (experts on task labels)", "LHG only (family LightGBM aggregated)": "LHG only (family LightGBM, aggregated)",
    "w/o FIG community experts": "w/o FIG community experts", "w/o SSG GraphSAGE expert": "w/o SSG GraphSAGE expert",
    "w/o meta-learner (mean fusion)": "w/o meta-learner (mean of experts)", "w/o flat fusion (graph path only)": "w/o flat head (graph path only)",
    "Alt: LightGBM + FIG diffusion/residual features": "Alt.: LightGBM + FIG diffusion/residual features",
    "Alt: LightGBM + SSG evidence features": "Alt.: LightGBM + SSG neighbourhood features",
}
if NAME != "MemTGF":  # v4+ wording: indicator groups instead of communities
    DISPLAY |= {"w/o LHG (task-level experts)": "w/o label hierarchy (experts on task labels)",
                "LHG only (family LightGBM aggregated)": "Label hierarchy only (family LightGBM, aggregated)",
                "w/o FIG community experts": "w/o indicator-group experts", "w/o SSG GraphSAGE expert": "w/o GraphSAGE expert",
                "w/o flat fusion (graph path only)": "w/o flat head (expert path only)"}
KEY = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc", "kappa", "roc_auc", "pr_auc",
       "malware_detection_rate", "false_positive_rate", "train_time_s", "infer_ms_per_sample"]

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_runs(task):
    files = sorted((RES / task).glob("seed_*/metrics.csv")) + sorted((RES / task).glob("seed_*/sota_reimpl_metrics.csv"))
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


def probs_of(seed_dir, name):
    key = name.replace("/", "_").replace(":", "")
    for fn in ("test_probs.npz", "sota_reimpl_probs.npz"):
        f = seed_dir / fn
        if f.exists():
            z = np.load(f)
            if key in z.files:
                return z["y_test"], z[key]
    return None, None


def mean_std(df):
    g = df.groupby(["model", "family"])[KEY]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy()
    for c in KEY:
        out[c + "_std"] = s[c]
    out["n_seeds"] = g.size()
    return out.reset_index()


def fmt(m, s, pct=True, bold=False):
    v = f"{100 * m:.2f}$\\pm${100 * s:.2f}" if pct else f"{m:.4f}$\\pm${s:.4f}"
    return f"\\textbf{{{v}}}" if bold else v


def significance(df, task):
    """Paired tests across seeds (same splits) + McNemar / bootstrap on seed 42."""
    rows = []
    piv = df.pivot_table(index="seed", columns="model", values=["accuracy", "f1_macro"])
    for b in [m for m in MAIN_MODELS + SOTA_MODELS if m != "MemTGF" and m in piv["f1_macro"].columns]:
        common = piv["f1_macro"][[b, "MemTGF"]].dropna().index
        if len(common) == 0:
            continue
        d_f1 = (piv["f1_macro"]["MemTGF"] - piv["f1_macro"][b]).loc[common]
        d_acc = (piv["accuracy"]["MemTGF"] - piv["accuracy"][b]).loc[common]
        t_p = stats.ttest_rel(piv["f1_macro"]["MemTGF"].loc[common], piv["f1_macro"][b].loc[common]).pvalue if len(common) > 1 and d_f1.std() > 0 else np.nan
        w_p = stats.wilcoxon(d_f1).pvalue if (d_f1 != 0).sum() >= 1 else np.nan
        # McNemar pooled over all seeds (independent test sets per seed are summed into one 2x2 table)
        n01 = n10 = 0
        boot = []
        for sd in sorted((RES / task).glob("seed_*")):
            y, pm = probs_of(sd, "MemTGF")
            _, pbb = probs_of(sd, b)
            if pm is None or pbb is None:
                continue
            a, c = pm.argmax(1) == y, pbb.argmax(1) == y
            n01 += int((a & ~c).sum())
            n10 += int((~a & c).sum())
            if sd.name == "seed_42":
                pa, pb = pm.argmax(1), pbb.argmax(1)
                rng = np.random.default_rng(0)
                for _ in range(1000):
                    i = rng.integers(0, len(y), len(y))
                    boot.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
        mc_p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        rows.append({"task": task, "baseline": b, "delta_f1_macro_mean": d_f1.mean(), "delta_acc_mean": d_acc.mean(),
                     "wins_f1": int((d_f1 > 0).sum()), "ties_f1": int((d_f1 == 0).sum()), "n_seeds": len(d_f1),
                     "paired_t_p": t_p, "wilcoxon_p": w_p, "mcnemar_memtgf_only_correct": n01, "mcnemar_baseline_only_correct": n10,
                     "mcnemar_exact_p": mc_p, "boot95_lo_seed42": np.percentile(boot, 2.5) if boot else np.nan,
                     "boot95_hi_seed42": np.percentile(boot, 97.5) if boot else np.nan})
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summaries, sigs = {}, []
    for task in TASKS:
        df = load_runs(task)
        if df is None:
            print(f"no results for {task}")
            continue
        summ = mean_std(df)
        summ.to_csv(RES / task / "summary_mean_std.csv", index=False)
        summaries[task] = (df, summ)
        sigs.append(significance(df, task))
        print(task, "seeds:", sorted(df.seed.unique()))
    sig = pd.concat(sigs)
    sig.to_csv(RES / "significance_tests.csv", index=False)

    # ------------------------------------------------------------ main table (LaTeX)
    for task, (df, s) in summaries.items():
        main = s[s.model.isin(MAIN_MODELS)].set_index("model").loc[[m for m in MAIN_MODELS if m in set(s.model)]]
        best = {c: main[c].max() for c in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc"]}
        lines = []
        for m, r in main.iterrows():
            name = DISPLAY.get(m, m)
            name = f"\\textbf{{{name} (ours)}}" if m == "MemTGF" else name
            # bold every value that equals the best one as printed, so that printed ties are treated alike
            cells = [fmt(r[c], r[c + "_std"], bold=f"{100 * r[c]:.2f}" == f"{100 * best[c]:.2f}") for c in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc"]]
            # the GraphSAGE expert is trained inside the MemHSE fit, which records the whole fit time for it
            train = "--$^{\\dagger}$" if m == "GraphSAGE (SSG)" else f"{r.train_time_s:.1f}"
            lines.append(f"{name} & {r.family.capitalize()} & " + " & ".join(cells) + f" & {train} & {1000 * r.infer_ms_per_sample:.1f} \\\\")
        (TAB / f"main_{task}.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------ compact main table (all tasks side by side)
    order = MAIN_MODELS[:-1] + SOTA_MODELS + ["MemTGF"]
    best = {(t, c): summaries[t][1][summaries[t][1].model.isin(order)][c].max() for t in summaries for c in ("accuracy", "f1_macro")}
    lines = []
    for m in order:
        cells = []
        for t in TASKS:
            s_ = summaries[t][1].set_index("model")
            for c in ("accuracy", "f1_macro"):
                r = s_.loc[m] if m in s_.index else None
                cells.append("--" if r is None else fmt(r[c], r[c + "_std"], bold=np.isclose(r[c], best[(t, c)])))
        fam = "Proposed" if m == "MemTGF" else ("Journal (re-impl.)" if m in SOTA_MODELS else summaries["4class"][1].set_index("model").loc[m, "family"].capitalize())
        name = r"\textbf{" + NAME + " (ours)}" if m == "MemTGF" else DISPLAY.get(m, m) + (r" \cite{" + SOTA_REF[m] + "}" if m in SOTA_MODELS else "")
        if m == SOTA_MODELS[0] or m == "MemTGF":
            lines.append(r"\midrule")
        lines.append(f"{name} & {fam} & " + " & ".join(cells) + r" \\")
    (TAB / "main_compact.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------ ablation table
    for task, (df, s) in summaries.items():
        full = s[s.model == "MemTGF"].iloc[0]
        ab = s[s.family.isin(["ablation", "alt-integration"])].copy()
        order = ["w/o LHG (task-level experts)", "LHG only (family LightGBM aggregated)", "w/o FIG community experts", "w/o SSG GraphSAGE expert",
                 "w/o meta-learner (mean fusion)", "w/o flat fusion (graph path only)",
                 "Alt: LightGBM + FIG diffusion/residual features", "Alt: LightGBM + SSG evidence features"]
        ab = ab.set_index("model").reindex([o for o in order if o in set(ab.model)])
        lines = [f"\\textbf{{{NAME} (full)}} & {fmt(full.accuracy, full.accuracy_std)} & {fmt(full.f1_macro, full.f1_macro_std)} & {fmt(full.mcc, full.mcc_std)} & -- & -- \\\\"]
        piv = df.pivot_table(index="seed", columns="model", values="f1_macro")
        for m, r in ab.iterrows():
            d = piv["MemTGF"] - piv[m]
            pv = stats.ttest_rel(piv["MemTGF"], piv[m]).pvalue if d.abs().sum() > 0 else np.nan
            ptxt = "--" if pd.isna(pv) else ("$<$0.001" if pv < 1e-3 else f"{pv:.3f}")
            lines.append(f"{DISPLAY.get(m, m)} & {fmt(r.accuracy, r.accuracy_std)} & {fmt(r.f1_macro, r.f1_macro_std)} & {fmt(r.mcc, r.mcc_std)} & {100 * (r.f1_macro - full.f1_macro):+.2f} & {ptxt} \\\\")
        (TAB / f"ablation_{task}.tex").write_text("\n".join(lines), encoding="utf-8")
        ab.reset_index().to_csv(RES / task / "ablation_summary.csv", index=False)

    # ------------------------------------------------------------ significance table
    lines = []
    for _, r in sig[sig.baseline.isin(["LightGBM", "XGBoost", "RandomForest", "Soft-vote ensemble (LGBM+XGB+RF)", "GraphSAGE (SSG)", "FeatureGCN (FIG)"])].iterrows():
        b = DISPLAY.get(r.baseline, r.baseline)
        p = lambda v: "--" if pd.isna(v) else ("$<$0.001" if v < 1e-3 else f"{v:.3f}")
        lines.append(f"{TASK_LABEL[r.task]} & {b} & {100 * r.delta_f1_macro_mean:+.2f} & {r.wins_f1}/{r.n_seeds} & {p(r.paired_t_p)} & "
                     f"{r.mcnemar_memtgf_only_correct}/{r.mcnemar_baseline_only_correct} & {p(r.mcnemar_exact_p)} & "
                     f"[{100 * r.boot95_lo_seed42:+.2f}, {100 * r.boot95_hi_seed42:+.2f}] \\\\")
    (TAB / "significance.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------ Fig: macro-F1 comparison
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=False)
    for ax, task in zip(axes, TASKS):
        if task not in summaries:
            continue
        s = summaries[task][1].set_index("model")
        ms = [m for m in MAIN_MODELS if m in s.index]
        v, e = s.loc[ms, "f1_macro"] * 100, s.loc[ms, "f1_macro_std"] * 100
        order = np.argsort(v.values)
        names = [DISPLAY.get(ms[i], ms[i]) for i in order]
        cols = [FAMILY_COLOR.get(s.loc[ms[i], "family"], GRAY) for i in order]
        ax.barh(range(len(ms)), v.values[order], xerr=e.values[order], color=cols, height=0.72, edgecolor="white", linewidth=1,
                error_kw=dict(lw=0.7, capsize=1.5, ecolor=MUTED))
        ax.set_yticks(range(len(ms)), names, fontsize=6.5)
        lo = max(0, v.min() - 5)
        ax.set_xlim(lo, min(100.5, v.max() + (100 - lo) * 0.08))
        ax.set_title(PANEL[task] + TASK_LABEL[task], fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("macro-F1 (%)")
        ax.grid(axis="x", alpha=0.25)
        top = order[-1]
        ax.text(v.values[top] + e.values[top], len(ms) - 1, f" {v.values[top]:.2f}", va="center", fontsize=6.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRAY)]
    fig.legend(handles, [f"{NAME} (proposed)", "graph baselines", "tabular baselines"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "comparison_f1_macro")

    # ------------------------------------------------------------ Fig: ablation deltas
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, task in zip(axes, ["4class", "16class"]):
        p = RES / task / "ablation_summary.csv"
        if not p.exists():
            continue
        ab = pd.read_csv(p)
        full = summaries[task][1].set_index("model").loc["MemTGF", "f1_macro"]
        d = (ab.f1_macro - full) * 100
        names = [m.replace("Alt: LightGBM + ", "Alt: +").replace(" (task-level experts)", "").replace(" (family LightGBM aggregated)", "")
                 .replace(" (graph path only)", "").replace(" (mean fusion)", "").replace(" diffusion/residual features", " diff/res feats")
                 .replace(" evidence features", " evidence feats") for m in ab.model]
        if NAME != "MemTGF":  # v4+ wording
            short_v4 = {"w/o LHG (task-level experts)": "w/o label hierarchy", "LHG only (family LightGBM aggregated)": "label hierarchy only",
                        "w/o FIG community experts": "w/o group experts", "w/o SSG GraphSAGE expert": "w/o GraphSAGE expert",
                        "w/o meta-learner (mean fusion)": "w/o meta-learner", "w/o flat fusion (graph path only)": "w/o flat head",
                        "Alt: LightGBM + FIG diffusion/residual features": "LightGBM + FIG features",
                        "Alt: LightGBM + SSG evidence features": "LightGBM + SSG features"}
            names = [short_v4.get(m, n) for m, n in zip(ab.model, names)]
        ax.barh(range(len(d)), d, color=[ORANGE if x < 0 else AQUA for x in d], height=0.65, edgecolor="white")
        span = max(abs(d.min()), 0.5)
        for i, x in enumerate(d):  # value labels, so that near-zero effects remain visible
            ax.text(x - 0.02 * span if x < 0 else x + 0.02 * span, i, f"{x:+.2f}" if abs(x) >= 0.005 else "0.00",
                    ha="right" if x < 0 else "left", va="center", fontsize=5.8, color=INK)
        ax.set_xlim(-1.18 * span, 0.12 * span + max(d.max(), 0))
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_yticks(range(len(d)), names, fontsize=6.2)
        ax.invert_yaxis()
        ax.set_title({"4class": "(a) ", "16class": "(b) "}[task] + TASK_LABEL[task], fontsize=8.5, loc="left")
        ax.set_xlabel(f"Δ macro-F1 vs. full {NAME} (pp)")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "ablation_delta")

    # ------------------------------------------------------------ Fig: robustness + sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    style = {"MemTGF": (BLUE, "-o"), "LightGBM": (GRAY, "--s"), "XGBoost": ("#6b6a65", "--^"), "GraphSAGE (SSG)": (ORANGE, ":d")}
    if NAME != "MemTGF":  # v4+: the bagged forests discussed in the text are shown as well
        style |= {"RandomForest": (AQUA, "-.v"), "ExtraTrees": ("#6250d6", "-.x")}
    nice = {"MemTGF": NAME, "RandomForest": "Random forest", "ExtraTrees": "Extra trees"}
    for ax, task in zip(axes, TASKS):
        p = RES / task / "robustness_masking.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        for m, (c, st) in style.items():
            d = r[r.model == m]
            ax.plot(100 * d.mask_rate, 100 * d.f1_macro, st, color=c, lw=1.6, ms=3.5, label=nice.get(m, m))
        ax.set_title(PANEL[task] + TASK_LABEL[task], fontsize=8.5, loc="left")
        ax.set_xlabel("masked feature values (%)")
        ax.set_ylabel("macro-F1 (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "robustness_masking")

    # ------------------------------------------------------------ confusion matrices (seed 42, MemTGF)
    for task in TASKS:
        f = RES / task / "seed_42" / "test_probs.npz"
        info = RES / task / "dataset_info.json"
        if not f.exists():
            continue
        import json

        classes = json.load(open(info))["classes"]
        z = np.load(f)
        cm = confusion_matrix(z["y_test"], z["MemTGF"].argmax(1))
        bounds = []
        if task == "16class":  # group families by category so within-category confusion forms blocks
            order = ["Benign", "Ako", "Conti", "Maze", "Pysa", "Shade", "180solutions", "CWS", "Gator", "TIBS", "Transponder",
                     "Emotet", "Reconyc", "Refroso", "Scar", "Zeus"]
            if sorted(order) == sorted(classes):
                idx = [classes.index(c) for c in order]
                cm, classes, bounds = cm[np.ix_(idx, idx)], order, [0.5, 5.5, 10.5]
        cmn = cm / cm.sum(1, keepdims=True)
        size = {"binary": 2.4, "4class": 3.0, "16class": 5.6}[task]
        fig, ax = plt.subplots(figsize=(size + 0.8, size))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes, fontsize=6.5)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=5.5 if task == "16class" else 7,
                            color="white" if cmn[i, j] > 0.55 else INK)
        for b in bounds:
            ax.axhline(b, color=INK, lw=0.6)
            ax.axvline(b, color=INK, lw=0.6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title({"binary": "", "4class": "(a) Category", "16class": "(b) Family"}[task], fontsize=8.5, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
        save(fig, f"confusion_{task}")

    # per-family F1: MemTGF vs LightGBM (16-class)
    f = RES / "16class" / "seed_42" / "test_probs.npz"
    if f.exists():
        import json

        classes = json.load(open(RES / "16class" / "dataset_info.json"))["classes"]
        F1 = {}
        for m in ("MemTGF", "LightGBM"):
            per = []
            for sf in sorted((RES / "16class").glob("seed_*/test_probs.npz")):
                z = np.load(sf)
                per.append(f1_score(z["y_test"], z[m].argmax(1), average=None, labels=range(len(classes))))
            F1[m] = np.mean(per, 0)
        order = np.argsort(F1["MemTGF"])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        yy = np.arange(len(classes))
        ax.hlines(yy, F1["LightGBM"][order] * 100, F1["MemTGF"][order] * 100, color=GRAY, lw=1.2)
        ax.plot(F1["LightGBM"][order] * 100, yy, "s", color=GRAY, ms=4, label="LightGBM")
        ax.plot(F1["MemTGF"][order] * 100, yy, "o", color=BLUE, ms=4.5, label=NAME)
        ax.set_yticks(yy, np.array(classes)[order], fontsize=6.5)
        ax.set_xlabel("F1 (%)" if NAME != "MemTGF" else "per-family F1 (%) — mean of 5 seeds")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        save(fig, "per_family_f1")

    # ------------------------------------------------------------ SOTA re-implementation table (same protocol)
    lines = []
    for m in SOTA_MODELS + ["MemTGF"]:
        cells = []
        for task in TASKS:
            s_ = summaries[task][1].set_index("model") if task in summaries else None
            if s_ is None or m not in s_.index:
                cells += ["--", "--"]
                continue
            r = s_.loc[m]
            cells += [fmt(r.accuracy, r.accuracy_std), fmt(r.f1_macro, r.f1_macro_std)]
        name = r"\textbf{" + NAME + " (ours)}" if m == "MemTGF" else DISPLAY[m] + r" \cite{" + SOTA_REF[m] + "}"
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    (TAB / "sota_reimpl.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------ LOFO (unseen family)
    p = RES / "lofo" / "lofo_results.csv"
    if p.exists():
        lo = pd.read_csv(p)
        lo.groupby("model")[["detection_rate", "category_acc", "benign_fpr"]].mean().to_csv(RES / "lofo" / "lofo_summary.csv")
        show = ["MemTGF", "LightGBM 4-class", "LightGBM binary detector", "GraphSAGE (SSG) expert"]
        cols = {"MemTGF": BLUE, "LightGBM 4-class": GRAY, "LightGBM binary detector": "#6b6a65", "GraphSAGE (SSG) expert": ORANGE}
        cat_of = lo.drop_duplicates("held_out_family").set_index("held_out_family")["true_category"]
        fams = sorted(lo.held_out_family.unique(), key=lambda f: (cat_of[f], f))
        labels = [f"{f} ({cat_of[f][0]})" for f in fams]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
        for ax, metric, ttl, lim in ((axes[0], "detection_rate", "detection of the unseen family (%)", (95, 100.3)),
                                     (axes[1], "category_acc", "correct category of the unseen family (%)", (0, 100))):
            pv = lo.pivot_table(index="held_out_family", columns="model", values=metric).reindex(fams)
            ms = [m for m in show if m in pv.columns and pv[m].notna().any()]
            h = 0.8 / len(ms)
            for k, m in enumerate(ms):
                ax.barh(np.arange(len(fams)) + k * h - 0.4 + h / 2, 100 * pv[m].values, height=h, color=cols[m], label=NAME if m == "MemTGF" else m, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(fams)), labels, fontsize=6.5)
            ax.set_xlabel(ttl, fontsize=7.5)
            ax.set_xlim(*lim)
            ax.grid(axis="x", alpha=0.25)
            ax.invert_yaxis()
        axes[1].axvline(100 / 3, color=INK, lw=0.8, ls="--")
        axes[1].text(100 / 3 + 1, len(fams) - 0.6, "chance (3 categories)", fontsize=6, color=MUTED)
        hd, lb = axes[0].get_legend_handles_labels()
        fig.legend(hd, lb, loc="upper center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "lofo_unseen_family")
        lines = []

        def c(g, m, k):
            return f"{100 * g.loc[m, k]:.1f}" if m in g.index and pd.notna(g.loc[m, k]) else "--"

        for fam in fams:
            g = lo[lo.held_out_family == fam].set_index("model")
            lines.append(f"{fam} & {g['true_category'].iloc[0]} & {c(g, 'MemTGF', 'detection_rate')} & {c(g, 'LightGBM 4-class', 'detection_rate')} & "
                         f"{c(g, 'LightGBM binary detector', 'detection_rate')} & {c(g, 'MemTGF', 'category_acc')} & {c(g, 'LightGBM 4-class', 'category_acc')} & "
                         f"{c(g, 'GraphSAGE (SSG) expert', 'category_acc')}" + r" \\")
        mean = lo.groupby("model")[["detection_rate", "category_acc", "benign_fpr"]].mean()
        lines.append(r"\midrule")
        lines.append(r"\textbf{Mean} & & " + f"{c(mean, 'MemTGF', 'detection_rate')} & {c(mean, 'LightGBM 4-class', 'detection_rate')} & "
                     f"{c(mean, 'LightGBM binary detector', 'detection_rate')} & {c(mean, 'MemTGF', 'category_acc')} & {c(mean, 'LightGBM 4-class', 'category_acc')} & "
                     f"{c(mean, 'GraphSAGE (SSG) expert', 'category_acc')}" + r" \\")
        fpr = lambda m: f"{100 * mean.loc[m, 'benign_fpr']:.3f}"
        lines.append(r"Benign FPR & & " + f"{fpr('MemTGF')} & {fpr('LightGBM 4-class')} & {fpr('LightGBM binary detector')} & & &" + r" \\")
        (TAB / "lofo.tex").write_text("\n".join(lines), encoding="utf-8")

    # sensitivity table
    rows = []
    for task in TASKS:
        p = RES / task / "sensitivity.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["task"] = task
            rows.append(d)
    if rows:
        sens = pd.concat(rows)
        sens.to_csv(RES / "sensitivity_all.csv", index=False)
        lines = []
        default = []  # reference row: the default model on the same split (seed 42)
        for t in TASKS:
            d42 = summaries[t][0] if t in summaries else None
            d42 = d42[(d42.seed == 42) & (d42.model == "MemTGF")] if d42 is not None else []
            default.append(f"{100 * d42.accuracy.iloc[0]:.2f} / {100 * d42.f1_macro.iloc[0]:.2f}" if len(d42) else "--")
        lines += ["Default ($k=15$, $\\tau=0.5$, resolution 1.0) & " + " & ".join(default) + " \\\\", "\\midrule"]
        for (param, val), g in sens.groupby(["param", "value"], sort=False):
            cells = []
            for t in TASKS:
                gg = g[g.task == t]
                cells.append(f"{100 * gg.accuracy.iloc[0]:.2f} / {100 * gg.f1_macro.iloc[0]:.2f}" if len(gg) else "--")
            pname = {"k": "$k$", "tau": "$\\tau$", "resolution": "Louvain resolution"}.get(param, param)
            lines.append(f"{pname} $=$ {val:g} & " + " & ".join(cells) + " \\\\")
        (TAB / "sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
