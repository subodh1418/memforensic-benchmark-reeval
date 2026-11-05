"""Version-4 tables: shortcut protocols (tables/leaky.tex), leave-one-category-out (tables/loco.tex),
and a printout of the plugin-grouping check on the development seeds."""
import os
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / os.environ.get("MEMTGF_PAPER", "paper") / "tables"
NAME = os.environ.get("MEMTGF_NAME", "MemTGF")
f = lambda m, s: f"{100 * m:.2f}$\\pm${100 * s:.2f}"

# shortcut protocols
p = RES / "review" / "leaky_protocols.csv"
if p.exists():
    d = pd.read_csv(p)
    g = d.groupby(["protocol", "task"])[["accuracy", "f1_macro"]].agg(["mean", "std"])
    label = {"clean": "Clean (this study)", "raw": "Raw file, random 80/20", "smote-in": "Raw, SMOTE on training part",
             "smote-pre": "Raw, SMOTE before the split"}
    lines = []
    for k in ("clean", "raw", "smote-in", "smote-pre"):
        if (k, "4class") not in g.index:
            continue
        cells = []
        for t in ("4class", "16class"):
            r = g.loc[(k, t)] if (k, t) in g.index else None
            cells += ["--", "--"] if r is None else [f(r[(c, "mean")], r[(c, "std")]) for c in ("accuracy", "f1_macro")]
        lines.append(f"{label[k]} & " + " & ".join(cells) + r" \\")
    (TAB / "leaky.tex").write_text("\n".join(lines), encoding="utf-8")
    print((g * 100).round(2))

# leave-one-category-out
p = RES / "loco" / "loco_results.csv"
if p.exists():
    d = pd.read_csv(p)
    order = [("MemTGF", NAME), ("MemTGF plugin partition", f"{NAME}, plugin groups"), ("LightGBM 3-class", "LightGBM (categories)"),
             ("LightGBM binary detector", "LightGBM (binary)"), ("RandomForest 3-class", "Random forest"), ("kNN-Graph 3-class", "$k$NN (SSG)")]
    cats = ["Ransomware", "Spyware", "Trojan"]
    lines = []
    for key, lab in order:
        cells = []
        for c in cats:
            r = d[(d.model == key) & (d.held_out_category == c)]
            cells += ["--", "--"] if r.empty else [f"{100 * r.detection_rate.iloc[0]:.2f}", f"{100 * r.benign_fpr.iloc[0]:.3f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (TAB / "loco.tex").write_text("\n".join(lines), encoding="utf-8")
    print(d.pivot_table(index="model", columns="held_out_category", values=["detection_rate", "benign_fpr"]).round(4))

# plugin grouping on development seeds (default Louvain = candidate P0 in plus_metrics.csv)
for task in ("4class", "16class"):
    rows = []
    for s in (7, 8, 9):
        pp = RES / task / f"dev_seed_{s}" / "v4_plugin_metrics.csv"
        if not pp.exists():
            continue
        base = pd.read_csv(RES / task / f"dev_seed_{s}" / "plus_metrics.csv")
        base = base[base.candidate.str.startswith("P0")].iloc[0]
        plug = pd.read_csv(pp).iloc[0]
        rows.append({"seed": s, "louvain_acc": base.accuracy, "louvain_f1": base.f1_macro, "plugin_acc": plug.accuracy, "plugin_f1": plug.f1_macro})
    if rows:
        r = pd.DataFrame(rows)
        r["d_f1"] = (r.plugin_f1 - r.louvain_f1) * 100
        p_val = stats.ttest_rel(r.plugin_f1, r.louvain_f1).pvalue if len(r) > 1 else float("nan")
        print(f"\n[dev seeds] {task}\n", (r * [1, 100, 100, 100, 100, 1]).round(2).to_string(index=False), f"\nmean dF1={r.d_f1.mean():+.2f}  p={p_val:.3f}")
