"""Builds the reviewer-control table (M1, M3, m2) for the paper: <paper>/tables/controls.tex and results/review/controls_summary.csv."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / os.environ.get("MEMTGF_PAPER", "paper") / "tables"
NAME = os.environ.get("MEMTGF_NAME", "MemTGF")
ROWS = [("MemTGF", f"{NAME}, Louvain groups (default)"),
        ("MemTGF random partition (Louvain sizes)", "Random groups, same sizes"),
        ("MemTGF plugin partition (no graph)", "Volatility-plugin groups (no graph)"),
        ("MemTGF Louvain resolution 1.5", "Louvain groups, resolution 1.5"),
        ("No-graph stacking (LGBM+XGB+RF+ET)", "No-structure stack"),
        ("Soft-vote ensemble (LGBM+XGB+RF)", "Soft-vote ensemble")]

out, lines = [], []
for task in ("4class", "16class"):
    frames = [pd.read_csv(f) for f in glob.glob(str(RES / task / "seed_*" / "metrics.csv"))]
    frames += [pd.read_csv(f) for f in glob.glob(str(RES / task / "seed_*" / "controls_metrics.csv"))]
    df = pd.concat(frames)
    piv = {m: df.pivot_table(index="seed", columns="model", values=m) for m in ("accuracy", "f1_macro")}
    for key, label in ROWS:
        if key not in piv["f1_macro"]:
            continue
        common = piv["f1_macro"][[key, "MemTGF"]].dropna().index
        a, f = piv["accuracy"].loc[common, key], piv["f1_macro"].loc[common, key]
        d = (piv["f1_macro"].loc[common, "MemTGF"] - f) * 100
        p = stats.ttest_rel(piv["f1_macro"].loc[common, "MemTGF"], f).pvalue if key != "MemTGF" and len(common) > 1 else np.nan
        out.append({"task": task, "variant": label, "n_seeds": len(common), "acc": a.mean() * 100, "acc_sd": a.std() * 100,
                    "f1": f.mean() * 100, "f1_sd": f.std() * 100, "memtgf_minus_variant_f1": d.mean(), "p": p, "memtgf_wins": int((d > 0).sum())})
s = pd.DataFrame(out)
(RES / "review").mkdir(exist_ok=True)
s.to_csv(RES / "review" / "controls_summary.csv", index=False)
print(s.round(3).to_string(index=False))
for task, title in (("4class", "Category task"), ("16class", "Family task")):
    lines.append(r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
    for r in s[s.task == task].itertuples():
        ptxt = "--" if pd.isna(r.p) else ("$<$0.001" if r.p < 1e-3 else f"{r.p:.3f}")
        dtxt = "--" if r.variant == ROWS[0][1] else f"{-r.memtgf_minus_variant_f1:+.2f}"
        lines.append(f"{r.variant} & {r.acc:.2f}$\\pm${r.acc_sd:.2f} & {r.f1:.2f}$\\pm${r.f1_sd:.2f} & {dtxt} & {ptxt} & {"--" if r.variant == ROWS[0][1] else f"{r.memtgf_wins}/{r.n_seeds}"} \\\\")
    lines.append(r"\midrule")
(TAB / "controls.tex").write_text("\n".join(lines[:-1]), encoding="utf-8")
