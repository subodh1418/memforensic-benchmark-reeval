"""Multi-seed development study (dev seeds 7, 8, 9 - disjoint from final seeds 42-46):
feature engineering and hybrid-selection variants with the MemTGF expert learner."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from features import HybridSelector, add_engineered
from memdgf_core import fit_lgbm, load_task, split_indices

task = sys.argv[1]
X, y, classes, info, y_fam, families, M = load_task(task)
XE = add_engineered(X)
rows = []
for seed in (7, 8, 9):
    tr, va, te = split_indices(y_fam, seed)

    def run(name, cols, Xsrc):
        m = fit_lgbm(Xsrc.iloc[tr][cols].to_numpy(np.float32), y[tr], Xsrc.iloc[va][cols].to_numpy(np.float32), y[va], len(classes), seed, 6)
        p = m.predict(Xsrc.iloc[te][cols].to_numpy(np.float32))
        rows.append({"seed": seed, "config": name, "n": len(cols), "acc": accuracy_score(y[te], p), "f1_macro": f1_score(y[te], p, average="macro")})
        print(f"[{task} s{seed}] {name:30s} n={len(cols):3d} acc={rows[-1]['acc']:.4f} f1m={rows[-1]['f1_macro']:.4f}", flush=True)

    run("raw", list(X.columns), X)
    run("raw+FE", list(XE.columns), XE)
    soft = HybridSelector(keep_frac=0.9, rho_max=0.98, min_hits=2, shadow_q=0.9, seed=seed, n_jobs=6).fit(XE.iloc[tr], y_fam[tr])
    run("HFS-soft(raw+FE)", soft.selected, XE)
    strict = HybridSelector(seed=seed, n_jobs=6).fit(XE.iloc[tr], y_fam[tr])
    run("HFS-strict(raw+FE)", strict.selected, XE)
df = pd.DataFrame(rows)
df.to_csv(f"../results/dev_feature_study2_{task}.csv", index=False)
print(df.groupby("config")[["n", "acc", "f1_macro"]].agg(["mean", "std"]).round(4).to_string())
