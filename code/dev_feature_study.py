"""Development study (dev seed 7, never the final seeds): feature engineering and hybrid feature selection.

Compares LightGBM (family-supervised for 4-class, as in MemTGF's experts) on:
 raw, raw+FE, HFS(raw), HFS(raw+FE), filter-only top-32 (ANOVA+MI), and reports dev-test metrics.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from features import HybridSelector, add_engineered
from memdgf_core import fit_lgbm, load_task, split_indices

task = sys.argv[1]
SEED = 7
X, y, classes, info, y_fam, families, M = load_task(task)
tr, va, te = split_indices(y_fam, SEED)
XE = add_engineered(X)
out = {}


def run(name, cols, Xsrc):
    t = time.time()
    m = fit_lgbm(Xsrc.iloc[tr][cols].to_numpy(np.float32), y[tr], Xsrc.iloc[va][cols].to_numpy(np.float32), y[va], len(classes), SEED, 6)
    p = m.predict(Xsrc.iloc[te][cols].to_numpy(np.float32))
    out[name] = {"n_features": len(cols), "acc": accuracy_score(y[te], p), "f1_macro": f1_score(y[te], p, average="macro"), "s": time.time() - t}
    print(f"[{task}] {name:34s} n={len(cols):3d} acc={out[name]['acc']:.4f} f1m={out[name]['f1_macro']:.4f}", flush=True)


run("raw", list(X.columns), X)
run("raw+FE", list(XE.columns), XE)
hs_raw = HybridSelector(seed=SEED, n_jobs=6).fit(X.iloc[tr], y_fam[tr])
run("HFS(raw)", hs_raw.selected, X)
hs_fe = HybridSelector(seed=SEED, n_jobs=6).fit(XE.iloc[tr], y_fam[tr])
run("HFS(raw+FE)", hs_fe.selected, XE)
filt = list(hs_fe.report.index[:32])
run("filter-only top-32 (raw+FE)", filt, XE)
filt_raw = list(hs_raw.report.index[:32])
run("filter-only top-32 (raw)", filt_raw, X)
json.dump({"results": out, "hfs_raw_fe_selected": hs_fe.selected, "dropped": [c for c in XE.columns if c not in hs_fe.selected]},
          open(f"../results/dev_feature_study_{task}.json", "w"), indent=2)
hs_fe.report.to_csv(f"../results/dev_hfs_report_{task}.csv")
