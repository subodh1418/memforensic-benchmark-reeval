"""Accuracy-push study on the development seed (7) - legitimate levers only, all selected on validation:
 A. hyper-parameter search for the family-level GBDT expert
 B. a more diverse out-of-fold stack (tuned LGBM, XGBoost-GPU, ExtraTrees, MLP, kNN, FIG community experts)
 C. per-class log-prior offsets tuned on validation for macro-F1
Reports dev-test accuracy / macro-F1 next to the current MemTGF-style stack.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from memdgf_core import GraphViews, MemDGFConfig, load_task, split_indices
from memtgf import fig_communities

task = sys.argv[1]
SEED, NJ = 7, 6
X, y, classes, info, y_fam, families, M = load_task(task)
tr, va, te = split_indices(y_fam, SEED)
views = GraphViews(MemDGFConfig(seed=SEED, n_jobs=NJ)).fit(X.iloc[tr], y_fam[tr], families)
B = {"tr": views.blocks(), "va": views.blocks(X.iloc[va]), "te": views.blocks(X.iloc[te])}
yf = {"tr": y_fam[tr], "va": y_fam[va]}
K = len(families)
log = {}


def rep(name, p_fam):
    p = p_fam @ M
    a, f = accuracy_score(y[te], p.argmax(1)), f1_score(y[te], p.argmax(1), average="macro")
    log[name] = {"acc": a, "f1_macro": f}
    print(f"[{task}] {name:48s} acc={a:.4f} f1m={f:.4f}", flush=True)


# ---------------------------------------------------------------- A. HPO of the GBDT expert
rng = np.random.default_rng(SEED)
grid = []
for _ in range(16):
    grid.append(dict(num_leaves=int(rng.choice([15, 31, 63, 127])), learning_rate=float(rng.choice([0.02, 0.03, 0.05])),
                     min_child_samples=int(rng.choice([5, 10, 20, 50])), colsample_bytree=float(rng.choice([0.4, 0.6, 0.8, 1.0])),
                     subsample=float(rng.choice([0.7, 0.8, 1.0])), reg_lambda=float(rng.choice([0.0, 1.0, 5.0])),
                     min_split_gain=float(rng.choice([0.0, 0.01]))))
best, best_ll = None, np.inf
for g in grid:
    m = LGBMClassifier(n_estimators=4000, subsample_freq=1, random_state=SEED, n_jobs=NJ, verbose=-1, **g)
    m.fit(B["tr"]["raw"], yf["tr"], eval_set=[(B["va"]["raw"], yf["va"])], callbacks=[early_stopping(100, verbose=False)])
    ll = log_loss(yf["va"], m.predict_proba(B["va"]["raw"]), labels=range(K))
    if ll < best_ll:
        best, best_ll, best_m = dict(g, n_estimators=int(m.best_iteration_)), ll, m
print(f"[{task}] best HPO config {best} val_logloss={best_ll:.4f}", flush=True)
rep("A. tuned family LGBM (single)", best_m.predict_proba(B["te"]["raw"]))

# ---------------------------------------------------------------- B. diverse OOF stack
comms = fig_communities(views.fig.A, SEED)


def lgb_tuned():
    return LGBMClassifier(subsample_freq=1, random_state=SEED, n_jobs=NJ, verbose=-1, **best)


experts = {
    "lgb_tuned": (lgb_tuned, "raw", None),
    "xgb_gpu": (lambda: XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7,
                                      tree_method="hist", device="cuda", random_state=SEED), "raw", None),
    "extratrees": (lambda: ExtraTreesClassifier(500, n_jobs=NJ, random_state=SEED), "raw", None),
    "mlp": (lambda: MLPClassifier((256, 128), early_stopping=True, max_iter=300, random_state=SEED), "std", None),
    "knn": (lambda: KNeighborsClassifier(15, weights="distance", n_jobs=NJ), "std", None),
}
for j, c in enumerate(comms):
    experts[f"comm{j}"] = (lgb_tuned, "raw", c)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(B["tr"]["raw"], yf["tr"]))
oof, pv, pt = {}, {}, {}
for name, (mk, view, cols) in experts.items():
    t = time.time()
    sel = (lambda A: A) if cols is None else (lambda A, c=cols: A[:, c])
    Atr, Ava, Ate = sel(B["tr"][view]), sel(B["va"][view]), sel(B["te"][view])
    o = np.zeros((len(tr), K), np.float32)
    for f, q in folds:
        o[q] = mk().fit(Atr[f], yf["tr"][f]).predict_proba(Atr[q])
    oof[name] = o
    full = mk().fit(Atr, yf["tr"])
    pv[name], pt[name] = full.predict_proba(Ava), full.predict_proba(Ate)
    print(f"[{task}]   expert {name:12s} done ({time.time() - t:.0f}s)", flush=True)
    rep(f"   expert {name}", pt[name])


def meta(names):
    Z = lambda P: np.hstack([np.log(np.clip(P[n], 1e-6, 1)) for n in names])
    m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       random_state=SEED, n_jobs=NJ, verbose=-1).fit(Z(oof), yf["tr"])
    return m.predict_proba(Z(pv)), m.predict_proba(Z(pt))


base_names = ["lgb_tuned"] + [n for n in experts if n.startswith("comm")]
pv_base, pt_base = meta(base_names)
rep("B0. stack: tuned LGBM + FIG communities", pt_base)
pv_div, pt_div = meta(list(experts))
rep("B1. stack: + XGB + ET + MLP + kNN (diverse)", pt_div)


# ---------------------------------------------------------------- C. macro-F1 log-prior offsets (task level)
def offsets(pv_task, y_val, iters=3):
    b = np.zeros(pv_task.shape[1])
    L = np.log(np.clip(pv_task, 1e-9, 1))
    for _ in range(iters):
        for c in range(len(b)):
            cand = np.linspace(-1.5, 1.5, 31)
            sc = []
            for v in cand:
                bb = b.copy()
                bb[c] = v
                sc.append(f1_score(y_val, (L + bb).argmax(1), average="macro"))
            b[c] = cand[int(np.argmax(sc))]
    return b


for nm, (pvv, ptt) in {"B1": (pv_div, pt_div), "B0": (pv_base, pt_base)}.items():
    b = offsets(pvv @ M, y[va])
    p = np.exp(np.log(np.clip(ptt @ M, 1e-9, 1)) + b)
    a, f = accuracy_score(y[te], p.argmax(1)), f1_score(y[te], p.argmax(1), average="macro")
    log[f"C. {nm} + macro-F1 prior offsets"] = {"acc": a, "f1_macro": f}
    print(f"[{task}] C. {nm} + macro-F1 prior offsets{'':18s} acc={a:.4f} f1m={f:.4f}", flush=True)
json.dump({"best_hpo": best, "results": log}, open(f"../results/dev_push_study_{task}.json", "w"), indent=2)
