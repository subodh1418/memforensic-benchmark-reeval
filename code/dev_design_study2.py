"""Design study 2 (dev seed 7): graph-community experts + out-of-fold stacking."""
from __future__ import annotations

import sys

import networkx as nx
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from memdgf_core import GraphViews, MemDGFConfig, load_task, split_indices

task = sys.argv[1]
SEED = 7
X, y, classes, _ = load_task(task)
tr, va, te = split_indices(y, SEED)
C = len(classes)
views = GraphViews(MemDGFConfig(seed=SEED, n_jobs=14)).fit(X.iloc[tr], y[tr], classes)
Btr, Bva, Bte = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
# development: merge train+val for OOF stacking, evaluate on dev test
Rtr = np.vstack([Btr["raw"], Bva["raw"]]); Str = np.vstack([Btr["std"], Bva["std"]]); ytr = np.concatenate([y[tr], y[va]])
Rte, Ste, yte = Bte["raw"], Bte["std"], y[te]

G = nx.from_numpy_array(views.fig.A)
comms = [sorted(c) for c in nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=SEED)]
cols = views.prep.columns
for c in comms:
    print("community:", [cols[i] for i in c])


def rep(name, p):
    print(f"{name:45s} acc={accuracy_score(yte, p.argmax(1)):.4f} f1m={f1_score(yte, p.argmax(1), average='macro'):.4f}", flush=True)


def lgb(n=600):
    return LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=14, verbose=-1)


experts = {"full_lgbm": (lambda: lgb(), "raw", None), "full_et": (lambda: ExtraTreesClassifier(400, n_jobs=14, random_state=SEED), "raw", None),
           "knn_graph": (lambda: KNeighborsClassifier(15, weights="distance", n_jobs=14), "std", None)}
for j, c in enumerate(comms):
    experts[f"comm{j}"] = (lambda: lgb(400), "raw", c)

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof, test_p = {}, {}
for name, (mk, view, idx) in experts.items():
    A_tr = Rtr if view == "raw" else Str
    A_te = Rte if view == "raw" else Ste
    if idx is not None:
        A_tr, A_te = A_tr[:, idx], A_te[:, idx]
    o = np.zeros((len(ytr), C))
    for f, q in skf.split(A_tr, ytr):
        o[q] = mk().fit(A_tr[f], ytr[f]).predict_proba(A_tr[q])
    oof[name] = o
    test_p[name] = mk().fit(A_tr, ytr).predict_proba(A_te)
    rep(name, test_p[name])


def meta(names):
    Z = lambda P: np.hstack([np.log(np.clip(P[n], 1e-6, 1)) for n in names])
    m = LogisticRegression(max_iter=3000, C=1.0).fit(Z(oof), ytr)
    return m.predict_proba(Z(test_p))


rep("stack full_lgbm+et+knn", meta(["full_lgbm", "full_et", "knn_graph"]))
rep("stack communities only", meta([n for n in experts if n.startswith("comm")]))
rep("stack communities+full_lgbm", meta(["full_lgbm"] + [n for n in experts if n.startswith("comm")]))
rep("stack all", meta(list(experts)))
mz = lambda P: np.hstack([P[n] for n in experts])
m2 = lgb(300).fit(mz(oof), ytr)
rep("LGBM-meta all", m2.predict_proba(mz(test_p)))
