"""Design study 3 (dev seed 7): distribution-consistent (cross-fitted) SSG evidence."""
from __future__ import annotations

import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from memdgf_core import GraphViews, MemDGFConfig, load_task, split_indices

task = sys.argv[1]
SEED = 7
X, y, classes, _ = load_task(task)
tr, va, te = split_indices(y, SEED)
C = len(classes)
fit_idx = np.concatenate([tr, va])
views = GraphViews(MemDGFConfig(seed=SEED, n_jobs=14)).fit(X.iloc[fit_idx], y[fit_idx], classes)
Btr, Bte = views.blocks(), views.blocks(X.iloc[te])
ytr, yte = y[fit_idx], y[te]
ssg = views.ssg

# cross-fitted test evidence: average over the same 5 fold-reference sets used for OOF
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(views.X_std_tr, ytr))
ev_te_cf = np.mean([ssg._evidence(views.X_std_tr[f], ytr[f], Bte["std"]) for f, _ in folds], 0)
Bte_cf = dict(Bte, ssg=ev_te_cf)


def rep(name, p):
    print(f"{name:45s} acc={accuracy_score(yte, p.argmax(1)):.4f} f1m={f1_score(yte, p.argmax(1), average='macro'):.4f}", flush=True)


def lgb(keys, Bt, n=600):
    m = LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=14, verbose=-1)
    m.fit(np.hstack([Btr[k] for k in keys]), ytr)
    return m.predict_proba(np.hstack([Bt[k] for k in keys]))


rep("LGBM raw", lgb(("raw",), Bte))
rep("LGBM raw+FIG", lgb(("raw", "diff", "res"), Bte))
rep("LGBM raw+SSG (full-ref test)", lgb(("raw", "ssg"), Bte))
rep("LGBM raw+SSG (cross-fitted test)", lgb(("raw", "ssg"), Bte_cf))
rep("LGBM raw+FIG+SSG (cross-fitted)", lgb(("raw", "diff", "res", "ssg"), Bte_cf))
rep("LGBM raw n=1000", lgb(("raw",), Bte, 1000))
