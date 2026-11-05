"""Design-development study on a held-out development seed (7), never on the final seeds 42-46."""
from __future__ import annotations

import sys
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, lgbm_params, load_task, split_indices, stack

task = sys.argv[1]
DEV_SEED = 7
X, y, classes, _ = load_task(task)
tr, va, te = split_indices(y, DEV_SEED)
C = len(classes)
views = GraphViews(MemDGFConfig(seed=DEV_SEED, n_jobs=14)).fit(X.iloc[tr], y[tr], classes)
Btr, Bva, Bte = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
ytr, yva, yte = y[tr], y[va], y[te]


def rep(name, p):
    print(f"{name:45s} acc={accuracy_score(yte, p.argmax(1)):.4f} f1m={f1_score(yte, p.argmax(1), average='macro'):.4f}", flush=True)


def gb(keys, Btr=Btr, Bva=Bva, Bte=Bte, **kw):
    params = lgbm_params(C, DEV_SEED, 14) | kw
    m = LGBMClassifier(**params)
    from lightgbm import early_stopping
    m.fit(stack(Btr, keys), ytr, eval_set=[(stack(Bva, keys), yva)], callbacks=[early_stopping(100, verbose=False)])
    return m.predict_proba(stack(Bte, keys)), m


p_raw, m_raw = gb(("raw",))
rep("LGBM raw", p_raw)
rep("LGBM raw+FIG", gb(("raw", "diff", "res"))[0])
rep("LGBM raw+FIG res only", gb(("raw", "res"))[0])
rep("LGBM raw+SSG(std)", gb(("raw", "ssg"))[0])
rep("LGBM raw nl31 lr.03", gb(("raw",), num_leaves=31, learning_rate=0.03)[0])
rep("LGBM raw nl127", gb(("raw",), num_leaves=127)[0])


# SSG evidence in an importance-weighted metric space (weights from train-only LGBM gain)
def evidence_space(W, name):
    Ztr, Zva, Zte = Btr["std"] * W, Bva["std"] * W, Bte["std"] * W
    ev = lambda R, yR, Q: _ev(R, yR, Q)
    oof = np.zeros((len(ytr), 2 * C + 1), np.float32)
    for f, q in StratifiedKFold(5, shuffle=True, random_state=DEV_SEED).split(Ztr, ytr):
        oof[q] = ev(Ztr[f], ytr[f], Ztr[q])
    B2tr, B2va, B2te = dict(Btr, ev=oof), dict(Bva, ev=ev(Ztr, ytr, Zva)), dict(Bte, ev=ev(Ztr, ytr, Zte))
    rep(f"kNN vote only ({name})", B2te["ev"][:, :C])
    rep(f"LGBM raw+ev({name})", gb(("raw", "ev"), B2tr, B2va, B2te)[0])
    rep(f"LGBM raw+FIG+ev({name})", gb(("raw", "diff", "res", "ev"), B2tr, B2va, B2te)[0])


def _ev(R, yR, Q, k=15):
    d, i = NearestNeighbors(n_neighbors=k).fit(R).kneighbors(Q)
    w = 1 / (d + 1e-6)
    vote = np.stack([(w * (yR[i] == c)).sum(1) for c in range(C)], 1) / w.sum(1, keepdims=True)
    cd = np.stack([np.log1p(NearestNeighbors(n_neighbors=1).fit(R[yR == c]).kneighbors(Q)[0][:, 0]) for c in range(C)], 1)
    return np.hstack([vote, cd - cd.min(1, keepdims=True), np.log1p(d.mean(1, keepdims=True))]).astype(np.float32)


gain = m_raw.booster_.feature_importance("gain")
evidence_space(np.sqrt(gain / gain.sum()).astype(np.float32), "gain-weighted")

# Probability-level fusion with other learners
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

et = ExtraTreesClassifier(500, n_jobs=14, random_state=DEV_SEED).fit(Btr["raw"], ytr)
rep("ExtraTrees", et.predict_proba(Bte["raw"]))
rep("LGBM+ET avg", (p_raw + et.predict_proba(Bte["raw"])) / 2)
