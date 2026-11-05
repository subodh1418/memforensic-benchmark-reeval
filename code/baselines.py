"""Tabular and graph baselines compared against MemDGF."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from memdgf_core import DEVICE, fit_lgbm


class SkModel:
    """Wraps a fitted estimator with the input view it expects ('raw' or 'std')."""

    def __init__(self, est, view):
        self.est, self.view = est, view

    def proba(self, blocks):
        return self.est.predict_proba(blocks[self.view])


def tabular_baselines(B_tr, y_tr, B_va, y_va, n_classes, seed, n_jobs):
    """Yield (name, family, fitted SkModel)."""
    yield "LogisticRegression", "tabular", SkModel(LogisticRegression(max_iter=3000, C=1.0).fit(B_tr["std"], y_tr), "std")
    yield "DecisionTree", "tabular", SkModel(DecisionTreeClassifier(random_state=seed).fit(B_tr["raw"], y_tr), "raw")
    yield "RandomForest", "tabular", SkModel(
        RandomForestClassifier(500, n_jobs=n_jobs, random_state=seed).fit(B_tr["raw"], y_tr), "raw"
    )
    yield "ExtraTrees", "tabular", SkModel(
        ExtraTreesClassifier(500, n_jobs=n_jobs, random_state=seed).fit(B_tr["raw"], y_tr), "raw"
    )
    xgb = XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7,
        tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu", early_stopping_rounds=100, random_state=seed, n_jobs=n_jobs, eval_metric="mlogloss" if n_classes > 2 else "logloss",
    )
    xgb.fit(B_tr["raw"], y_tr, eval_set=[(B_va["raw"], y_va)], verbose=False)
    yield "XGBoost", "tabular", SkModel(xgb, "raw")
    yield "LightGBM", "tabular", SkModel(fit_lgbm(B_tr["raw"], y_tr, B_va["raw"], y_va, n_classes, seed, n_jobs), "raw")
    yield "MLP", "tabular", SkModel(
        MLPClassifier((256, 128), early_stopping=True, max_iter=300, random_state=seed).fit(B_tr["std"], y_tr), "std"
    )
    yield "kNN-Graph", "graph", SkModel(KNeighborsClassifier(15, weights="distance", n_jobs=n_jobs).fit(B_tr["std"], y_tr), "std")


# ----------------------------------------------------------------- SGC on the SSG
class SGCModel:
    """Simplified Graph Convolution: 2-hop mean propagation over the SSG + softmax regression."""

    def __init__(self, views):
        self.views = views
        Xs = views.X_std_tr
        self.Mtr = _sp(views.ssg.train_nbr, len(Xs))
        self.S1_tr = 0.5 * (Xs + self.Mtr @ Xs)
        self.S2_tr = 0.5 * (self.S1_tr + self.Mtr @ self.S1_tr)

    def features(self, std=None):
        if std is None:
            return np.hstack([self.views.X_std_tr, self.S1_tr, self.S2_tr])
        nbr, _ = self.views.ssg.neighbours(std)
        Me = _sp(nbr, len(self.views.X_std_tr))
        s1 = 0.5 * (std + Me @ self.views.X_std_tr)
        s2 = 0.5 * (s1 + Me @ self.S1_tr)
        return np.hstack([std, s1, s2])

    def fit(self, y_tr):
        self.lr = LogisticRegression(max_iter=3000).fit(self.features(), y_tr)
        return self

    def proba(self, blocks):
        return self.lr.predict_proba(self.features(blocks["std"]))


def _sp(nbr, n_ref):
    from scipy import sparse

    n, k = nbr.shape
    return sparse.csr_matrix((np.full(n * k, 1.0 / k, dtype=np.float32), (np.repeat(np.arange(n), k), nbr.ravel())), shape=(n, n_ref))


# ---------------------------------------------------------- FeatureGCN on the FIG
class FeatureGCNNet(nn.Module):
    """Each memory dump is a graph over the FIG; node signal = feature value."""

    def __init__(self, Ahat, n_classes, d=32, dropout=0.2):
        super().__init__()
        n = Ahat.shape[0]
        self.register_buffer("A", torch.tensor(Ahat))
        self.emb = nn.Parameter(torch.randn(n, d) * 0.1)
        self.val = nn.Parameter(torch.randn(n, d) * 0.1)
        self.w1, self.w2 = nn.Linear(d, d), nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        h = x.unsqueeze(-1) * self.val + self.emb
        h = F.relu(self.A @ self.w1(h))
        h = F.relu(self.A @ self.w2(h)) + h
        return self.head(torch.cat([h.mean(1), h.max(1).values], 1))


class FeatureGCN:
    def __init__(self, Ahat, n_classes, seed):
        torch.manual_seed(seed)
        self.net = FeatureGCNNet(Ahat, n_classes).to(DEVICE)
        self.seed = seed

    def fit(self, X_tr, y_tr, X_va, y_va, epochs=60, patience=8, bs=512):
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(X_tr, device=DEVICE), torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
        xv, yv = torch.tensor(X_va, device=DEVICE), torch.tensor(y_va, dtype=torch.long, device=DEVICE)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            self.net.train()
            for b in torch.randperm(len(xt), generator=g).to(DEVICE).split(bs):
                opt.zero_grad()
                F.cross_entropy(self.net(xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = F.cross_entropy(self.net(xv), yv).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def proba(self, blocks):
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(blocks["std"], device=DEVICE)), 1).cpu().numpy()
