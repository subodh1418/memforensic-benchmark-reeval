"""Re-implementations of recent journal methods for CIC-MalMem-2022, run under the MemTGF protocol
(same de-duplicated corpus, same family-stratified seeds 42-46, same test sets).

Implemented from the published descriptions (not the authors' code):
  Zakaria2025-SoftVote   Zakaria et al., Appl. Comput. Inform. 2025  - MI/ANOVA selection + soft voting RF + kNN + XGBoost
  AlGhanem2025-MADANET   Al-Ghanem et al., CMES 2025                 - attention DNN-CNN; PCA (binary) / SMOTE (multi-class)
  Hussain2024-GNBiLSTM   Hussain et al., PeerJ Comput. Sci. 2024    - group-normalised BiLSTM
  Mahdi2025-1DCNN        Mahdi & Trabelsi, IJIES 2025               - Pearson-correlation selection + 1D-CNN
  Hadjila2024-ANOVADNN   Hadjila et al., Sci. Tech. J. ITMO 2024    - ANOVA selection + DNN
  Cevallos2024-DNN       Cevallos-Salas et al., IEEE Access 2024     - DNN + SMOTE
  Farfoura2026-HAGEN     Farfoura et al., Electronics 2026           - hierarchical attention-gated network + prototypes
Simplifications: GA hyper-parameter search (Zakaria) replaced by the paper's reported model families with
standard settings; all oversampling is applied to the training split only.
Usage: python sota_reimpl.py --task 4class --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from memdgf_core import DEVICE, Preprocessor, load_task, split_indices
from metrics_utils import compute_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ------------------------------------------------------------------------ helpers
def anova_mi_rank(X, y, seed, k):
    f, _ = f_classif(X, y)
    sub = np.random.default_rng(seed).choice(len(X), min(15000, len(X)), replace=False)
    mi = mutual_info_classif(X[sub], y[sub], random_state=seed)
    r = (pd.Series(np.nan_to_num(f)).rank(ascending=False) + pd.Series(mi).rank(ascending=False)).to_numpy()
    return np.argsort(r)[:k]


def pearson_filter(X, thr=0.9):
    C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    keep = []
    for j in range(X.shape[1]):
        if all(C[j, k] < thr for k in keep):
            keep.append(j)
    return np.array(keep)


def train_torch(net, Xtr, ytr, Xva, yva, seed, epochs=50, patience=5, bs=256, lr=1e-3):
    torch.manual_seed(seed)
    net = net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE), torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    xv, yv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE), torch.tensor(yva, dtype=torch.long, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(xt), generator=g).to(DEVICE).split(bs):
            opt.zero_grad()
            F.cross_entropy(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = F.cross_entropy(net(xv), yv).item()
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return lambda X: torch.softmax(net(torch.tensor(X, dtype=torch.float32, device=DEVICE)), 1).detach().cpu().numpy()


# ------------------------------------------------------------------------ networks
class MADANet(nn.Module):
    """Conv block (ReLU, max-pool) -> attention over positions -> flatten -> dense 32/64/128 with BN."""

    def __init__(self, d, C):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2) if d >= 4 else nn.Identity()
        L = d // 2 if d >= 4 else d
        self.att = nn.Linear(64, 1)
        dims = [64 * L, 32, 64, 128]
        self.dense = nn.Sequential(*[m for i in range(3) for m in (nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU())])
        self.out = nn.Linear(128, C)

    def forward(self, x):
        h = self.pool(F.relu(self.conv(x.unsqueeze(1))))  # B x 64 x L
        a = torch.softmax(self.att(h.transpose(1, 2)), 1)  # B x L x 1
        h = h * a.transpose(1, 2) * h.shape[2]
        return self.out(self.dense(h.flatten(1)))


class GNBiLSTM(nn.Module):
    def __init__(self, d, C, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.gn = nn.GroupNorm(8, 2 * hidden)
        self.fc = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, C))

    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(self.gn(o.mean(1)))


class CNN1D(nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * (d // 2), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, C))

    def forward(self, x):
        return self.c(self.f(x.unsqueeze(1)))


class HAGENNet(nn.Module):
    """Hierarchical attention-gated network with a prototype classifier (after Farfoura et al., 2026):
    feature tokens -> gated self-attention -> pooled per Volatility plugin group -> gated group attention
    -> attention pooling -> distance to learnable class prototypes."""

    def __init__(self, d, C, groups, dim=32, heads=4, protos=3):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(d, dim) * 0.1), nn.Parameter(torch.zeros(d, dim))
        G = torch.zeros(len(set(groups)), d)
        for j, g in enumerate(groups):
            G[sorted(set(groups)).index(g), j] = 1.0
        self.register_buffer("G", G / G.sum(1, keepdim=True))
        self.att1, self.att2 = nn.MultiheadAttention(dim, heads, batch_first=True), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.g1, self.g2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, 1)
        self.proto = nn.Parameter(torch.randn(C, protos, dim) * 0.5)

    def forward(self, x):
        h = x.unsqueeze(-1) * self.w + self.b                       # B x d x dim (feature tokens)
        a, _ = self.att1(h, h, h)
        h = self.n1(h + torch.sigmoid(self.g1(h)) * a)              # gated residual, feature level
        g = torch.einsum("kd,bde->bke", self.G, h)                  # plugin-group tokens
        a, _ = self.att2(g, g, g)
        g = self.n2(g + torch.sigmoid(self.g2(g)) * a)              # gated residual, group level
        z = (torch.softmax(self.pool(g), 1) * g).sum(1)             # attention pooling
        dist = ((z[:, None, None, :] - self.proto[None]) ** 2).sum(-1)  # B x C x protos
        return -dist.min(-1).values                                 # prototype logits


def dnn(d, C, widths=(128, 64, 32), p=0.2):
    layers, prev = [], d
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(p)]
        prev = w
    return nn.Sequential(*layers, nn.Linear(prev, C))


# ------------------------------------------------------------------------ runner
def run(task, seed, data):
    X, y, classes, info, y_fam, families, M = data
    tr, va, te = split_indices(y_fam, seed)
    C = len(classes)
    prep = Preprocessor().fit(X.iloc[tr])
    (_, Str), (_, Sva), (_, Ste) = prep.transform(X.iloc[tr]), prep.transform(X.iloc[va]), prep.transform(X.iloc[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    smote = lambda A, b: SMOTE(random_state=seed).fit_resample(A, b) if C > 2 else (A, b)
    rows, probs = [], {}

    def rec(name, fit_fn):
        t = time.time()
        predict = fit_fn()
        tt = time.time() - t
        t = time.time()
        p = predict()
        m = compute_metrics(yte, p, classes)
        m.update(model=name, family="sota-reimpl", seed=seed, train_time_s=tt, infer_ms_per_sample=1000 * (time.time() - t) / len(yte))
        rows.append(m)
        probs[name] = p.astype(np.float32)
        print(f"[{task} s{seed}] {name:24s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)

    def zakaria():
        idx = anova_mi_rank(Str, ytr, seed, 30)
        rf = RandomForestClassifier(300, n_jobs=6, random_state=seed).fit(Str[:, idx], ytr)
        kn = KNeighborsClassifier(5, n_jobs=6).fit(Str[:, idx], ytr)
        xg = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, tree_method="hist", device="cuda", random_state=seed).fit(Str[:, idx], ytr)
        return lambda: np.mean([m.predict_proba(Ste[:, idx]) for m in (rf, kn, xg)], 0)

    def madanet():
        if C == 2:
            pca = PCA(2, random_state=seed).fit(Str)
            A, Av, At = pca.transform(Str), pca.transform(Sva), pca.transform(Ste)
            Ar, yr = A, ytr
        else:
            A, Av, At = Str, Sva, Ste
            Ar, yr = smote(A, ytr)
        f = train_torch(MADANet(A.shape[1], C), Ar, yr, Av, yva, seed)
        return lambda: f(At)

    def gnbilstm():
        f = train_torch(GNBiLSTM(Str.shape[1], C), Str, ytr, Sva, yva, seed)
        return lambda: f(Ste)

    def mahdi():
        keep = pearson_filter(Str)
        f = train_torch(CNN1D(len(keep), C), Str[:, keep], ytr, Sva[:, keep], yva, seed)
        return lambda: f(Ste[:, keep])

    def hadjila():
        f_, _ = f_classif(Str, ytr)
        idx = np.argsort(-np.nan_to_num(f_))[:30]
        f = train_torch(dnn(30, C), Str[:, idx], ytr, Sva[:, idx], yva, seed)
        return lambda: f(Ste[:, idx])

    def cevallos():
        Ar, yr = smote(Str, ytr)
        f = train_torch(dnn(Str.shape[1], C, (256, 128, 64)), Ar, yr, Sva, yva, seed)
        return lambda: f(Ste)

    def hagen():
        groups = [c.split(".")[0] for c in prep.columns]
        f = train_torch(HAGENNet(Str.shape[1], C, groups), Str, ytr, Sva, yva, seed)
        return lambda: f(Ste)

    for name, fn in [("Farfoura2026-HAGEN", hagen), ("Zakaria2025-SoftVote", zakaria), ("AlGhanem2025-MADANET", madanet), ("Hussain2024-GNBiLSTM", gnbilstm),
                     ("Mahdi2025-1DCNN", mahdi), ("Hadjila2024-ANOVADNN", hadjila), ("Cevallos2024-DNN", cevallos)]:
        rec(name, fn)
    d = RESULTS / task / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "sota_reimpl_metrics.csv", index=False)
    np.savez_compressed(d / "sota_reimpl_probs.npz", y_test=yte, test_idx=te, **probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    a = ap.parse_args()
    torch.set_num_threads(4)
    data = load_task(a.task)
    for s in a.seeds:
        done = RESULTS / a.task / f"seed_{s}" / "sota_reimpl_metrics.csv"
        if done.exists() and "Farfoura2026-HAGEN" in done.read_text():
            print(f"skip {a.task} seed {s} (done)", flush=True)
            continue
        run(a.task, s, data)


if __name__ == "__main__":
    main()
