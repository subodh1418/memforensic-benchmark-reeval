"""MemDGF: Memory-forensic Dual-Graph Fusion for fileless malware detection.

Two graphs are learned from the training split only:

* Feature Interaction Graph (FIG): nodes are Volatility memory-forensic
  features, edges are |Spearman rho| dependencies plus same-plugin domain
  edges. Graph diffusion (X A_hat, X A_hat^2) and the graph-Laplacian residual
  (X - X A_hat) expose cross-plugin inconsistencies.
* Sample Similarity Graph (SSG): an inductive kNN graph over memory dumps,
  giving out-of-fold neighbourhood class evidence and an inductive GraphSAGE.

A gradient-boosted learner is trained on the graph-augmented representation
and its probabilities are fused with GraphSAGE using a validation-tuned weight.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier, early_stopping
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_ROOT = Path(os.environ.get("MEMHSE_DATA", r"D:\python text\Gan\data"))  # folder with the three CIC-MalMem-2022 CSV files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_FILES = {
    "binary": "binarycic.csv",
    "4class": "4clascic.csv",
    "16class": "16classcicmalmem.csv",
}


# --------------------------------------------------------------------------- data
def load_corpus():
    """One de-duplicated CIC-MalMem-2022 corpus carrying all three label granularities.

    The three task files are row-aligned, so family, category and binary labels
    come from the same memory dump. De-duplication is done once (full rows, then
    exact feature vectors) so every task uses identical samples and splits.
    """
    fam_df = pd.read_csv(DATA_ROOT / TASK_FILES["16class"])
    cat = pd.read_csv(DATA_ROOT / TASK_FILES["4class"])["Class"].astype(str)
    binl = pd.read_csv(DATA_ROOT / TASK_FILES["binary"])["Class"].astype(str)
    n_initial = len(fam_df)
    full_dup = fam_df.duplicated().to_numpy()
    fam = fam_df.pop("Class").astype(str)
    X = fam_df.apply(pd.to_numeric, errors="coerce")
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=zero_var)
    keep = ~full_dup
    # exact feature-vector duplicates (incl. label-conflicting ones) keep their first occurrence
    h = pd.util.hash_pandas_object(X.loc[keep], index=False)
    first = ~h.duplicated(keep="first").to_numpy()
    rows = np.where(keep)[0][first]
    X = X.iloc[rows].reset_index(drop=True)
    labels = {"16class": fam.iloc[rows].to_numpy(), "4class": cat.iloc[rows].to_numpy(), "binary": binl.iloc[rows].to_numpy()}
    info = {
        "initial_rows": int(n_initial),
        "full_duplicate_rows": int(full_dup.sum()),
        "rows_after_dedup": int(len(X)),
        "n_features": int(X.shape[1]),
        "zero_variance_removed": zero_var,
        "class_counts": {t: pd.Series(v).value_counts().to_dict() for t, v in labels.items()},
    }
    return X, labels, info


def load_task(task: str):
    """Return X, encoded task labels, task classes, info and the label hierarchy."""
    X, labels, info = load_corpus()
    enc = LabelEncoder().fit(labels[task])
    fam_enc = LabelEncoder().fit(labels["16class"])
    y = enc.transform(labels[task])
    y_fam = fam_enc.transform(labels["16class"])
    # family -> task aggregation matrix (edges of the label-hierarchy graph)
    M = np.zeros((len(fam_enc.classes_), len(enc.classes_)), dtype=np.float32)
    for f, t in set(zip(y_fam, y)):
        M[f, t] = 1.0
    assert (M.sum(1) == 1).all(), "families must map to exactly one task class"
    info = dict(info, task=task, classes=enc.classes_.tolist(), families=fam_enc.classes_.tolist())
    return X, y, enc.classes_.tolist(), info, y_fam, fam_enc.classes_.tolist(), M


def split_indices(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te


class Preprocessor:
    """Median imputation + signed log1p + standardisation, fitted on train only."""

    def fit(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        self.median = X.median()
        self.scaler = StandardScaler().fit(self._log(X.fillna(self.median)))
        return self

    @staticmethod
    def _log(X: pd.DataFrame) -> np.ndarray:
        v = X.to_numpy(dtype=np.float64)
        return np.sign(v) * np.log1p(np.abs(v))

    def transform(self, X: pd.DataFrame):
        Xi = X[self.columns].fillna(self.median)
        raw = Xi.to_numpy(dtype=np.float32)
        std = np.clip(self.scaler.transform(self._log(Xi)), -8, 8).astype(np.float32)
        return raw, std


# ---------------------------------------------------------- feature interaction graph
@dataclass
class FeatureInteractionGraph:
    tau: float = 0.5
    topk: int = 3
    domain_prior: bool = True
    prior_weight: float = 0.3

    def fit(self, Xs: np.ndarray, columns: list[str]):
        n = Xs.shape[1]
        rho = pd.DataFrame(Xs).rank().corr().abs().fillna(0.0).to_numpy().copy()
        np.fill_diagonal(rho, 0.0)
        A = np.where(rho >= self.tau, rho, 0.0)
        for i in range(n):
            nbr = np.argsort(-rho[i])[: self.topk]
            A[i, nbr] = np.maximum(A[i, nbr], rho[i, nbr])
        self.groups = np.array([c.split(".")[0] for c in columns])
        if self.domain_prior:
            same = self.groups[:, None] == self.groups[None, :]
            np.fill_diagonal(same, False)
            A = np.where(same, np.maximum(A, self.prior_weight), A)
        A = np.maximum(A, A.T)
        At = A + np.eye(n)
        dinv = 1.0 / np.sqrt(At.sum(1))
        self.rho = rho
        self.A = A
        self.Ahat = (At * dinv[:, None] * dinv[None, :]).astype(np.float32)
        self.Ahat2 = (self.Ahat @ self.Ahat).astype(np.float32)
        self.columns = list(columns)
        return self

    def transform(self, Xs: np.ndarray):
        d1 = Xs @ self.Ahat
        return {"diff": np.hstack([d1, Xs @ self.Ahat2]), "res": Xs - d1}

    def names(self, prefix: str = ""):
        return {
            "diff": [f"{prefix}diff1::{c}" for c in self.columns] + [f"{prefix}diff2::{c}" for c in self.columns],
            "res": [f"{prefix}res::{c}" for c in self.columns],
        }


# ------------------------------------------------------------- sample similarity graph
def _drop_self(idx: np.ndarray, dist: np.ndarray, k: int):
    keep = idx != np.arange(len(idx))[:, None]
    order = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(idx, order, 1), np.take_along_axis(dist, order, 1)


def mean_operator(nbr: np.ndarray, n_ref: int) -> torch.Tensor:
    n, k = nbr.shape
    rows = np.repeat(np.arange(n), k)
    M = sparse.csr_matrix((np.full(n * k, 1.0 / k, dtype=np.float32), (rows, nbr.ravel())), shape=(n, n_ref)).tocoo()
    return torch.sparse_coo_tensor(np.vstack([M.row, M.col]), M.data, M.shape).coalesce().to(DEVICE)


@dataclass
class SampleSimilarityGraph:
    k: int = 15
    n_folds: int = 5
    seed: int = 42

    def _evidence(self, Xref, yref, Xq):
        C = self.n_classes
        dist, idx = NearestNeighbors(n_neighbors=self.k).fit(Xref).kneighbors(Xq)
        w = 1.0 / (dist + 1e-6)
        lab = yref[idx]
        vote = np.stack([(w * (lab == c)).sum(1) for c in range(C)], 1) / w.sum(1, keepdims=True)
        cdist = np.stack(
            [NearestNeighbors(n_neighbors=1).fit(Xref[yref == c]).kneighbors(Xq)[0][:, 0] for c in range(C)], 1
        )
        logd = np.log1p(cdist)
        ent = -(vote * np.log(vote + 1e-12)).sum(1, keepdims=True)
        return np.hstack([vote, logd, logd - logd.min(1, keepdims=True), np.log1p(dist.mean(1, keepdims=True)), ent]).astype(np.float32)

    def names(self, classes):
        return (
            [f"ssg_vote::{c}" for c in classes]
            + [f"ssg_logdist::{c}" for c in classes]
            + [f"ssg_reldist::{c}" for c in classes]
            + ["ssg_mean_knn_dist", "ssg_vote_entropy"]
        )

    def fit(self, Xs_tr: np.ndarray, y_tr: np.ndarray, n_classes: int):
        self.n_classes = n_classes
        self.X_ref, self.y_ref = Xs_tr, y_tr
        oof = np.zeros((len(y_tr), 3 * n_classes + 2), dtype=np.float32)
        for fit_i, q_i in StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed).split(Xs_tr, y_tr):
            oof[q_i] = self._evidence(Xs_tr[fit_i], y_tr[fit_i], Xs_tr[q_i])
        self.train_evidence = oof
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Xs_tr)
        d, i = self.nn.kneighbors(Xs_tr)
        self.train_nbr, _ = _drop_self(i, d, self.k)
        return self

    def evidence(self, Xq):
        return self._evidence(self.X_ref, self.y_ref, Xq)

    def neighbours(self, Xq):
        d, i = self.nn.kneighbors(Xq, n_neighbors=self.k)
        return i, d


# -------------------------------------------------------------------- GraphSAGE (SSG)
class InductiveSAGE(nn.Module):
    """Two-layer GraphSAGE-mean; query nodes aggregate from their training neighbours."""

    def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(2 * d_in, hidden)
        self.lin2 = nn.Linear(2 * hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.drop = nn.Dropout(dropout)

    def _l1(self, x, x_ref, M):
        return self.drop(F.relu(self.norm1(self.lin1(torch.cat([x, torch.sparse.mm(M, x_ref)], 1)))))

    def forward(self, x_ref, M_ref, x_q=None, M_q=None):
        h_ref = self._l1(x_ref, x_ref, M_ref)
        if x_q is None:
            h_q, M_q = h_ref, M_ref
        else:
            h_q = self._l1(x_q, x_ref, M_q)
        h2 = F.relu(self.norm2(self.lin2(torch.cat([h_q, torch.sparse.mm(M_q, h_ref)], 1))))
        return self.head(self.drop(h2))


def train_sage(Xs_tr, y_tr, nbr_tr, Xs_va, y_va, nbr_va, n_classes, seed, hidden=128, epochs=400, patience=40):
    torch.manual_seed(seed)
    xr, xv = torch.tensor(Xs_tr, device=DEVICE), torch.tensor(Xs_va, device=DEVICE)
    Mr, Mv = mean_operator(nbr_tr, len(Xs_tr)), mean_operator(nbr_va, len(Xs_tr))
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yv = torch.tensor(y_va, dtype=torch.long, device=DEVICE)
    model = InductiveSAGE(Xs_tr.shape[1], hidden, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xr, Mr), yt)
        loss.backward()
        opt.step()
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(xr, Mr, xv, Mv), yv).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 5
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model


def sage_proba(model, Xs_tr, nbr_tr, Xq, nbr_q):
    with torch.no_grad():
        logits = model(
            torch.tensor(Xs_tr, device=DEVICE), mean_operator(nbr_tr, len(Xs_tr)),
            torch.tensor(Xq, device=DEVICE), mean_operator(nbr_q, len(Xs_tr)),
        )
    return torch.softmax(logits, 1).cpu().numpy()


# ------------------------------------------------------------------------- learners
def lgbm_params(n_classes: int, seed: int, n_jobs: int):
    return dict(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def fit_lgbm(Xtr, ytr, Xva, yva, n_classes, seed, n_jobs):
    m = LGBMClassifier(**lgbm_params(n_classes, seed, n_jobs))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(100, verbose=False)])
    return m


def tune_fusion_weight(p_gbdt, p_graph, y_val):
    """Pick w in [0,1] maximising validation macro-F1 (ties favour the GBDT)."""
    best_w, best_f = 1.0, -1.0
    for w in np.round(np.linspace(1.0, 0.0, 21), 2):
        f = f1_score(y_val, (w * p_gbdt + (1 - w) * p_graph).argmax(1), average="macro")
        if f > best_f + 1e-9:
            best_w, best_f = float(w), f
    return best_w


# ------------------------------------------------------------------- the full model
BLOCKS_ALL = ("raw", "diff", "res", "ssg")


@dataclass
class MemDGFConfig:
    k: int = 15
    tau: float = 0.5
    topk: int = 3
    domain_prior: bool = True
    seed: int = 42
    n_jobs: int = 4


@dataclass
class GraphViews:
    """Everything fitted on the training split; produces per-block representations."""

    cfg: MemDGFConfig
    timings: dict = field(default_factory=dict)

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, classes: list[str]):
        t = time.time()
        self.classes = classes
        self.prep = Preprocessor().fit(X_tr)
        raw, std = self.prep.transform(X_tr)
        cols = self.prep.columns
        self.fig = FeatureInteractionGraph(self.cfg.tau, self.cfg.topk, self.cfg.domain_prior).fit(std, cols)
        self.fig_np = FeatureInteractionGraph(self.cfg.tau, self.cfg.topk, False).fit(std, cols)
        self.timings["fig_s"] = time.time() - t
        t = time.time()
        self.ssg = SampleSimilarityGraph(self.cfg.k, 5, self.cfg.seed).fit(std, y_tr, len(classes))
        self.timings["ssg_s"] = time.time() - t
        self.X_std_tr = std
        self._train_blocks = self._blocks(raw, std, self.ssg.train_evidence)
        return self

    def _blocks(self, raw, std, ev):
        f, fn = self.fig.transform(std), self.fig_np.transform(std)
        return {"raw": raw, "std": std, "diff": f["diff"], "res": f["res"], "diff_np": fn["diff"], "res_np": fn["res"], "ssg": ev}

    def blocks(self, X: pd.DataFrame | None = None):
        """Blocks for the training split (OOF evidence) when X is None, else for new data."""
        if X is None:
            return self._train_blocks
        raw, std = self.prep.transform(X)
        return self._blocks(raw, std, self.ssg.evidence(std))

    def names(self):
        cols = self.prep.columns
        n, nn_ = self.fig.names(), self.fig_np.names("np_")
        return {"raw": cols, "std": cols, "diff": n["diff"], "res": n["res"], "diff_np": nn_["diff"], "res_np": nn_["res"], "ssg": self.ssg.names(self.classes)}


def stack(blocks: dict, keys) -> np.ndarray:
    return np.hstack([blocks[k] for k in keys]).astype(np.float32)
