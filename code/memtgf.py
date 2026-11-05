"""MemTGF: Memory-forensic Tri-Graph Fusion for fileless malware detection.

Graph 1 - Label-Hierarchy Graph (LHG): Benign/Malware -> category -> family.
          Experts learn at the family level; probabilities are aggregated along
          LHG edges to the target granularity (binary, 4-class or 16-class).
Graph 2 - Feature-Interaction Graph (FIG): train-only dependency graph over the
          Volatility features; Louvain communities define subgraph experts.
Graph 3 - Sample-Similarity Graph (SSG): inductive kNN graph over memory dumps;
          an inductive GraphSAGE (GPU) is trained on it as a further expert.

Expert probabilities are fused by an out-of-fold meta-learner, then blended with
a flat task-level learner through a validation-tuned weight.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from memdgf_core import _drop_self, fit_lgbm, lgbm_params, sage_proba, train_sage, tune_fusion_weight


@dataclass
class TGFConfig:
    seed: int = 42
    n_jobs: int = 5
    n_folds: int = 5
    resolution: float = 1.0
    use_lhg: bool = True


def fig_communities(A: np.ndarray, seed: int, resolution: float = 1.0):
    G = nx.from_numpy_array(A)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return sorted([sorted(c) for c in comms], key=lambda c: (-len(c), c[0]))


_UID = itertools.count()


def _log(p):
    return np.log(np.clip(p, 1e-6, 1.0))


class MemTGF:
    def __init__(self, views, cfg: TGFConfig, M: np.ndarray, communities=None):
        self.views, self.cfg = views, cfg
        self.M = M if cfg.use_lhg else np.eye(M.shape[1], dtype=np.float32)
        # communities: None -> Louvain on the FIG; otherwise an explicit partition (used for controls)
        self.comms = communities if communities is not None else fig_communities(views.fig.A, cfg.seed, cfg.resolution)
        self.experts = {"E_full": None} | {f"E_comm{j}": c for j, c in enumerate(self.comms)} | {"E_sage": "sage"}
        self._meta, self._w = {}, {}
        self._uid = next(_UID)  # unique cache key; id() can be reused after garbage collection

    # --------------------------------------------------------------------- fit
    def fit(self, B_tr, y_task_tr, y_fam_tr, B_va, y_task_va, y_fam_va):
        cfg = self.cfg
        y_tr = y_fam_tr if cfg.use_lhg else y_task_tr
        y_va = y_fam_va if cfg.use_lhg else y_task_va
        self.n_exp_classes = int(y_tr.max()) + 1
        self.y_task_va = y_task_va
        folds = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(B_tr["raw"], y_tr))
        self.models, self.oof = {}, {}
        for name, cols in self.experts.items():
            if cols == "sage":
                self.models[name] = self._fit_sage_full(B_tr, y_tr, B_va, y_va)
                self.oof[name] = self._sage_oof(B_tr, y_tr, B_va, y_va, folds)
                continue
            Xtr = B_tr["raw"] if cols is None else B_tr["raw"][:, cols]
            Xva = B_va["raw"] if cols is None else B_va["raw"][:, cols]
            m = fit_lgbm(Xtr, y_tr, Xva, y_va, self.n_exp_classes, cfg.seed, cfg.n_jobs)
            n_best = max(50, int(m.best_iteration_ or m.n_estimators))
            self.models[name] = (m, cols)
            oof = np.zeros((len(y_tr), self.n_exp_classes), np.float32)
            for f, q in folds:
                fm = LGBMClassifier(**(lgbm_params(self.n_exp_classes, cfg.seed, cfg.n_jobs) | {"n_estimators": n_best}))
                oof[q] = fm.fit(Xtr[f], y_tr[f]).predict_proba(Xtr[q])
            self.oof[name] = oof
        self.y_exp_tr = y_tr
        # flat task-level head (same learner, task labels)
        self.flat = fit_lgbm(B_tr["raw"], y_task_tr, B_va["raw"], y_task_va, int(y_task_tr.max()) + 1, cfg.seed, cfg.n_jobs)
        self.B_va = B_va
        return self

    def _fit_sage_full(self, B_tr, y_tr, B_va, y_va):
        ssg = self.views.ssg
        nbr_va, _ = ssg.neighbours(B_va["std"])
        return train_sage(self.views.X_std_tr, y_tr, ssg.train_nbr, B_va["std"], y_va, nbr_va, self.n_exp_classes, self.cfg.seed)

    def _sage_oof(self, B_tr, y_tr, B_va, y_va, folds):
        Xs, k = self.views.X_std_tr, self.views.ssg.k
        oof = np.zeros((len(y_tr), self.n_exp_classes), np.float32)
        for f, q in folds:
            nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Xs[f])
            d, i = nn_.kneighbors(Xs[f])
            nbr_f, _ = _drop_self(i, d, k)
            nbr_va = nn_.kneighbors(B_va["std"], n_neighbors=k)[1]
            nbr_q = nn_.kneighbors(Xs[q], n_neighbors=k)[1]
            sm = train_sage(Xs[f], y_tr[f], nbr_f, B_va["std"], y_va, nbr_va, self.n_exp_classes, self.cfg.seed)
            oof[q] = sage_proba(sm, Xs[f], nbr_f, Xs[q], nbr_q)
        return oof

    # ----------------------------------------------------------------- predict
    def expert_proba(self, blocks):
        key = f"_exp{self._uid}"
        if key in blocks:
            return blocks[key]
        out = {}
        for name, cols in self.experts.items():
            if cols == "sage":
                nbr, _ = self.views.ssg.neighbours(blocks["std"])
                out[name] = sage_proba(self.models[name], self.views.X_std_tr, self.views.ssg.train_nbr, blocks["std"], nbr)
            else:
                m, c = self.models[name]
                out[name] = m.predict_proba(blocks["raw"] if c is None else blocks["raw"][:, c])
        out["_flat"] = self.flat.predict_proba(blocks["raw"])
        blocks[key] = out
        return out

    def _meta_model(self, subset):
        key = tuple(subset)
        if key not in self._meta:
            Z = np.hstack([_log(self.oof[n]) for n in subset])
            m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, random_state=self.cfg.seed, n_jobs=self.cfg.n_jobs, verbose=-1)
            self._meta[key] = m.fit(Z, self.y_exp_tr)
        return self._meta[key]

    def graph_proba(self, blocks, subset=None, meta=True):
        """Expert fusion at the expert level, aggregated along LHG edges to the task."""
        subset = list(subset or self.experts)
        P = self.expert_proba(blocks)
        if meta:
            p = self._meta_model(subset).predict_proba(np.hstack([_log(P[n]) for n in subset]))
        else:
            p = np.mean([P[n] for n in subset], 0)
        return p @ self.M

    def proba(self, blocks, subset=None, meta=True, fuse=True):
        key = (tuple(subset or self.experts), meta, fuse)
        p_graph = self.graph_proba(blocks, subset, meta)
        if not fuse:
            return p_graph
        if key not in self._w:
            self._w[key] = tune_fusion_weight(self.graph_proba(self.B_va, subset, meta), self.expert_proba(self.B_va)["_flat"], self.y_task_va)
        w = self._w[key]
        return w * p_graph + (1 - w) * self.expert_proba(blocks)["_flat"]


class Variant:
    """A fixed (subset, meta, fuse) configuration of a fitted MemTGF, with a .proba(blocks) API."""

    def __init__(self, model: MemTGF, subset=None, meta=True, fuse=True):
        self.model, self.subset, self.meta, self.fuse = model, subset, meta, fuse

    def proba(self, blocks):
        return self.model.proba(blocks, self.subset, self.meta, self.fuse)

    @property
    def fusion_weight(self):
        return self.model._w.get((tuple(self.subset or self.model.experts), self.meta, self.fuse), 0.0 if not self.fuse else None)
