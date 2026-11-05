"""Reviewer controls (v3), 4-class and 16-class, seeds 42-46:
  M1  MemTGF with random feature partitions of the Louvain sizes, and with Volatility-plugin partitions (no graph)
  M3  No-graph stacking: same out-of-fold meta-learner over LightGBM, XGBoost, random forest and extra trees
  m2  MemTGF with Louvain resolution 1.5 on every seed
Results: results/<task>/seed_<s>/controls_metrics.csv (+ probabilities in controls_probs.npz). Resumable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from memdgf_core import DEVICE, GraphViews, MemDGFConfig, fit_lgbm, lgbm_params, load_task, split_indices
from memtgf import MemTGF, TGFConfig, Variant, fig_communities
from metrics_utils import compute_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


def random_partition(n, sizes, seed):
    perm = np.random.default_rng(seed).permutation(n)
    out, k = [], 0
    for s in sizes:
        out.append(sorted(perm[k:k + s].tolist()))
        k += s
    return out


def plugin_partition(cols):
    groups = {}
    for j, c in enumerate(cols):
        groups.setdefault(c.split(".")[0], []).append(j)
    return [sorted(v) for _, v in sorted(groups.items())]


class NoGraphStack:
    """OOF stacking over four tabular learners trained on task labels; no FIG, SSG or LHG."""

    def __init__(self, seed, n_jobs):
        self.seed, self.n_jobs = seed, n_jobs

    def _makers(self, C):
        s, j = self.seed, self.n_jobs
        return {
            "lgbm": lambda: LGBMClassifier(**lgbm_params(C, s, j) | {"n_estimators": self.n_lgbm}),
            "xgb": lambda: XGBClassifier(n_estimators=self.n_xgb, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7,
                                         tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu", random_state=s, n_jobs=j),
            "rf": lambda: RandomForestClassifier(500, n_jobs=j, random_state=s),
            "et": lambda: ExtraTreesClassifier(500, n_jobs=j, random_state=s),
        }

    def fit(self, Xtr, ytr, Xva, yva, C):
        self.n_lgbm = max(50, int(fit_lgbm(Xtr, ytr, Xva, yva, C, self.seed, self.n_jobs).best_iteration_ or 300))
        xg = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                           device="cuda" if DEVICE.type == "cuda" else "cpu", early_stopping_rounds=100, random_state=self.seed, n_jobs=self.n_jobs)
        xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        self.n_xgb = max(50, int(xg.best_iteration or 300))
        mk = self._makers(C)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=self.seed).split(Xtr, ytr))
        oof = {n: np.zeros((len(ytr), C), np.float32) for n in mk}
        self.full = {}
        for n, f in mk.items():
            for a, q in folds:
                oof[n][q] = f().fit(Xtr[a], ytr[a]).predict_proba(Xtr[q])
            self.full[n] = f().fit(Xtr, ytr)
        Z = np.hstack([np.log(np.clip(oof[n], 1e-6, 1)) for n in mk])
        self.meta = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                   random_state=self.seed, n_jobs=self.n_jobs, verbose=-1).fit(Z, ytr)
        return self

    def proba(self, X):
        Z = np.hstack([np.log(np.clip(self.full[n].predict_proba(X), 1e-6, 1)) for n in self.full])
        return self.meta.predict_proba(Z)


def run(task, seed, data, n_jobs, out):
    X, y, classes, info, y_fam, families, M = data
    tr, va, te = split_indices(y_fam, seed)
    rows, probs = [], {}

    def rec(name, p, t0):
        m = compute_metrics(y[te], p, classes)
        m.update(model=name, family="control", seed=seed, train_time_s=time.time() - t0)
        rows.append(m)
        probs[name] = p.astype(np.float32)
        print(f"[{task} s{seed}] {name:40s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)

    views = GraphViews(MemDGFConfig(seed=seed, n_jobs=n_jobs)).fit(X.iloc[tr], y_fam[tr], families)
    B_tr, B_va, B_te = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
    sizes = [len(c) for c in fig_communities(views.fig.A, seed)]
    cols = views.prep.columns
    variants = [("MemTGF", None, 1.0)] if seed < 42 else []  # development seeds also need the default model
    for name, comms, res in variants + [("MemTGF random partition (Louvain sizes)", random_partition(len(cols), sizes, seed), 1.0),
                             ("MemTGF plugin partition (no graph)", plugin_partition(cols), 1.0),
                             ("MemTGF Louvain resolution 1.5", None, 1.5)]:
        t0 = time.time()
        tgf = MemTGF(views, TGFConfig(seed=seed, n_jobs=n_jobs, resolution=res), M, communities=comms).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
        rec(name, Variant(tgf).proba(dict(B_te)), t0)
    t0 = time.time()
    st = NoGraphStack(seed, n_jobs).fit(B_tr["raw"], y[tr], B_va["raw"], y[va], len(classes))
    rec("No-graph stacking (LGBM+XGB+RF+ET)", st.proba(B_te["raw"]), t0)
    pd.DataFrame(rows).to_csv(out / "controls_metrics.csv", index=False)
    np.savez_compressed(out / "controls_probs.npz", y_test=y[te], **{k.replace("/", "_").replace(":", ""): v for k, v in probs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["16class", "4class"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    for task in a.tasks:
        data = load_task(task)
        for s in a.seeds:
            out = RESULTS / task / (f"dev_seed_{s}" if s < 42 else f"seed_{s}")  # development seeds kept apart
            out.mkdir(parents=True, exist_ok=True)
            if (out / "controls_metrics.csv").exists():
                print(f"skip {task} {s}", flush=True)
                continue
            run(task, s, data, a.n_jobs, out)


if __name__ == "__main__":
    main()
