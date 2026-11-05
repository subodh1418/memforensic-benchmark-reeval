"""MemTGF+ : candidate improved model (design selected on development seeds 7-9 only).

Experts are specified explicitly as name -> (learner, columns):
  ("lgbm", cols)  LightGBM on a column subset (evidence partition) or all columns (cols=None)
  ("xgb", None)   XGBoost (GPU) on all indicators
  ("et", None)    ExtraTrees on all indicators
  ("sage", None)  inductive GraphSAGE on the sample-similarity graph
All experts learn family labels (label-hierarchy supervision), are fused by the out-of-fold meta-learner and
aggregated along the label hierarchy; a validated blend with a flat head follows (inherited from MemTGF).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from memdgf_core import DEVICE, fit_lgbm, lgbm_params, sage_proba
from memtgf import MemTGF


def plugin_partition(cols):
    groups = {}
    for j, c in enumerate(cols):
        groups.setdefault(c.split(".")[0], []).append(j)
    return [sorted(v) for _, v in sorted(groups.items())]


def build_experts(cols, communities=None, diverse=True, sage=True, full=True):
    ex = {}
    if full:
        ex["E_full"] = ("lgbm", None)
    for j, c in enumerate(communities if communities is not None else plugin_partition(cols)):
        ex[f"E_part{j}"] = ("lgbm", c)
    if diverse:
        ex["E_xgb"] = ("xgb", None)
        ex["E_et"] = ("et", None)
    if sage:
        ex["E_sage"] = ("sage", None)
    return ex


class MemTGFPlus(MemTGF):
    def __init__(self, views, cfg, M, experts: dict):
        super().__init__(views, cfg, M, communities=[])  # Louvain not used; experts given explicitly
        self.experts = experts

    def _xgb(self, n_est):
        return XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                             device="cuda" if DEVICE.type == "cuda" else "cpu", random_state=self.cfg.seed, n_jobs=self.cfg.n_jobs)

    def _et(self):
        return ExtraTreesClassifier(500, n_jobs=self.cfg.n_jobs, random_state=self.cfg.seed)

    def fit(self, B_tr, y_task_tr, y_fam_tr, B_va, y_task_va, y_fam_va):
        cfg = self.cfg
        y_tr = y_fam_tr if cfg.use_lhg else y_task_tr
        y_va = y_fam_va if cfg.use_lhg else y_task_va
        K = self.n_exp_classes = int(y_tr.max()) + 1
        self.y_task_va = y_task_va
        folds = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(B_tr["raw"], y_tr))
        self.models, self.oof = {}, {}
        for name, (kind, cols) in self.experts.items():
            if kind == "sage":
                self.models[name] = self._fit_sage_full(B_tr, y_tr, B_va, y_va)
                self.oof[name] = self._sage_oof(B_tr, y_tr, B_va, y_va, folds)
                continue
            Xtr = B_tr["raw"] if cols is None else B_tr["raw"][:, cols]
            Xva = B_va["raw"] if cols is None else B_va["raw"][:, cols]
            if kind == "lgbm":
                full = fit_lgbm(Xtr, y_tr, Xva, y_va, K, cfg.seed, cfg.n_jobs)
                n_best = max(50, int(full.best_iteration_ or full.n_estimators))
                make = lambda n=n_best: LGBMClassifier(**(lgbm_params(K, cfg.seed, cfg.n_jobs) | {"n_estimators": n}))
            elif kind == "xgb":
                es = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                                   device="cuda" if DEVICE.type == "cuda" else "cpu", early_stopping_rounds=100, random_state=cfg.seed, n_jobs=cfg.n_jobs)
                es.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
                n_best = max(50, int(es.best_iteration or 300))
                full = self._xgb(n_best).fit(Xtr, y_tr)
                make = lambda n=n_best: self._xgb(n)
            else:
                full = self._et().fit(Xtr, y_tr)
                make = self._et
            oof = np.zeros((len(y_tr), K), np.float32)
            for f, q in folds:
                oof[q] = make().fit(Xtr[f], y_tr[f]).predict_proba(Xtr[q])
            self.models[name] = (full, cols)
            self.oof[name] = oof
        self.y_exp_tr = y_tr
        self.flat = fit_lgbm(B_tr["raw"], y_task_tr, B_va["raw"], y_task_va, int(y_task_tr.max()) + 1, cfg.seed, cfg.n_jobs)
        self.B_va = B_va
        return self

    def expert_proba(self, blocks):
        key = f"_exp{self._uid}"
        if key in blocks:
            return blocks[key]
        out = {}
        for name, (kind, cols) in self.experts.items():
            if kind == "sage":
                nbr, _ = self.views.ssg.neighbours(blocks["std"])
                out[name] = sage_proba(self.models[name], self.views.X_std_tr, self.views.ssg.train_nbr, blocks["std"], nbr)
            else:
                m, c = self.models[name]
                out[name] = m.predict_proba(blocks["raw"] if c is None else blocks["raw"][:, c])
        out["_flat"] = self.flat.predict_proba(blocks["raw"])
        blocks[key] = out
        return out
