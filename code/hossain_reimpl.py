"""Re-implementation of Hossain & Islam (Cybersecurity 2024), the highest published CIC-MalMem-2022 figures (>99 % on all
three tasks), under the clean protocol and under the shortcut protocol that oversamples before splitting.

Implemented from the published outline (the full text was not accessible offline, so unreported details use standard
settings): features ranked by the mean of their chi-square (on min-max scaled values) and mutual-information ranks, the
top 30 kept, then a correlation filter dropping the later member of any pair with |r| >= 0.9; SMOTE; soft-voting ensemble
of random forest (300 trees), extra trees (300 trees) and XGBoost (500 rounds, depth 6, learning rate 0.1).
  clean      de-duplicated corpus, family-stratified 70/15/15 split (train + validation used for fitting), SMOTE on training only
  smote-pre  released file with duplicates, SMOTE on the whole file before a random stratified 80/20 split
Output: results/review/hossain_reimpl.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from leaky_protocols import raw_task
from memdgf_core import load_task, split_indices
from run_controls import RESULTS


def select(X, y, seed, k=30, thr=0.9):
    Xm = (X - X.min(0)) / np.maximum(X.max(0) - X.min(0), 1e-12)
    c, _ = chi2(Xm, y)
    sub = np.random.default_rng(seed).choice(len(X), min(15000, len(X)), replace=False)
    mi = mutual_info_classif(X[sub], y[sub], random_state=seed)
    rank = pd.Series(np.nan_to_num(c)).rank(ascending=False) + pd.Series(mi).rank(ascending=False)
    top = np.argsort(rank.to_numpy())[:k]
    C = np.abs(np.nan_to_num(np.corrcoef(X[:, top], rowvar=False)))
    keep = []
    for j in range(len(top)):
        if all(C[j, i] < thr for i in keep):
            keep.append(j)
    return top[keep]


def fit_predict(Xtr, ytr, Xte, seed, n_jobs, oversample=True):
    idx = select(Xtr, ytr, seed)
    A, b = (SMOTE(random_state=seed).fit_resample(Xtr[:, idx], ytr) if oversample and ytr.max() > 1 else (Xtr[:, idx], ytr))
    models = [RandomForestClassifier(300, n_jobs=n_jobs, random_state=seed), ExtraTreesClassifier(300, n_jobs=n_jobs, random_state=seed),
              XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, tree_method="hist", n_jobs=n_jobs, random_state=seed)]
    p = np.mean([m.fit(A, b).predict_proba(Xte[:, idx]) for m in models], 0)
    return p.argmax(1), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    a = ap.parse_args()
    rows = []
    for task in ("binary", "4class", "16class"):
        Xc, yc, _, _, y_fam, _, _ = load_task(task)
        Xc = Xc.fillna(Xc.median()).to_numpy(np.float32)
        Xr, yr = raw_task(task)
        for s in a.seeds:
            tr, va, te = split_indices(y_fam, s)
            tr = np.concatenate([tr, va])
            res = {"clean": (fit_predict(Xc[tr], yc[tr], Xc[te], s, a.n_jobs), yc[te])}
            Xa, ya = SMOTE(random_state=s).fit_resample(Xr, yr) if yr.max() > 1 else (Xr, yr)
            jtr, jte = train_test_split(np.arange(len(ya)), test_size=0.2, stratify=ya, random_state=s)
            res["smote-pre"] = (fit_predict(Xa[jtr], ya[jtr], Xa[jte], s, a.n_jobs, oversample=False), ya[jte])
            for k, ((p, nf), yt) in res.items():
                rows.append({"task": task, "seed": s, "protocol": k, "n_features": nf,
                             "accuracy": accuracy_score(yt, p), "f1_macro": f1_score(yt, p, average="macro")})
            print(task, s, {k: round(rows[-2 + i]["accuracy"], 4) for i, k in enumerate(res)}, flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "review" / "hossain_reimpl.csv", index=False)
    d = pd.DataFrame(rows)
    print((d.groupby(["task", "protocol"])[["accuracy", "f1_macro"]].agg(["mean", "std"]) * 100).round(2))


if __name__ == "__main__":
    main()
