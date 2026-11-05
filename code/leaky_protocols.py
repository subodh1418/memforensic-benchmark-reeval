"""How much do common evaluation shortcuts inflate CIC-MalMem-2022 results? (version 4)

The same LightGBM learner is evaluated under four protocols, five seeds each:
  clean      de-duplicated corpus, family-stratified 70/15/15 split, train-only fitting (the paper's protocol)
  raw        released file as is (duplicates kept), random stratified 80/20 split
  smote-in   raw, SMOTE applied to the training part only (correct use of oversampling)
  smote-pre  raw, SMOTE applied to the whole file before the 80/20 split (oversampling leaks into the test set)
Early stopping uses 15 % of the training part in every protocol. Output: results/review/leaky_protocols.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from memdgf_core import DATA_ROOT, TASK_FILES, fit_lgbm, load_task, split_indices
from run_controls import RESULTS


def raw_task(task):
    df = pd.read_csv(DATA_ROOT / TASK_FILES[task])
    y = df.pop("Class").astype(str)
    X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.loc[:, X.nunique() > 1]
    return X.to_numpy(np.float32), pd.factorize(y, sort=True)[0]


def evaluate(Xtr, ytr, Xte, yte, seed, n_jobs):
    a, v = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=seed)
    m = fit_lgbm(Xtr[a], ytr[a], Xtr[v], ytr[v], int(ytr.max()) + 1, seed, n_jobs)
    p = m.predict(Xte)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for task in ("4class", "16class"):
        Xc, yc, _, _, y_fam, _, _ = load_task(task)
        Xc = Xc.fillna(Xc.median()).to_numpy(np.float32)
        Xr, yr = raw_task(task)
        for s in (42, 43, 44, 45, 46):
            tr, va, te = split_indices(y_fam, s)
            tr = np.concatenate([tr, va])  # validation carved out inside evaluate(), as for the other protocols
            res = {"clean": evaluate(Xc[tr], yc[tr], Xc[te], yc[te], s, a.n_jobs)}
            itr, ite = train_test_split(np.arange(len(yr)), test_size=0.2, stratify=yr, random_state=s)
            res["raw"] = evaluate(Xr[itr], yr[itr], Xr[ite], yr[ite], s, a.n_jobs)
            Xs, ys = SMOTE(random_state=s).fit_resample(Xr[itr], yr[itr])
            res["smote-in"] = evaluate(Xs, ys, Xr[ite], yr[ite], s, a.n_jobs)
            Xa, ya = SMOTE(random_state=s).fit_resample(Xr, yr)
            jtr, jte = train_test_split(np.arange(len(ya)), test_size=0.2, stratify=ya, random_state=s)
            res["smote-pre"] = evaluate(Xa[jtr], ya[jtr], Xa[jte], ya[jte], s, a.n_jobs)
            for k, (acc, f1) in res.items():
                rows.append({"task": task, "seed": s, "protocol": k, "accuracy": acc, "f1_macro": f1})
            print(task, s, {k: round(v[0], 4) for k, v in res.items()}, flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "review" / "leaky_protocols.csv", index=False)
    d = pd.DataFrame(rows)
    print((d.groupby(["task", "protocol"])[["accuracy", "f1_macro"]].agg(["mean", "std"]) * 100).round(2))


if __name__ == "__main__":
    main()
