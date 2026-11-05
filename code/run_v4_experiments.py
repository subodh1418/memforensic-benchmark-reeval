"""Version-4 experiments.

  dev   Volatility-plugin partition on development seeds 7-9 (both multi-class tasks), to check on
        data never used for reporting whether the plugin grouping beats the Louvain grouping
        (the Louvain default on these seeds is candidate "P0" in dev_seed_<s>/plus_metrics.csv).
        Output: results/<task>/dev_seed_<s>/v4_plugin_metrics.csv
  loco  Leave-one-category-out: every family of one malware category is removed from training and
        validation (seed-42 split); the model is trained on benign + the two other categories and
        scored on all dumps of the withheld category and on the benign test dumps.
        Output: results/loco/loco_results.csv
Usage: python run_v4_experiments.py dev|loco [--n_jobs 14]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, load_task, split_indices
from memtgf import MemTGF, TGFConfig, Variant
from metrics_utils import compute_metrics
from run_controls import plugin_partition

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run_dev(n_jobs):
    for task in ("16class", "4class"):
        X, y, classes, info, y_fam, families, M = load_task(task)
        for s in (7, 8, 9):
            out = RESULTS / task / f"dev_seed_{s}" / "v4_plugin_metrics.csv"
            if out.exists():
                print(f"skip {task} {s}", flush=True)
                continue
            t0 = time.time()
            tr, va, te = split_indices(y_fam, s)
            views = GraphViews(MemDGFConfig(seed=s, n_jobs=n_jobs)).fit(X.iloc[tr], y_fam[tr], families)
            B_tr, B_va, B_te = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
            tgf = MemTGF(views, TGFConfig(seed=s, n_jobs=n_jobs), M, communities=plugin_partition(views.prep.columns))
            tgf.fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
            m = compute_metrics(y[te], Variant(tgf).proba(dict(B_te)), classes)
            m.update(model="MemTGF plugin partition (no graph)", seed=s, task=task, train_time_s=time.time() - t0)
            pd.DataFrame([m]).to_csv(out, index=False)
            print(f"[dev {task} s{s}] plugin acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def run_loco(n_jobs):
    out_dir = RESULTS / "loco"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "loco_results.csv"
    X, y4, classes4, info, y_fam, families, M = load_task("4class")
    assert classes4[0] == "Benign"
    tr, va, te = split_indices(y_fam, 42)
    te_benign = te[y4[te] == 0]
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["held_out_category"] for r in rows}
    for c in range(1, len(classes4)):
        if classes4[c] in done:
            print(f"skip {classes4[c]}", flush=True)
            continue
        t0 = time.time()
        held = [f for f in range(len(families)) if M[f, c] == 1]
        keep_f = [f for f in range(len(families)) if f not in held]
        keep_c = [k for k in range(len(classes4)) if k != c]
        fmap, cmap = -np.ones(len(families), int), -np.ones(len(classes4), int)
        fmap[keep_f], cmap[keep_c] = np.arange(len(keep_f)), np.arange(len(keep_c))
        mask = lambda idx: idx[~np.isin(y_fam[idx], held)]
        trh, vah = mask(tr), mask(va)
        U = np.where(np.isin(y_fam, held))[0]
        yf, yc = fmap[y_fam], cmap[y4]
        Mk = M[np.ix_(keep_f, keep_c)]
        views = GraphViews(MemDGFConfig(seed=42, n_jobs=n_jobs)).fit(X.iloc[trh], yf[trh], [families[k] for k in keep_f])
        B_tr, B_va = views.blocks(), views.blocks(X.iloc[vah])
        B_U, B_B = views.blocks(X.iloc[U]), views.blocks(X.iloc[te_benign])
        models = {}
        for name, comms in (("MemTGF", None), ("MemTGF plugin partition", plugin_partition(views.prep.columns))):
            tgf = MemTGF(views, TGFConfig(seed=42, n_jobs=n_jobs), Mk, communities=comms).fit(B_tr, yc[trh], yf[trh], B_va, yc[vah], yf[vah])
            models[name] = Variant(tgf).proba
        yb = (yc != 0).astype(int)
        lgb3 = fit_lgbm(B_tr["raw"], yc[trh], B_va["raw"], yc[vah], 3, 42, n_jobs)
        lgb2 = fit_lgbm(B_tr["raw"], yb[trh], B_va["raw"], yb[vah], 2, 42, n_jobs)
        rf = RandomForestClassifier(500, n_jobs=n_jobs, random_state=42).fit(B_tr["raw"], yc[trh])
        knn = KNeighborsClassifier(15, weights="distance", n_jobs=n_jobs).fit(B_tr["std"], yc[trh])
        models["LightGBM 3-class"] = lambda B: lgb3.predict_proba(B["raw"])
        models["LightGBM binary detector"] = lambda B: lgb2.predict_proba(B["raw"])
        models["RandomForest 3-class"] = lambda B: rf.predict_proba(B["raw"])
        models["kNN-Graph 3-class"] = lambda B: knn.predict_proba(B["std"])
        for name, f in models.items():
            pu, pb = f(dict(B_U))[:, 0], f(dict(B_B))[:, 0]  # benign probability (benign is class 0 in every model)
            rows.append({"held_out_category": classes4[c], "model": name, "n_unseen": len(U),
                         "detection_rate": float((pu < 0.5).mean()), "benign_fpr": float((pb < 0.5).mean()),
                         "median_p_benign_unseen": float(np.median(pu))})
        pd.DataFrame(rows).to_csv(out, index=False)
        r = [x for x in rows if x["held_out_category"] == classes4[c]]
        print(f"[LOCO {classes4[c]}] " + " | ".join(f"{x['model']}: det={x['detection_rate']:.4f} fpr={x['benign_fpr']:.5f}" for x in r)
              + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dev", "loco"])
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    run_dev(a.n_jobs) if a.what == "dev" else run_loco(a.n_jobs)
