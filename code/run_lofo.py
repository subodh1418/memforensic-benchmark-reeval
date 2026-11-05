"""Leave-one-family-out (LOFO) study: detection and category attribution of UNSEEN malware families.

For every malware family h: remove h from training and validation, train on the remaining
14 families + Benign (seed-42 split), and score every dump of h (never seen) plus the benign
test dumps (false alarms). Metrics:
  detection_rate  - share of unseen-family dumps predicted as malware (any non-benign class)
  category_acc    - share of unseen-family dumps attributed to their true category (R/S/T)
  benign_fpr      - share of benign test dumps predicted as malware
Usage: python run_lofo.py [--families Conti Zeus ...] [--n_jobs 14]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, load_task, split_indices
from memtgf import MemTGF, TGFConfig, Variant

OUT = Path(__file__).resolve().parents[1] / "results" / "lofo"
SEED = 42


def scores(P, y_cat_true, benign_mask_eval, benign_idx=0):
    """P: 4-class probabilities (Benign first). Returns per-row decisions."""
    mal = P[:, benign_idx] < 0.5
    cat = np.argmax(np.delete(P, benign_idx, 1), 1) + 1  # best malware category, ignoring benign
    return mal, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    X, y4, classes4, info, y_fam, families, M = load_task("4class")
    assert classes4[0] == "Benign"
    yb = (y4 != 0).astype(int)
    tr, va, te = split_indices(y_fam, SEED)
    benign_fam = families.index("Benign")
    te_benign = te[y4[te] == 0]
    targets = a.families or [f for f in families if f != "Benign"]
    prev = OUT / "lofo_results.csv"
    rows = pd.read_csv(prev).to_dict("records") if prev.exists() else []
    done = {r["held_out_family"] for r in rows}
    for fam in targets:
        if fam in done:
            print(f"skip {fam} (done)", flush=True)
            continue
        t0 = time.time()
        h = families.index(fam)
        true_cat = int(np.argmax(M[h]))
        keep = [f for f in range(len(families)) if f != h]
        remap = -np.ones(len(families), int)
        remap[keep] = np.arange(len(keep))
        trh, vah = tr[y_fam[tr] != h], va[y_fam[va] != h]
        U = np.where(y_fam == h)[0]
        yf = remap[y_fam]
        views = GraphViews(MemDGFConfig(seed=SEED, n_jobs=a.n_jobs)).fit(X.iloc[trh], yf[trh], [families[k] for k in keep])
        B_tr, B_va = views.blocks(), views.blocks(X.iloc[vah])
        B_U, B_B = views.blocks(X.iloc[U]), views.blocks(X.iloc[te_benign])
        tgf = MemTGF(views, TGFConfig(seed=SEED, n_jobs=a.n_jobs), M[keep]).fit(B_tr, y4[trh], yf[trh], B_va, y4[vah], yf[vah])

        models = {
            "MemTGF": Variant(tgf),
            "MemTGF graph path (no flat fusion)": Variant(tgf, fuse=False),
            "GraphSAGE (SSG) expert": Variant(tgf, ["E_sage"], meta=False, fuse=False),
        }
        lgb4 = fit_lgbm(B_tr["raw"], y4[trh], B_va["raw"], y4[vah], 4, SEED, a.n_jobs)
        lgb2 = fit_lgbm(B_tr["raw"], yb[trh], B_va["raw"], yb[vah], 2, SEED, a.n_jobs)
        rf = RandomForestClassifier(500, n_jobs=a.n_jobs, random_state=SEED).fit(B_tr["raw"], y4[trh])
        knn = KNeighborsClassifier(15, weights="distance", n_jobs=a.n_jobs).fit(B_tr["std"], y4[trh])
        wrap = lambda f: type("W", (), {"proba": staticmethod(f)})
        models["LightGBM 4-class"] = wrap(lambda B: lgb4.predict_proba(B["raw"]))
        models["RandomForest 4-class"] = wrap(lambda B: rf.predict_proba(B["raw"]))
        models["kNN-Graph 4-class"] = wrap(lambda B: knn.predict_proba(B["std"]))
        models["LightGBM binary detector"] = wrap(lambda B: np.column_stack([lgb2.predict_proba(B["raw"])[:, 0], lgb2.predict_proba(B["raw"])[:, 1], np.zeros((len(B["raw"]), 2))]))

        for name, m in models.items():
            PU, PB = m.proba(dict(B_U)), m.proba(dict(B_B))
            mal_u, cat_u = scores(PU, true_cat, None)
            mal_b, _ = scores(PB, None, None)
            rows.append({
                "held_out_family": fam, "true_category": classes4[true_cat], "model": name, "n_unseen": len(U),
                "detection_rate": float(mal_u.mean()),
                "category_acc": float((cat_u == true_cat).mean()) if "binary" not in name else np.nan,
                "category_acc_given_detected": float((cat_u[mal_u] == true_cat).mean()) if mal_u.any() and "binary" not in name else np.nan,
                "benign_fpr": float(mal_b.mean()),
            })
        r = [x for x in rows if x["held_out_family"] == fam]
        print(f"[LOFO {fam:13s}] " + " | ".join(f"{x['model'].split(' ')[0]}: det={x['detection_rate']:.3f} cat={x['category_acc']:.3f}" for x in r[:1] + r[3:4]) + f"  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "lofo_results.csv", index=False)
        json.dump({"fusion_weight": tgf._w.get((tuple(tgf.experts), True, True))}, open(OUT / f"fit_{fam}.json", "w"))


if __name__ == "__main__":
    main()
