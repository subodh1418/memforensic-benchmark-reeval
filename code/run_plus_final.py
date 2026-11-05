"""Final evaluation of the selected MemTGF+ design on the test seeds 42-46 (binary, 4-class, 16-class),
with its own ablation. Run only after the design was fixed on development seeds (dev_plus.py).
Usage: python run_plus_final.py --candidate "P3 plugin + diverse"
Output: results/<task>/seed_<s>/plus_final_metrics.csv and plus_final_probs.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dev_plus import candidates
from memdgf_core import GraphViews, MemDGFConfig, load_task, split_indices
from memtgf import TGFConfig, Variant, fig_communities
from memtgf_plus import MemTGFPlus, plugin_partition
from metrics_utils import compute_metrics

RES = Path(__file__).resolve().parents[1] / "results"


def run(task, seed, cand, n_jobs):
    X, y, classes, info, y_fam, families, M = load_task(task)
    tr, va, te = split_indices(y_fam, seed)
    views = GraphViews(MemDGFConfig(seed=seed, n_jobs=n_jobs)).fit(X.iloc[tr], y_fam[tr], families)
    B_tr, B_va, B_te = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
    cols = views.prep.columns
    union = {"E_full": ("lgbm", None)} | {f"E_part{j}": ("lgbm", c) for j, c in enumerate(plugin_partition(cols))}
    union |= {f"E_lv{j}": ("lgbm", c) for j, c in enumerate(fig_communities(views.fig.A, seed))}
    union |= {"E_xgb": ("xgb", None), "E_et": ("et", None), "E_sage": ("sage", None)}
    names = list(union)
    chosen = candidates(names)[cand]
    assert all(n in union for n in chosen), chosen
    part = [n for n in names if n.startswith("E_part")]
    lv = [n for n in names if n.startswith("E_lv")]
    t0 = time.time()
    model = MemTGFPlus(views, TGFConfig(seed=seed, n_jobs=n_jobs), M, union).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
    fit_s = time.time() - t0
    with_sage = chosen + ["E_sage"] if "E_sage" not in chosen else chosen
    variants = {
        "MemTGF+": Variant(model, chosen),
        "MemTGF+ w/o plugin experts": Variant(model, [n for n in chosen if n not in part]),
        "MemTGF+ w/o Louvain (FIG) experts": Variant(model, [n for n in chosen if n not in lv]),
        "MemTGF+ w/o diverse experts (XGB, ET)": Variant(model, [n for n in chosen if n not in ("E_xgb", "E_et")]),
        "MemTGF+ w/o full-feature LightGBM": Variant(model, [n for n in chosen if n != "E_full"]),
        "MemTGF+ w/o meta-learner (mean)": Variant(model, chosen, meta=False),
        "MemTGF+ w/o flat head": Variant(model, chosen, fuse=False),
    }
    variants["MemTGF+ with SAGE expert" if "E_sage" not in chosen else "MemTGF+ w/o SAGE expert"] = (
        Variant(model, with_sage) if "E_sage" not in chosen else Variant(model, [n for n in chosen if n != "E_sage"]))
    rows, probs = [], {}
    for name, v in variants.items():
        t = time.time()
        p = v.proba(dict(B_te))
        m = compute_metrics(y[te], p, classes)
        m.update(model=name, family="proposed-v3" if name == "MemTGF+" else "ablation-v3", seed=seed, train_time_s=fit_s,
                 infer_ms_per_sample=1000 * (time.time() - t) / len(te))
        rows.append(m)
        probs[name] = p.astype(np.float32)
        print(f"[{task} s{seed}] {name:42s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)
    if task != "16class":  # label-hierarchy ablation needs a refit on task labels
        t0 = time.time()
        nolhg = MemTGFPlus(views, TGFConfig(seed=seed, n_jobs=n_jobs, use_lhg=False), M, {n: union[n] for n in chosen})
        nolhg.fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
        p = Variant(nolhg, chosen).proba(dict(B_te))
        m = compute_metrics(y[te], p, classes)
        m.update(model="MemTGF+ w/o LHG (task-level experts)", family="ablation-v3", seed=seed, train_time_s=time.time() - t0)
        rows.append(m)
        probs[m["model"]] = p.astype(np.float32)
        print(f"[{task} s{seed}] {m['model']:42s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)
    out = RES / task / f"seed_{seed}"
    pd.DataFrame(rows).to_csv(out / "plus_final_metrics.csv", index=False)
    np.savez_compressed(out / "plus_final_probs.npz", y_test=y[te], **{k.replace("/", "_").replace(":", ""): v for k, v in probs.items()})
    json.dump({"candidate": cand, "experts": chosen, "fusion_weights": {str(k): v for k, v in model._w.items()}},
              open(out / "plus_final_info.json", "w"), indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tasks", nargs="+", default=["16class", "4class", "binary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    for task in a.tasks:
        for s in a.seeds:
            if (RES / task / f"seed_{s}" / "plus_final_metrics.csv").exists():
                print(f"skip {task} {s}", flush=True)
                continue
            run(task, s, a.candidate, a.n_jobs)


if __name__ == "__main__":
    main()
