"""Model selection for MemTGF+ on development seeds 7, 8, 9 (never used for reported results).
One fit trains the union of experts; each candidate is a subset with its own meta-learner and blend weight.
Output: results/<task>/dev_seed_<s>/plus_metrics.csv and results/review/dev_plus_summary.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from memdgf_core import GraphViews, MemDGFConfig, load_task, split_indices
from memtgf import TGFConfig, Variant, fig_communities
from memtgf_plus import MemTGFPlus, plugin_partition
from metrics_utils import compute_metrics

RES = Path(__file__).resolve().parents[1] / "results"


def candidates(names):
    part = [n for n in names if n.startswith("E_part")]
    lv = [n for n in names if n.startswith("E_lv")]
    return {
        "P0 MemTGF (full + Louvain + SAGE)": ["E_full", *lv, "E_sage"],
        "P1 plugin (full + plugin + SAGE)": ["E_full", *part, "E_sage"],
        "P2 plugin + diverse + SAGE": ["E_full", *part, "E_xgb", "E_et", "E_sage"],
        "P3 plugin + diverse": ["E_full", *part, "E_xgb", "E_et"],
        "P4 plugin + Louvain + diverse": ["E_full", *part, *lv, "E_xgb", "E_et"],
        "P5 plugin + Louvain + diverse + SAGE": ["E_full", *part, *lv, "E_xgb", "E_et", "E_sage"],
        "P6 Louvain + diverse": ["E_full", *lv, "E_xgb", "E_et"],
        "P7 diverse only (full + XGB + ET)": ["E_full", "E_xgb", "E_et"],
    }


def run(task, seed, n_jobs):
    X, y, classes, info, y_fam, families, M = load_task(task)
    tr, va, te = split_indices(y_fam, seed)
    views = GraphViews(MemDGFConfig(seed=seed, n_jobs=n_jobs)).fit(X.iloc[tr], y_fam[tr], families)
    B_tr, B_va, B_te = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
    cols = views.prep.columns
    experts = {"E_full": ("lgbm", None)}
    experts |= {f"E_part{j}": ("lgbm", c) for j, c in enumerate(plugin_partition(cols))}
    experts |= {f"E_lv{j}": ("lgbm", c) for j, c in enumerate(fig_communities(views.fig.A, seed))}
    experts |= {"E_xgb": ("xgb", None), "E_et": ("et", None), "E_sage": ("sage", None)}
    t0 = time.time()
    model = MemTGFPlus(views, TGFConfig(seed=seed, n_jobs=n_jobs), M, experts).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
    print(f"[{task} dev{seed}] union of {len(experts)} experts fitted in {time.time() - t0:.0f}s", flush=True)
    rows = []
    Bt = dict(B_te)
    for name, subset in candidates(list(experts)).items():
        m = compute_metrics(y[te], Variant(model, subset).proba(Bt), classes)
        m.update(candidate=name, seed=seed, task=task, n_experts=len(subset))
        rows.append(m)
        print(f"[{task} dev{seed}] {name:40s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)
    out = RES / task / f"dev_seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "plus_metrics.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["16class", "4class"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    ap.add_argument("--n_jobs", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    for task in a.tasks:
        for s in a.seeds:
            if (RES / task / f"dev_seed_{s}" / "plus_metrics.csv").exists():
                print(f"skip {task} dev{s}", flush=True)
                continue
            run(task, s, a.n_jobs)
    frames = [pd.read_csv(f) for f in RES.glob("*/dev_seed_*/plus_metrics.csv")]
    if frames:
        d = pd.concat(frames)
        s = d.groupby(["task", "candidate"])[["accuracy", "f1_macro"]].agg(["mean", "std"]).round(4)
        (RES / "review").mkdir(exist_ok=True)
        s.to_csv(RES / "review" / "dev_plus_summary.csv")
        print(s.to_string())


if __name__ == "__main__":
    main()
