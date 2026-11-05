"""Main comparison, ablation, sensitivity and robustness experiments for one task.

Usage: python run_experiments.py --task 16class --seeds 42 43 44 45 46 --n_jobs 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from baselines import FeatureGCN, SGCModel, SkModel, tabular_baselines
from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, load_task, split_indices, stack
from memtgf import MemTGF, TGFConfig, Variant
from metrics_utils import compute_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


class SoftVote:
    """Non-graph ensemble baseline: mean probability of fitted tabular models."""

    def __init__(self, members):
        self.members = members

    def proba(self, blocks):
        return np.mean([m.proba(blocks) for m in self.members], 0)


class StackedLGBM:
    """LightGBM on a stack of graph-derived blocks (alternative integration strategies)."""

    def __init__(self, model, keys):
        self.model, self.keys = model, keys

    def proba(self, blocks):
        return self.model.predict_proba(stack(blocks, self.keys))


def fit_views(X, y_fam, tr, seed, n_jobs, families, **cfg_kw):
    cfg = replace(MemDGFConfig(seed=seed, n_jobs=n_jobs), **cfg_kw)
    # SSG evidence and GraphSAGE use the finest (family) labels of the training split only
    return GraphViews(cfg).fit(X.iloc[tr], y_fam[tr], families)


def run_seed(task, seed, n_jobs, data, out_dir):
    X, y, classes, info, y_fam, families, M = data
    tr, va, te = split_indices(y_fam, seed)
    C = len(classes)
    t0 = time.time()
    views = fit_views(X, y_fam, tr, seed, n_jobs, families)
    views_time = time.time() - t0
    B_tr, B_va = views.blocks(), views.blocks(X.iloc[va])
    t = time.time()
    B_te = views.blocks(X.iloc[te])
    view_infer = time.time() - t
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    rows, probs, models, extra = [], {}, {}, {}

    def record(name, family, model, train_time, extra_infer=0.0):
        B = dict(B_te)  # fresh cache per model so timings are honest
        t = time.time()
        p = model.proba(B)
        infer = time.time() - t + extra_infer
        m = compute_metrics(y_te, p, classes)
        m.update(model=name, family=family, seed=seed, train_time_s=train_time, infer_ms_per_sample=1000 * infer / len(y_te))
        rows.append(m)
        probs[name] = p.astype(np.float32)
        models[name] = model
        print(f"[{task} s{seed}] {name:42s} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} mcc={m['mcc']:.4f}", flush=True)

    # ---------------------------------------------------------------- baselines
    t = time.time()
    for name, fam, mdl in tabular_baselines(B_tr, y_tr, B_va, y_va, C, seed, n_jobs):
        record(name, fam, mdl, time.time() - t)
        t = time.time()
    record("Soft-vote ensemble (LGBM+XGB+RF)", "tabular", SoftVote([models[n] for n in ("LightGBM", "XGBoost", "RandomForest")]),
           sum(r["train_time_s"] for r in rows if r["model"] in ("LightGBM", "XGBoost", "RandomForest")))
    t = time.time()
    record("SGC (SSG)", "graph", SGCModel(views).fit(y_tr), time.time() - t + views_time, view_infer)
    t = time.time()
    fgcn = FeatureGCN(views.fig.Ahat, C, seed).fit(B_tr["std"], y_tr, B_va["std"], y_va)
    record("FeatureGCN (FIG)", "graph", fgcn, time.time() - t + views_time)

    # ---------------------------------------------------------------- MemTGF
    t = time.time()
    tgf = MemTGF(views, TGFConfig(seed=seed, n_jobs=n_jobs), M).fit(B_tr, y_tr, y_fam[tr], B_va, y_va, y_fam[va])
    tgf_time = time.time() - t + views_time
    full = Variant(tgf)
    record("MemTGF", "proposed", full, tgf_time, view_infer)
    record("GraphSAGE (SSG)", "graph", Variant(tgf, ["E_sage"], meta=False, fuse=False), tgf_time, view_infer)

    # ---------------------------------------------------------------- ablations
    no_fig = ["E_full", "E_sage"]
    no_ssg = [n for n in tgf.experts if n != "E_sage"]
    record("w/o FIG community experts", "ablation", Variant(tgf, no_fig), tgf_time, view_infer)
    record("w/o SSG GraphSAGE expert", "ablation", Variant(tgf, no_ssg), tgf_time, view_infer)
    record("w/o meta-learner (mean fusion)", "ablation", Variant(tgf, meta=False), tgf_time, view_infer)
    record("w/o flat fusion (graph path only)", "ablation", Variant(tgf, fuse=False), tgf_time, view_infer)
    if task != "16class":
        t = time.time()
        tgf_nolhg = MemTGF(views, TGFConfig(seed=seed, n_jobs=n_jobs, use_lhg=False), M).fit(B_tr, y_tr, y_fam[tr], B_va, y_va, y_fam[va])
        record("w/o LHG (task-level experts)", "ablation", Variant(tgf_nolhg), time.time() - t + views_time, view_infer)
        record("LHG only (family LightGBM aggregated)", "ablation", Variant(tgf, ["E_full"], meta=False, fuse=False), tgf_time, view_infer)
    for name, keys in (("Alt: LightGBM + FIG diffusion/residual features", ("raw", "diff", "res")),
                       ("Alt: LightGBM + SSG evidence features", ("raw", "ssg"))):
        t = time.time()
        g = fit_lgbm(stack(B_tr, keys), y_tr, stack(B_va, keys), y_va, C, seed, n_jobs)
        record(name, "alt-integration", StackedLGBM(g, keys), time.time() - t + views_time, view_infer)

    extra["fusion_weights"] = {f"{'+'.join(k[0])}|meta={k[1]}": w for k, w in tgf._w.items()}
    extra["communities"] = [[views.prep.columns[i] for i in c] for c in tgf.comms]
    extra["graph_timings_s"] = views.timings
    extra["expert_best_iterations"] = {n: int(m[0].best_iteration_ or 0) for n, m in tgf.models.items() if isinstance(m, tuple)}

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(seed_dir / "metrics.csv", index=False)
    np.savez_compressed(seed_dir / "test_probs.npz", y_test=y_te, test_idx=te, **{k.replace("/", "_").replace(":", ""): v for k, v in probs.items()})
    json.dump(extra, open(seed_dir / "fit_info.json", "w"), indent=2, default=str)
    return views, tgf, models, (tr, va, te)


def sensitivity(task, seed, n_jobs, data, out_dir):
    X, y, classes, info, y_fam, families, M = data
    tr, va, te = split_indices(y_fam, seed)
    rows = []
    grid = [("k", v) for v in (5, 25)] + [("tau", v) for v in (0.3, 0.7)] + [("resolution", v) for v in (0.5, 1.5)]
    for param, val in grid:
        vk = {param: val} if param in ("k", "tau") else {}
        views = fit_views(X, y_fam, tr, seed, n_jobs, families, **vk)
        B_tr, B_va, B_te = views.blocks(), views.blocks(X.iloc[va]), views.blocks(X.iloc[te])
        tc = TGFConfig(seed=seed, n_jobs=n_jobs, resolution=val if param == "resolution" else 1.0)
        tgf = MemTGF(views, tc, M).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
        m = compute_metrics(y[te], Variant(tgf).proba(B_te), classes)
        m.update(param=param, value=val, n_communities=len(tgf.comms))
        rows.append(m)
        print(f"[{task}] sensitivity {param}={val} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "sensitivity.csv", index=False)


def robustness(task, views, models, data, te, out_dir, seed):
    """Randomly mask a fraction of test feature values (median-imputed) and re-evaluate."""
    X, y, classes = data[0], data[1], data[2]
    rng = np.random.default_rng(seed)
    X_te = X.iloc[te]
    rows = []
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
        Xm = X_te.mask(rng.random(X_te.shape) < rate) if rate > 0 else X_te
        B = views.blocks(Xm)
        for name in ("MemTGF", "LightGBM", "XGBoost", "RandomForest", "ExtraTrees", "GraphSAGE (SSG)", "FeatureGCN (FIG)", "MLP"):
            m = compute_metrics(y[te], models[name].proba(B), classes)
            m.update(model=name, mask_rate=rate)
            rows.append(m)
        print(f"[{task}] robustness mask={rate} done", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "robustness_masking.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["binary", "4class", "16class"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_jobs", type=int, default=5)
    ap.add_argument("--skip_extras", action="store_true")
    ap.add_argument("--skip_sensitivity", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    out_dir = RESULTS / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_task(a.task)
    json.dump(data[3], open(out_dir / "dataset_info.json", "w"), indent=2, default=str)
    ref_seed = 42  # bundle, robustness and sensitivity always use the seed-42 split
    for seed in a.seeds:
        need_extras = seed == ref_seed and not a.skip_extras and not (out_dir / "robustness_masking.csv").exists()
        if (out_dir / f"seed_{seed}" / "metrics.csv").exists() and not need_extras:
            print(f"skip seed {seed} (done)", flush=True)
            continue
        t = time.time()
        views, tgf, models, (tr, va, te) = run_seed(a.task, seed, a.n_jobs, data, out_dir)
        print(f"[{a.task}] seed {seed} finished in {time.time() - t:.0f}s", flush=True)
        if seed == ref_seed:
            joblib.dump({"views": views, "memtgf": tgf, "lightgbm": models["LightGBM"].est, "split": (tr, va, te),
                         "classes": data[2], "families": data[5], "M": data[6]}, out_dir / f"bundle_seed{seed}.joblib")
            if need_extras:
                robustness(a.task, views, models, data, te, out_dir, seed)
        del views, tgf, models
    pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("seed_*/metrics.csv"))]).to_csv(out_dir / "all_runs.csv", index=False)
    if not a.skip_extras and not a.skip_sensitivity and not (out_dir / "sensitivity.csv").exists():
        sensitivity(a.task, ref_seed, a.n_jobs, data, out_dir)


if __name__ == "__main__":
    main()
