"""Quick analyses requested in the review (v3). Writes results/review/review_analyses.json."""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from memdgf_core import Preprocessor, load_task, split_indices
from memtgf import Variant

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "review"
OUT.mkdir(exist_ok=True)
report = {}

# ---------------------------------------------------------------- M4: isolated inference timing
timing = {}
for task in ("binary", "4class", "16class"):
    b = joblib.load(RES / task / "bundle_seed42.joblib")
    X, *_ = load_task(task)
    te = b["split"][2]
    Xte = X.iloc[te]
    tgf, lgb, views = b["memtgf"], b["lightgbm"], b["views"]
    Variant(tgf).proba({"raw": views.prep.transform(Xte.iloc[:200])[0], "std": views.prep.transform(Xte.iloc[:200])[1]})  # warm-up
    t = time.perf_counter()
    raw, std = views.prep.transform(Xte)
    Variant(tgf).proba({"raw": raw, "std": std})
    t_mem = (time.perf_counter() - t) / len(te)
    t = time.perf_counter()
    lgb.predict_proba(views.prep.transform(Xte)[0])
    t_lgb = (time.perf_counter() - t) / len(te)
    timing[task] = {"memtgf_ms_per_dump": 1000 * t_mem, "lightgbm_ms_per_dump": 1000 * t_lgb, "n_test": int(len(te))}
    print(task, timing[task], flush=True)
report["M4_inference_timing"] = timing

# ---------------------------------------------------------------- M8: Cover-Hart bound from 1-NN error
bounds = {}
for task in ("4class", "16class"):
    X, y, classes, *_rest = load_task(task)
    y_fam = _rest[1]
    tr, va, te = split_indices(y_fam, 42)
    L = len(classes)
    out = {}
    pre = Preprocessor().fit(X.iloc[tr])
    std_tr, std_te = pre.transform(X.iloc[tr])[1], pre.transform(X.iloc[te])[1]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform").fit(X.iloc[tr].fillna(pre.median))
    q_tr, q_te = qt.transform(X.iloc[tr].fillna(pre.median)), qt.transform(X.iloc[te].fillna(pre.median))
    for name, (A, B, metric) in {"standardised_euclidean": (std_tr, std_te, "euclidean"), "quantile_manhattan": (q_tr, q_te, "manhattan")}.items():
        idx = NearestNeighbors(n_neighbors=1, metric=metric).fit(A).kneighbors(B)[1][:, 0]
        r1 = float(np.mean(y[tr][idx] != y[te]))
        lower = (L - 1) / L * (1 - np.sqrt(max(0.0, 1 - L / (L - 1) * r1)))  # Cover-Hart lower bound on Bayes error
        out[name] = {"one_nn_error": r1, "bayes_error_lower_bound": float(lower), "accuracy_upper_bound": float(1 - lower)}
    bounds[task] = out
    print(task, out, flush=True)
report["M8_cover_hart"] = bounds

# ---------------------------------------------------------------- m1: per-seed McNemar + Fisher
mc = {}
for task in ("4class", "16class"):
    for base in ("Soft-vote ensemble (LGBM+XGB+RF)", "LightGBM", "XGBoost"):
        ps, per = [], []
        for f in sorted(glob.glob(str(RES / task / "seed_*" / "test_probs.npz"))):
            z = np.load(f)
            y = z["y_test"]
            a, c = z["MemTGF"].argmax(1) == y, z[base].argmax(1) == y
            n01, n10 = int((a & ~c).sum()), int((~a & c).sum())
            p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            ps.append(p)
            per.append({"seed": Path(f).parent.name, "memtgf_only": n01, "baseline_only": n10, "p": p})
        fisher = stats.combine_pvalues(ps, method="fisher").pvalue
        mc[f"{task}|{base}"] = {"per_seed": per, "fisher_combined_p": float(fisher), "seeds_p_lt_0.05": int(sum(p < 0.05 for p in ps))}
        print(task, base, "fisher", round(fisher, 5), [round(p, 4) for p in ps], flush=True)
report["m1_mcnemar_per_seed"] = mc

# ---------------------------------------------------------------- m4: Zeus vs categories
X, y16, fam, info, yf, families, M = load_task("16class")
X4, y4, cats, *_ = load_task("4class")
top = ["handles.nfile", "svcscan.nservices", "handles.avg_handles_per_proc", "dlllist.avg_dlls_per_proc", "handles.nsection", "handles.nkey", "malfind.ninjections", "ldrmodules.not_in_load"]
fam_names = np.array(families)[yf]
cat_names = np.array(cats)[y4]
groups = {"Zeus": fam_names == "Zeus", "Trojan (other families)": (cat_names == "Trojan") & (fam_names != "Zeus"),
          "Ransomware": cat_names == "Ransomware", "Spyware": cat_names == "Spyware"}
med = pd.DataFrame({g: X.loc[m, top].median() for g, m in groups.items()})
report["m4_zeus_medians"] = med.round(2).to_dict()
# distance of Zeus centroid (standardised) to each category centroid
pre = Preprocessor().fit(X)
S = pre.transform(X)[1]
cent = {g: S[m].mean(0) for g, m in groups.items()}
report["m4_zeus_centroid_distance"] = {g: float(np.linalg.norm(cent["Zeus"] - cent[g])) for g in groups if g != "Zeus"}
print(med.round(1).to_string(), report["m4_zeus_centroid_distance"], flush=True)

# ---------------------------------------------------------------- m5: psxview signal
psx = [c for c in X.columns if c.startswith("psxview.") and not c.endswith("_false_avg")]
nz = {}
for g in ("Benign",) + tuple(c for c in cats if c != "Benign"):
    m = cat_names == g
    nz[g] = {"share_any_hidden": float((X.loc[m, psx] > 0).any(axis=1).mean()), "mean_total_hidden": float(X.loc[m, psx].sum(axis=1).mean()),
             "share_hidden_ge2": float((X.loc[m, psx].sum(axis=1) >= 2).mean())}
report["m5_psxview"] = nz
print(nz, flush=True)

# ---------------------------------------------------------------- Q1: fusion weights
fw = {}
for task in ("binary", "4class", "16class"):
    ws = []
    for f in sorted(glob.glob(str(RES / task / "seed_*" / "fit_info.json"))):
        d = json.load(open(f))["fusion_weights"]
        ws.append([v for k, v in d.items() if k.startswith("E_full+E_comm0+E_comm1+E_comm2+E_sage|meta=True")][0])
    fw[task] = ws
report["Q1_fusion_weights"] = fw
print(fw)
json.dump(report, open(OUT / "review_analyses.json", "w"), indent=2)
