"""M6: deletion fidelity of the graph-smoothed ranking vs the plain TreeSHAP ranking (seed 42)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from memdgf_core import load_task
from memtgf import Variant

RES = Path(__file__).resolve().parents[1] / "results"
rows = []
for task in ("binary", "4class", "16class"):
    b = joblib.load(RES / task / "bundle_seed42.joblib")
    views, tgf, te = b["views"], b["memtgf"], b["split"][2]
    X, y, *_ = load_task(task)
    Xte, med = X.iloc[te], views.prep.median
    nodes = pd.read_csv(RES / task / "explainability" / "node_importance.csv")
    ranks = {"graph-smoothed (R)": list(nodes.sort_values("graph_importance", ascending=False).feature),
             "plain TreeSHAP (r)": list(nodes.sort_values("shap_importance", ascending=False).feature)}

    def f1_after(cols):
        Xd = Xte.copy()
        for c in cols:
            Xd[c] = med[c]
        raw, std = views.prep.transform(Xd)
        return f1_score(y[te], Variant(tgf).proba({"raw": raw, "std": std}).argmax(1), average="macro")

    for name, order in ranks.items():
        curve = {k: f1_after(order[:k]) for k in (1, 2, 4, 8, 12, 16, 24)}
        aopc = float(np.mean([curve[0] if 0 in curve else 0 for _ in [0]]))  # placeholder, replaced below
        base = f1_after([])
        aopc = float(np.mean([base - v for v in curve.values()]))  # mean drop over k (higher = more faithful)
        rows.append({"task": task, "ranking": name, "base_f1": base, "mean_drop_aopc": aopc, **{f"k{k}": v for k, v in curve.items()}})
        print(task, name, round(aopc * 100, 2), {k: round(v * 100, 1) for k, v in curve.items()}, flush=True)
    overlap = len(set(ranks["graph-smoothed (R)"][:10]) & set(ranks["plain TreeSHAP (r)"][:10]))
    rows.append({"task": task, "ranking": "top-10 overlap", "base_f1": overlap})
pd.DataFrame(rows).to_csv(RES / "review" / "fidelity_graph_vs_shap.csv", index=False)
