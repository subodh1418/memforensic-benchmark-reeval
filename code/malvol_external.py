"""External test on MalVol-25: train on the de-duplicated CIC-MalMem-2022 corpus, score the MalVol-25 dumps.

Input: results/malvol/malvol25_features.csv (from malvol_extract.py). Runs fully offline.
Reported per model:
  detection   share of infected dumps with P(malware) >= 0.5
  fpr         share of clean dumps with P(malware) >= 0.5
  auc         ROC-AUC over all dumps (threshold-free)
  pair_acc    share of clean/infected pairs (same machine and OS) in which the infected dump scores higher;
              this controls for the machine and Windows version, which differ completely from CIC-MalMem
Also writes a covariate-shift report: per feature, the share of MalVol dumps outside the CIC 1st-99th percentile.
Output: results/malvol/external_scores.csv, external_summary.csv, shift_report.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, load_task, split_indices
from memtgf import MemTGF, TGFConfig, Variant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "malvol"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="malvol25_features.csv", help="feature file in results/malvol")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _vol2")
    ap.add_argument("--pairs", nargs="*", help="restrict to these pairs (default: all with features)")
    ap.add_argument("--exclude", nargs="*", default=["W32.MyDoom.A."], help="pairs to drop")
    a = ap.parse_args()
    mv = pd.read_csv(OUT / a.features)
    if a.pairs:
        mv = mv[mv.pair.isin(a.pairs)]
    mv = mv[~mv.pair.isin(a.exclude or [])].reset_index(drop=True)
    X, y, classes, info, y_fam, families, M = load_task("binary")
    cols = list(X.columns)
    missing = [c for c in cols if c not in mv.columns or mv[c].isna().all()]
    print(f"MalVol dumps: {len(mv)} | CIC features: {len(cols)} | unavailable in Volatility 3: {missing}")
    Xm = mv.reindex(columns=cols).astype(float)

    tr, va, _ = split_indices(y_fam, 42)  # train/validation from CIC only; the whole of MalVol-25 is the test set
    views = GraphViews(MemDGFConfig(seed=42, n_jobs=14)).fit(X.iloc[tr], y_fam[tr], families)
    B_tr, B_va, B_mv = views.blocks(), views.blocks(X.iloc[va]), views.blocks(Xm)  # medians from CIC training fill the gaps
    scores = {}
    lgb = fit_lgbm(B_tr["raw"], y[tr], B_va["raw"], y[va], 2, 42, 14)
    scores["LightGBM"] = lgb.predict_proba(B_mv["raw"])[:, 1]
    scores["Random forest"] = RandomForestClassifier(500, n_jobs=14, random_state=42).fit(B_tr["raw"], y[tr]).predict_proba(B_mv["raw"])[:, 1]
    scores["Extra trees"] = ExtraTreesClassifier(500, n_jobs=14, random_state=42).fit(B_tr["raw"], y[tr]).predict_proba(B_mv["raw"])[:, 1]
    tgf = MemTGF(views, TGFConfig(seed=42, n_jobs=14), M).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
    scores["MemHSE"] = Variant(tgf).proba(dict(B_mv))[:, 1]

    sc = mv[["dump", "pair", "label"]].copy()
    rows = []
    for name, p in scores.items():
        sc[name] = p
        lab = mv.label.to_numpy()
        pairs = sc.pivot_table(index="pair", columns="label", values=name).dropna()
        rows.append({"model": name, "n_dumps": len(lab), "detection": float((p[lab == 1] >= 0.5).mean()),
                     "fpr": float((p[lab == 0] >= 0.5).mean()),
                     "auc": float(roc_auc_score(lab, p)) if len(set(lab)) == 2 else np.nan,
                     "pair_acc": float((pairs[1] > pairs[0]).mean()) if len(pairs) else np.nan, "n_pairs": len(pairs)})
    summ = pd.DataFrame(rows)
    sc.to_csv(OUT / f"external_scores{a.tag}.csv", index=False)
    summ.to_csv(OUT / f"external_summary{a.tag}.csv", index=False)

    lo, hi = X.iloc[tr].quantile(0.01), X.iloc[tr].quantile(0.99)
    shift = pd.DataFrame({"feature": cols, "available": [c not in missing for c in cols],
                          "share_outside_cic_p1_p99": [float(((Xm[c] < lo[c]) | (Xm[c] > hi[c])).mean()) if c not in missing else np.nan for c in cols],
                          "cic_median": X.iloc[tr].median().values, "malvol_median": Xm.median().values})
    shift.to_csv(OUT / f"shift_report{a.tag}.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print("\nmost shifted features:\n", shift.sort_values("share_outside_cic_p1_p99", ascending=False).head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
