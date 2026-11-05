"""Within-corpus check on MalVol-25: can infection be detected at all when training on MalVol-25 itself?

Leave-one-pair-out over the clean/infected pairs. For each held-out pair we report whether the infected image is ranked
above its clean twin (pair accuracy; chance 0.5, one-sided binomial test).
  absolute   classifier on the image indicators (signed log, standardised on the training pairs)
  paired     classifier on within-pair differences d = x_infected - x_clean (signed log); training uses d -> infected and
             -d -> clean, and a held-out pair is correct if d is classified as infected. Logistic regression without
             intercept keeps the problem symmetric.
  loader     paired, restricted to the six ldrmodules indicators, chosen a priori from the CIC-MalMem-2022 explanations
Models: L2 logistic regression (C=0.1) and random forest (500 trees). Rows with a missing indicator are median-filled.
Output: results/malvol/within_malvol.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "results" / "malvol"
SETS = {"Volatility 3": ("malvol25_features.csv", ["W32.MyDoom.A."]), "Volatility 2.6": ("malvol25_features_vol2.csv", [])}


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def lopo(X0, X1, model, paired):
    """X0, X1: (pairs, features) clean and infected images. Returns per-pair correctness."""
    n, ok = len(X0), []
    for i in range(n):
        tr = np.arange(n) != i
        if paired:
            D = X1[tr] - X0[tr]
            sc = StandardScaler(with_mean=False).fit(np.vstack([D, -D]))
            A, b = sc.transform(np.vstack([D, -D])), np.r_[np.ones(len(D)), np.zeros(len(D))]
            m = model().fit(A, b)
            d = sc.transform((X1[i] - X0[i])[None])
            ok.append(m.predict_proba(d)[0, 1] > m.predict_proba(-d)[0, 1])
        else:
            A = np.vstack([X0[tr], X1[tr]])
            sc = StandardScaler().fit(A)
            m = model().fit(sc.transform(A), np.r_[np.zeros(tr.sum()), np.ones(tr.sum())])
            p0, p1 = m.predict_proba(sc.transform(X0[i:i + 1]))[0, 1], m.predict_proba(sc.transform(X1[i:i + 1]))[0, 1]
            ok.append(p1 > p0)
    return np.array(ok)


def main():
    models = {"Logistic regression": lambda paired=False: LogisticRegression(C=0.1, fit_intercept=not paired, max_iter=5000),
              "Random forest": lambda paired=False: RandomForestClassifier(500, min_samples_leaf=2, random_state=42, n_jobs=8)}
    rows = []
    for tool, (fname, excl) in SETS.items():
        mv = pd.read_csv(OUT / fname)
        mv = mv[~mv.pair.isin(excl)]
        feats = [c for c in mv.columns if c not in ("dump", "label", "pair", "profile") and mv[c].notna().any()]
        mv[feats] = mv[feats].fillna(mv[feats].median())
        pairs = sorted(mv.pair.unique())
        get = lambda lab, cols: slog(mv[mv.label == lab].set_index("pair").loc[pairs, cols].to_numpy(float))
        loader = [c for c in feats if c.startswith("ldrmodules.")]
        for variant, cols, paired in (("absolute", feats, False), ("paired", feats, True), ("loader (paired)", loader, True)):
            X0, X1 = get(0, cols), get(1, cols)
            for mname, mk in models.items():
                ok = lopo(X0, X1, lambda: mk(paired), paired)
                rows.append({"extractor": tool, "variant": variant, "model": mname, "n_pairs": len(ok), "correct": int(ok.sum()),
                             "pair_acc": ok.mean(), "p_one_sided": binomtest(int(ok.sum()), len(ok), 0.5, alternative="greater").pvalue})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "within_malvol.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
