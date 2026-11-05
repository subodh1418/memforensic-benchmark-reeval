"""Category check on MalVol-25: do category models trained on CIC-MalMem-2022 recognise the infected ransomware images?

MalVol-25 shares no family with CIC-MalMem-2022 and contains ransomware almost only, so only one category can be
tested. For the infected images of the ten ransomware pairs we report
  argmax_ransomware   share whose most probable class (of four, incl. benign) is ransomware
  among_malware       share for which ransomware is the most probable of the three malware categories (benign
                      ignored, as in the leave-one-family-out category test); chance: one third
Output: results/malvol/category_check.csv
Usage: python malvol_category.py   (runs offline; uses both feature files when present)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memdgf_core import GraphViews, MemDGFConfig, fit_lgbm, load_task, split_indices
from memtgf import MemTGF, TGFConfig, Variant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "malvol"
RANSOMWARE = ["Cerber", "WannaCry", "GandCrab", "Locky.AZ", "Dharma", "SporaRansomware", "GoldenEye", "InfinityCrypt",
              "DeriaLock", "LuckyLcoker"]
FEATURES = {"Volatility 3": "malvol25_features.csv", "Volatility 2.6": "malvol25_features_vol2.csv"}


def main():
    X, y, classes, info, y_fam, families, M = load_task("4class")
    cols = list(X.columns)
    r, mal = classes.index("Ransomware"), [i for i, c in enumerate(classes) if c != "Benign"]
    tr, va, _ = split_indices(y_fam, 42)
    views = GraphViews(MemDGFConfig(seed=42, n_jobs=14)).fit(X.iloc[tr], y_fam[tr], families)
    B_tr, B_va = views.blocks(), views.blocks(X.iloc[va])
    lgb = fit_lgbm(B_tr["raw"], y[tr], B_va["raw"], y[va], len(classes), 42, 14)
    tgf = MemTGF(views, TGFConfig(seed=42, n_jobs=14), M).fit(B_tr, y[tr], y_fam[tr], B_va, y[va], y_fam[va])
    rows = []
    for tool, fname in FEATURES.items():
        if not (OUT / fname).exists():
            continue
        mv = pd.read_csv(OUT / fname)
        mv = mv[(mv.label == 1) & mv.pair.isin(RANSOMWARE) & (mv.pair != "W32.MyDoom.A.")].reset_index(drop=True)
        B_mv = views.blocks(mv.reindex(columns=cols).astype(float))
        for name, p in (("LightGBM", lgb.predict_proba(B_mv["raw"])), ("MemHSE", Variant(tgf).proba(dict(B_mv)))):
            rows.append({"extractor": tool, "model": name, "n_images": len(mv),
                         "argmax_ransomware": float((p.argmax(1) == r).mean()),
                         "among_malware": float((np.array(mal)[p[:, mal].argmax(1)] == r).mean()),
                         "malware_category_predicted": ";".join(f"{classes[k]}={int(n)}" for k, n in
                                                                 zip(*np.unique(np.array(mal)[p[:, mal].argmax(1)], return_counts=True))),
                         "pairs": ";".join(mv.pair)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "category_check.csv", index=False)
    print(res.drop(columns="pairs").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
