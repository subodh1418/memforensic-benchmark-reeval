"""Design study 4 (dev seed 7): label-hierarchy graph (family -> category -> malware)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

from memdgf_core import DATA_ROOT, load_task, split_indices

SEED = 7
X, y16, fam, _ = load_task("16class")
cat_df = pd.read_csv(DATA_ROOT / "4clascic.csv")
fam_df = pd.read_csv(DATA_ROOT / "16classcicmalmem.csv")
fam2cat = dict(zip(fam_df["Class"], cat_df["Class"]))
cats = sorted(set(fam2cat.values()))
F2C = np.zeros((len(fam), len(cats)))
for i, f in enumerate(fam):
    F2C[i, cats.index(fam2cat[f])] = 1
y4 = F2C[y16].argmax(1)
tr, va, te = split_indices(y16, SEED)
Xtr, Xva, Xte = X.iloc[tr].to_numpy(np.float32), X.iloc[va].to_numpy(np.float32), X.iloc[te].to_numpy(np.float32)


def lgb():
    return LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=14, verbose=-1)


def rep(name, y, p):
    print(f"{name:50s} acc={accuracy_score(y, p.argmax(1)):.4f} f1m={f1_score(y, p.argmax(1), average='macro'):.4f}", flush=True)


def fuse(pa_va, pb_va, y_va, pa, pb):
    ws = np.linspace(0, 1, 21)
    w = max(ws, key=lambda w: f1_score(y_va, (w * pa_va + (1 - w) * pb_va).argmax(1), average="macro"))
    return w * pa + (1 - w) * pb, w


m4 = lgb().fit(Xtr, y4[tr]); p4_va, p4 = m4.predict_proba(Xva), m4.predict_proba(Xte)
m16 = lgb().fit(Xtr, y16[tr]); p16_va, p16 = m16.predict_proba(Xva), m16.predict_proba(Xte)
print("--- 4-class")
rep("flat 4-class LGBM", y4[te], p4)
rep("family-trained LGBM aggregated to category", y4[te], p16 @ F2C)
pf, w = fuse(p4_va, p16_va @ F2C, y4[va], p4, p16 @ F2C)
rep(f"fusion flat+aggregated (w={w:.2f})", y4[te], pf)

print("--- 16-class")
rep("flat 16-class LGBM", y16[te], p16)
# hierarchical: P(fam) = P(cat) * P(fam | cat) with category-specialised family experts
cond_va, cond = np.zeros_like(p16_va), np.zeros_like(p16)
for c in range(len(cats)):
    members = np.where(F2C[:, c] == 1)[0]
    if len(members) == 1:
        cond_va[:, members] = 1; cond[:, members] = 1
        continue
    mask = np.isin(y16[tr], members)
    e = lgb().fit(Xtr[mask], y16[tr][mask])
    cond_va[:, e.classes_] = e.predict_proba(Xva); cond[:, e.classes_] = e.predict_proba(Xte)
ph_va = (p4_va @ F2C.T) * cond_va
ph = (p4 @ F2C.T) * cond
rep("hierarchical P(cat)*P(fam|cat)", y16[te], ph)
pf, w = fuse(p16_va, ph_va, y16[va], p16, ph)
rep(f"fusion flat+hierarchical (w={w:.2f})", y16[te], pf)
ph2 = ((p4 + p16 @ F2C) / 2) @ F2C.T * cond
rep("hierarchical with fused category prob", y16[te], ph2)
