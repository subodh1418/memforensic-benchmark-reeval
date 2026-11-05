"""Label-overlap analysis: how many malware test dumps have a near-identical training twin with a different label."""
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from memdgf_core import Preprocessor, load_task, split_indices

X, y4, _, _, yf, _, _ = load_task("4class")
tr, va, te = split_indices(yf, 42)
p = Preprocessor().fit(X.iloc[tr])
d, i = NearestNeighbors(n_neighbors=1).fit(p.transform(X.iloc[tr])[1]).kneighbors(p.transform(X.iloc[te])[1])
d, i = d[:, 0], i[:, 0]
mal = y4[te] != 0
rows = []
for thr in (0.05, 0.1, 0.25, 0.5):
    close = (d < thr) & mal
    rows.append({"threshold": thr, "share_with_twin": float(close.sum() / mal.sum()),
                 "category_differs": float(np.mean(y4[tr][i][close] != y4[te][close])),
                 "family_differs": float(np.mean(yf[tr][i][close] != yf[te][close]))})
json.dump({"rows": rows, "median_nn_dist_malware": float(np.median(d[mal])), "median_nn_dist_benign": float(np.median(d[~mal]))},
          open(Path(__file__).resolve().parents[1] / "results" / "ceiling_analysis.json", "w"), indent=2)
print(rows)
