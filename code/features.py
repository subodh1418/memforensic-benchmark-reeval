"""Memory-forensic feature engineering and hybrid (filter -> redundancy -> embedded/wrapper) feature selection.

Engineered features are deterministic row-wise transforms (no fitted statistics), so they cannot
leak information across splits. The hybrid selector is fitted on the training split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif

HANDLE_TYPES = ["nfile", "nevent", "ndesktop", "nkey", "nthread", "ndirectory", "nsemaphore", "ntimer", "nsection", "nmutant"]
PSX_VIEWS = ["not_in_pslist", "not_in_eprocess_pool", "not_in_ethread_pool", "not_in_pspcid_list", "not_in_csrss_handles", "not_in_session", "not_in_deskthrd"]


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b != 0, a / np.where(b == 0, 1, b), 0.0)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Domain features derived from Volatility plugin semantics (row-wise only)."""
    g = lambda c: X[c].to_numpy(float) if c in X else np.zeros(len(X))
    nproc = g("pslist.nproc")
    F = {}
    # per-process normalisation and handle-type composition (handles plugin)
    for h in HANDLE_TYPES:
        F[f"fe.handles_{h}_per_proc"] = _div(g(f"handles.{h}"), nproc)
        F[f"fe.handles_{h}_share"] = _div(g(f"handles.{h}"), g("handles.nhandles"))
    F["fe.threads_total"] = g("pslist.avg_threads") * nproc
    F["fe.thread_handles_per_thread"] = _div(g("handles.nthread"), F["fe.threads_total"])
    F["fe.orphan_processes"] = nproc - g("pslist.nppid")
    # loader-list inconsistencies: modules mapped in memory but unlinked from PEB lists (ldrmodules)
    unl = g("ldrmodules.not_in_load") + g("ldrmodules.not_in_init") + g("ldrmodules.not_in_mem")
    F["fe.ldr_unlinked_total"] = unl
    F["fe.ldr_unlinked_per_dll"] = _div(unl, g("dlllist.ndlls"))
    F["fe.ldr_load_minus_init"] = g("ldrmodules.not_in_load") - g("ldrmodules.not_in_init")
    F["fe.ldr_mem_share"] = _div(g("ldrmodules.not_in_mem"), unl)
    # code-injection density (malfind)
    F["fe.injections_per_proc"] = _div(g("malfind.ninjections"), nproc)
    F["fe.unique_injection_ratio"] = _div(g("malfind.uniqueInjections"), g("malfind.ninjections"))
    F["fe.commit_per_injection"] = _div(g("malfind.commitCharge"), g("malfind.ninjections"))
    F["fe.protection_per_injection"] = _div(g("malfind.protection"), g("malfind.ninjections"))
    # cross-view hidden-process consensus (psxview)
    psx = np.stack([g(f"psxview.{v}") for v in PSX_VIEWS], 1)
    psx_fa = np.stack([g(f"psxview.{v}_false_avg") for v in PSX_VIEWS], 1)
    F["fe.psx_hidden_total"] = psx.sum(1)
    F["fe.psx_hidden_max"] = psx.max(1)
    F["fe.psx_views_disagreeing"] = (psx > 0).sum(1)
    F["fe.psx_false_avg_total"] = psx_fa.sum(1)
    F["fe.psx_pslist_vs_pool"] = g("psxview.not_in_pslist") - g("psxview.not_in_eprocess_pool")
    F["fe.psx_hidden_per_proc"] = _div(psx.sum(1), nproc)
    # services, drivers and kernel callbacks (svcscan, modules, callbacks)
    F["fe.svc_active_ratio"] = _div(g("svcscan.nactive"), g("svcscan.nservices"))
    F["fe.svc_kernel_driver_ratio"] = _div(g("svcscan.kernel_drivers"), g("svcscan.nservices"))
    F["fe.svc_process_ratio"] = _div(g("svcscan.process_services"), g("svcscan.nservices"))
    F["fe.svc_shared_vs_own"] = _div(g("svcscan.shared_process_services"), g("svcscan.process_services"))
    F["fe.modules_minus_drivers"] = g("modules.nmodules") - g("svcscan.kernel_drivers") - g("svcscan.fs_drivers")
    F["fe.callbacks_anonymous_ratio"] = _div(g("callbacks.nanonymous"), g("callbacks.ncallbacks"))
    F["fe.callbacks_generic_ratio"] = _div(g("callbacks.ngeneric"), g("callbacks.ncallbacks"))
    F["fe.dlls_per_module"] = _div(g("dlllist.ndlls"), g("modules.nmodules"))
    F["fe.handles_per_dll"] = _div(g("handles.nhandles"), g("dlllist.ndlls"))
    return pd.DataFrame(F, index=X.index).astype(np.float32)


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([X, engineer(X)], axis=1)


class HybridSelector:
    """Stage 1 filter (MI + ANOVA rank fusion) -> stage 2 Spearman redundancy pruning
    -> stage 3 Boruta-style shadow test with LightGBM gain (embedded/wrapper)."""

    def __init__(self, keep_frac=0.8, rho_max=0.97, n_shadow_runs=5, min_hits=3, shadow_q=1.0, seed=42, n_jobs=4):
        self.keep_frac, self.rho_max, self.n_shadow_runs, self.min_hits = keep_frac, rho_max, n_shadow_runs, min_hits
        self.shadow_q = shadow_q  # 1.0 = classic Boruta (beat the best shadow); <1 = softer quantile threshold
        self.seed, self.n_jobs = seed, n_jobs

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xf = X.fillna(X.median())
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(Xf), size=min(15000, len(Xf)), replace=False)
        mi = mutual_info_classif(Xf.iloc[sub], y[sub], random_state=self.seed)
        f, _ = f_classif(Xf, y)
        score = pd.DataFrame({"mi": mi, "anova": np.nan_to_num(f)}, index=X.columns)
        score["rank"] = (score.mi.rank(ascending=False) + score.anova.rank(ascending=False)) / 2
        score = score.sort_values("rank")
        stage1 = list(score.index[: max(10, int(np.ceil(self.keep_frac * len(score))))])
        # stage 2: keep the better-ranked feature of near-duplicate pairs
        rho = Xf[stage1].rank().corr().abs().to_numpy()
        stage2 = []
        for i, c in enumerate(stage1):
            if all(rho[i, stage1.index(k)] < self.rho_max for k in stage2):
                stage2.append(c)
        # stage 3: Boruta-style shadow features with LightGBM gain
        hits = pd.Series(0, index=stage2)
        A = Xf[stage2].to_numpy(np.float32)
        for r in range(self.n_shadow_runs):
            sh = np.column_stack([rng.permutation(A[:, j]) for j in range(A.shape[1])])  # shadow copies
            m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               importance_type="gain", random_state=self.seed + r, n_jobs=self.n_jobs, verbose=-1)
            m.fit(np.hstack([A, sh])[sub], y[sub])
            imp = m.feature_importances_
            hits += (imp[: A.shape[1]] > np.quantile(imp[A.shape[1]:], self.shadow_q)).astype(int)
        self.selected = [c for c in stage2 if hits[c] >= self.min_hits]
        self.report = score.assign(stage1=score.index.isin(stage1), stage2=score.index.isin(stage2),
                                   shadow_hits=hits.reindex(score.index).fillna(0).astype(int), selected=score.index.isin(self.selected))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected]
