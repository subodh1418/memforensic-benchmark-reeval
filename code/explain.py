"""Graph-based explainability for MemTGF (run after run_experiments.py has saved bundle_seed42).

Outputs per task (results/<task>/explainability/):
  node_importance.csv         FIG node attributions (TreeSHAP of the family expert, aggregated along LHG)
  edge_importance.csv         FIG edge co-attribution  e_ij = A_ij * sqrt(I_i * I_j)
  community_importance.csv    subgraph (Louvain community) attribution + meta-level expert contribution
  class_feature_importance.csv per-class node attribution
  deletion_fidelity.csv       macro-F1 after deleting top-k graph-ranked nodes vs random nodes / communities
  local_explanation.json      worked example: LHG path, FIG ego-subgraph, SSG prototypes
  figures (*.png, *.pdf)
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score

from memdgf_core import load_task
from memtgf import Variant, _log

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = Path(__file__).resolve().parents[1] / __import__("os").environ.get("MEMTGF_PAPER", "paper") / "figures"
MINIMAL = __import__("os").environ.get("MEMTGF_MINIMAL") == "1"  # submission style
GROUP_COLORS = {
    "pslist": "#4C78A8", "dlllist": "#F58518", "handles": "#54A24B", "ldrmodules": "#B279A2", "malfind": "#E45756",
    "psxview": "#72B7B2", "modules": "#EECA3B", "svcscan": "#9D755D", "callbacks": "#BAB0AC",
}
COMM_COLORS = ["#2a78d6", "#e87ba4", "#1baf7a", "#6250d6", "#eda100", "#008300", "#e34948", "#52514e"]


def short(c):
    return c.replace("psxview.", "psx.").replace("ldrmodules.", "ldr.").replace("handles.", "hdl.").replace("svcscan.", "svc.").replace("callbacks.", "cb.").replace("_false_avg", "_fa")


def task_shap(model, X, M):
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 2:  # binary objective -> single margin
        sv = np.stack([-sv, sv], -1)
    return np.einsum("nfk,kt->nft", sv, M)  # margin-space aggregation along LHG edges


def main(task, n_shap=2000, seed=42):
    out = RESULTS / task / "explainability"
    out.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    b = joblib.load(RESULTS / task / f"bundle_seed{seed}.joblib")
    views, tgf, (tr, va, te) = b["views"], b["memtgf"], b["split"]
    classes, families, M = b["classes"], b["families"], tgf.M
    X, y, *_ = load_task(task)
    cols = views.prep.columns
    rng = np.random.default_rng(seed)
    te_s = rng.choice(te, size=min(n_shap, len(te)), replace=False)
    Xs = views.prep.transform(X.iloc[te_s])[0]

    # ---------------------------------------------------------------- node level
    full_model = tgf.models["E_full"][0]
    S = task_shap(full_model, Xs, M)  # n x F x T
    I = np.abs(S).mean((0, 2))
    I = I / I.sum()
    A = views.fig.A
    Ahat = views.fig.Ahat
    I_graph = 0.5 * I + 0.5 * (Ahat @ I) / (Ahat @ I).sum()  # one-hop graph-smoothed relevance
    comm_of = np.zeros(len(cols), int)
    for j, c in enumerate(tgf.comms):
        comm_of[c] = j
    nodes = pd.DataFrame({"feature": cols, "plugin": [c.split(".")[0] for c in cols], "community": comm_of,
                          "shap_importance": I, "graph_importance": I_graph, "degree": (A > 0).sum(1)})
    nodes.sort_values("graph_importance", ascending=False).to_csv(out / "node_importance.csv", index=False)

    ii, jj = np.triu_indices(len(cols), 1)
    w = A[ii, jj]
    keep = w > 0
    edges = pd.DataFrame({"u": np.array(cols)[ii[keep]], "v": np.array(cols)[jj[keep]], "weight": w[keep],
                          "co_attribution": w[keep] * np.sqrt(I[ii[keep]] * I[jj[keep]])})
    edges.sort_values("co_attribution", ascending=False).to_csv(out / "edge_importance.csv", index=False)

    # per-class node attribution (samples of that class only)
    yt = y[te_s]
    cls_imp = pd.DataFrame({c: np.abs(S[yt == t, :, t]).mean(0) for t, c in enumerate(classes)}, index=cols)
    cls_imp.to_csv(out / "class_feature_importance.csv")

    # ------------------------------------------------------- subgraph / expert level
    experts = list(tgf.experts)
    meta = tgf._meta_model(experts)
    B = views.blocks(X.iloc[te_s])
    P = tgf.expert_proba(B)
    Z = np.hstack([_log(P[n]) for n in experts])
    mv = shap.TreeExplainer(meta).shap_values(Z[:1000])
    mv = np.stack(mv, -1) if isinstance(mv, list) else np.asarray(mv)
    K = tgf.n_exp_classes
    pred_fam = meta.predict(Z[:1000])
    contrib = np.stack([np.abs(mv[np.arange(len(pred_fam)), e * K:(e + 1) * K, pred_fam]).sum(1) for e in range(len(experts))], 1)
    share = contrib / contrib.sum(1, keepdims=True)
    exp_share = pd.Series(share.mean(0), index=experts)
    comm_rows = []
    for j, c in enumerate(tgf.comms):
        comm_rows.append({"expert": f"E_comm{j}", "n_nodes": len(c), "plugins": ",".join(sorted({cols[i].split('.')[0] for i in c})),
                          "node_attribution_sum": float(I[c].sum()), "meta_contribution_share": float(exp_share[f"E_comm{j}"]),
                          "top_nodes": ";".join([cols[i] for i in np.array(c)[np.argsort(-I[c])][:5]])})
    comm_rows.append({"expert": "E_full", "n_nodes": len(cols), "plugins": "all", "node_attribution_sum": 1.0,
                      "meta_contribution_share": float(exp_share["E_full"]), "top_nodes": ""})
    comm_rows.append({"expert": "E_sage", "n_nodes": 0, "plugins": "SSG neighbourhood", "node_attribution_sum": np.nan,
                      "meta_contribution_share": float(exp_share["E_sage"]), "top_nodes": ""})
    pd.DataFrame(comm_rows).to_csv(out / "community_importance.csv", index=False)
    # per-task-class expert share
    task_pred = (np.eye(K)[pred_fam] @ M).argmax(1)
    cls_share = pd.DataFrame({classes[t]: share[task_pred == t].mean(0) if (task_pred == t).any() else np.nan for t in range(len(classes))}, index=experts)
    cls_share.to_csv(out / "class_expert_share.csv")

    # ------------------------------------------------------------- deletion fidelity
    Xte = X.iloc[te]
    med = views.prep.median
    full = Variant(tgf)

    def f1_after(drop_cols):
        Xd = Xte.copy()
        for c in drop_cols:
            Xd[c] = med[c]
        return f1_score(y[te], full.proba(views.blocks(Xd)).argmax(1), average="macro")

    rank_graph = list(nodes.sort_values("graph_importance", ascending=False)["feature"])
    rows = []
    for k in (0, 1, 2, 4, 6, 8, 12, 16, 24):
        rows.append({"k": k, "ranking": "graph (MemTGF-XAI)", "f1_macro": f1_after(rank_graph[:k])})
        rr = [f1_after(list(rng.choice(cols, k, replace=False))) for _ in range(3)] if k else [rows[0]["f1_macro"]]
        rows.append({"k": k, "ranking": "random", "f1_macro": float(np.mean(rr)), "f1_std": float(np.std(rr))})
        print(f"[{task}] deletion k={k} graph={rows[-2]['f1_macro']:.4f} random={rows[-1]['f1_macro']:.4f}", flush=True)
    for j, c in enumerate(tgf.comms):
        rows.append({"k": len(c), "ranking": f"delete community E_comm{j}", "f1_macro": f1_after([cols[i] for i in c])})
    del_df = pd.DataFrame(rows)
    del_df.to_csv(out / "deletion_fidelity.csv", index=False)

    # ------------------------------------------------------------ local explanation
    benign = classes.index("Benign")
    full_p = full.proba(views.blocks(Xte))
    pred = full_p.argmax(1)
    cand = np.where((y[te] != benign) & (pred == y[te]))[0]
    pick = cand[np.argmax(full_p[cand].max(1))] if task == "binary" else cand[np.argsort(full_p[cand].max(1))[len(cand) // 2]]
    xi = Xte.iloc[[pick]]
    Bi = views.blocks(xi)
    p_fam = tgf._meta_model(experts).predict_proba(np.hstack([_log(tgf.expert_proba(Bi)[n]) for n in experts]))[0]
    s_i = task_shap(full_model, views.prep.transform(xi)[0], M)[0, :, pred[pick]]
    top = np.argsort(-np.abs(s_i))[:8]
    nbr, dist = views.ssg.neighbours(Bi["std"])
    fam_names = np.array(families)[views.ssg.y_ref[nbr[0]]]
    local = {
        "test_row": int(te[pick]), "true_class": classes[y[te][pick]], "predicted_class": classes[pred[pick]],
        "task_probabilities": dict(zip(classes, map(float, full_p[pick]))),
        "lhg_path": {"P(malware)": float(1 - p_fam[families.index("Benign")]),
                     "top_families": {families[k]: float(p_fam[k]) for k in np.argsort(-p_fam)[:3]}},
        "fig_evidence": [{"feature": cols[f], "value": float(xi.iloc[0][cols[f]]), "shap": float(s_i[f]), "community": int(comm_of[f])} for f in top],
        "fig_evidence_edges": [{"u": cols[a], "v": cols[c], "weight": float(A[a, c])} for a in top for c in top if a < c and A[a, c] > 0],
        "ssg_prototypes": [{"family": str(f), "distance": float(d)} for f, d in zip(fam_names[:5], dist[0][:5])],
    }
    json.dump(local, open(out / "local_explanation.json", "w"), indent=2)

    args = (task, out, cols, A, I, I_graph, comm_of, edges, cls_imp, exp_share, cls_share, del_df, local, tgf.comms)
    joblib.dump(args, out / "plot_cache.joblib")
    plot_all(*args)
    print(f"[{task}] explainability done -> {out}")


def plot_all(task, out, cols, A, I, I_graph, comm_of, edges, cls_imp, exp_share, cls_share, del_df, local, comms):
    name = {"binary": "Binary task", "4class": "Category task", "16class": "Family task"}.get(task, task)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    # 1. FIG explanation graph
    G = nx.Graph()
    for j, c in enumerate(cols):
        G.add_node(j)
    for _, r in edges.head(70).iterrows():
        G.add_edge(cols.index(r.u), cols.index(r.v), w=r.co_attribution)
    # community-sector layout: each Louvain community on its own ring, nodes ordered by attribution
    pos, centre_of = {}, {}
    m = len(comms)
    R = 1.25 * max(0.4 + 0.075 * len(c) for c in comms) / np.sin(np.pi / max(m, 2)) if m > 1 else 0.0
    for j, c in enumerate(comms):
        cx, cy = R * np.cos(2 * np.pi * j / m + np.pi / 2), R * np.sin(2 * np.pi * j / m + np.pi / 2)
        order = np.array(c)[np.argsort(-I_graph[c])]
        r = 0.4 + 0.075 * len(c)
        for t, node in enumerate(order):
            a = 2 * np.pi * t / len(order)
            pos[node] = (cx + r * np.cos(a), cy + r * np.sin(a))
            centre_of[node] = (cx, cy)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ew = np.array([G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3 + 4 * ew / ew.max(), edge_color="#7f7f7f", alpha=0.45)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25 + 650 * I_graph / I_graph.max(),
                           node_color=[COMM_COLORS[c % len(COMM_COLORS)] for c in comm_of], edgecolors="white", linewidths=0.7)
    placed = []  # label anchors already used, to keep labels apart (minimal style)
    for k in np.argsort(-I_graph)[:10 if MINIMAL else 14]:
        (x, y_), (cx, cy) = pos[k], centre_of[k]
        d = np.array([x - cx, y_ - cy])
        d = d / (np.linalg.norm(d) + 1e-9)
        tx, ty = x + 0.5 * d[0], y_ + 0.5 * d[1]
        if MINIMAL:
            while any(abs(ty - py) < 0.16 and (tx >= x) == (px >= qx) and abs(tx - px) < 1.6 for px, py, qx in placed):
                ty += 0.16 if d[1] >= 0 else -0.16
            placed.append((tx, ty, x))
        ax.annotate(short(cols[k]), (x, y_), xytext=(tx, ty), fontsize=7 if MINIMAL else 6.5,
                    ha="left" if d[0] >= 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"))
    for j, c in enumerate(comms):
        if MINIMAL:
            top = [p for p, _ in Counter(cols[i].split(".")[0] for i in c).most_common(3)]
            label = f"$C_{j}$: {', '.join(top)} ({len(c)})"
        else:
            plugins = sorted({cols[i].split(".")[0] for i in c})
            label = f"C{j} ({len(c)} nodes): {', '.join(plugins[:4])}{'…' if len(plugins) > 4 else ''}"
        ax.scatter([], [], c=COMM_COLORS[j % len(COMM_COLORS)], s=40, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=7.5 if MINIMAL else 7, frameon=False,
              ncol=3 if MINIMAL else 1)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"xai_fig_graph_{task}.{ext}", dpi=300, **({"bbox_inches": "tight", "pad_inches": 0.05} if MINIMAL else {}))
    plt.close(fig)

    # family figures: benign first, then families grouped by category (same order as the confusion matrix)
    order16 = ["Benign", "Ako", "Conti", "Maze", "Pysa", "Shade", "180solutions", "CWS", "Gator", "TIBS", "Transponder",
               "Emotet", "Reconyc", "Refroso", "Scar", "Zeus"]
    seps = []
    if sorted(order16) == sorted(cls_imp.columns):
        cls_imp, cls_share, seps = cls_imp[order16], cls_share[order16], [0.5, 5.5, 10.5]

    # 2. per-class heatmap (top 15 nodes)
    topf = cls_imp.sum(1).sort_values(ascending=False).index[:15]
    H = cls_imp.loc[topf]
    H = H / H.max(0)
    fig, ax = plt.subplots(figsize=(max(4, 0.42 * H.shape[1] + 2.5), 4.6))
    im = ax.imshow(H.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(topf)), [short(c) for c in topf], fontsize=7)
    ax.set_xticks(range(H.shape[1]), H.columns, rotation=60, ha="right", fontsize=7)
    for s in seps:
        ax.axvline(s, color="white", lw=1.2)
    fig.colorbar(im, ax=ax, label="relative attribution (per class)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"xai_class_heatmap_{task}.{ext}", dpi=300)
    plt.close(fig)

    # 3. deletion fidelity
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for rk, st, colr, lab in (("graph (MemTGF-XAI)", "-o", "#2a78d6", "top-ranked" if MINIMAL else "graph-ranked"), ("random", "--s", "#a3a29c", "random")):
        d = del_df[del_df.ranking == rk]
        ax.plot(d.k, 100 * d.f1_macro, st, ms=3.2, lw=1.6, color=colr, label=lab)
    ax.set_xlabel("deleted indicators $k$", fontsize=8)
    ax.set_ylabel("macro-F1 (%)", fontsize=8)
    ax.set_title({"binary": "(a) Binary", "4class": "(b) Category", "16class": "(c) Family"}.get(task, ""), fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"xai_deletion_{task}.{ext}", dpi=300)
    plt.close(fig)

    # 4. expert (subgraph / SSG) share per class
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * cls_share.shape[1] + 2.5), 3.2))
    bottom = np.zeros(cls_share.shape[1])
    for k, (ename, row) in enumerate(cls_share.iterrows()):
        if ename.startswith("E_comm"):
            members = [cols[i].split(".")[0] for i in comms[int(ename[-1])]]
            top = [p for p, _ in Counter(members).most_common(3)]
            nice = "$E_{C_" + ename[-1] + "}$ (" + ", ".join(top) + ")"
        else:
            nice = {"E_full": "$E_{full}$ (all indicators)", "E_sage": "$E_{SAGE}$ (SSG)"}[ename]
        colr = {"E_full": "#a3a29c", "E_sage": "#eb6834"}.get(ename) or COMM_COLORS[int(ename[-1]) % len(COMM_COLORS)]
        ax.bar(range(cls_share.shape[1]), 100 * row.values, bottom=bottom, label=nice, color=colr, edgecolor="white", linewidth=0.6)
        bottom += 100 * np.nan_to_num(row.values)
    ax.set_xticks(range(cls_share.shape[1]), cls_share.columns, rotation=60, ha="right", fontsize=7)
    for s in seps:
        ax.axvline(s, color="#52514e", lw=0.6, ls=":")
    ax.set_ylabel("share of decision (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"xai_expert_share_{task}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot_all(*joblib.load(RESULTS / a.task / "explainability" / "plot_cache.joblib"))
    else:
        main(a.task)
