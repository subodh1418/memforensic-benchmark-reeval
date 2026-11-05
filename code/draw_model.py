"""Proposed-model (MemTGF architecture) figure: paper/figures/model.pdf/png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / __import__("os").environ.get("MEMTGF_PAPER", "paper") / "figures"
MINIMAL = __import__("os").environ.get("MEMTGF_MINIMAL") == "1"  # submission style: labels only, details in the caption
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
GRAY, LIGHT, INK, MUTED = "#a3a29c", "#e9e8e4", "#0b0b0b", "#52514e"
COMM = [BLUE, "#e87ba4", AQUA]


def panel(ax, x, y, w, h, title, color, alpha=0.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=color, ec="none", alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc="none", ec=color, lw=1.1))
    ax.text(x + 0.008, y + h - 0.022, title, fontsize=6.6, weight="bold", color=INK, va="center")


def arrow(ax, p0, p1, color=MUTED, ls="-", rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def probbar(ax, x, y, color, n=8, seed=0):
    v = np.random.default_rng(seed).dirichlet(np.ones(n) * 0.6)
    for i, p in enumerate(v):
        ax.add_patch(Rectangle((x + i * 0.0062, y), 0.0048, (0.02 if MINIMAL else 0.045) * p / v.max(), fc=color, ec="none"))


fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
rng = np.random.default_rng(4)

# ---- input vector
panel(ax, 0.005, 0.25, 0.085, 0.55, "Input", GRAY, 0.05)
cells = np.repeat([0, 1, 2], [9, 5, 5])
rng.shuffle(cells)
for i, c in enumerate(cells):
    ax.add_patch(Rectangle((0.03, 0.29 + i * 0.023), 0.035, 0.019, fc=COMM[c], ec="white", lw=0.4))
if not MINIMAL:
    ax.text(0.0475, 0.75, "x ∈ ℝᵈ", fontsize=6.6, ha="center", color=INK)
if not MINIMAL:  # explained in the caption; the label overflowed the box at print size
    ax.text(0.0475, 0.265, "memory\nindicators", fontsize=5.4, ha="center", va="top", color=MUTED)

# ---- FIG panel
panel(ax, 0.115, 0.68, 0.24, 0.30, "① Feature graph (FIG)" if MINIMAL else "① Feature-Interaction Graph", BLUE)
centres = [(0.17, 0.80), (0.245, 0.86), (0.30, 0.775)]
if MINIMAL:  # centre the clusters vertically once the annotation line is gone
    centres = [(cx, cy - 0.03) for cx, cy in centres]
pos = []
for c, (cx, cy) in enumerate(centres):
    for k in range(5):
        a = 2 * np.pi * k / 5 + c
        pos.append((cx + 0.028 * np.cos(a), cy + 0.042 * np.sin(a), c))
for i, (x1, y1, c1) in enumerate(pos):
    for j, (x2, y2, c2) in enumerate(pos):
        if i < j and (c1 == c2 or rng.random() < 0.04):
            ax.plot([x1, x2], [y1, y2], color=GRAY if c1 != c2 else COMM[c1], lw=0.6, alpha=0.7, zorder=1)
for x1, y1, c1 in pos:
    ax.scatter([x1], [y1], s=34, c=COMM[c1], edgecolors="white", linewidths=0.6, zorder=2)
if not MINIMAL:
    ax.text(0.235, 0.705, "|ρ|≥τ ∪ top-κ ∪ plugin prior → Louvain C₀,C₁,C₂", fontsize=5.3, ha="center", color=MUTED)

# ---- SSG panel
panel(ax, 0.115, 0.36, 0.24, 0.29, "② Sample graph (SSG)" if MINIMAL else "② Sample-Similarity Graph", ORANGE)
pts = rng.uniform([-0.10, -0.085], [0.10, 0.085], (30, 2)) + [0.235, 0.505]
lab = rng.integers(0, 3, 26)
q = np.array([0.235, 0.505])
d = np.linalg.norm(pts - q, axis=1)
for i in np.argsort(d)[:5]:
    ax.plot([q[0], pts[i, 0]], [q[1], pts[i, 1]], color=ORANGE, lw=0.9, zorder=1)
for (px, py), l in zip(pts, lab):
    ax.scatter([px], [py], s=16, c=[GRAY, "#c9c8c2", "#7f7e79"][l], edgecolors="white", linewidths=0.4, zorder=2)
ax.plot(*q, marker="*", ms=11, color=RED, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
if not MINIMAL:
    ax.text(0.235, 0.385, "unseen dump ★ attaches to k train neighbours", fontsize=5.3, ha="center", color=MUTED)

# ---- LHG panel
panel(ax, 0.115, 0.03, 0.24, 0.30, "③ Label hierarchy (LHG)" if MINIMAL else "③ Label-Hierarchy Graph", VIOLET)
root = (0.235, 0.265)
lvl1 = [(0.165, 0.205), (0.28, 0.205)]
lvl2 = [(0.235, 0.14), (0.28, 0.14), (0.325, 0.14)]
ax.plot(*zip(root, lvl1[0]), color=VIOLET, lw=0.8)
ax.plot(*zip(root, lvl1[1]), color=VIOLET, lw=0.8)
for p in lvl2:
    ax.plot(*zip(lvl1[1], p), color=VIOLET, lw=0.8)
    for k in range(5):
        fx = p[0] - 0.016 + k * 0.008
        ax.plot([p[0], fx], [p[1], 0.075], color=VIOLET, lw=0.4, alpha=0.7)
        ax.scatter([fx], [0.072], s=5, c=VIOLET, zorder=3)
ax.plot([lvl1[0][0], lvl1[0][0]], [lvl1[0][1], 0.075], color=VIOLET, lw=0.6)
ax.scatter([lvl1[0][0]], [0.072], s=5, c=VIOLET, zorder=3)
for p, t in [(root, ""), (lvl1[0], "Benign"), (lvl1[1], "Malware")] + [(p, n) for p, n in zip(lvl2, ["R", "S", "T"])]:
    ax.scatter([p[0]], [p[1]], s=30, c="white", edgecolors=VIOLET, linewidths=1.0, zorder=3)
    if t and MINIMAL and t in ("R", "S", "T"):  # beside the node, so the Malware->S edge does not cross the letter
        ax.text(p[0] - 0.009, p[1], t, fontsize=5.3, ha="right", va="center", color=INK)
    elif t:
        ax.text(p[0] + (0.012 if t not in ("R", "S", "T") else 0), p[1] + (0.0 if t not in ("R", "S", "T") else 0.016), t, fontsize=5.3,
                ha="left" if t not in ("R", "S", "T") else "center", va="center", color=INK)
if not MINIMAL:
    ax.text(0.235, 0.045, r"benign + 15 family leaves $\rightarrow M_t$", fontsize=5.3, ha="center", color=MUTED)

# input to graphs
for yy in ((0.83, 0.50) if MINIMAL else (0.83, 0.50, 0.18)):  # the LHG comes from labels, not from the input dump
    arrow(ax, (0.092, 0.52), (0.113, yy))

# ---- experts
if MINIMAL:  # evenly spaced experts in a tighter panel
    panel(ax, 0.385, 0.29, 0.20, 0.69, "Experts", GRAY, 0.04)
else:
    panel(ax, 0.385, 0.20, 0.20, 0.78, "Experts (family-supervised)", GRAY, 0.04)
experts = [(r"$E_{C_0}$: GBDT on $C_0$", COMM[0]), (r"$E_{C_1}$: GBDT on $C_1$", COMM[1]), (r"$E_{C_2}$: GBDT on $C_2$", COMM[2]),
           (r"$E_{full}$: GBDT on all $C_j$" if MINIMAL else r"$E_{full}$: GBDT, all", GRAY), (r"$E_{SAGE}$: GraphSAGE", ORANGE)]
ey = [0.88, 0.76, 0.64, 0.52, 0.40] if MINIMAL else [0.88, 0.76, 0.64, 0.52, 0.32]
for (t, c), y in zip(experts, ey):
    ax.add_patch(FancyBboxPatch((0.395, y - 0.035), 0.18, 0.075, boxstyle="round,pad=0.003,rounding_size=0.01", fc="white", ec=c, lw=1.2))
    ax.text(0.402, y + (0.016 if MINIMAL else 0.012), t, fontsize=5.6, color=INK, va="center")
    probbar(ax, 0.50, y - 0.028, c, seed=int(y * 100))
    if not MINIMAL:
        ax.text(0.455, y - 0.018, "p̂ ∈ Δ¹⁶", fontsize=5, color=MUTED, va="center")
for y in ey[:3]:
    arrow(ax, (0.357, 0.83), (0.393, y), BLUE)
arrow(ax, (0.357, 0.80), (0.393, ey[3]), GRAY)
arrow(ax, (0.357, 0.50), (0.393, ey[4]), ORANGE)
arrow(ax, (0.357, 0.205), (0.393, 0.31) if MINIMAL else (0.393, 0.245), VIOLET, ls="--")
if not MINIMAL:
    ax.text(0.398, 0.245, "family labels f\nsupervise every expert", fontsize=5.2, color=VIOLET, style="italic", va="center")

# ---- meta learner
panel(ax, 0.615, 0.45, 0.13, 0.40, "Stacking", AQUA)
for k in range(5):
    ax.add_patch(Rectangle((0.63 + k * 0.021, 0.66), 0.017, 0.11, fc=AQUA if k else "white", ec=AQUA, lw=0.8, alpha=0.6 if k else 1))
ax.text(0.68, 0.635, "out-of-fold\nlog-probs z" if MINIMAL else "5-fold OOF\nlog-probabilities z", fontsize=5.4, ha="center", va="top", color=INK)
ax.add_patch(FancyBboxPatch((0.622, 0.475) if MINIMAL else (0.628, 0.475), 0.117 if MINIMAL else 0.105, 0.06, boxstyle="round,pad=0.003,rounding_size=0.01", fc=AQUA, ec="none", alpha=0.85))
ax.text(0.68, 0.505, "meta GBDT h" if MINIMAL else "meta GBDT h(z)", fontsize=5.8, ha="center", va="center", color="white", weight="bold")
for y in ey:
    arrow(ax, (0.577, y), (0.613, 0.66), AQUA)

# ---- aggregation + fusion + output
panel(ax, 0.775, 0.55, 0.215, 0.30, "LHG aggregation", VIOLET)
# family -> category map: benign plus an equal number of families per category, as in the LHG tree
rows = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3] if MINIMAL else [0, 1, 1, 2, 2, 3, 3, 3]
n, ch = len(rows), (0.021 if MINIMAL else 0.026)
for i, r in enumerate(rows):
    for j in range(4):
        ax.add_patch(Rectangle((0.79 + j * 0.016, 0.585 + (n - 1 - i) * ch), 0.014, ch - 0.003, fc=VIOLET if j == r else "white", ec="#cfcde6", lw=0.4))
ax.text(0.905, 0.71, r"$\mathbf{p}^{t}=\mathbf{p}^{\mathcal{F}}M_t$", fontsize=7.2, ha="center", va="center", color=INK)
arrow(ax, (0.747, 0.505), (0.80, 0.585), AQUA)

FY = (0.31, 0.19, 0.395) if MINIMAL else (0.25, 0.26, 0.40)  # fusion box: y, height, formula y
panel(ax, 0.775, FY[0], 0.215, FY[1], "Flat-head blend" if MINIMAL else "Validated fusion", INK, 0.03)  # v6: section name
ax.text(0.8825, FY[2], r"$w\,\mathbf{p}^{t}+(1-w)\,\mathbf{q}^{t}$", fontsize=7.4, ha="center", color=INK)
if MINIMAL:  # say where q^t comes from; it has no box of its own
    ax.text(0.8825, 0.345, r"$\mathbf{q}^{t}$: flat LightGBM head", fontsize=5.4, ha="center", va="center", color=MUTED)
else:
    ax.text(0.8825, 0.33, r"$\mathbf{q}^{t}$: flat head;  $w$: tuned on validation", fontsize=5.4, ha="center", va="center", color=MUTED)
arrow(ax, (0.8825, 0.548), (0.8825, FY[0] + FY[1] + 0.002), VIOLET)

PY = (0.09, 0.16) if MINIMAL else (0.04, 0.16)  # prediction box: y, height (smaller and lighter in submission style)
ax.add_patch(FancyBboxPatch((0.775, PY[0]), 0.215, PY[1], boxstyle="round,pad=0.004,rounding_size=0.012",
                            fc="#3d3c39" if MINIMAL else INK, ec="none"))
ax.text(0.8825, PY[0] + PY[1] / 2, "Prediction\n2 / 4 / 16 classes\n+ explanation" if MINIMAL else "Prediction\nbinary · 4-class · 16-class\n+ graph explanation",
        fontsize=6.2, ha="center", va="center", color="white", weight="bold")
arrow(ax, (0.8825, FY[0] - 0.002), (0.8825, PY[0] + PY[1] + 0.003), INK)
if MINIMAL:  # print-size text: at least about 6 pt after scaling to the text width
    for t in ax.texts:
        t.set_fontsize(t.get_fontsize() * 1.3)
fig.tight_layout(pad=0.1)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"model.{ext}", dpi=300, bbox_inches="tight")
