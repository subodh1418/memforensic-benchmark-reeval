"""Proposed-methodology (research workflow) figure: paper/figures/methodology.pdf/png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / __import__("os").environ.get("MEMTGF_PAPER", "paper") / "figures"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"

PHASES = [
    ("① Data acquisition", GRAY, ["CIC-MalMem-2022 dumps", "55 Volatility plugin features", "3 row-aligned label files", "→ one corpus, 3 granularities"]),
    ("② Leak-safe preprocessing", GRAY, ["full-row + feature-vector de-dup", "zero-variance removal (52 kept)", "family-stratified 70/15/15, 5 seeds", "train-only impute + signed-log z"]),
    ("③ Feature eng. & selection", YELLOW, ["46 memory-forensic features", "hybrid FS: filter→redund.→shadow", "3 dev seeds, validation only", "→ no consistent gain: keep 52 raw"]),
    ("④ Tri-graph construction", BLUE, ["FIG: feature dependency graph", "  + Louvain communities", "SSG: inductive k-NN graph", "LHG: benign→category→family"]),
    ("⑤ MemTGF training", VIOLET, ["community / full / GraphSAGE", "experts on family labels", "5-fold OOF meta-learner", "LHG aggregation + val fusion"]),
    ("⑥ Evaluation & statistics", ORANGE, ["binary · 4-class · 16-class", "12 baselines + 7 SOTA re-impl.", "13 metrics, mean ± sd", "paired t, McNemar, bootstrap CI"]),
    ("⑦ Ablation & robustness", RED, ["ablation + alt. graph integrations", "k / τ / resolution sensitivity", "masked-feature robustness", "unseen-family (LOFO) test"]),
    ("⑧ Graph explainability", AQUA, ["node · edge · community", "SSG prototypes · LHG path", "per-class attribution", "deletion-fidelity test"]),
]


def phase(ax, x, y, w, h, title, color, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc="white", ec=color, lw=1.3))
    ax.add_patch(Rectangle((x, y + h - 0.052), w, 0.052, fc=color, ec="none", alpha=0.9))
    ax.text(x + w / 2, y + h - 0.026, title, ha="center", va="center", fontsize=6.9, color="white" if color not in (YELLOW, GRAY) else INK, weight="bold")
    for i, l in enumerate(lines):
        ax.text(x + 0.008, y + h - 0.085 - i * 0.041, ("• " if not l.startswith(" ") else "   ") + l.strip(), fontsize=5.9, color=INK, va="center")


def arrow(ax, p0, p1, color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(7.4, 4.9))
ax.set_xlim(-0.005, 1.005)
ax.set_ylim(0, 1)
ax.axis("off")
w, h, gap = 0.226, 0.25, 0.0253
xs = [0.005 + i * (w + gap) for i in range(4)]
ytop, ybot = 0.69, 0.33
for i in range(4):
    phase(ax, xs[i], ytop, w, h, *PHASES[i])
    if i < 3:
        arrow(ax, (xs[i] + w + 0.002, ytop + h / 2), (xs[i + 1] - 0.002, ytop + h / 2))
for j in range(4):  # second row runs right -> left
    i = 4 + j
    x = xs[3 - j]
    phase(ax, x, ybot, w, h, *PHASES[i])
    if j < 3:
        arrow(ax, (x - 0.002, ybot + h / 2), (xs[3 - j - 1] + w + 0.002, ybot + h / 2))
arrow(ax, (xs[3] + w / 2, ytop - 0.004), (xs[3] + w / 2, ybot + h + 0.004))

# SOTA banner
ax.add_patch(FancyBboxPatch((0.005, 0.03), 0.99, 0.17, boxstyle="round,pad=0.006,rounding_size=0.012", fc="#f4f3f0", ec=GREEN, lw=1.3))
ax.add_patch(Rectangle((0.005, 0.03), 0.012, 0.17, fc=GREEN, ec="none"))
ax.text(0.03, 0.165, "⑨ State-of-the-art analysis", fontsize=7.6, weight="bold", color=INK, va="center")
ax.text(0.03, 0.115, "Peer-reviewed journals 2024–2026 only (IEEE, Springer, Elsevier/ScienceDirect, Emerald, PeerJ, …), Crossref-verified;", fontsize=6.4, va="center")
ax.text(0.03, 0.072, "7 recent methods re-implemented under the same protocol; comparability audit (de-dup, SMOTE order, granularity, splits)", fontsize=6.4, va="center")
arrow(ax, (xs[0] + w / 2, ybot - 0.004), (xs[0] + w / 2, 0.205))
ax.text(xs[0] + w / 2 + 0.01, 0.245, "results", fontsize=6, color=MUTED, style="italic")
fig.tight_layout(pad=0.2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"methodology.{ext}", dpi=300, bbox_inches="tight")
