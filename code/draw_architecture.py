"""Draws the MemTGF architecture figure (paper/figures/architecture.pdf/png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
BLUE, ORANGE, AQUA, VIOLET, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#a3a29c", "#0b0b0b"


def box(ax, x, y, w, h, text, color, fs=7.2, fill_alpha=0.12, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=color, ec=color, lw=1.2, alpha=fill_alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc="none", ec=color, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=1.0, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(-0.015, 1.015)
ax.set_ylim(0, 1)
ax.axis("off")

# input
box(ax, 0.005, 0.40, 0.13, 0.22, "Memory dump\n(Volatility 2)\n55 plugin\nfeatures", GRAY, bold=False)
box(ax, 0.165, 0.40, 0.13, 0.22, "Leak-safe prep.\nde-dup → split\n→ median impute\n→ signed log\n→ z-score (train)", GRAY, fs=6.5)
arrow(ax, 0.137, 0.51, 0.163, 0.51)

# three graphs
box(ax, 0.33, 0.74, 0.20, 0.20, "① Feature-Interaction Graph\nnodes = 52 features\nedges = |Spearman ρ| ≥ τ,\ntop-k, same-plugin prior\n→ Louvain communities", BLUE, fs=6.4)
box(ax, 0.33, 0.41, 0.20, 0.20, "② Sample-Similarity Graph\nnodes = memory dumps\nedges = k-NN (train only)\ninductive attachment\nof unseen dumps", ORANGE, fs=6.4)
box(ax, 0.33, 0.08, 0.20, 0.20, "③ Label-Hierarchy Graph\nBenign / Malware\n→ 3 categories\n→ 15 families\n(family supervision)", VIOLET, fs=6.4)
for yy in (0.84, 0.51, 0.18):
    arrow(ax, 0.297, 0.51, 0.328, yy)

# experts
box(ax, 0.57, 0.80, 0.17, 0.13, "Community experts\nGBDT on FIG-C₀ … C_m", BLUE, fs=6.4)
box(ax, 0.57, 0.62, 0.17, 0.12, "Full-feature expert\nGBDT on all nodes", GRAY, fs=6.4)
box(ax, 0.57, 0.43, 0.17, 0.13, "GraphSAGE expert\n2-layer, mean aggr.\n(GPU)", ORANGE, fs=6.4)
arrow(ax, 0.531, 0.84, 0.568, 0.865, BLUE)
arrow(ax, 0.531, 0.80, 0.568, 0.68)
arrow(ax, 0.531, 0.51, 0.568, 0.495, ORANGE)
arrow(ax, 0.531, 0.24, 0.62, 0.425, VIOLET, ls="--")
ax.text(0.636, 0.383, "family-level\nsupervision", fontsize=5.8, color=VIOLET, ha="left", va="center", style="italic")

# meta + aggregation + fusion
box(ax, 0.77, 0.60, 0.11, 0.20, "Out-of-fold\nmeta-learner\n(5-fold\nstacking)", AQUA, fs=6.4)
for yy in (0.865, 0.68, 0.495):
    arrow(ax, 0.742, yy, 0.768, 0.70, AQUA)
box(ax, 0.77, 0.33, 0.11, 0.17, "LHG\naggregation\np_task = p_fam·M", VIOLET, fs=6.4)
arrow(ax, 0.825, 0.598, 0.825, 0.502, AQUA)
arrow(ax, 0.531, 0.255, 0.768, 0.36, VIOLET, ls="--")
box(ax, 0.905, 0.33, 0.09, 0.17, "Val-tuned\nfusion with\nflat head", INK, fs=6.2, fill_alpha=0.06)
arrow(ax, 0.881, 0.415, 0.903, 0.415)
box(ax, 0.905, 0.07, 0.09, 0.19, "Prediction\nbinary /\n4-class /\n16-class", INK, fs=6.2, fill_alpha=0.06, bold=True)
arrow(ax, 0.95, 0.328, 0.95, 0.262)

# explainability band
box(ax, 0.57, 0.04, 0.31, 0.17, "Graph explanations\nFIG node / edge / community attribution\nSSG nearest-prototype memory dumps\nLHG decision path + deletion-fidelity check", AQUA, fs=5.9, fill_alpha=0.08)
arrow(ax, 0.903, 0.13, 0.882, 0.13, AQUA, style="<|-")

fig.tight_layout(pad=0.2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"architecture.{ext}", dpi=300, bbox_inches="tight")
