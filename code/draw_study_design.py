"""Study-design figure (version 6, Fig. 1): data and protocol mapped to the four research questions.
Writes <paper>/figures/methodology_flowchart.{pdf,png}; the v2-v4 flowchart is draw_flowchart.py."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / os.environ.get("MEMTGF_PAPER", "paper") / "figures"
NAME = os.environ.get("MEMTGF_NAME", "MemHSE")
INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#a3a29c"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"


def box(ax, x, y, w, h, color, alpha=0.10, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=color, ec="none", alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc="none", ec=color, lw=lw, ls=ls))


def arrow(ax, p0, p1, color=MUTED):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.95)
ax.axis("off")

# ---- top band: data -> de-duplication -> clean protocol (one flow)
top_y, top_h = 0.76, 0.17
steps = [(0.01, 0.235, "CIC-MalMem-2022", "58,596 dumps · 55 indicators"),
         (0.29, 0.235, "De-duplication", "58,027 dumps · 52 indicators"),
         (0.57, 0.42, "Clean protocol", "5 family-stratified splits · train-only fitting\npaired tests · design on development seeds")]
for x, bw, title, sub in steps:
    box(ax, x, top_y, bw, top_h, GRAY, 0.12)
    ax.text(x + bw / 2, top_y + top_h - 0.04, title, ha="center", va="center", fontsize=8, weight="bold", color=INK)
    ax.text(x + bw / 2, top_y + 0.05, sub, ha="center", va="center", fontsize=6.8, color=MUTED, linespacing=1.2)
arrow(ax, (0.245, top_y + top_h / 2), (0.29, top_y + top_h / 2))
arrow(ax, (0.525, top_y + top_h / 2), (0.57, top_y + top_h / 2))
xs = [0.01, 0.255, 0.50, 0.755]
w = 0.235
bus_y = 0.665  # one bus from the protocol to all four research questions
ax.plot([0.78, 0.78], [top_y, bus_y], color=MUTED, lw=1.1)
ax.plot([xs[0] + w / 2, xs[3] + w / 2], [bus_y, bus_y], color=MUTED, lw=1.1)

# ---- four research-question columns
rq = [("RQ1", "Does published\naccuracy survive?", BLUE,
       ["7 journal methods\nre-implemented", "3 shortcut protocols\n(duplicates, SMOTE)"]),
      ("RQ2", "How much accuracy\nis attainable?", ORANGE,
       ["Nearest-neighbour\nlabel audit", "Cover–Hart bound"]),
      ("RQ3", "Where does\nstructure help?", AQUA,
       [f"{NAME} (Fig. 2)\nvs. 12 baselines", "Ablations ·\ngrouping controls"]),
      ("RQ4", "Does it generalise,\nand can it explain?", VIOLET,
       ["Unseen families,\ncategories · MalVol-25", "Explanations ·\ndeletion fidelity"])]
col_y, col_h = 0.17, 0.42
for (tag, q, c, items), x in zip(rq, xs):
    box(ax, x, col_y, w, col_h, c, 0.07)
    ax.text(x + 0.014, col_y + col_h - 0.05, tag, ha="left", va="center", fontsize=9, weight="bold", color=c)
    ax.text(x + 0.075, col_y + col_h - 0.05, q, ha="left", va="center", fontsize=7.2, color=INK, linespacing=1.1)
    for k, it in enumerate(items):
        iy = col_y + col_h - 0.22 - k * 0.145
        ax.add_patch(FancyBboxPatch((x + 0.015, iy), w - 0.03, 0.115, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec=c, lw=0.9))
        ax.text(x + w / 2, iy + 0.0575, it, ha="center", va="center", fontsize=7, color=INK, linespacing=1.1)
    arrow(ax, (x + w / 2, bus_y), (x + w / 2, col_y + col_h + 0.003), c)

# ---- bottom: synthesis
box(ax, 0.03, 0.03, 0.94, 0.085, INK, 0.05)
ax.text(0.5, 0.0725, "Findings  →  reporting checklist for the benchmark", ha="center", va="center", fontsize=8, weight="bold", color=INK)
for x in xs:
    arrow(ax, (x + w / 2, col_y), (x + w / 2, 0.117))

fig.tight_layout(pad=0.1)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"methodology_flowchart.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
