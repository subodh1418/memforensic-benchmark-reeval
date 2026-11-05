"""Methodology flowchart (v2): standard flowchart shapes, minimal text. Writes <paper>/figures/methodology_flowchart.{pdf,png}."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / os.environ.get("MEMTGF_PAPER", "paper") / "figures"
NAME = os.environ.get("MEMTGF_NAME", "MemTGF")
V4 = NAME != "MemTGF"  # version-4 wording (re-evaluation study)
INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#a3a29c"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
FS = 7.0


def proc(ax, cx, cy, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", fc=color, ec="none", alpha=0.13))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", fc="none", ec=color, lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK, weight="bold" if bold else "normal", linespacing=1.2)


def term(ax, cx, cy, w, h, text):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc="#ecebe7", ec=MUTED, lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS - 0.3, color=INK, weight="bold", linespacing=1.2)


def diamond(ax, cx, cy, w, h, text):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=YELLOW, ec="none", alpha=0.25))
    ax.add_patch(Polygon(pts, closed=True, fc="none", ec="#c98500", lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS - 0.6, color=INK, linespacing=1.15)


def arr(ax, p0, p1, ls="-", label=None, lpos=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=MUTED, linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=FS - 0.8, color=MUTED, ha="center", va="center", style="italic")


fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
W, H = 0.155, 0.13
y1, y2, y3 = 0.85, 0.50, 0.13

# row 1 -- data preparation and the feature-engineering decision
term(ax, 0.08, y1, 0.15, 0.13, "CIC-MalMem\n2022 dumps")
proc(ax, 0.265, y1, W, H, "De-duplicate,\nsplit (5 seeds)", GRAY)
proc(ax, 0.45, y1, W, H, "Train-only\nimpute + scaling", GRAY)
diamond(ax, 0.655, y1, 0.19, 0.25, "FE + hybrid FS\nhelps on dev\nseeds?")
proc(ax, 0.885, y1, 0.17, H, "Keep the 52\nVolatility indicators", GRAY)
arr(ax, (0.155, y1), (0.187, y1))
arr(ax, (0.343, y1), (0.372, y1))
arr(ax, (0.528, y1), (0.56, y1))
arr(ax, (0.75, y1), (0.80, y1), label="no", lpos=(0.775, y1 + 0.04))
arr(ax, (0.655, y1 - 0.125), (0.655, y1 - 0.19), ls=(0, (3, 2)))
ax.text(0.668, y1 - 0.16, "yes: add FE / FS", fontsize=FS - 0.8, color=MUTED, ha="left", va="center", style="italic")

# row 2 -- proposed model, right to left
ax.add_patch(FancyBboxPatch((0.175, y2 - 0.125), 0.81, 0.25, boxstyle="round,pad=0.005,rounding_size=0.015", fc="none", ec=BLUE, lw=0.8, ls=(0, (4, 3))))
proc(ax, 0.885, y2, 0.17, 0.17, "Learn structures\nFIG · SSG · LHG" if V4 else "Build three graphs\nFIG · SSG · LHG", BLUE, bold=True)
proc(ax, 0.655, y2, 0.19, 0.17, "Family-level experts\n$E_{C_j}$, $E_{full}$, $E_{SAGE}$", VIOLET)
proc(ax, 0.45, y2, W, 0.17, "Out-of-fold\nmeta-learner", AQUA)
proc(ax, 0.265, y2, W, 0.17, "LHG aggregation\n+ flat-head blend", VIOLET)
term(ax, 0.08, y2, 0.15, 0.13, NAME + "\nprediction")
arr(ax, (0.885, y1 - H / 2), (0.885, y2 + 0.085))
arr(ax, (0.80, y2), (0.75, y2))
arr(ax, (0.56, y2), (0.528, y2))
arr(ax, (0.372, y2), (0.343, y2))
arr(ax, (0.187, y2), (0.155, y2))
ax.text(0.98, y2 - 0.105, "reference model" if V4 else "proposed model", fontsize=FS - 1, color=BLUE, ha="right", va="center", style="italic")

# row 3 -- evaluation
arr(ax, (0.08, y2 - 0.065), (0.08, y3 + 0.065))
proc(ax, 0.08, y3, 0.15, H, "Evaluate\n3 tasks, 5 seeds", ORANGE, bold=True)
if V4:
    proc(ax, 0.285, y3, 0.175, H, "12 baselines, 7 re-impl.,\nshortcut protocols", ORANGE)
    proc(ax, 0.495, y3, 0.175, H, "Ceiling audit,\nstructure controls", RED)
    proc(ax, 0.705, y3, 0.175, H, "LOFO, LOCO, robust-\nness, explanations", AQUA)
    term(ax, 0.905, y3, 0.16, 0.13, "Findings and\nchecklist")
else:
    proc(ax, 0.285, y3, 0.175, H, "12 baselines +\n7 re-implemented", ORANGE)
    proc(ax, 0.495, y3, 0.175, H, "Ablation, sensitivity\nrobustness, LOFO", RED)
    proc(ax, 0.705, y3, 0.175, H, "Graph explanations\n+ deletion fidelity", AQUA)
    term(ax, 0.905, y3, 0.16, 0.13, "Findings and\nSOTA analysis")
arr(ax, (0.155, y3), (0.197, y3))
arr(ax, (0.373, y3), (0.407, y3))
arr(ax, (0.583, y3), (0.617, y3))
arr(ax, (0.793, y3), (0.825, y3))

fig.tight_layout(pad=0.1)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"methodology_flowchart.{ext}", dpi=300, bbox_inches="tight")
