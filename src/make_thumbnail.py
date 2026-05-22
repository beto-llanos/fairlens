"""Generate a branded 1280x720 Devpost thumbnail -> outputs/thumbnail.png"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
metrics = json.loads((OUT / "metrics.json").read_text())

INDIGO = "#312E81"
INDIGO_D = "#1E1B4B"
WHITE = "#FFFFFF"
LIGHT = "#C7D2FE"
ACCENT = "#818CF8"
RED = "#F87171"
GREEN = "#34D399"

# ---- selection rates for the hero chart (sex: baseline vs mitigated) ----
base = {r["sex"]: r["selection_rate"] for r in metrics["fairness"]["sex"]["baseline"]["by_group"]}
mit = {r["sex"]: r["selection_rate"] for r in metrics["fairness"]["sex"]["mitigated"]["by_group"]}
di_b = metrics["fairness"]["sex"]["baseline"]["disparate_impact_ratio"]
di_m = metrics["fairness"]["sex"]["mitigated"]["disparate_impact_ratio"]
acc = metrics["performance_baseline"]["accuracy"]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(INDIGO_D)

# ---------- LEFT: text ----------
fig.text(0.055, 0.80, "FairLens", fontsize=72, fontweight="bold", color=WHITE)
# accent underline
fig.add_artist(plt.Line2D([0.057, 0.34], [0.745, 0.745], color=ACCENT, lw=5))

fig.text(0.057, 0.66, "See the bias your model is hiding.",
         fontsize=23, color=LIGHT)

fig.text(0.057, 0.50, f"{acc*100:.1f}% accurate.", fontsize=40, fontweight="bold", color=WHITE)
fig.text(0.057, 0.40, "Still discriminates by sex & race.", fontsize=30,
         fontweight="bold", color=RED)

# four-fifths callout pill
pill = FancyBboxPatch((0.057, 0.235), 0.46, 0.085, boxstyle="round,pad=0.012",
                      transform=fig.transFigure, facecolor="#3B1D2B",
                      edgecolor=RED, linewidth=1.5)
fig.add_artist(pill)
fig.text(0.075, 0.265, f"Disparate impact {di_b:.2f}  —  fails the EEOC four-fifths rule",
         fontsize=16.5, color="#FCA5A5", fontweight="bold")

fig.text(0.057, 0.105, "AI For Good Hackathon   ·   ACM-W Data Science Ethics Track",
         fontsize=15, color=ACCENT, fontweight="bold")

# ---------- RIGHT: hero chart ----------
ax = fig.add_axes([0.60, 0.22, 0.345, 0.5])
ax.set_facecolor(INDIGO_D)
groups = ["Female", "Male"]
x = range(len(groups))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], [base[g] for g in groups], w,
            label="Biased model", color=RED)
b2 = ax.bar([i + w / 2 for i in x], [mit[g] for g in groups], w,
            label="After fix", color=GREEN)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.0%}", ha="center", color=WHITE, fontsize=12,
                fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(groups, color=WHITE, fontsize=14, fontweight="bold")
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Who gets approved for high income?", color=WHITE, fontsize=15,
             fontweight="bold", pad=14)
ax.legend(loc="upper left", frameon=False, labelcolor=WHITE, fontsize=11,
          handlelength=1.1)
ax.set_ylim(0, max(base.values()) * 1.30)
ax.margins(x=0.18)

fig.text(0.605, 0.135, f"disparate impact  {di_b:.2f}  →  {di_m:.2f}   (≈1 acc. point cost)",
         fontsize=13, color=LIGHT)

fig.savefig(OUT / "thumbnail.png", facecolor=fig.get_facecolor(), bbox_inches=None)
print("wrote", OUT / "thumbnail.png")
