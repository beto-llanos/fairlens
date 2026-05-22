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

INDIGO_D = "#1E1B4B"
WHITE = "#FFFFFF"
LIGHT = "#C7D2FE"
ACCENT = "#818CF8"
RED = "#F87171"
GREEN = "#34D399"

base = {r["sex"]: r["selection_rate"] for r in metrics["fairness"]["sex"]["baseline"]["by_group"]}
di_b = metrics["fairness"]["sex"]["baseline"]["disparate_impact_ratio"]
acc = metrics["performance_baseline"]["accuracy"]
ratio = base["Male"] / base["Female"]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(INDIGO_D)

# ---------- LEFT: text, evenly spaced ----------
fig.text(0.055, 0.82, "FairLens", fontsize=70, fontweight="bold", color=WHITE)
fig.add_artist(plt.Line2D([0.058, 0.34], [0.765, 0.765], color=ACCENT, lw=5))
fig.text(0.058, 0.685, "See the bias your model is hiding.", fontsize=22, color=LIGHT)

fig.text(0.058, 0.545, f"{acc*100:.1f}% accurate.", fontsize=38, fontweight="bold", color=WHITE)
fig.text(0.058, 0.455, "Still biased by sex & race.", fontsize=30, fontweight="bold", color=RED)

pill = FancyBboxPatch((0.058, 0.285), 0.40, 0.08, boxstyle="round,pad=0.012",
                      transform=fig.transFigure, facecolor="#3B1D2B",
                      edgecolor=RED, linewidth=1.5)
fig.add_artist(pill)
fig.text(0.075, 0.313, f"Fails the EEOC four-fifths rule  ·  DI {di_b:.2f}",
         fontsize=16.5, color="#FCA5A5", fontweight="bold")

fig.text(0.058, 0.205, "One toggle closes most of the gap — for ~1 accuracy point.",
         fontsize=15, color=GREEN, fontweight="bold")

fig.text(0.058, 0.085, "AI For Good Hackathon   ·   ACM-W Data Science Ethics Track",
         fontsize=15, color=ACCENT, fontweight="bold")

# ---------- RIGHT: headline + one clear bias chart (no bottom caption) ----------
fig.text(0.60, 0.775, f"Men approved {ratio:.1f}× more",
         fontsize=22, fontweight="bold", color=WHITE)
fig.text(0.60, 0.712, "share the model predicts as high-income",
         fontsize=14, color=LIGHT)

ax = fig.add_axes([0.62, 0.245, 0.325, 0.40])
ax.set_facecolor(INDIGO_D)
groups = ["Women", "Men"]
vals = [base["Female"], base["Male"]]
bars = ax.bar(groups, vals, width=0.58, color=[ACCENT, RED])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.009, f"{v:.0%}",
            ha="center", color=WHITE, fontsize=23, fontweight="bold")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, color=WHITE, fontsize=18, fontweight="bold")
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylim(0, max(vals) * 1.30)
ax.margins(x=0.20)

fig.savefig(OUT / "thumbnail.png", facecolor=fig.get_facecolor())
print("wrote", OUT / "thumbnail.png", "| ratio", round(ratio, 2))
