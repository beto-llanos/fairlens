"""Generate static PNG charts for the README and analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ACCENT = "#4F46E5"
DANGER = "#EF4444"
OK = "#10B981"

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25})

metrics = json.loads((OUT / "metrics.json").read_text())
df = pd.read_csv(OUT / "test_predictions.csv")


def fig_base_rates():
    """Disparity that already exists in the data itself."""
    rep = metrics["cleaning_report"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [
        (axes[0], "positive_rate_by_sex", "By sex"),
        (axes[1], "positive_rate_by_race", "By race"),
    ]:
        d = rep[key]
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        bars = ax.bar(labels, vals, color=ACCENT)
        ax.axhline(rep["positive_rate_overall"], color=DANGER, ls="--", lw=1,
                   label=f"overall {rep['positive_rate_overall']:.0%}")
        ax.set_title(title)
        ax.set_ylabel("share earning >$50k")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.0%}",
                    ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Ground-truth income disparity in the data (before any model)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_base_rates.png", bbox_inches="tight")
    plt.close(fig)


def fig_before_after(attr: str):
    """Selection rate per group, baseline vs mitigated."""
    base = {r[attr]: r["selection_rate"] for r in metrics["fairness"][attr]["baseline"]["by_group"]}
    mit = {r[attr]: r["selection_rate"] for r in metrics["fairness"][attr]["mitigated"]["by_group"]}
    groups = list(base.keys())
    x = range(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([i - w / 2 for i in x], [base[g] for g in groups], w,
           label="Baseline (biased)", color=DANGER)
    ax.bar([i + w / 2 for i in x], [mit[g] for g in groups], w,
           label="After mitigation", color=OK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("selection rate")
    b = metrics["fairness"][attr]["baseline"]["disparate_impact_ratio"]
    m = metrics["fairness"][attr]["mitigated"]["disparate_impact_ratio"]
    ax.set_title(f"Selection rate by {attr} — disparate impact "
                 f"{b:.2f} → {m:.2f}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"fig_before_after_{attr}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_base_rates()
    fig_before_after("sex")
    fig_before_after("race")
    print("wrote charts to", OUT)
