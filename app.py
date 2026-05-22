"""
FairLens — interactive bias-audit dashboard.

Reads the artifacts produced by src/pipeline.py and lets a judge:
  * pick a protected attribute (sex / race),
  * slide the model's decision threshold and watch fairness move in real time,
  * toggle a bias mitigation on/off and see the before/after — including the
    (small) accuracy cost of being fair.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fairlearn.metrics import equalized_odds_difference

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

FOUR_FIFTHS = 0.8  # EEOC adverse-impact threshold for the disparate-impact ratio.
ACCENT = "#4F46E5"
DANGER = "#EF4444"
OK = "#10B981"

st.set_page_config(page_title="FairLens — Bias Audit", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------- #
# Data + metric helpers
# --------------------------------------------------------------------------- #
@st.cache_data
def load_artifacts():
    df = pd.read_csv(OUT / "test_predictions.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())
    return df, metrics


def group_rates(df: pd.DataFrame, attr: str, pred_col: str) -> pd.DataFrame:
    """Selection rate, TPR, FPR and count per group for the given predictions."""
    rows = []
    for g, sub in df.groupby(attr):
        pos = sub["y_true"] == 1
        neg = sub["y_true"] == 0
        rows.append(
            {
                "group": g,
                "selection_rate": sub[pred_col].mean(),
                "tpr": sub.loc[pos, pred_col].mean() if pos.any() else 0.0,
                "fpr": sub.loc[neg, pred_col].mean() if neg.any() else 0.0,
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_rate", ascending=False)


def disparate_impact(rates: pd.DataFrame) -> float:
    s = rates["selection_rate"]
    return float(s.min() / s.max()) if s.max() > 0 else 0.0


def accuracy_of(df: pd.DataFrame, pred_col: str) -> float:
    return float((df[pred_col] == df["y_true"]).mean())


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
df, metrics = load_artifacts()

st.sidebar.title("🔍 FairLens")
st.sidebar.caption("Audit an income classifier for demographic bias — and fix it.")

attr = st.sidebar.radio("Protected attribute", ["sex", "race"], index=0)
threshold = st.sidebar.slider(
    "Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01,
    help="Probability above which the model predicts 'high income'. The same "
         "single threshold is applied to everyone.",
)
mitigate = st.sidebar.toggle(
    "Apply bias mitigation", value=False,
    help="Post-processing (Fairlearn ThresholdOptimizer) that equalizes error "
         "rates across groups using group-aware thresholds.",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Model:** HistGradientBoosting  \n"
    f"**Test set:** {len(df):,} people  \n"
    "**Data:** UCI Adult / Census Income"
)

# --------------------------------------------------------------------------- #
# Compute current predictions
# --------------------------------------------------------------------------- #
df = df.copy()
df["y_pred_threshold"] = (df["y_score"] >= threshold).astype(int)

if mitigate:
    pred_col = f"y_pred_mitigated_{attr}"
    mode_label = "Mitigated (group-aware thresholds)"
else:
    pred_col = "y_pred_threshold"
    mode_label = f"Baseline @ threshold {threshold:.2f}"

rates = group_rates(df, attr, pred_col)
di = disparate_impact(rates)
eod = float(equalized_odds_difference(df["y_true"], df[pred_col], sensitive_features=df[attr]))
acc = accuracy_of(df, pred_col)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("FairLens 🔍")
st.markdown(
    "##### An accurate model is not the same as a fair one. "
    "This dashboard shows an 87%-accurate income classifier quietly "
    "discriminating — and a one-toggle fix."
)
st.caption(f"Currently viewing: **{mode_label}** · protected attribute: **{attr}**")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Accuracy", f"{acc*100:.1f}%")
k2.metric("ROC-AUC", f"{metrics['performance_baseline']['roc_auc']:.3f}")
di_flag = "✅ passes" if di >= FOUR_FIFTHS else "⚠️ adverse impact"
k3.metric("Disparate-impact ratio", f"{di:.2f}", di_flag,
          delta_color="off")
k4.metric("Equalized-odds gap", f"{eod:.3f}",
          "lower is fairer", delta_color="off")

if di < FOUR_FIFTHS:
    st.error(
        f"**Four-fifths rule violated.** The least-selected group is approved at "
        f"**{di:.0%}** the rate of the most-selected group. The U.S. EEOC treats a "
        f"ratio below 0.80 as evidence of *adverse impact* — i.e. potential illegal "
        f"discrimination.",
        icon="⚖️",
    )
else:
    st.success(
        f"**Four-fifths rule satisfied** (ratio {di:.2f} ≥ 0.80). Selection rates are "
        f"within the EEOC's acceptable range across {attr} groups.",
        icon="⚖️",
    )

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("Who gets approved?")
    st.caption("Selection rate = share of each group the model labels 'high income'.")
    fig = go.Figure()
    fig.add_bar(
        x=rates["group"], y=rates["selection_rate"],
        marker_color=ACCENT, text=[f"{v:.0%}" for v in rates["selection_rate"]],
        textposition="outside",
    )
    fig.update_layout(
        yaxis_tickformat=".0%", yaxis_title="Selection rate",
        height=380, margin=dict(t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Equal opportunity?")
    st.caption("True-positive rate = of the people who truly earn >$50k, what "
               "share does the model correctly approve? Unequal bars = unequal chances.")
    fig2 = go.Figure()
    fig2.add_bar(name="True-positive rate", x=rates["group"], y=rates["tpr"],
                 marker_color=OK)
    fig2.add_bar(name="False-positive rate", x=rates["group"], y=rates["fpr"],
                 marker_color=DANGER)
    fig2.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_title="Rate",
        height=380, margin=dict(t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------- #
# Before / after mitigation comparison
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("The cost of fairness: before vs after mitigation")

base = metrics["fairness"][attr]["baseline"]
mit = metrics["fairness"][attr]["mitigated"]
mit_perf = metrics["fairness"][attr]["mitigated_perf"]
base_perf = metrics["performance_baseline"]

comp = pd.DataFrame(
    {
        "Metric": [
            "Disparate-impact ratio (→ 1.0 is fair)",
            "Demographic-parity gap (→ 0 is fair)",
            "Equalized-odds gap (→ 0 is fair)",
            "Accuracy",
        ],
        "Before": [
            base["disparate_impact_ratio"],
            base["demographic_parity_difference"],
            base["equalized_odds_difference"],
            base_perf["accuracy"],
        ],
        "After": [
            mit["disparate_impact_ratio"],
            mit["demographic_parity_difference"],
            mit["equalized_odds_difference"],
            mit_perf["accuracy"],
        ],
    }
)
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(
        comp.style.format({"Before": "{:.3f}", "After": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
with c2:
    acc_drop = base_perf["accuracy"] - mit_perf["accuracy"]
    st.metric("Accuracy traded for fairness", f"{acc_drop*100:.2f} pts")
    st.caption(
        f"Equalizing error rates across **{attr}** groups cost only "
        f"**{acc_drop*100:.2f} accuracy points** — a small price for closing the gap."
    )

st.divider()
st.caption(
    "FairLens · UCI Adult/Census Income · scikit-learn + Fairlearn · "
    "Built for the AI For Good Hackathon (ACM-W Data Science Ethics)."
)
