"""
FairLens — core data-science pipeline.

Loads the UCI Adult / Census Income dataset, cleans it, trains an income
classifier, audits it for demographic bias, and applies a post-processing
mitigation. All artifacts are written to outputs/ so the Streamlit dashboard
and the analysis notebook can read them without recomputing.

Run:  python src/pipeline.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Columns we keep out of the model and why (documented for the rubric).
DROP_COLS = {
    "fnlwgt": "survey sampling weight — not a person-level feature, leaks nothing useful",
    "education": "redundant with education-num (same info, ordinal)",
}
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
SENSITIVE = ["sex", "race"]


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Load Adult, clean it, and return (clean_df, cleaning_report)."""
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    report: dict = {"rows_raw": int(len(raw)), "cols_raw": int(raw.shape[1])}

    df = raw.copy()

    # 1. Normalise text: strip whitespace but preserve true NaNs, then map the
    #    several spellings of "missing" (?, nan, None, blank) to a single NaN.
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("string").str.strip()
    df = df.replace({"?": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})

    # 2. Record missingness (the headline data-cleaning story).
    missing = (
        df[CATEGORICAL].isna().sum().sort_values(ascending=False)
    )
    report["missing_by_col"] = {k: int(v) for k, v in missing.items() if v > 0}
    report["rows_with_any_missing"] = int(df[CATEGORICAL].isna().any(axis=1).sum())

    # 3. Impute categoricals with their mode (keeps all 48k rows).
    for col in CATEGORICAL:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    # 4. Binarise the target: >50K -> 1.
    df["class"] = df["class"].astype(str).str.strip()
    df["income_gt50k"] = (df["class"] == ">50K").astype(int)

    # 5. Drop documented leak/redundant columns.
    df = df.drop(columns=list(DROP_COLS) + ["class"])

    # 6. Cast numerics.
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC)

    report["rows_clean"] = int(len(df))
    report["dropped_cols"] = DROP_COLS
    report["positive_rate_overall"] = round(float(df["income_gt50k"].mean()), 4)
    report["positive_rate_by_sex"] = {
        k: round(float(v), 4)
        for k, v in df.groupby("sex")["income_gt50k"].mean().items()
    }
    report["positive_rate_by_race"] = {
        k: round(float(v), 4)
        for k, v in df.groupby("race")["income_gt50k"].mean().items()
    }
    return df, report


def train_model(df: pd.DataFrame):
    """Train a HistGradientBoosting classifier inside a preprocessing pipeline."""
    X = df[NUMERIC + CATEGORICAL]
    y = df["income_gt50k"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="passthrough",
    )
    clf = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


def perf_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }


def fairness_block(y_true, y_pred, sensitive) -> dict:
    """Per-group rates + summary disparity metrics for one sensitive attribute."""
    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy_score,
        "count": count,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    by_group = mf.by_group.copy()
    by_group = by_group.round(4)
    return {
        "by_group": by_group.reset_index().to_dict(orient="records"),
        "demographic_parity_difference": round(
            float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        "disparate_impact_ratio": round(
            float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        "equalized_odds_difference": round(
            float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
    }


def main():
    print(">> loading + cleaning Adult dataset")
    df, report = load_and_clean()
    print(f"   clean rows: {report['rows_clean']:,}  positive rate: {report['positive_rate_overall']}")

    print(">> training model")
    clf, X_train, X_test, y_train, y_test = train_model(df)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    perf = perf_metrics(y_test, y_pred, y_score)
    print(f"   accuracy {perf['accuracy']}  roc_auc {perf['roc_auc']}")

    # Baseline fairness audit for each sensitive attribute.
    fairness = {}
    for attr in SENSITIVE:
        fairness[attr] = {"baseline": fairness_block(y_test, y_pred, X_test[attr])}

    # Mitigation: post-processing ThresholdOptimizer to equalize odds.
    mitigated_preds = {}
    for attr in SENSITIVE:
        print(f">> mitigating bias for '{attr}' (equalized odds)")
        topt = ThresholdOptimizer(
            estimator=clf,
            constraints="equalized_odds",
            objective="accuracy_score",
            predict_method="predict_proba",
            prefit=True,
        )
        topt.fit(X_train, y_train, sensitive_features=X_train[attr])
        y_pred_mit = topt.predict(X_test, sensitive_features=X_test[attr])
        mitigated_preds[attr] = y_pred_mit
        fairness[attr]["mitigated"] = fairness_block(y_test, y_pred_mit, X_test[attr])
        fairness[attr]["mitigated_perf"] = perf_metrics(y_test, y_pred_mit, y_score)

    # Persist test-set scores + sensitive features for the interactive dashboard.
    test_out = X_test[SENSITIVE].copy()
    test_out["y_true"] = y_test.values
    test_out["y_score"] = y_score
    test_out["y_pred_baseline"] = y_pred
    for attr in SENSITIVE:
        test_out[f"y_pred_mitigated_{attr}"] = mitigated_preds[attr]
    test_out.to_csv(OUT / "test_predictions.csv", index=False)

    results = {
        "cleaning_report": report,
        "model": "HistGradientBoostingClassifier",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "performance_baseline": perf,
        "fairness": fairness,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f">> wrote {OUT/'metrics.json'} and {OUT/'test_predictions.csv'}")
    print("\n=== SUMMARY ===")
    for attr in SENSITIVE:
        b = fairness[attr]["baseline"]
        m = fairness[attr]["mitigated"]
        print(
            f"{attr:5s}  disparate-impact  before {b['disparate_impact_ratio']:.3f} "
            f"-> after {m['disparate_impact_ratio']:.3f}   |  eq-odds diff "
            f"before {b['equalized_odds_difference']:.3f} -> after {m['equalized_odds_difference']:.3f}"
        )


if __name__ == "__main__":
    main()
