"""Programmatically build notebooks/analysis.ipynb (then execute it separately)."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "analysis.ipynb"

md = new_markdown_cell
code = new_code_cell

cells = [
    md(
        "# FairLens — Auditing an Income Classifier for Demographic Bias\n"
        "### AI For Good Hackathon · ACM-W Data Science Ethics track\n\n"
        "**The question:** machine-learning models increasingly decide who gets a "
        "loan, an interview, or a benefit. They are optimized for *accuracy* — but "
        "accuracy says nothing about *fairness*. A model can be 87% accurate and "
        "still systematically deny opportunities to women or to a racial group.\n\n"
        "In this notebook we:\n"
        "1. clean the **UCI Adult / Census Income** dataset,\n"
        "2. train a strong income classifier,\n"
        "3. **audit** it for bias by `sex` and `race` using formal fairness metrics,\n"
        "4. apply a **mitigation** and measure the (small) accuracy cost of fairness.\n\n"
        "> The interactive version of these results lives in the FairLens dashboard "
        "(`streamlit run app.py`)."
    ),
    code(
        "import sys, json\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from src.pipeline import (load_and_clean, train_model, fairness_block,\n"
        "                          perf_metrics, NUMERIC, CATEGORICAL, SENSITIVE)\n"
        "pd.set_option('display.max_columns', 40)\n"
        "plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25})"
    ),
    md(
        "## 1. Load & clean the data\n\n"
        "The raw Adult dataset encodes missing values inconsistently (whitespace, "
        "`?`, and true `NaN`). A subtle but common bug is to call `.str.strip()` "
        "*before* handling nulls, which turns `NaN` into the literal string `'nan'` "
        "and hides the missingness. Our pipeline normalizes all of these to a single "
        "`NaN`, then imputes categorical gaps with the column mode (keeping all "
        "48,842 rows). We also drop two columns:\n"
        "- `fnlwgt` — a survey sampling weight, not a person-level signal;\n"
        "- `education` — redundant with the ordinal `education-num`."
    ),
    code(
        "df, report = load_and_clean()\n"
        "print(f\"rows: {report['rows_raw']:,} raw -> {report['rows_clean']:,} clean\")\n"
        "print(f\"rows with >=1 missing value: {report['rows_with_any_missing']:,}\")\n"
        "pd.Series(report['missing_by_col'], name='missing_count').to_frame()"
    ),
    md(
        "## 2. The data is *already* unequal\n\n"
        "Before we train anything, the ground-truth positive rate (share earning "
        ">\\$50k) differs sharply across groups. A model trained naively will learn — "
        "and often *amplify* — this historical inequality."
    ),
    code(
        "rates = pd.DataFrame({\n"
        "    'positive_rate': {**{f'sex={k}': v for k,v in report['positive_rate_by_sex'].items()},\n"
        "                      **{f'race={k}': v for k,v in report['positive_rate_by_race'].items()}}\n"
        "})\n"
        "ax = rates.sort_values('positive_rate').plot.barh(legend=False, color='#4F46E5', figsize=(7,4))\n"
        "ax.axvline(report['positive_rate_overall'], color='#EF4444', ls='--', label='overall')\n"
        "ax.set_xlabel('share earning >$50k'); ax.set_title('Ground-truth income disparity'); ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3. Train the classifier\n\n"
        "A `HistGradientBoostingClassifier` on one-hot-encoded features. We hold out "
        "30% of the data for evaluation. This is a genuinely strong model — the point "
        "is that strength and fairness are independent."
    ),
    code(
        "clf, X_train, X_test, y_train, y_test = train_model(df)\n"
        "y_score = clf.predict_proba(X_test)[:, 1]\n"
        "y_pred = (y_score >= 0.5).astype(int)\n"
        "perf = perf_metrics(y_test, y_pred, y_score)\n"
        "pd.Series(perf, name='baseline').to_frame()"
    ),
    md(
        "## 4. Audit for bias\n\n"
        "We use [Fairlearn](https://fairlearn.org/) to compute, per group:\n"
        "- **selection rate** — how often the model predicts 'high income';\n"
        "- **true / false positive rate** — does the model give equal *chances* and "
        "make equal *mistakes* across groups?\n\n"
        "Summary metrics:\n"
        "- **Disparate-impact ratio** = least-selected group's rate ÷ most-selected. "
        "The U.S. EEOC's **four-fifths rule** flags any ratio **< 0.80** as evidence "
        "of *adverse impact*.\n"
        "- **Equalized-odds difference** — the largest gap in TPR/FPR across groups "
        "(0 = perfectly equal error rates)."
    ),
    code(
        "for attr in SENSITIVE:\n"
        "    fb = fairness_block(y_test, y_pred, X_test[attr])\n"
        "    print(f'\\n=== {attr.upper()} ===')\n"
        "    print('disparate-impact ratio :', fb['disparate_impact_ratio'],\n"
        "          '  (four-fifths rule: PASS)' if fb['disparate_impact_ratio']>=0.8 else '  (four-fifths rule: FAIL)')\n"
        "    print('equalized-odds gap     :', fb['equalized_odds_difference'])\n"
        "    display(pd.DataFrame(fb['by_group']))"
    ),
    md(
        "Both attributes **fail the four-fifths rule** by a wide margin. Women are "
        "selected at roughly a third of men's rate; the least-selected racial group "
        "at about a fifth of the most-selected. Even where accuracy looks fine, the "
        "*errors are unequally distributed* — that is the ethical harm."
    ),
    md(
        "## 5. Mitigation — and the cost of fairness\n\n"
        "We apply Fairlearn's **`ThresholdOptimizer`**, a post-processing method that "
        "chooses *group-specific* decision thresholds to satisfy **equalized odds** "
        "while maximizing accuracy. Crucially, this requires no retraining and works "
        "on top of the existing model."
    ),
    code(
        "from fairlearn.postprocessing import ThresholdOptimizer\n"
        "summary = []\n"
        "for attr in SENSITIVE:\n"
        "    topt = ThresholdOptimizer(estimator=clf, constraints='equalized_odds',\n"
        "                              objective='accuracy_score', predict_method='predict_proba', prefit=True)\n"
        "    topt.fit(X_train, y_train, sensitive_features=X_train[attr])\n"
        "    y_mit = topt.predict(X_test, sensitive_features=X_test[attr])\n"
        "    b = fairness_block(y_test, y_pred, X_test[attr])\n"
        "    m = fairness_block(y_test, y_mit, X_test[attr])\n"
        "    mp = perf_metrics(y_test, y_mit, y_score)\n"
        "    summary.append({'attribute': attr,\n"
        "                    'DI before': b['disparate_impact_ratio'], 'DI after': m['disparate_impact_ratio'],\n"
        "                    'eq-odds before': b['equalized_odds_difference'], 'eq-odds after': m['equalized_odds_difference'],\n"
        "                    'accuracy before': perf['accuracy'], 'accuracy after': mp['accuracy']})\n"
        "pd.DataFrame(summary)"
    ),
    md(
        "## 6. Takeaways\n\n"
        "- A model can be **highly accurate and clearly discriminatory at the same "
        "time** — accuracy is not a fairness guarantee.\n"
        "- Bias here is **measurable** with standard, legally-grounded metrics "
        "(four-fifths rule, equalized odds), not a matter of opinion.\n"
        "- Mitigation **closes most of the gap for a few accuracy points** — fairness "
        "is affordable, but it is a deliberate choice someone has to make.\n"
        "- 'Fairness through unawareness' (just dropping `sex`/`race`) does **not** "
        "work, because proxies like `relationship` and `marital-status` re-encode the "
        "protected attribute.\n\n"
        "**FairLens turns this audit into a tool anyone can use** — explore it live in "
        "the dashboard."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NB_PATH.parent.mkdir(exist_ok=True)
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH)
