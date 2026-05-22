# FairLens — Devpost submission copy

Copy/paste these into the Devpost fields. Written to hit the ACM-W Data Science
Ethics rubric: Social Impact 30% · Technical Quality 30% · Clarity 30% · Innovation 10%.

---

## Project name
FairLens — see the bias your model is hiding

## Elevator pitch (one line)
An interactive audit that proves a "good" 87%-accurate income model quietly
discriminates by sex and race — and fixes most of it with one toggle.

## Inspiration
Models now decide who gets loans, interviews, and benefits, and they're judged on
one number: accuracy. But a model can be highly accurate and still systematically
deny opportunities to a protected group — the harm is invisible in the headline
metric. We wanted to make that invisible harm **visible, measurable, and fixable**
for anyone, not just ML researchers.

## What it does
FairLens audits a real income classifier trained on the UCI Adult / Census Income
dataset (48,842 people). It:
- trains a strong gradient-boosting model (87.6% accuracy, 0.93 ROC-AUC);
- audits it for bias by **sex** and **race** with legally-grounded fairness
  metrics (the EEOC four-fifths rule, equalized odds);
- shows it **fails the four-fifths rule badly** (disparate-impact 0.32 by sex,
  0.21 by race — anything under 0.80 is legal "adverse impact");
- applies a mitigation that closes most of the gap for ~1 accuracy point;
- puts it all in a dashboard where a judge slides the decision threshold and
  toggles the fix to watch fairness change live.

## How we built it
- **Data cleaning** (pandas): normalized three different spellings of "missing,"
  recovered 3,620 hidden missing values, mode-imputed, dropped a leak column
  (`fnlwgt`) and a redundant one (`education`).
- **Modeling** (scikit-learn): `HistGradientBoostingClassifier` on one-hot
  features, 70/30 stratified split.
- **Fairness audit** (Fairlearn): `MetricFrame` for per-group selection/TPR/FPR,
  plus disparate-impact ratio and equalized-odds difference.
- **Mitigation** (Fairlearn `ThresholdOptimizer`): group-aware thresholds for
  equalized odds, no retraining.
- **Dashboard** (Streamlit + Plotly): live threshold slider, attribute selector,
  before/after mitigation comparison with the accuracy cost.
- A fully-executed `analysis.ipynb` documents the whole data-science narrative.

## Challenges we ran into
The sneakiest bug was in *cleaning*: calling `.str.strip()` before handling nulls
silently turned every `NaN` into the literal string `'nan'`, hiding 3,620 missing
values and making the dataset look clean when it wasn't. Catching that was a good
reminder that data-ethics work starts with honest data.

## Accomplishments we're proud of
Turning an abstract "AI ethics" concern into concrete, defensible numbers tied to
actual US anti-discrimination law (the four-fifths rule) — and shipping it as a
tool a non-expert can actually operate.

## What we learned
- Accuracy and fairness are independent; you have to measure fairness on purpose.
- "Fairness through unawareness" (dropping sex/race) fails because of proxies.
- There's no single definition of fairness — the honest move is to show the
  trade-off, not hide it.

## What's next for FairLens
Upload-your-own-CSV auditing, more mitigation methods (reweighing, exponentiated
gradient), and an exportable PDF "fairness report" for compliance teams.

## Built with
python, pandas, scikit-learn, fairlearn, plotly, streamlit, jupyter

---

## 🎬 2-minute demo video script

**[0:00–0:20] Hook.** "This model predicts who earns over \$50k a year with 87%
accuracy. Sounds great. But watch what it's actually doing." — show the dashboard
header.

**[0:20–0:45] The harm.** Point to the KPI row: disparate-impact ratio 0.32 by
sex, the red four-fifths-rule banner. "U.S. anti-discrimination law flags anything
under 0.80 as adverse impact. This model approves women for high income at a third
the rate of men." Show the selection-rate bar chart.

**[0:45–1:10] It's not just selection.** Switch to the equal-opportunity chart.
"Even among people who *truly* earn over \$50k, women are correctly approved less
often. The errors are unequally distributed." Switch the attribute to **race** —
"same story, even worse: ratio 0.21."

**[1:10–1:40] The fix.** Flip the **Apply bias mitigation** toggle. "One toggle.
The bars even out, the four-fifths banner turns green-ish, the equalized-odds gap
drops from 0.08 to 0.03." Show the before/after table.

**[1:40–2:00] The point.** "And the accuracy cost? About one point. Fairness was
affordable the whole time — it just had to be measured and chosen. FairLens makes
that choice visible to anyone." End on the repo + live link.
