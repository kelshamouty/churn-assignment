"""Generate self-contained HTML reports for EDA and modelling.

Run as: python -m src.generate_reports
  (requires figures to already exist — run src.eda and src.modeling first)

Both files embed figures as base64 so they can be shared as single files.
  - reports/eda_report.html
  - reports/modelling_report.html
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import pandas as pd

from src.eda_supplement import compute as eda_supplement_compute
from src.modeling import (
    MODEL_NAME, SEED,
    business_savings_curve,
    confusion_at_thresholds,
    cv_scores,
    holdout_scores,
    make_model,
    permutation_importances,
)
from src.preprocess import TARGET, feature_columns, prepare_frame

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ecommerce_customer_churn_dataset.csv"
FIG_EDA = ROOT / "figures" / "eda"
FIG_MOD = ROOT / "figures" / "modeling"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def img64(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f7f8fa;
  color: #1a1a2e;
  line-height: 1.7;
}
.page { max-width: 900px; margin: 0 auto; padding: 48px 24px 80px; }

/* header */
.hero {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
  color: #fff;
  padding: 56px 48px;
  border-radius: 16px;
  margin-bottom: 48px;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin-bottom: 10px; }
.hero p  { font-size: 1.05rem; opacity: .8; max-width: 620px; }
.hero .meta { margin-top: 20px; font-size: .85rem; opacity: .6; }

/* section headers */
h2 {
  font-size: 1.45rem;
  font-weight: 700;
  margin: 56px 0 8px;
  color: #0f3460;
  border-left: 4px solid #e94560;
  padding-left: 14px;
}
h3 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 28px 0 8px;
  color: #1a1a2e;
}

/* inline code */
code {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: .85em;
  background: #eef2f7;
  color: #0f3460;
  padding: 1px 5px;
  border-radius: 4px;
  white-space: nowrap;
}

/* prose */
p  { margin-bottom: 14px; font-size: .97rem; }
ul { margin: 8px 0 14px 22px; }
li { margin-bottom: 6px; font-size: .97rem; }

/* highlight boxes */
.callout {
  border-radius: 10px;
  padding: 18px 22px;
  margin: 20px 0;
  font-size: .95rem;
}
.callout.insight  { background: #eef6ff; border-left: 4px solid #3a86ff; }
.callout.warning  { background: #fff8ec; border-left: 4px solid #f9a825; }
.callout.decision { background: #edfaf1; border-left: 4px solid #43a047; }
.callout.result   { background: #fdeef4; border-left: 4px solid #e94560; }
.callout strong   { display: block; margin-bottom: 4px; font-size: .9rem; text-transform: uppercase; letter-spacing: .05em; opacity: .6; }

/* figures */
.fig-wrap {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  margin: 24px 0;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
  text-align: center;
}
.fig-wrap img { max-width: 100%; border-radius: 6px; }
.fig-caption {
  margin-top: 10px;
  font-size: .82rem;
  color: #666;
  font-style: italic;
}

/* stat cards */
.cards { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
.card {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  flex: 1 1 160px;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
  text-align: center;
}
.card .val { font-size: 2rem; font-weight: 700; color: #0f3460; }
.card .lbl { font-size: .78rem; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }

/* tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  background: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
  font-size: .88rem;
}
th {
  background: #0f3460;
  color: #fff;
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
}
td { padding: 9px 14px; border-bottom: 1px solid #f0f0f0; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f9f9fb; }

/* divider */
.divider { height: 1px; background: #e5e7eb; margin: 40px 0; }

/* footer */
.footer { text-align: center; font-size: .8rem; color: #aaa; margin-top: 60px; }
"""


# Column / feature names that should render as inline code throughout the reports.
_COLUMN_NAMES = [
    "Session_Duration_Avg", "Email_Open_Rate", "Customer_Service_Calls",
    "Cart_Abandonment_Rate", "Lifetime_Value", "Membership_Years",
    "Days_Since_Last_Purchase", "Login_Frequency", "Pages_Per_Session",
    "Mobile_App_Usage", "Social_Media_Engagement_Score", "Product_Reviews_Written",
    "Wishlist_Items", "Total_Purchases", "Average_Order_Value", "Discount_Usage_Rate",
    "Returns_Rate", "Payment_Method_Diversity", "Credit_Balance", "Signup_Quarter",
    "Engagement_Index", "HistGradientBoostingClassifier",
]
# Build a single alternation pattern, longest names first to avoid partial matches.
_COL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(_COLUMN_NAMES, key=len, reverse=True)) + r")\b"
)


def _wrap_code_tags(html: str) -> str:
    """Wrap known column/feature names in <code> tags, touching only text nodes.

    Splits on existing <code> blocks and HTML tags so that attributes,
    tag content, and already-wrapped names are never double-processed.
    """
    # Splitting pattern: existing <code>…</code> blocks OR any HTML tag.
    # Odd-indexed chunks in the result are tags/code blocks — leave them alone.
    parts = re.split(r"(<code>[^<]*</code>|<[^>]+>)", html)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)          # tag or existing code block — untouched
        else:
            out.append(_COL_PATTERN.sub(r"<code>\1</code>", part))
    return "".join(out)


def _clean_prose(text: str) -> str:
    """Normalise voice, punctuation, and column-name formatting."""
    # Split on base64 blobs first so no replacement ever touches PNG bytes.
    parts = re.split(r"(data:image/png;base64,[A-Za-z0-9+/=]+)", text)
    cleaned = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            cleaned.append(part)
        else:
            part = part.replace("—", ":")
            part = re.sub(r" -- ", ": ", part)
            part = re.sub(r"\bWe\b", "I", part)
            part = re.sub(r"\bwe\b", "I", part)
            part = _wrap_code_tags(part)
            cleaned.append(part)
    return "".join(cleaned)


def html_doc(title: str, body: str) -> str:
    body = _clean_prose(body)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page">
{body}
<div class="footer">Generated by src/generate_reports.py · 2026-05-21</div>
</div>
</body>
</html>"""


def fig(path: Path, caption: str) -> str:
    return f"""<div class="fig-wrap">
  <img src="{img64(path)}" alt="{caption}">
  <div class="fig-caption">{caption}</div>
</div>"""


def cards(*items) -> str:
    inner = "".join(
        f'<div class="card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for v, l in items
    )
    return f'<div class="cards">{inner}</div>'


def callout(kind: str, label: str, text: str) -> str:
    return f'<div class="callout {kind}"><strong>{label}</strong>{text}</div>'


# ── EDA REPORT ────────────────────────────────────────────────────────────────

def build_eda_report(supp: dict) -> str:

    body = """
<div class="hero">
  <h1>Customer Churn: EDA Report</h1>
  <p>Exploratory data analysis on a synthetic e-commerce dataset of 50,000 customers.
     This report documents what the data looked like, what problems we found, and the
     decisions those findings drove — before a single model was trained.</p>
  <div class="meta">Author: Khaled El-Shamouty &nbsp;·&nbsp; 2026-05-21</div>
</div>

<h2>1. The dataset at a glance</h2>

<p>We start with 50,000 customers, 24 features, and a binary target: did the customer
churn? Before diving into signal, a few basics.</p>
"""

    body += cards(
        ("50,000", "Customers"),
        ("24", "Features"),
        ("28.9%", "Churn rate"),
        ("0", "Duplicates"),
        ("14 / 24", "Features with missing values"),
    )

    body += """
<p>A 28.9% positive class is moderately imbalanced — meaningful enough that plain
accuracy is a poor metric, but not so extreme that resampling is needed.
We'll use PR-AUC (precision-recall) as the primary evaluation metric downstream,
since it is sensitive to performance on the minority class.</p>
"""

    body += fig(FIG_EDA / "01_numeric_histograms.png",
                "Distribution of all 20 numeric features. Note the heavy-tailed "
                "Average_Order_Value and skewed Membership_Years — and the near-uniform "
                "spread of most engagement features.")

    body += """
<h2>2. Data quality issues</h2>

<p>This is a synthetic dataset, and it contains intentional noise designed to test
how the analyst handles dirty data. Three obvious problems surfaced immediately:</p>

<ul>
  <li><strong>Age:</strong> minimum of 5, maximum of 200. Neither value is plausible
      for an e-commerce account. About 36 rows fall outside a reasonable [18, 90]
      window.</li>
  <li><strong>Total_Purchases:</strong> 40 negative values — counts cannot be negative.</li>
  <li><strong>Cart_Abandonment_Rate:</strong> 30 values above 100% — it's a rate.</li>
</ul>
"""

    body += callout("warning", "Decision",
                    "Cap Age to [18, 90], clip Total_Purchases at 0, cap "
                    "Cart_Abandonment_Rate at 100. Other heavy-tailed features "
                    "(Average_Order_Value, Returns_Rate) are left untouched — their "
                    "distributions are broad but individually plausible, and the "
                    "chosen model handles fat tails natively.")

    body += """
<h2>3. Missingness — and why it matters</h2>

<p>Fourteen features carry missing values between 0.3% and 12%, concentrated in
behavioural fields: session data, mobile usage, social engagement, email opens.
The identifier-like fields (country, signup quarter, lifetime value) are fully
populated.</p>

<p>The natural question to ask is: <em>is missingness random, or does it correlate
with churn?</em> The answer turned out to be the single most useful finding in the
entire EDA.</p>
"""

    miss = supp["missing_flag_churn_signal"][:5]
    rows = "".join(
        f"<tr><td>{r['feature']}</td>"
        f"<td>{r['churn_when_missing']:.1%}</td>"
        f"<td>{r['churn_when_present']:.1%}</td>"
        f"<td style='color:#e94560;font-weight:600'>+{r['delta']:.1%}</td></tr>"
        for r in miss
    )
    body += f"""
<table>
  <thead><tr><th>Feature</th><th>Churn (missing)</th><th>Churn (present)</th><th>Δ</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""

    body += callout("insight", "Key insight",
                    "When <strong>Session_Duration_Avg</strong> is missing, the churn "
                    "rate jumps from 28% to 41% — a 13 percentage-point lift. The same "
                    "pattern holds for Email_Open_Rate (+12 pp) and Customer_Service_Calls "
                    "(+7 pp). Customers who stopped engaging stopped generating telemetry. "
                    "Missingness is not noise — it is a leading signal. "
                    "I add explicit NA-flag columns for these three features.")

    body += fig(FIG_EDA / "06_missingness_signal.png",
                "Churn rate when each behavioural feature is missing vs present. "
                "The gap is 7–13 percentage points above the 28.9% base rate.")

    body += """
<h2>4. What predicts churn — the linear view</h2>

<p>Point-biserial correlation tells us how strongly each numeric feature moves
with the churn label, linearly. Two clear groups emerge:</p>

<ul>
  <li><strong>Frustrated customers churn:</strong> high Customer_Service_Calls (+0.29),
      high Cart_Abandonment_Rate (+0.28), long Days_Since_Last_Purchase (+0.15).</li>
  <li><strong>Engaged customers stay:</strong> high Pages_Per_Session (−0.23),
      Session_Duration_Avg (−0.23), Mobile_App_Usage (−0.22), Email_Open_Rate (−0.22),
      Login_Frequency (−0.20).</li>
</ul>

<p>Seven engagement features tell essentially the same story. They're mutually
correlated at 0.60–0.76 with each other, which would destabilise linear model
coefficients. Rather than let them fight over the same signal, we compute a single
<strong>Engagement_Index</strong> — the mean z-score of all seven — so the
narrative later has one clean axis to talk about.</p>
"""

    body += fig(FIG_EDA / "04_top_features_by_churn.png",
                "Distribution of the top 12 features by |r| split by churn label. "
                "Churners (orange) visibly shift on Customer_Service_Calls, "
                "Cart_Abandonment_Rate, and Days_Since_Last_Purchase.")

    body += """
<h2>5. What predicts churn — the non-linear view</h2>

<p>Linear correlation only captures straight-line relationships. Mutual information
captures <em>any</em> statistical dependence. The ranking changed in one important way:</p>
"""

    mi = supp["mutual_information_ranking"]
    ltv = next(r for r in mi if r["feature"] == "Lifetime_Value")
    mem = next(r for r in mi if r["feature"] == "Membership_Years")

    body += f"""
<table>
  <thead><tr><th>Feature</th><th>Linear r</th><th>Mutual Information</th><th>Verdict</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Lifetime_Value</strong></td>
      <td>−0.01</td>
      <td style="font-weight:600;color:#0f3460">{ltv['mi']:.4f} (3rd highest)</td>
      <td>Strong non-linear signal — invisible to linear models</td>
    </tr>
    <tr>
      <td>Membership_Years</td>
      <td>0.00</td>
      <td>{mem['mi']:.4f}</td>
      <td>Genuinely no signal at all</td>
    </tr>
  </tbody>
</table>
"""

    body += fig(FIG_EDA / "07_mi_vs_correlation.png",
                "Each dot is a feature. X-axis: absolute linear correlation with churn. "
                "Y-axis: mutual information. Lifetime_Value sits isolated top-left: "
                "high MI, near-zero linear correlation. The shaded zone is where linear "
                "models are blind.")

    body += callout("decision", "The modelling decision this drives",
                    "<strong>Lifetime_Value</strong> has near-zero linear correlation "
                    "with churn but ranks 3rd in mutual information. The relationship "
                    "is non-linear — a linear model cannot use it. This is the empirical "
                    "justification for choosing a tree-based gradient boosting model "
                    "rather than logistic regression. The choice is forced by the data, "
                    "not preference.")

    body += fig(FIG_EDA / "03_correlation_heatmap.png",
                "Correlation heatmap of all numeric features. The engagement cluster "
                "(top-left block) is clearly visible at r = 0.60–0.76. "
                "Cart_Abandonment_Rate sits as a negative mirror of that cluster.")

    body += """
<h2>6. Geography and demographics</h2>

<p>Categorical features were checked for predictive signal by comparing churn rates
across their levels.</p>
"""

    body += fig(FIG_EDA / "02_churn_by_categorical.png",
                "Churn rate by categorical feature. The dashed line is the overall "
                "28.9% base rate. All categories hover close to it — the spread is "
                "small relative to sampling noise.")

    body += """
<p>The max-minus-min churn rate spread across levels was 6.5 pp for City (40 levels),
2.6 pp for Country (8 levels), 2.5 pp for Gender, and 1.5 pp for Signup_Quarter.
These deltas are within the noise range for the per-level sample sizes involved.</p>
"""

    body += callout("decision", "Decision",
                    "Keep Country, Gender, and Signup_Quarter (cheap to one-hot encode, "
                    "harmless). Drop City — 40 OHE columns for essentially zero mutual "
                    "information is a bad trade. Geography and demographics are not "
                    "meaningful segmentation axes for this churn problem.")

    body += """
<h2>7. Recency as a signal</h2>

<p>Days_Since_Last_Purchase shows a clear distributional shift between churners and
non-churners: churners have a median of 26 days since their last purchase versus
19 days for stayers, with a heavier right tail.</p>
"""

    body += fig(FIG_EDA / "05_days_since_last_purchase.png",
                "Distribution of days since last purchase by churn label. "
                "Churners (orange) skew toward longer recency gaps, as expected.")

    body += """
<h2>8. Summary of decisions going into modelling</h2>
"""

    body += """
<table>
  <thead><tr><th>Decision</th><th>Rationale</th></tr></thead>
  <tbody>
    <tr>
      <td>Cap Age [18,90], Total_Purchases ≥0, Cart_Abandonment_Rate ≤100</td>
      <td>The dataset contains obvious data-entry errors (Age = 200, negative purchase counts, abandonment rates above 100%). Capping to physically meaningful ranges removes noise without discarding rows.</td>
    </tr>
    <tr>
      <td>Add NA-flag columns for Session_Duration_Avg, Email_Open_Rate, Customer_Service_Calls</td>
      <td>When these fields are missing, churn rate jumps 7–13 percentage points above the base rate. The absence of data is itself a signal — customers who disengaged stopped generating telemetry — so it should be explicitly modelled rather than silently imputed away.</td>
    </tr>
    <tr>
      <td>Build Engagement_Index composite (mean z-score of 7 features)</td>
      <td>Seven engagement features are mutually correlated at r = 0.60–0.76 and tell essentially the same story. Collapsing them into a single index avoids redundancy in the model and makes the headline driver narrative cleaner for stakeholders.</td>
    </tr>
    <tr>
      <td>Use tree-based gradient boosting, not logistic regression</td>
      <td>Mutual information analysis revealed that Lifetime_Value has near-zero linear correlation with churn (r = −0.01) but ranks 3rd in mutual information — a clear sign of a non-linear relationship. A linear model cannot use this signal; a tree-based model can, and this turned out to be the dominant driver.</td>
    </tr>
    <tr>
      <td>Drop City (40 levels); keep Country, Gender, Signup_Quarter</td>
      <td>City has 40 levels but near-zero mutual information with churn. Adding 40 one-hot columns for essentially no predictive gain is a bad trade. The remaining categoricals are cheap to encode and harmless to include.</td>
    </tr>
    <tr>
      <td>No feature scaling</td>
      <td>Tree models split features by finding thresholds, an operation unaffected by the scale of the input. Scaling would change the numbers but not what the model learns, so it adds complexity with no benefit here.</td>
    </tr>
    <tr>
      <td>Median imputation; no row-dropping</td>
      <td>Dropping rows with any missing value would remove a disproportionate share of churners (since missingness correlates with churn) and bias the training set. Median imputation keeps all 50,000 rows and makes a conservative, symmetric fill that introduces no structural bias.</td>
    </tr>
  </tbody>
</table>
"""

    return html_doc("EDA Report: Customer Churn", body)


# ── MODELLING REPORT ──────────────────────────────────────────────────────────

def build_modelling_report(m: dict) -> str:
    holdout = m["holdout_results"]
    cv = m["cv_results"]
    imp = m["permutation_importance_top20"]
    cm = m["confusion_at_thresholds"]
    savings = m["business_savings_curve"]

    body = """
<div class="hero">
  <h1>Customer Churn: Modelling Report</h1>
  <p>What model we built, why, how well it works, what it says about churn drivers,
     and what it means for the business. The EDA choices that led here are documented
     in the companion EDA report.</p>
  <div class="meta">Author: Khaled El-Shamouty &nbsp;·&nbsp; 2026-05-21</div>
</div>

<h2>1. The modelling brief (in one sentence)</h2>

<p>Rank 50,000 customers by their probability of churning, so a retention team with a
finite budget can decide which customers to contact — and where to draw the line.</p>

<p>That framing matters. It makes this a <strong>ranking problem</strong>, not a
classification problem. The right metrics are therefore precision@K and PR-AUC
(how good is the top of the ranked list?), not accuracy (how often is the
binary label right?).</p>

<h2>2. Model choice</h2>

<p>The EDA found that <strong>Lifetime_Value</strong> — the feature with the highest
mutual information with churn — has essentially zero linear correlation with it
(r = −0.01). The signal is non-linear. A logistic regression model can't use it.
A gradient-boosted tree model can, trivially.</p>
"""

    body += callout("decision", "Why HistGradientBoosting",
                    "sklearn's HistGradientBoostingClassifier: handles non-linear "
                    "feature effects, native missing-value support, scale-invariant, "
                    "and zero extra dependencies (ships with scikit-learn). "
                    "Trained with 400 boosting rounds at learning rate 0.05. "
                    "No grid search — CV variance was too small to justify it.")

    body += """
<h2>3. Validation approach</h2>

<ul>
  <li>Stratified 80/20 split — 40,000 train, 10,000 test set (set aside until final eval).</li>
  <li>5-fold stratified CV on the train set to check stability before committing to the test set.</li>
  <li>All reported numbers below are on the untouched 20% test set.</li>
</ul>

<h2>4. Results</h2>
"""

    body += cards(
        (f"{holdout['roc_auc']:.3f}", "ROC-AUC"),
        (f"{holdout['pr_auc']:.3f}", "PR-AUC"),
        (f"{holdout['brier']:.3f}", "MSE (calibration)"),
        (f"{cv['cv_pr_auc_mean']:.3f} ± {cv['cv_pr_auc_std']:.3f}", "CV PR-AUC (5-fold)"),
    )

    body += f"""
<p>CV and test set agree closely — the result isn't a lucky split. A MSE of
{holdout['brier']:.3f} means the predicted probabilities are well-calibrated:
the model isn't just ranking, it's outputting meaningful probabilities that can
be used directly in expected-value calculations.</p>
"""

    body += fig(FIG_MOD / "roc_curve.png",
                "ROC curve on the test set. AUC = 0.929.")

    body += fig(FIG_MOD / "pr_curve.png",
                "Precision-Recall curve. Average precision = 0.914. The dashed line "
                "is the baseline (random classifier at the 28.9% churn rate). "
                "The model sits dramatically above it across all recall levels.")

    body += fig(FIG_MOD / "calibration.png",
                f"Calibration plot. A perfectly calibrated model follows the dashed diagonal. "
                f"MSE = {holdout['brier']:.3f}: predicted probabilities track empirical "
                f"churn rates closely, which matters for downstream cost calculations.")

    body += """
<h2>5. What drives the predictions</h2>

<p>Permutation importance measures how much model performance (PR-AUC) drops when
each feature's values are shuffled. Features the model relies on most show the
largest drop.</p>
"""

    top10 = imp[:10]
    rows = "".join(
        f"<tr><td>{r['feature']}</td>"
        f"<td><div style='background:#0f3460;height:10px;border-radius:4px;"
        f"width:{int(r['importance_mean']/imp[0]['importance_mean']*100)}%'></div></td>"
        f"<td style='text-align:right'>{r['importance_mean']:.3f}</td></tr>"
        for r in top10
    )
    body += f"""
<table>
  <thead><tr><th>Feature</th><th>Relative importance</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""

    body += callout("insight", "Surprise #1 — Lifetime Value",
                    "<strong>Lifetime_Value</strong> is the #2 driver despite having "
                    "r = −0.01 linear correlation with churn. The relationship is highly "
                    "non-linear — confirming the EDA's prediction and justifying the "
                    "choice of a tree-based model. 'Our most valuable customers are safe' "
                    "is not supported by the data.")

    body += callout("insight", "Surprise #2 — Membership_Years is noise",
                    "Tenure contributes essentially zero to the model. Long-tenured "
                    "customers are not meaningfully safer. Retention spend based on "
                    "'they've been with us for years' as a proxy is not well-founded "
                    "in this data.")

    body += """
<h2>6. How engaged customers differ from churners</h2>
"""

    body += fig(FIG_MOD / "cumulative_gains.png",
                "Cumulative gains chart. Contacting the top 20% of customers ranked "
                "by predicted churn risk captures ~67% of all true churners. "
                "A random contact strategy would need to contact 20% to reach 20%.")

    body += """
<h2>7. Choosing an operating point</h2>

<p>The model outputs a probability for each customer. The business picks a threshold
above which a customer gets a retention offer. That choice trades precision against
recall — and budget against coverage.</p>
"""

    rows = "".join(
        f"<tr><td>{r['threshold']}</td>"
        f"<td>{r['precision']:.1%}</td>"
        f"<td>{r['recall']:.1%}</td>"
        f"<td>{r['f1']:.3f}</td>"
        f"<td>{r['positives_flagged_pct']:.1%} of base</td></tr>"
        for r in cm
    )
    body += f"""
<table>
  <thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Customers contacted</th></tr></thead>
  <tbody>{rows}</tbody>
</table>

<p>At <strong>threshold 0.30</strong>: cast wide, capture 85% of true churners,
flag ~29% of the base — a high-recall campaign. At <strong>0.70</strong>: surgical,
96% precision, but miss 27% of true churners and flag only 22% of the base.
The right threshold depends on the relative cost of a missed churner vs a wasted
offer — a business decision, not a modelling decision.</p>
"""

    body += """
<h2>8. The business case — savings curve</h2>

<p>To translate model performance into business terms, we compute expected net savings
at each possible targeting cutoff under three transparent assumptions:</p>

<ul>
  <li>$100 retained revenue per churner successfully saved</li>
  <li>$15 cost per retention offer sent</li>
  <li>30% of contacted true churners are actually retained</li>
</ul>

<p><em>These numbers are placeholders. The shape of the curve is the finding;
the dollar levels only become meaningful when real marketing economics are plugged in.</em></p>
"""

    body += fig(FIG_MOD / "savings_curve.png",
                "Expected net savings vs fraction of customers contacted (sorted by "
                "predicted churn risk). The curve peaks around 25% — beyond that, "
                "the cost of contacting lower-precision customers outweighs the returns.")

    # find peak
    peak = max(savings, key=lambda r: r["net_savings"])

    body += callout("result", "The actionable finding",
                    f"Under these toy economics, the optimal campaign targets the "
                    f"<strong>top ~{int(peak['k_fraction']*100)}% of customers</strong> "
                    f"by predicted risk — contacting {peak['contacted']:,} customers, "
                    f"capturing ~{peak['true_churners_in_topK']:,} true churners, "
                    f"and generating estimated net savings of "
                    f"<strong>${peak['net_savings']:,.0f}</strong> per campaign cycle. "
                    f"Beyond that point the campaign spends more than it recovers.")

    body += """
<h2>9. Recommendations</h2>

<h3>For the retention team</h3>
<ul>
  <li><strong>Target the top 25–30% of customers by predicted churn risk</strong> each
      campaign cycle. That captures ~80% of true churners at &gt;82% precision.</li>
  <li><strong>Treat service-call volume as a front-line signal.</strong> It's the
      #1 model driver and readable directly from the CRM — no model needed. Customers
      with 7+ calls in the window are at sharply elevated risk.</li>
  <li><strong>Flag customers whose behavioural data goes missing</strong>, not just
      those whose metrics decline. A lapsed session or missing email-open record is
      a leading indicator, not just a gap.</li>
</ul>

<h3>For Product / Analytics leadership</h3>
<ul>
  <li><strong>Lifetime_Value is not a proxy for churn safety.</strong> The model
      shows a strong non-linear relationship — high-LTV customers churn in ways a
      simple ranking by value won't reveal.</li>
  <li><strong>Tenure doesn't predict churn here.</strong> If production data agrees,
      long-tenure customers should not be deprioritised in retention spend.</li>
  <li><strong>Geography and signup cohort carry no signal.</strong> Campaign
      segmentation by country or quarter is not statistically justified.</li>
</ul>

<h2>10. Limitations</h2>

<table>
  <thead><tr><th>Limitation</th><th>Implication</th></tr></thead>
  <tbody>
    <tr>
      <td>Synthetic data — performance is unrealistically clean</td>
      <td>Headline AUC numbers should not be used as production targets. Re-evaluate on real data.</td>
    </tr>
    <tr>
      <td>No event timestamp on Churned</td>
      <td>Can't validate temporal stability or build time-to-event models.</td>
    </tr>
    <tr>
      <td>Toy economics in savings curve</td>
      <td>Replace with real offer costs and LTV estimates before making budget decisions.</td>
    </tr>
    <tr>
      <td>Lifetime_Value may be contemporaneous</td>
      <td>If LTV includes post-churn revenue, it's a leakage risk. Audit computation window before deploying.</td>
    </tr>
    <tr>
      <td>No hyperparameter tuning</td>
      <td>Marginal performance left on table — deliberate given brief's guidance. An Optuna sweep is a natural next step.</td>
    </tr>
  </tbody>
</table>
"""

    return html_doc("Modelling Report: Customer Churn", body)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    """Load data, run EDA supplement, fit model, return (supp, metrics) dicts.

    Re-trains the model from scratch using the same seed as src.modeling so
    the metrics are identical to the figures already on disk.
    """
    from sklearn.model_selection import train_test_split

    print("  Loading data ...")
    df = pd.read_csv(DATA)

    print("  Running EDA supplement ...")
    supp = eda_supplement_compute(df)

    print("  Preparing features + splitting ...")
    df = prepare_frame(df)
    y = df[TARGET].astype(int)
    X = df[feature_columns()].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )

    print("  Running 5-fold CV ...")
    model = make_model()
    cv = cv_scores(model, X_train, y_train)

    print("  Fitting model ...")
    model.fit(X_train, y_train)
    holdout, proba = holdout_scores(model, X_test, y_test)

    cm_table = confusion_at_thresholds(proba, y_test)
    perm = permutation_importances(model, X_test, y_test, n_repeats=4)
    savings = business_savings_curve(y_test.values, proba)

    m = {
        "model": MODEL_NAME,
        "cv_results": {k: round(v, 4) for k, v in cv.items()},
        "holdout_results": {k: round(v, 4) for k, v in holdout.items()},
        "confusion_at_thresholds": cm_table,
        "permutation_importance_top20": perm.head(20).round(5).to_dict(orient="records"),
        "business_savings_curve": savings.to_dict(orient="records"),
    }
    return supp, m


def main():
    print("Computing data ...")
    supp, m = load_data()

    print("Generating EDA report ...")
    eda_html = build_eda_report(supp)
    out = REPORT_DIR / "eda_report.html"
    out.write_text(eda_html, encoding="utf-8")
    print(f"  → {out}")

    print("Generating modelling report ...")
    mod_html = build_modelling_report(m)
    out = REPORT_DIR / "modelling_report.html"
    out.write_text(mod_html, encoding="utf-8")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
