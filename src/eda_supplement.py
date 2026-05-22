"""Supplementary EDA: data quality issues, missing-flag predictive power, MI ranking.

Run as: python -m src.eda_supplement   (prints summary to stdout)
Import: from src.eda_supplement import compute
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ecommerce_customer_churn_dataset.csv"


def compute(df: pd.DataFrame) -> dict:
    """Run supplementary EDA checks and return results as a dict.

    Covers four areas:
      - dirty_value_diagnostics: known data-quality issues (impossible ages,
        negative counts, rates above 100%) with counts and extremes.
      - missing_flag_churn_signal: for each feature with missing values,
        the churn rate when the field is missing vs present. A large gap
        means the missingness itself is predictive and should be modelled.
      - mutual_information_ranking: features ranked by MI with the target,
        using a KNN estimator that captures non-linear dependence that
        point-biserial correlation would miss.
      - engagement_composite: correlation of the mean z-score of the seven
        engagement features with churn, as a sanity check on the composite.

    Returns:
        dict with keys: dirty_value_diagnostics, missing_flag_churn_signal,
        mutual_information_ranking, engagement_composite.
    """
    target = "Churned"

    dirty = {
        "Age": {
            "n_below_10": int((df["Age"] < 10).sum()),
            "n_above_100": int((df["Age"] > 100).sum()),
            "max": float(df["Age"].max()),
            "min": float(df["Age"].min()),
        },
        "Total_Purchases": {
            "n_negative": int((df["Total_Purchases"] < 0).sum()),
            "min": float(df["Total_Purchases"].min()),
            "n_above_100": int((df["Total_Purchases"] > 100).sum()),
        },
        "Cart_Abandonment_Rate": {
            "n_above_100": int((df["Cart_Abandonment_Rate"] > 100).sum()),
            "max": float(df["Cart_Abandonment_Rate"].max()),
        },
        "Returns_Rate": {
            "n_above_50": int((df["Returns_Rate"] > 50).sum()),
            "n_above_100": int((df["Returns_Rate"] > 100).sum()),
            "max": float(df["Returns_Rate"].max()),
        },
        "Average_Order_Value": {
            "p99": float(df["Average_Order_Value"].quantile(0.99)),
            "p999": float(df["Average_Order_Value"].quantile(0.999)),
            "max": float(df["Average_Order_Value"].max()),
        },
        "Lifetime_Value": {
            "n_zero": int((df["Lifetime_Value"] == 0).sum()),
            "min": float(df["Lifetime_Value"].min()),
        },
    }

    miss_signal = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].isna().any():
            flag = df[c].isna().astype(int)
            rate_missing = df.loc[flag == 1, target].mean()
            rate_present = df.loc[flag == 0, target].mean()
            miss_signal.append({
                "feature": c,
                "missing_pct": round(float(flag.mean() * 100), 3),
                "churn_when_missing": round(float(rate_missing), 4),
                "churn_when_present": round(float(rate_present), 4),
                "delta": round(float(rate_missing - rate_present), 4),
            })
    miss_signal.sort(key=lambda x: abs(x["delta"]), reverse=True)

    work = df.copy()
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(target)
    work[num_cols] = work[num_cols].fillna(work[num_cols].median(numeric_only=True))
    cat_cols = ["Gender", "Country", "City", "Signup_Quarter"]
    for c in cat_cols:
        work[c] = work[c].astype("category").cat.codes
    X = work[num_cols + cat_cols]
    y = work[target]
    mi = mutual_info_classif(
        X, y,
        discrete_features=[X.columns.get_loc(c) for c in cat_cols],
        random_state=42,
    )
    mi_rank = sorted(
        [{"feature": f, "mi": round(float(v), 5)} for f, v in zip(X.columns, mi)],
        key=lambda x: x["mi"], reverse=True,
    )

    eng_cols = [
        "Session_Duration_Avg", "Pages_Per_Session", "Mobile_App_Usage",
        "Login_Frequency", "Email_Open_Rate", "Wishlist_Items",
        "Social_Media_Engagement_Score",
    ]
    e = df[eng_cols].copy()
    e = (e - e.mean()) / e.std()
    engagement = e.mean(axis=1)
    engagement_summary = {
        "feature": "engagement_idx (mean z-score of 7 engagement features)",
        "r_pb_with_churn": round(float(engagement.corr(df[target])), 4),
        "missing_count_when_any_missing": int(e.isna().any(axis=1).sum()),
    }

    return {
        "dirty_value_diagnostics": dirty,
        "missing_flag_churn_signal": miss_signal,
        "mutual_information_ranking": mi_rank,
        "engagement_composite": engagement_summary,
    }


if __name__ == "__main__":
    import json
    df = pd.read_csv(DATA)
    print(json.dumps(compute(df), indent=2, default=str))
