"""Preprocessing utilities — decisions justified in the writeups.

Provides `build_tree_preprocessor()` (median-impute numerics, OHE categoricals,
no scaling). Cleans known dirty values and adds missingness-indicator
columns for the three behavioural features where NA-vs-present itself
predicts churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Churned"

# Features that — per EDA — are dirty: cap to physically meaningful ranges.
# (Done in clean_dirty_values; the model preprocessors run after this.)
NUMERIC_FEATURES = [
    "Age", "Membership_Years", "Login_Frequency", "Session_Duration_Avg",
    "Pages_Per_Session", "Cart_Abandonment_Rate", "Wishlist_Items",
    "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase",
    "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
    "Customer_Service_Calls", "Product_Reviews_Written",
    "Social_Media_Engagement_Score", "Mobile_App_Usage",
    "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance",
]

# Categorical features kept for OHE. City dropped: 40 levels, ~0 MI with target.
CATEGORICAL_FEATURES = ["Gender", "Country", "Signup_Quarter"]

# Missingness flags that carry strong signal (per EDA supplement).
MISSING_FLAG_SOURCES = ["Session_Duration_Avg", "Email_Open_Rate", "Customer_Service_Calls"]


def clean_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physically-motivated caps to obviously dirty values.

    Caps justified by EDA:
      - Age: capped to [18, 90]. ~36/47,505 rows outside this range, including max=200.
      - Total_Purchases: clipped at 0 (40 negative entries; counts cannot be negative).
      - Cart_Abandonment_Rate: capped at 100 (rate is %; 30 entries > 100).
    Other heavy-tailed features (AOV, Returns_Rate) are NOT capped — distribution
    is broad but values are individually plausible; tree models tolerate the tail.
    """
    out = df.copy()
    out["Age"] = out["Age"].clip(lower=18, upper=90)
    out["Total_Purchases"] = out["Total_Purchases"].clip(lower=0)
    out["Cart_Abandonment_Rate"] = out["Cart_Abandonment_Rate"].clip(lower=0, upper=100)
    return out


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add NA-indicator columns for the three features whose missingness predicts churn."""
    out = df.copy()
    for col in MISSING_FLAG_SOURCES:
        out[f"{col}__missing"] = out[col].isna().astype("int8")
    return out


def add_engagement_index(df: pd.DataFrame) -> pd.DataFrame:
    """Composite z-score of engagement features.

    The seven engagement features are mutually correlated at r = 0.60–0.76.
    Collapsing them into a single index avoids redundancy and makes the
    feature importance narrative cleaner — one 'engagement' axis rather than
    seven correlated ones.
    """
    eng = [
        "Session_Duration_Avg", "Pages_Per_Session", "Mobile_App_Usage",
        "Login_Frequency", "Email_Open_Rate", "Wishlist_Items",
        "Social_Media_Engagement_Score",
    ]
    out = df.copy()
    z = out[eng]
    z = (z - z.mean(skipna=True)) / z.std(skipna=True)
    out["Engagement_Index"] = z.mean(axis=1, skipna=True)
    return out


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all pre-modelling transformations in the correct order.

    Runs: clean_dirty_values → add_missing_flags → add_engagement_index.
    Called by both modeling.py and generate_reports.py before any splitting.
    """
    df = clean_dirty_values(df)
    df = add_missing_flags(df)
    df = add_engagement_index(df)
    return df


def _all_numeric_after_prep() -> list[str]:
    # Original numeric features + the three NA-flag columns + the engagement composite.
    return NUMERIC_FEATURES + [f"{c}__missing" for c in MISSING_FLAG_SOURCES] + ["Engagement_Index"]


def build_tree_preprocessor() -> ColumnTransformer:
    """Median-impute numerics (no scaling) → OHE categoricals.

    No scaling: tree-based models split on thresholds and are scale-invariant.
    Median imputation is applied even though HistGradientBoosting handles NaN
    natively — the explicit NA-flag columns added by add_missing_flags() already
    capture the missingness signal, so imputed values are just fill-ins.
    """
    num_cols = _all_numeric_after_prep()
    cat_cols = CATEGORICAL_FEATURES
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_columns() -> list[str]:
    """Full list of feature columns passed to the model after prepare_frame()."""
    return _all_numeric_after_prep() + CATEGORICAL_FEATURES
