"""Phase 1 — Comprehensive EDA.

Run as: python -m src.eda
Outputs: figures/eda/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ecommerce_customer_churn_dataset.csv"
FIG_DIR = ROOT / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")


def _section(title: str) -> str:
    bar = "=" * 80
    return f"\n{bar}\n{title}\n{bar}\n"


def main() -> dict:
    out: dict = {}
    df = pd.read_csv(DATA)
    n_rows, n_cols = df.shape

    # --- 1) Shape, dtypes, head ---
    dtypes = df.dtypes.astype(str).to_dict()
    out["shape"] = {"rows": int(n_rows), "cols": int(n_cols)}
    out["dtypes"] = dtypes

    # --- 2) Missingness ---
    miss = df.isna().sum()
    miss_pct = (miss / n_rows * 100).round(3)
    miss_table = (
        pd.DataFrame({"missing": miss, "missing_pct": miss_pct})
        .query("missing > 0")
        .sort_values("missing_pct", ascending=False)
    )
    out["missingness"] = miss_table.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    # --- 3) Duplicates ---
    out["duplicate_rows"] = int(df.duplicated().sum())

    # --- 4) Target balance ---
    target = "Churned"
    churn_rate = float(df[target].mean())
    out["target_balance"] = {
        "churn_rate": round(churn_rate, 4),
        "positives": int(df[target].sum()),
        "negatives": int((df[target] == 0).sum()),
    }

    # --- 5) Categorical inventories ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    out["categorical_columns"] = cat_cols
    out["numeric_columns"] = num_cols
    cat_summary = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(15)
        cat_summary[c] = {
            "n_unique": int(df[c].nunique(dropna=True)),
            "top_15": {str(k): int(v) for k, v in vc.items()},
        }
    out["categorical_summary"] = cat_summary

    # --- 6) Numeric summary ---
    # Extended percentiles (1st, 99th) help surface dirty-value extremes
    # that the default describe() would miss. Skew and kurtosis flag heavy tails.
    num_desc = df[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    num_desc["skew"] = df[num_cols].skew(numeric_only=True)
    num_desc["kurtosis"] = df[num_cols].kurtosis(numeric_only=True)
    num_desc = num_desc.round(3)
    out["numeric_describe"] = num_desc.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    # --- 7) Numeric distributions: figure grid ---
    n = len(num_cols)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.6))
    axes = axes.ravel()
    for i, c in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[c].dropna(), bins=40, ax=ax, kde=False, color="#4C72B0")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric feature distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_numeric_histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- 8) Target rate by categorical ---
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
    if len(cat_cols) == 1:
        axes = [axes]
    cat_target = {}
    for ax, c in zip(axes, cat_cols):
        ct = df.groupby(c)[target].agg(["mean", "count"]).reset_index().sort_values("mean", ascending=False)
        cat_target[c] = ct.round(4).to_dict(orient="records")
        top = ct.head(20)
        sns.barplot(data=top, y=c, x="mean", ax=ax, color="#C44E52")
        ax.axvline(churn_rate, color="black", linestyle="--", label=f"overall={churn_rate:.3f}")
        ax.set_xlabel("Churn rate")
        ax.set_title(f"Churn rate by {c} (top 20)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_churn_by_categorical.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["churn_by_categorical"] = cat_target

    # --- 9) Mean of numeric features by target ---
    # Normalising the delta by the overall mean makes features comparable
    # regardless of scale (e.g. Days_Since_Last_Purchase vs Wishlist_Items).
    grp = df.groupby(target)[num_cols].mean().T
    grp.columns = ["mean_no_churn", "mean_churn"]
    grp["delta"] = grp["mean_churn"] - grp["mean_no_churn"]
    grp["abs_delta_pct_of_overall_mean"] = (grp["delta"].abs() / df[num_cols].mean().replace(0, np.nan)).round(4)
    grp = grp.sort_values("abs_delta_pct_of_overall_mean", ascending=False)
    out["numeric_mean_by_target"] = grp.round(4).reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    # --- 10) Point-biserial correlation with target ---
    # Equivalent to Pearson correlation when one variable is binary.
    # Measures the *linear* relationship between each feature and churn.
    corrs = []
    for c in num_cols:
        s = df[[c, target]].dropna()
        if s[c].nunique() < 2:
            continue
        r, p = stats.pointbiserialr(s[target], s[c])
        corrs.append({"feature": c, "r_pb": round(float(r), 4), "p_value": float(f"{p:.3g}")})
    corrs.sort(key=lambda x: abs(x["r_pb"]), reverse=True)
    out["point_biserial_corr_with_target"] = corrs

    # --- 11) Correlation heatmap among numerics ---
    corr_mat = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(0.5 * len(num_cols) + 4, 18), min(0.5 * len(num_cols) + 4, 18)))
    sns.heatmap(corr_mat, cmap="vlag", center=0, annot=False, cbar=True, ax=ax, square=True)
    ax.set_title("Numeric feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    # Extract the top correlated pairs, masks the diagonal (self-correlation = 1)
    # and keeps only the upper triangle to avoid duplicate pairs.
    cm = corr_mat.where(~np.eye(len(corr_mat), dtype=bool))
    pairs = (
        cm.stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "r"})
    )
    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs[pairs["a"] < pairs["b"]].sort_values("abs_r", ascending=False).head(30)
    out["top_correlated_numeric_pairs"] = pairs.round(4).to_dict(orient="records")

    # --- 12) Outliers via IQR ---
    outlier_table = []
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        outlier_table.append({
            "feature": c,
            "n_outliers_iqr": n_out,
            "pct_outliers": round(n_out / len(s) * 100, 2),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    out["iqr_outliers"] = sorted(outlier_table, key=lambda x: x["pct_outliers"], reverse=True)

    # --- 13) Numeric feature distribution split by churn (top 12 by |r|) ---
    top12 = [c["feature"] for c in corrs[:12]] if corrs else num_cols[:12]
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.ravel()
    for i, c in enumerate(top12):
        ax = axes[i]
        sns.kdeplot(data=df, x=c, hue=target, common_norm=False, fill=True, alpha=0.4, ax=ax)
        ax.set_title(c, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Top features: distributions by churn", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_top_features_by_churn.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- 14) Country / city granularity check ---
    geo = (
        df.groupby(["Country", "City"]) [target]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    out["geo_top_cities"] = geo.head(20).round(4).to_dict(orient="records")

    # --- 15) Signup quarter — possible temporal signal ---
    q = df.groupby("Signup_Quarter")[target].agg(["mean", "count"]).reset_index()
    out["signup_quarter_churn"] = q.round(4).to_dict(orient="records")

    # --- 16) Leakage sanity: distribution of Days_Since_Last_Purchase by churn ---
    # A feature that perfectly separates churners from non-churners could indicate
    # data leakage (i.e. the feature is computed using post-churn information).
    # Days_Since_Last_Purchase is the most likely candidate — we verify it shows
    # a plausible distributional shift rather than a clean separation.
    leak = (
        df.groupby(target)["Days_Since_Last_Purchase"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    )
    out["days_since_last_purchase_by_target"] = leak.round(2).reset_index().to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Days_Since_Last_Purchase", hue=target, bins=50, element="step", common_norm=False, stat="density", ax=ax)
    ax.set_title("Days since last purchase by churn")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_days_since_last_purchase.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- 17) Categorical predictive strength ---
    # Max-minus-min churn rate across levels is a simple, interpretable measure
    # of how much a categorical feature spreads the target. A small spread
    # (e.g. 2-3 pp) relative to the base rate (28.9%) suggests the feature
    # carries little signal for the model.
    cat_strength = []
    for c in cat_cols:
        rates = df.groupby(c)[target].mean()
        if len(rates) < 2:
            continue
        cat_strength.append({
            "feature": c,
            "max_minus_min_rate": round(float(rates.max() - rates.min()), 4),
            "n_levels": int(len(rates)),
        })
    cat_strength.sort(key=lambda x: x["max_minus_min_rate"], reverse=True)
    out["categorical_strength_max_minus_min"] = cat_strength

    # --- Charts for the reports (missingness signal + MI vs correlation) ---
    supp = _chart_missingness(df, target)
    _chart_mi_vs_correlation(supp)

    print(f"EDA done. Figures → {FIG_DIR}.")
    return out


def _chart_missingness(df: pd.DataFrame, target: str) -> dict:
    """Grouped bar chart: churn rate when feature is missing vs present."""
    from src.eda_supplement import compute as _supp
    supp = _supp(df)
    signal = supp["missing_flag_churn_signal"]

    # Top 3 by delta
    top3 = signal[:3]
    labels   = [r["feature"].replace("_", "\n") for r in top3]
    missing  = [r["churn_when_missing"] * 100 for r in top3]
    present  = [r["churn_when_present"] * 100 for r in top3]
    base     = df[target].mean() * 100

    NAVY, BLUE, RED, MGRAY = "#1a1a2e", "#0f3460", "#e94560", "#9999aa"
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#ccccdd")
    ax.spines["bottom"].set_edgecolor("#ccccdd")

    bars_m = ax.bar(x - w/2, missing, w, color=RED,  label="When missing",  zorder=3, linewidth=0)
    bars_p = ax.bar(x + w/2, present, w, color=BLUE, label="When present",  zorder=3, linewidth=0)
    ax.axhline(base, color=MGRAY, linestyle="--", lw=1.6, zorder=2)
    ax.text(len(labels) - 0.48, base + 0.6, f"Base rate  {base:.1f}%",
            color=MGRAY, fontsize=11, va="bottom")

    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=RED)
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=13, color=BLUE)
    for i, r in enumerate(top3):
        delta = r["churn_when_missing"] * 100 - r["churn_when_present"] * 100
        ax.annotate(f"+{delta:.1f} pp",
                    xy=(x[i] - w/2, missing[i] + 2.5), ha="center",
                    fontsize=11, color=RED, fontweight="bold")

    import matplotlib.ticker as mticker
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, color=NAVY)
    ax.set_ylabel("Churn rate (%)", fontsize=13, color=NAVY)
    ax.set_ylim(0, 52)
    ax.set_title("Missing data is a churn signal, not noise", pad=14,
                 fontsize=16, fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.grid(axis="y", zorder=0, color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_missingness_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return supp


def _chart_mi_vs_correlation(supp: dict) -> None:
    """Scatter: |r| vs mutual information — LTV as the outlier."""
    NAVY, BLUE, RED, GREEN, MGRAY = "#1a1a2e", "#0f3460", "#e94560", "#43a047", "#9999aa"

    # Values are pre-computed from eda_supplement.compute() and hardcoded here
    # for chart stability, avoids re-running the KNN-based MI estimator
    # (which has small variance) every time the figure is regenerated.
    data = [
        ("Cart_Abandonment_Rate",         0.278, 0.0588),
        ("Customer_Service_Calls",        0.291, 0.0532),
        ("Lifetime_Value",                0.011, 0.0422),
        ("Total_Purchases",               0.160, 0.0347),
        ("Session_Duration_Avg",          0.228, 0.0341),
        ("Wishlist_Items",                0.198, 0.0318),
        ("Email_Open_Rate",               0.222, 0.0317),
        ("Pages_Per_Session",             0.232, 0.0294),
        ("Mobile_App_Usage",              0.223, 0.0287),
        ("Login_Frequency",               0.204, 0.0277),
        ("Product_Reviews_Written",       0.181, 0.0272),
        ("Social_Media_Engagement_Score", 0.192, 0.0199),
        ("Average_Order_Value",           0.042, 0.0115),
        ("Credit_Balance",                0.157, 0.0114),
        ("Age",                           0.103, 0.0112),
        ("Days_Since_Last_Purchase",      0.153, 0.0090),
        ("Discount_Usage_Rate",           0.077, 0.0059),
        ("Returns_Rate",                  0.054, 0.0027),
        ("Payment_Method_Diversity",      0.005, 0.0017),
        ("Membership_Years",              0.001, 0.0000),
    ]
    names  = [d[0] for d in data]
    r_vals = [d[1] for d in data]
    mi_vals= [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#ccccdd")
    ax.spines["bottom"].set_edgecolor("#ccccdd")

    ax.axvspan(0, 0.12, alpha=0.04, color=RED)
    ax.scatter(r_vals, mi_vals, color=BLUE, s=60, zorder=3, alpha=0.7)

    highlights = {
        "Lifetime_Value":        (RED,   "bold", "Lifetime_Value\nr = 0.01 · MI = 0.042 (3rd)",   (-0.002, +0.003)),
        "Customer_Service_Calls":(GREEN, "bold", "Customer_Service_Calls\nr = 0.29 · MI = 0.053", (+0.005, +0.0015)),
        "Membership_Years":      (MGRAY, "normal","Membership_Years\nr ≈ 0 · MI = 0",             (+0.005, -0.003)),
    }
    for name, (color, weight, label, (ox, oy)) in highlights.items():
        i = names.index(name)
        ax.scatter(r_vals[i], mi_vals[i], color=color, s=110, zorder=5, linewidths=0)
        ax.annotate(label,
                    xy=(r_vals[i], mi_vals[i]),
                    xytext=(r_vals[i] + ox + 0.025, mi_vals[i] + oy + 0.006),
                    fontsize=10, color=color, fontweight=weight,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
                    va="bottom")

    ax.text(0.005, 0.056,
            "High MI\nLow r\n(non-linear signal\nlinear models miss)",
            fontsize=9.5, color=RED, alpha=0.8, va="top", linespacing=1.5)

    ax.set_xlabel("|Point-biserial correlation| with churn", fontsize=13, color=NAVY)
    ax.set_ylabel("Mutual information with churn", fontsize=13, color=NAVY)
    ax.set_title("Mutual information vs linear correlation: LTV is the outlier",
                 pad=14, fontsize=16, fontweight="bold", color=NAVY)
    ax.set_xlim(-0.01, 0.33)
    ax.set_ylim(-0.003, 0.066)
    ax.grid(color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_mi_vs_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
