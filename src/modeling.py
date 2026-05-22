"""Phase 2 — Train and evaluate the churn model (HistGradientBoosting).

Run as: python -m src.modeling

Outputs:
  - figures/modeling/*.png    ROC, PR, calibration, gains, importance, savings

Model choice is sklearn's HistGradientBoostingClassifier. The rationale lives
in the writeups; in short: the EDA showed non-linear signal in `Lifetime_Value`
that a linear model cannot capture, and HGB is a strong default for tabular
data with mixed dtypes, missing values, and non-linear feature effects.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    TARGET,
    build_tree_preprocessor,
    feature_columns,
    prepare_frame,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ecommerce_customer_churn_dataset.csv"
FIG_DIR = ROOT / "figures" / "modeling"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42          # fixed seed for reproducibility across split, CV, and importance
MODEL_NAME = "HistGradientBoosting"
sns.set_theme(context="notebook", style="whitegrid")


def make_model() -> Pipeline:
    """Build the preprocessing + model pipeline.

    Hyperparameters are deliberately conservative defaults. CV variance across
    folds is small (≈ 0.004 PR-AUC SD), so the model is not on a knife-edge
    of capacity where a grid search would change conclusions.
    """
    return Pipeline([
        ("pre", build_tree_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=None,
            l2_regularization=0.0,
            random_state=SEED,
        )),
    ])


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Precision among the top-K% of customers ranked by predicted churn score.

    Answers: of the customers we'd contact in a campaign targeting the top K%,
    what fraction are actual churners?
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_fraction)))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Lift over random targeting at the top-K% cutoff.

    A lift of 3× means the model's top-K list contains 3× as many true churners
    as you'd expect from randomly selecting K% of the customer base.
    """
    base = float(y_true.mean()) if y_true.mean() > 0 else 1.0
    return precision_at_k(y_true, y_score, k_fraction) / base


def cv_scores(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run 5-fold stratified CV on the training set and return mean scores.

    Used as a stability check before committing to the test set evaluation.
    If CV and test set results agree closely, the model is not overfit to the split.
    Stratified folds preserve the 28.9% churn rate in each fold.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["roc_auc", "average_precision", "neg_log_loss"],
        n_jobs=-1, return_train_score=False,
    )
    return {
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
        "cv_pr_auc_mean": float(scores["test_average_precision"].mean()),
        "cv_pr_auc_std": float(scores["test_average_precision"].std()),
        "cv_log_loss_mean": float(-scores["test_neg_log_loss"].mean()),
    }


def holdout_scores(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    """Compute all evaluation metrics on the untouched test set.

    Returns a dict of metrics and the raw predicted probabilities.
    Probabilities are clipped away from 0/1 before log-loss to avoid log(0).
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_test, proba)),
        "f1_at_0_5": float(f1_score(y_test, pred)),
        "precision@10%": precision_at_k(y_test.values, proba, 0.10),
        "precision@20%": precision_at_k(y_test.values, proba, 0.20),
        "precision@30%": precision_at_k(y_test.values, proba, 0.30),
        "lift@10%": lift_at_k(y_test.values, proba, 0.10),
        "lift@20%": lift_at_k(y_test.values, proba, 0.20),
    }, proba


def plot_curves(proba: np.ndarray, y_test: pd.Series) -> dict:
    """Generate and save the four standard evaluation figures.

    ROC curve, Precision-Recall curve, calibration plot, and cumulative gains.
    All saved to figures/modeling/ and paths returned as a dict.
    """
    paths = {}

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{MODEL_NAME} (AUC={roc_auc_score(y_test, proba):.3f})",
            lw=1.8, color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC: test set"); ax.legend(fontsize=9)
    p = FIG_DIR / "roc_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["roc"] = str(p)

    # Precision-Recall
    base_rate = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, label=f"{MODEL_NAME} (AP={average_precision_score(y_test, proba):.3f})",
            lw=1.8, color="#C44E52")
    ax.axhline(base_rate, color="black", linestyle="--", lw=0.8, label=f"baseline={base_rate:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: test set"); ax.legend(fontsize=9)
    p = FIG_DIR / "pr_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["pr"] = str(p)

    # Calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=15, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker="o", label=MODEL_NAME, lw=1.6, markersize=5, color="#55A868")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Empirical fraction of churners")
    ax.set_title(f"Calibration: MSE = {brier_score_loss(y_test, proba):.3f}")
    ax.legend(fontsize=9)
    p = FIG_DIR / "calibration.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["calibration"] = str(p)

    # Cumulative gains
    fig, ax = plt.subplots(figsize=(6, 5))
    order = np.argsort(-proba)
    cum = np.cumsum(y_test.values[order])
    gain = cum / y_test.sum()
    frac = np.arange(1, len(y_test) + 1) / len(y_test)
    ax.plot(frac, gain, label=MODEL_NAME, lw=1.8, color="#8172B2")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set_xlabel("Fraction of customers contacted (sorted by score)")
    ax.set_ylabel("Fraction of true churners captured")
    ax.set_title("Cumulative gains: test set"); ax.legend(fontsize=9)
    p = FIG_DIR / "cumulative_gains.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["cumulative_gains"] = str(p)

    return paths


def confusion_at_thresholds(proba: np.ndarray, y_test: pd.Series, thresholds=(0.30, 0.50, 0.70)) -> list[dict]:
    """Evaluate precision, recall, and F1 at multiple probability thresholds.

    Illustrates the precision-recall trade-off: a lower threshold catches more
    churners (higher recall) at the cost of more false positives (lower precision),
    and vice versa. The right threshold is a business decision based on the
    relative cost of a missed churner vs a wasted retention offer.
    """
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_test, pred).tolist()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        rows.append({
            "threshold": t,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": f1_score(y_test, pred),
            "positives_flagged_pct": (tp + fp) / len(y_test),
        })
    return rows


def permutation_importances(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    """Compute permutation importance scored by average precision (PR-AUC).

    For each feature, shuffles its values n_repeats times and measures how much
    PR-AUC drops — larger drop means the model relies on that feature more.
    Runs on a sample of 8,000 rows for speed; results are stable at that size.
    """
    sample_idx = np.random.RandomState(SEED).choice(len(X_test), size=min(8000, len(X_test)), replace=False)
    Xs = X_test.iloc[sample_idx]
    ys = y_test.iloc[sample_idx]
    r = permutation_importance(
        model, Xs, ys, n_repeats=n_repeats, random_state=SEED,
        scoring="average_precision", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": Xs.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def business_savings_curve(y_true: np.ndarray, proba: np.ndarray,
                           value_per_save: float = 100.0,
                           cost_per_contact: float = 15.0,
                           save_rate: float = 0.30) -> pd.DataFrame:
    """Toy business savings curve.

    Placeholders the retention team would replace with real numbers. The
    *shape* of the curve (clear optimum around 25-30% targeting) is the
    actionable finding, not the dollar levels.
    """
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    rows = []
    for k_frac in np.linspace(0.05, 1.0, 20):
        k = int(np.ceil(len(y_true) * k_frac))
        contacted = k
        tp = cum_tp[k - 1]
        saves = tp * save_rate
        revenue = saves * value_per_save
        spend = contacted * cost_per_contact
        rows.append({
            "k_fraction": round(k_frac, 3),
            "contacted": contacted,
            "true_churners_in_topK": int(tp),
            "expected_saves": round(float(saves), 1),
            "spend": round(spend, 0),
            "revenue_retained": round(float(revenue), 0),
            "net_savings": round(float(revenue - spend), 0),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading and preparing data ...")
    df = pd.read_csv(DATA)
    df = prepare_frame(df)
    y = df[TARGET].astype(int)
    X = df[feature_columns()].copy()

    # Stratify on y to preserve the 28.9% churn rate in both train and test splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}   churn rate (train)={y_train.mean():.4f}")

    print("Building model ...")
    model = make_model()

    print("Running 5-fold CV (stability check) ...")
    cv = cv_scores(model, X_train, y_train)
    print(f"  CV ROC-AUC: {cv['cv_roc_auc_mean']:.4f} ± {cv['cv_roc_auc_std']:.4f}")
    print(f"  CV PR-AUC : {cv['cv_pr_auc_mean']:.4f} ± {cv['cv_pr_auc_std']:.4f}")

    print("Fitting on full train, evaluating on test set ...")
    model.fit(X_train, y_train)
    holdout, proba = holdout_scores(model, X_test, y_test)
    print(f"  Test set ROC-AUC: {holdout['roc_auc']:.4f}")
    print(f"  Test set PR-AUC : {holdout['pr_auc']:.4f}")
    print(f"  Test set MSE  : {holdout['brier']:.4f}")

    print("Plotting curves ...")
    fig_paths = plot_curves(proba, y_test)

    print("Computing operating-point trade-offs ...")
    cm_table = confusion_at_thresholds(proba, y_test, thresholds=(0.30, 0.50, 0.70))

    print("Computing permutation importance ...")
    perm = permutation_importances(model, X_test, y_test, n_repeats=4)

    top = perm.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C72B0")
    ax.set_title(f"Permutation importance: {MODEL_NAME}\n(scoring=average_precision, n_repeats=4)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=120)
    plt.close(fig)

    print("Computing business savings curve ...")
    savings = business_savings_curve(y_test.values, proba)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(savings["k_fraction"], savings["net_savings"], marker="o", color="#55A868")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Fraction of customers contacted (sorted by churn risk)")
    ax.set_ylabel("Expected net savings ($)")
    ax.set_title("Business savings curve (toy economics)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "savings_curve.png", dpi=120)
    plt.close(fig)

    print(f"Done. Figures → {FIG_DIR}.")
    return {
        "model": MODEL_NAME,
        "cv_results": {k: round(v, 4) for k, v in cv.items()},
        "holdout_results": {k: round(v, 4) for k, v in holdout.items()},
        "confusion_at_thresholds": cm_table,
        "permutation_importance_top20": perm.head(20).round(5).to_dict(orient="records"),
        "business_savings_curve": savings.to_dict(orient="records"),
        "data": {
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "churn_rate_train": round(float(y_train.mean()), 4),
            "churn_rate_test": round(float(y_test.mean()), 4),
        },
    }


if __name__ == "__main__":
    main()
