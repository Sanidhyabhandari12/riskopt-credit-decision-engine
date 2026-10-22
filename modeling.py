"""
Step 4 + 5: train the credit-risk models and validate them.

Models (simple -> complex):
  1. Logistic Regression  - the classic scorecard-style baseline, easy to explain
  2. Random Forest        - non-linear, robust, little tuning
  3. XGBoost              - gradient boosting, usually the strongest on tabular data

Data split (stratified, so each part keeps the ~8% bad rate):
  train 60%  -> fit the models
  calib 20%  -> pick the champion, calibrate PDs, tune the approval threshold
  test  20%  -> touched ONCE at the end for the final, unbiased numbers

I deliberately do NOT use class weights / oversampling: they distort the
predicted probabilities, and the financial model needs PDs that mean what
they say (a PD of 5% should default about 5% of the time).
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CALIB_SIZE, CATEGORICAL_COLS, ID_COL, MODEL_DIR, OUTPUT_DIR,
                    RANDOM_STATE, REDUNDANT_COLS, TARGET_COL, TEST_SIZE)


EXPLAINABLE_MODELS = ["logistic_regression", "xgboost"]


# ---------------------------------------------------------------------------
# Data split + preprocessing
# ---------------------------------------------------------------------------
def split_data(df):
    train_calib, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE)
    train, calib = train_test_split(
        train_calib, test_size=CALIB_SIZE / (1 - TEST_SIZE),
        stratify=train_calib[TARGET_COL], random_state=RANDOM_STATE)
    return train, calib, test


def get_feature_columns(df):
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + [ID_COL, TARGET_COL] + REDUNDANT_COLS]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, scale):
    """Median-impute numbers (+ a missing flag), one-hot encode categories.
    Fitted on the training set only, so no information leaks from calib/test."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:   # only logistic regression needs scaling; trees don't care
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=0.005, sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# A small wrapper so every model is used the same way downstream
# ---------------------------------------------------------------------------
class CreditModel:
    def __init__(self, name, preprocessor, estimator):
        self.name = name
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.calibrator = None
        self.feature_names = None

    def fit(self, X, y, X_es=None, y_es=None):
        Xt = self.preprocessor.fit_transform(X)
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        if self.name == "xgboost":
            # early stopping on a slice of the TRAINING data, not on calib/test
            self.estimator.fit(Xt, y, eval_set=[(self.preprocessor.transform(X_es), y_es)],
                               verbose=False)
        else:
            self.estimator.fit(Xt, y)
        return self

    def predict_raw(self, X):
        return self.estimator.predict_proba(self.preprocessor.transform(X))[:, 1]

    def calibrate(self, X, y):
        """Platt scaling: a 1-variable logistic regression on the model's log-odds.
        It only re-scales the probabilities, so the ranking (AUC) does not change."""
        self.calibrator = LogisticRegression(C=1e6)   # effectively no regularisation
        self.calibrator.fit(_logit(self.predict_raw(X)).reshape(-1, 1), y)
        return self

    def predict_pd(self, X):
        p = self.predict_raw(X)
        if self.calibrator is None:
            return p
        return self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def contributions(self, X):
        """Per-applicant contribution of each feature to the log-odds of default.
        XGBoost: exact TreeSHAP values (built into xgboost).
        Logistic regression: coefficient x standardised value.
        Positive = pushes the applicant towards default."""
        Xt = self.preprocessor.transform(X)
        if self.name == "xgboost":
            contrib = self.estimator.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
            contrib = contrib[:, :-1]     # last column is the bias term
        elif self.name == "logistic_regression":
            contrib = Xt * self.estimator.coef_[0]
        else:
            raise ValueError(f"no contributions for {self.name}")
        return pd.DataFrame(contrib, columns=self.feature_names, index=X.index)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_models(num_cols, cat_cols):
    return {
        "logistic_regression": CreditModel(
            "logistic_regression",
            build_preprocessor(num_cols, cat_cols, scale=True),
            LogisticRegression(C=0.1, max_iter=2000)),
        "random_forest": CreditModel(
            "random_forest",
            build_preprocessor(num_cols, cat_cols, scale=False),
            RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        "xgboost": CreditModel(
            "xgboost",
            build_preprocessor(num_cols, cat_cols, scale=False),
            xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=5,
                              min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                              # stop on log-loss, not AUC: it is smoother on a small slice and
                              # measures probability quality, which the profit model relies on
                              reg_lambda=2.0, eval_metric="logloss", early_stopping_rounds=150,
                              n_jobs=-1, random_state=RANDOM_STATE)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ks_statistic(y, p):
    """Max gap between the cumulative distributions of goods and bads - a standard
    banking metric for how well a score separates the two groups."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p):
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 5),
        "log_loss": round(log_loss(y, p), 4),
        "avg_predicted_pd": round(float(np.mean(p)), 4),
        "actual_bad_rate": round(float(np.mean(y)), 4),
    }


def decile_table(y, p):
    """Rank applicants by PD into 10 equal groups. A good model concentrates the
    bads in the top deciles - this is what a credit committee usually asks to see."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": p})
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(
        applications=("y", "size"), bads=("y", "sum"),
        avg_pd=("pd", "mean"), min_pd=("pd", "min"), max_pd=("pd", "max"))
    t["bad_rate"] = t["bads"] / t["applications"]
    t = t.sort_index(ascending=False)          # riskiest decile first
    t["cum_bads_captured"] = t["bads"].cumsum() / t["bads"].sum()
    t["lift"] = t["bad_rate"] / df["y"].mean()
    return t.round(4).reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc_pr(y, preds, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(y), color="k", ls="--", alpha=0.4, label="random")
    axes[1].set(title="Precision-Recall curve (test set)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)


def plot_calibration(y, p_raw, p_cal, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in [("before calibration", p_raw), ("after calibration", p_cal)]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label)
    lim = max(0.3, float(np.max(p_cal)) * 0.6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="perfect")
    ax.set(title="Calibration (test set)", xlabel="Predicted PD", ylabel="Observed bad rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=120)
    plt.close(fig)


def plot_importance(importance, out_dir, top_n=20):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["mean_abs_contribution"], color="#2a78d6")
    ax.set(title="Top drivers of default risk (mean |contribution|)", xlabel="Mean |log-odds contribution|")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------
def base_feature(name):
    """Map an encoded column back to the original feature
    (e.g. 'NAME_INCOME_TYPE_Working' -> 'NAME_INCOME_TYPE')."""
    if name.startswith("missingindicator_"):
        return name.replace("missingindicator_", "")
    for col in CATEGORICAL_COLS:
        if name.startswith(col + "_"):
            return col
    return name


def global_importance(contrib):
    grouped = contrib.abs().T.groupby(base_feature).sum().T   # sum one-hot parts per feature
    imp = grouped.mean().sort_values(ascending=False)
    return imp.rename("mean_abs_contribution").rename_axis("feature").reset_index()


def reason_codes(contrib, top_n=3):
    """Top features pushing each applicant TOWARDS default - the basis for
    telling a declined customer why (adverse-action reasons)."""
    grouped = contrib.T.groupby(base_feature).sum().T
    values = grouped.to_numpy()
    names = np.array(grouped.columns)
    top_idx = np.argsort(-values, axis=1)[:, :top_n]     # biggest positive contributions first
    codes = ["; ".join(names[idx][values[i, idx] > 0]) for i, idx in enumerate(top_idx)]
    return pd.Series(codes, index=contrib.index)


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------
def run(df):
    out_dir = OUTPUT_DIR / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[5] Train / calibration / test split")
    train, calib, test = split_data(df)
    for name, part in [("train", train), ("calib", calib), ("test", test)]:
        print(f"  {name:<6} {len(part):>8,} rows  bad rate {part[TARGET_COL].mean():.2%}")

    num_cols, cat_cols = get_feature_columns(df)
    print(f"  features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    # small early-stopping slice for xgboost, taken from train only
    fit_part, es_part = train_test_split(train, test_size=0.1, stratify=train[TARGET_COL],
                                         random_state=RANDOM_STATE)
    X_fit, y_fit = fit_part[num_cols + cat_cols], fit_part[TARGET_COL]
    X_es, y_es = es_part[num_cols + cat_cols], es_part[TARGET_COL]
    X_cal, y_cal = calib[num_cols + cat_cols], calib[TARGET_COL]
    X_test, y_test = test[num_cols + cat_cols], test[TARGET_COL]

    print("\n[6] Training models")
    models = make_models(num_cols, cat_cols)
    rows, test_preds = [], {}
    for name, model in models.items():
        model.fit(X_fit, y_fit, X_es, y_es)
        p_cal, p_test = model.predict_raw(X_cal), model.predict_raw(X_test)
        test_preds[name] = p_test
        m_cal, m_test = evaluate(y_cal, p_cal), evaluate(y_test, p_test)
        rows.append({"model": name, "calib_auc": m_cal["roc_auc"],
                     **{f"test_{k}": v for k, v in m_test.items()}})
        extra = ""
        if name == "xgboost":
            extra = f"  (best iteration {model.estimator.best_iteration})"
        print(f"  {name:<20} calib AUC {m_cal['roc_auc']:.4f}  test AUC {m_test['roc_auc']:.4f}"
              f"  KS {m_test['ks']:.3f}{extra}")

    comparison = pd.DataFrame(rows).sort_values("calib_auc", ascending=False)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    # Champion = best calibration-set AUC (never test) AMONG models that can give
    # per-applicant reason codes. A declined customer has to be told why, so the
    # random forest stays as a benchmark / challenger only.
    explainable = comparison[comparison["model"].isin(EXPLAINABLE_MODELS)]
    champion_name = explainable.iloc[0]["model"]
    champion = models[champion_name]
    print(f"\n  champion (best calib AUC among explainable models): {champion_name}")

    print("\n[7] Calibrating PDs (Platt scaling on calib set)")
    p_test_raw = champion.predict_raw(X_test)
    champion.calibrate(X_cal, y_cal)
    p_test = champion.predict_pd(X_test)
    before, after = evaluate(y_test, p_test_raw), evaluate(y_test, p_test)
    print(f"  test Brier  {before['brier']:.5f} -> {after['brier']:.5f}")
    print(f"  avg PD {before['avg_predicted_pd']:.2%} -> {after['avg_predicted_pd']:.2%} "
          f"(actual bad rate {after['actual_bad_rate']:.2%})")

    plot_roc_pr(y_test, test_preds, out_dir)
    plot_calibration(y_test, p_test_raw, p_test, out_dir)
    decile_table(y_test, p_test).to_csv(out_dir / "decile_table.csv", index=False)

    print("\n[8] Explainability")
    contrib_test = champion.contributions(X_test)
    importance = global_importance(contrib_test)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    plot_importance(importance, out_dir)
    reasons = reason_codes(contrib_test)
    print("  top 5 risk drivers: " + ", ".join(importance["feature"].head(5)))
    print(f"  example reason codes: {reasons.iloc[0]}")

    metrics = {"champion": champion_name, "test_raw": before, "test_calibrated": after,
               "n_train": len(fit_part), "n_calib": len(calib), "n_test": len(test)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(champion, MODEL_DIR / "champion_model.joblib")

    # scored datasets used by the decision engine and the dashboard
    keep = [ID_COL, TARGET_COL, "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN", "AGE_YEARS",
            "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "BUR_HAS_OVERDUE", "PREV_REFUSED_COUNT"]
    scored_calib = calib[keep].assign(PD=champion.predict_pd(X_cal))
    scored_test = test[keep].assign(PD=p_test, RISK_REASONS=reasons.values)
    return champion, scored_calib, scored_test, metrics
