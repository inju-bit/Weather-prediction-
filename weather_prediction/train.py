"""
Rain Tomorrow Prediction with CatBoost (advanced pipeline)
==========================================================
Dataset : "Rain in Australia" -- ~10 years of daily observations from the
          Australian Bureau of Meteorology (145k rows, 49 stations).
Task    : Binary classification, will it rain tomorrow? (RainTomorrow yes/no)

Pipeline:
  1. Cleaning: drop rows with missing target, engineer Month from Date,
     keep weather-station Location and wind directions as native categoricals.
  2. Model comparison: majority-class baseline, Logistic Regression, CatBoost.
  3. CatBoost baseline (defaults) -> GridSearchCV hyperparameter tuning
     (tuned on a stratified subsample, refit on the full training set).
  4. Evaluation: accuracy, ROC-AUC, PR-AUC, precision, recall, F1 (the rain
     class is the minority at ~22%, so accuracy alone is not enough).
  5. Explainability with SHAP.

CatBoost is a natural fit here because it handles the categorical columns
(Location, wind directions, RainToday) and missing numeric values directly,
with no one-hot encoding or manual imputation.

Source: Australian Bureau of Meteorology, Daily Weather Observations.
        Kaggle "Rain in Australia" (weather-dataset-rattle-package).

Usage:
    python src/train.py
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             f1_score, classification_report, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
WINE_RED = "#1f5d7a"   # storm blue for the weather theme
ACCENT = "#7aa5b5"
CAT = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]


def load_data():
    df = pd.read_csv("data/weatherAUS.csv").dropna(subset=["RainTomorrow"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month.astype("Int64").astype(str)
    df = df.drop(columns=["Date"])
    y = (df["RainTomorrow"] == "Yes").astype(int)
    X = df.drop(columns=["RainTomorrow"])
    for c in CAT:
        X[c] = X[c].astype(str).fillna("missing")
    return X, y


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "f1_rain": round(f1_score(y_true, y_pred), 4),
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    X, y = load_data()
    print(f"=== Rain in Australia | rows={len(X)} | features={X.shape[1]} ===")
    print(f"Rain-tomorrow rate (positive class): {y.mean():.3f}")
    cat_idx = [X.columns.get_loc(c) for c in CAT]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    rows = []

    # ---- Majority-class baseline ----
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    rows.append(evaluate("Majority baseline", yte, dummy.predict(Xte),
                         np.full(len(yte), ytr.mean())))

    # ---- Logistic Regression (with preprocessing) ----
    num = [c for c in X.columns if c not in CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=300, solver='liblinear', random_state=RANDOM_STATE))])
    lr.fit(Xtr, ytr)
    rows.append(evaluate("Logistic Regression", yte, lr.predict(Xte),
                         lr.predict_proba(Xte)[:, 1]))

    # ---- CatBoost baseline (defaults, capped iterations) ----
    base = CatBoostClassifier(iterations=300, random_state=RANDOM_STATE, verbose=0)
    base.fit(Xtr, ytr, cat_features=cat_idx)
    base_metrics = evaluate("CatBoost (default)", yte, base.predict(Xte),
                            base.predict_proba(Xte)[:, 1])
    rows.append(base_metrics)
    print(f"\n[Baseline] CatBoost default accuracy = {base_metrics['accuracy']}")

    # ---- GridSearchCV (tune on a stratified subsample, refit on full train) ----
    sub = Xtr.sample(15000, random_state=RANDOM_STATE)
    ysub = ytr.loc[sub.index]
    grid = GridSearchCV(
        CatBoostClassifier(iterations=150, random_state=RANDOM_STATE, verbose=0),
        {"depth": [6, 8], "learning_rate": [0.05, 0.1]},
        scoring="accuracy", cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1)
    grid.fit(sub, ysub, cat_features=cat_idx)
    print(f"[Tuned] best params: {grid.best_params_}")

    tuned = CatBoostClassifier(iterations=400, random_state=RANDOM_STATE,
                               verbose=0, **grid.best_params_)
    tuned.fit(Xtr, ytr, cat_features=cat_idx)
    tuned_pred = tuned.predict(Xte)
    tuned_proba = tuned.predict_proba(Xte)[:, 1]
    tuned_metrics = evaluate("CatBoost (tuned)", yte, tuned_pred, tuned_proba)
    rows.append(tuned_metrics)
    print(f"[Tuned] accuracy = {tuned_metrics['accuracy']} | "
          f"ROC-AUC = {tuned_metrics['roc_auc']} | PR-AUC = {tuned_metrics['pr_auc']}")
    print("\nClassification report (tuned CatBoost):")
    print(classification_report(yte, tuned_pred, target_names=["no rain", "rain"]))

    # ---- Figures ----
    # Confusion matrix
    cm = confusion_matrix(yte, tuned_pred)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 13, "weight": "bold"},
                xticklabels=["no rain", "rain"], yticklabels=["no rain", "rain"],
                linewidths=1, linecolor="white")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix | tuned CatBoost", fontsize=12, weight="bold")
    plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=140); plt.close()

    # ROC + PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    RocCurveDisplay.from_predictions(yte, tuned_proba, ax=ax1, color=WINE_RED,
                                     linewidth=2.5, name="Tuned CatBoost")
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    ax1.set_title(f"ROC | AUC = {tuned_metrics['roc_auc']}", weight="bold"); ax1.legend(loc="lower right")
    PrecisionRecallDisplay.from_predictions(yte, tuned_proba, ax=ax2, color=WINE_RED,
                                            linewidth=2.5, name="Tuned CatBoost")
    ax2.set_title(f"Precision-Recall | AP = {tuned_metrics['pr_auc']}", weight="bold")
    plt.tight_layout(); plt.savefig("outputs/roc_pr_curves.png", dpi=140); plt.close()

    # Model comparison
    comp = pd.DataFrame(rows).set_index("model")
    plt.figure(figsize=(8, 4.2))
    order = comp["accuracy"].sort_values().index
    bars = plt.barh(order, comp.loc[order, "accuracy"], color=WINE_RED, edgecolor="white")
    for b, m in zip(bars, comp.loc[order, "accuracy"]):
        plt.text(m - 0.01, b.get_y() + b.get_height()/2, f"{m:.3f}",
                 va="center", ha="right", color="white", weight="bold", fontsize=10)
    plt.xlim(0.74, 0.88)
    plt.title("Model comparison | test accuracy", fontsize=12, weight="bold")
    plt.xlabel("Accuracy"); sns.despine(left=True)
    plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=140); plt.close()

    # Feature importance
    imp = pd.Series(tuned.get_feature_importance(), index=X.columns).sort_values().tail(15)
    plt.figure(figsize=(7.5, 5.5))
    bars = plt.barh(imp.index, imp.values, color=WINE_RED, edgecolor="white")
    for b, v in zip(bars, imp.values):
        plt.text(v + max(imp.values)*0.01, b.get_y()+b.get_height()/2,
                 f"{v:.1f}", va="center", fontsize=9, color="#333")
    plt.title("Feature Importance | tuned CatBoost", fontsize=13, weight="bold")
    plt.xlabel("Importance"); plt.xlim(0, max(imp.values)*1.15); sns.despine(left=True)
    plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=140); plt.close()

    # SHAP (native CatBoost SHAP on a test sample)
    samp = Xte.sample(min(3000, len(Xte)), random_state=RANDOM_STATE)
    pool = Pool(samp, cat_features=cat_idx)
    sv = tuned.get_feature_importance(pool, type="ShapValues")
    sv = sv[:, :-1]  # drop base value column
    plt.figure()
    shap.summary_plot(sv, samp, show=False, plot_size=(8, 5.5))
    plt.title("SHAP summary | tuned CatBoost", fontsize=12, weight="bold")
    plt.tight_layout(); plt.savefig("outputs/shap_summary.png", dpi=140, bbox_inches="tight"); plt.close()

    # ---- Save results ----
    results = {
        "dataset": "Rain in Australia (BOM / Kaggle)",
        "n_samples": int(len(X)),
        "positive_rate_rain": round(float(y.mean()), 4),
        "best_params": grid.best_params_,
        "models": rows,
    }
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nModel comparison:")
    print(comp.to_string())
    print("\nSaved figures + results.json to outputs/")


if __name__ == "__main__":
    main()
