"""
SIH26006 - Bulk Freight Rate Forecasting
Model Training Pipeline: Linear Regression + XGBoost (or GradientBoosting fallback)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

# Try to import XGBoost, fall back to sklearn GradientBoosting
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("[OK] XGBoost found - using XGBRegressor")
except ImportError:
    USE_XGBOOST = False
    print("[WARN] XGBoost not found - using sklearn GradientBoostingRegressor as fallback")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_dataset, save_dataset
from src.feature_engineering import (
    engineer_features, build_preprocessor, save_preprocessor,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
)
from src.utils import compute_metrics, print_metrics_table

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "freight_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "app", "static", "plots")

BOOSTING_LABEL = "XGBoost" if USE_XGBOOST else "GradientBoosting"


def ensure_dirs():
    for d in [MODEL_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


# ??????????????????????????? Data Preparation ?????????????????????????????
def prepare_data():
    if not os.path.exists(DATA_PATH):
        print("[PLOT] Generating synthetic dataset...")
        df = generate_dataset(n_samples=5000)
        save_dataset(df, DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        print(f"[LOAD] Loaded existing dataset: {len(df)} rows")

    df = engineer_features(df)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Time-aware split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[OK] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df


# ????????????????????????? Linear Regression ??????????????????????????????
def train_linear_model(X_train, y_train, preprocessor):
    print("\n[INFO] Training Linear Regression (Ridge + Polynomial features)...")

    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("poly",         PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ("ridge",        Ridge(alpha=10.0)),
    ])

    cv_scores = cross_val_score(lr_pipeline, X_train, y_train,
                                cv=5, scoring="r2", n_jobs=-1)
    print(f"   CV R? scores: {cv_scores.round(4)} | Mean: {cv_scores.mean():.4f}")

    lr_pipeline.fit(X_train, y_train)
    path = os.path.join(MODEL_DIR, "linear_regression.pkl")
    joblib.dump(lr_pipeline, path)
    print(f"   [OK] Saved ? {path}")
    return lr_pipeline


# ??????????????????????????? Boosting Model ???????????????????????????????
def train_boosting_model(X_train, y_train, preprocessor):
    print(f"\n[INFO] Training {BOOSTING_LABEL} Regressor...")

    if USE_XGBOOST:
        base_model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        param_grid = {
            "booster__n_estimators": [300, 400],
            "booster__max_depth":    [5, 6],
            "booster__learning_rate":[0.05, 0.08],
        }
    else:
        base_model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        param_grid = {
            "booster__n_estimators": [300, 400],
            "booster__max_depth":    [4, 5],
            "booster__learning_rate":[0.05, 0.08],
        }

    boost_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("booster",      base_model),
    ])

    print("   Running GridSearchCV (3-fold)...")
    grid_search = GridSearchCV(
        boost_pipeline, param_grid, cv=3,
        scoring="r2", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV R?:  {grid_search.best_score_:.4f}")

    path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    joblib.dump(best_model, path)
    print(f"   [OK] Saved ? {path}")

    # Save model info
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    with open(info_path, "w") as f:
        json.dump({"boosting_backend": BOOSTING_LABEL}, f)

    return best_model


# ??????????????????????????? Evaluation & Plots ???????????????????????????
def evaluate_models(models: dict, X_test, y_test):
    all_metrics = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        m = compute_metrics(np.array(y_test), y_pred, name)
        all_metrics.append(m)
        predictions[name] = y_pred

    print_metrics_table(all_metrics)
    return all_metrics, predictions


def plot_predictions(y_test, predictions: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    colors = {"Linear Regression": "#4fc3f7", BOOSTING_LABEL: "#a5d6a7"}

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.set_facecolor("#161b22")
        ax.scatter(y_test, y_pred, alpha=0.4, s=15,
                   color=colors.get(name, "#a5d6a7"), edgecolors="none")
        min_val = min(float(y_test.min()), float(y_pred.min()))
        max_val = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_title(name, color="white", fontsize=13, pad=12)
        ax.set_xlabel("Actual Freight Rate ($/t)", color="#8b949e", fontsize=10)
        ax.set_ylabel("Predicted Freight Rate ($/t)", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#161b22", edgecolor="#30363d")

    fig.suptitle("Actual vs Predicted Freight Rates", color="white",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   [PLOT] Saved ? {path}")


def plot_feature_importance(boost_model, X_train):
    """Feature importance from boosting model."""
    boost_step    = boost_model.named_steps["booster"]
    preprocessor  = boost_model.named_steps["preprocessor"]
    preprocessor.fit(X_train)

    num_names = NUMERICAL_FEATURES.copy()
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_names   = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    importances = boost_step.feature_importances_
    n_feat = min(len(importances), len(feature_names))
    feat_imp = pd.Series(importances[:n_feat], index=feature_names[:n_feat])
    feat_imp = feat_imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    colors_bar = plt.cm.plasma(np.linspace(0.3, 0.9, len(feat_imp)))
    bars = ax.barh(feat_imp.index, feat_imp.values, color=colors_bar, edgecolor="none")
    ax.set_xlabel("Feature Importance (Gain)", color="#8b949e", fontsize=11)
    ax.set_title(f"Top 20 Feature Importances - {BOOSTING_LABEL}",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    for bar, val in zip(bars, feat_imp.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="#8b949e", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   [PLOT] Saved ? {path}")

    top_features = feat_imp.tail(15).to_dict()
    json_path = os.path.join(MODEL_DIR, "feature_importance.json")
    with open(json_path, "w") as f:
        json.dump({"labels": list(top_features.keys()),
                   "values": [round(v, 5) for v in top_features.values()]}, f)
    print(f"   [SAVE] Saved ? {json_path}")


def plot_metrics_comparison(all_metrics: list):
    df_m = pd.DataFrame(all_metrics).set_index("model")
    metrics_to_plot = ["RMSE", "MAE", "R2", "MAPE_%"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    colors_map = {"Linear Regression": "#4fc3f7", BOOSTING_LABEL: "#a5d6a7"}

    for ax, metric in zip(axes, metrics_to_plot):
        ax.set_facecolor("#161b22")
        values = df_m[metric]
        bars = ax.bar(values.index, values.values,
                      color=[colors_map.get(m, "#7e57c2") for m in values.index],
                      edgecolor="none", width=0.5)
        ax.set_title(metric, color="white", fontsize=12, pad=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(values),
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=10)

    fig.suptitle("Model Performance Comparison", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   [PLOT] Saved ? {path}")


def save_metrics_json(all_metrics: list):
    # Rename boosting model to "XGBoost" for dashboard consistency
    for m in all_metrics:
        if m["model"] == BOOSTING_LABEL and BOOSTING_LABEL != "XGBoost":
            m["model"] = "XGBoost"
    path = os.path.join(MODEL_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"   [SAVE] Saved ? {path}")


def save_historical_json(df: pd.DataFrame):
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (df.groupby("year_month")["freight_rate_usd_t"]
                 .agg(["mean", "min", "max"])
                 .reset_index())
    monthly.columns = ["month", "avg_rate", "min_rate", "max_rate"]
    monthly = monthly.round(2)
    path = os.path.join(MODEL_DIR, "historical_rates.json")
    monthly.to_json(path, orient="records")
    print(f"   [SAVE] Saved ? {path}")


# ??????????????????????????? Main Training Entry ??????????????????????????
def main():
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  SIH26006 - Freight Rate Forecasting Training Pipeline")
    print(f"  Boosting backend: {BOOSTING_LABEL}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, df = prepare_data()
    preprocessor = build_preprocessor()

    lr_model    = train_linear_model(X_train, y_train, preprocessor)
    boost_model = train_boosting_model(X_train, y_train, preprocessor)

    save_preprocessor(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    print("\n[EVAL] Evaluating on test set...")
    models = {"Linear Regression": lr_model, BOOSTING_LABEL: boost_model}
    all_metrics, predictions = evaluate_models(models, X_test, y_test)

    print("\n[PLOT] Generating plots...")
    plot_predictions(y_test, predictions)
    plot_feature_importance(boost_model, X_train)
    plot_metrics_comparison(all_metrics)
    save_metrics_json(all_metrics)
    save_historical_json(df)

    print("\n[OK] Training complete! All models and plots saved.")
    print(f"   Run:  python run.py   to start the web dashboard.")


if __name__ == "__main__":
    main()
