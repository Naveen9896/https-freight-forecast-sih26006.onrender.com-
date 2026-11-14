"""
SIH26006 ? Bulk Freight Rate Forecasting
Feature Engineering Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


# ????????????????????????? Column Groups ??????????????????????????????????
CATEGORICAL_FEATURES = ["route", "vessel_type", "cargo_type"]
NUMERICAL_FEATURES = [
    "cargo_volume_mt", "bunker_fuel_usd_mt", "distance_nm",
    "vessel_capacity_mt", "utilisation_pct", "port_congestion_idx",
    "seasonal_demand_idx", "month", "year",
    # Engineered features (added below)
    "fuel_x_distance", "vol_per_capacity", "sin_month", "cos_month",
    "fuel_per_nm", "effective_speed_factor",
]
TARGET = "freight_rate_usd_t"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and temporal features."""
    df = df.copy()

    # Interaction: fuel cost ? distance (key cost driver)
    df["fuel_x_distance"] = df["bunker_fuel_usd_mt"] * df["distance_nm"]

    # Volume efficiency relative to vessel capacity
    df["vol_per_capacity"] = df["cargo_volume_mt"] / df["vessel_capacity_mt"]

    # Fuel cost per nautical mile (cost density)
    df["fuel_per_nm"] = df["bunker_fuel_usd_mt"] / (df["distance_nm"] + 1)

    # Cyclical encoding of month (preserves Dec?Jan continuity)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Effective speed factor proxy
    df["effective_speed_factor"] = df["distance_nm"] / (df["vessel_capacity_mt"] / 1000 + 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer preprocessor."""
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


def load_and_prepare(csv_path: str):
    """Load raw CSV, engineer features, return X and y."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = engineer_features(df)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    return X, y, df


def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"? Preprocessor saved ? {path}")


def load_preprocessor(path: str = "models/preprocessor.pkl"):
    return joblib.load(path)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after transformation."""
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            cat_names = list(transformer.named_steps["encoder"].get_feature_names_out(columns))
    return num_names + cat_names
