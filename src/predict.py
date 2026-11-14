"""
SIH26006 ? Bulk Freight Rate Forecasting
Inference Module: load trained models and predict
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# XGBoost is optional ? joblib can load the pkl regardless of which backend was used
try:
    import xgboost  # noqa: F401
except ImportError:
    pass  # GradientBoostingRegressor fallback was used at training time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import vessel_procurement_recommendation

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_lr_model  = None
_xgb_model = None


def _load_models():
    global _lr_model, _xgb_model
    if _lr_model is None:
        _lr_model  = joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl"))
    if _xgb_model is None:
        _xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))


def build_input_df(
    route: str,
    vessel_type: str,
    cargo_type: str,
    cargo_volume_mt: float,
    bunker_fuel_usd_mt: float,
    month: int,
    year: int = 2025,
    port_congestion_idx: float = 0.3,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training schema."""
    from src.data_generator import ROUTES, VESSEL_TYPES

    route_info  = ROUTES.get(route, {"distance_nm": 1000, "base_rate": 15.0})
    vessel_info = VESSEL_TYPES.get(vessel_type, {"capacity_mt": 55000})

    distance_nm        = route_info["distance_nm"]
    vessel_capacity_mt = vessel_info["capacity_mt"]
    utilisation_pct    = round(cargo_volume_mt / vessel_capacity_mt * 100, 2)
    seasonal_demand_idx = 1.0 + 0.4 * np.sin((month - 3) * np.pi / 6)

    row = {
        "date":               pd.Timestamp(f"{year}-{month:02d}-01"),
        "year":               year,
        "month":              month,
        "day":                1,
        "route":              route,
        "origin":             route.split("->")[0],
        "destination":        route.split("->")[1],
        "vessel_type":        vessel_type,
        "cargo_type":         cargo_type,
        "cargo_volume_mt":    cargo_volume_mt,
        "bunker_fuel_usd_mt": bunker_fuel_usd_mt,
        "distance_nm":        distance_nm,
        "vessel_capacity_mt": vessel_capacity_mt,
        "utilisation_pct":    utilisation_pct,
        "port_congestion_idx":port_congestion_idx,
        "seasonal_demand_idx":round(float(seasonal_demand_idx), 3),
    }
    df = pd.DataFrame([row])
    return df


def predict(
    route: str,
    vessel_type: str,
    cargo_type: str,
    cargo_volume_mt: float,
    bunker_fuel_usd_mt: float,
    month: int,
    year: int = 2025,
    port_congestion_idx: float = 0.3,
    model_name: str = "xgboost",
) -> dict:
    """
    Predict freight rate and return procurement recommendation.
    model_name: 'xgboost' or 'linear'
    """
    _load_models()

    df = build_input_df(
        route, vessel_type, cargo_type,
        cargo_volume_mt, bunker_fuel_usd_mt,
        month, year, port_congestion_idx
    )
    df_eng = engineer_features(df)
    X = df_eng[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    if model_name == "linear":
        rate = float(_lr_model.predict(X)[0])
        alt_rate = float(_xgb_model.predict(X)[0])
    else:
        rate = float(_xgb_model.predict(X)[0])
        alt_rate = float(_lr_model.predict(X)[0])

    rate     = max(round(rate, 2), 1.0)
    alt_rate = max(round(alt_rate, 2), 1.0)

    procurement = vessel_procurement_recommendation(rate, cargo_volume_mt)

    return {
        "predicted_rate_usd_t":  rate,
        "alt_model_rate_usd_t":  alt_rate,
        "primary_model":         model_name,
        "route":                 route,
        "vessel_type":           vessel_type,
        "cargo_volume_mt":       cargo_volume_mt,
        "bunker_fuel_usd_mt":    bunker_fuel_usd_mt,
        "month":                 month,
        "year":                  year,
        "procurement":           procurement,
    }


if __name__ == "__main__":
    # Sample prediction
    result = predict(
        route="Paradip?Visakhapatnam",
        vessel_type="Supramax",
        cargo_type="Iron Ore",
        cargo_volume_mt=42000,
        bunker_fuel_usd_mt=620,
        month=11,
        year=2025,
        model_name="xgboost",
    )
    print("\n? Sample Prediction:")
    print(f"   Route:         {result['route']}")
    print(f"   XGBoost Rate:  ${result['predicted_rate_usd_t']:.2f}/tonne")
    print(f"   Linear Rate:   ${result['alt_model_rate_usd_t']:.2f}/tonne")
    print(f"   Total Cost:    ${result['procurement']['total_freight_usd']:,.0f}")
    print(f"   Decision:      {result['procurement']['decision']}")
