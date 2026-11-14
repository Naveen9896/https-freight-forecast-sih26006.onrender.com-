"""
SIH26006 ? Bulk Freight Rate Forecasting
Utility helpers
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    metrics = {
        "model":  model_name,
        "RMSE":   round(rmse, 4),
        "MAE":    round(mae, 4),
        "R2":     round(r2, 4),
        "MAPE_%": round(mape, 4),
    }
    return metrics


def print_metrics_table(metrics_list: list):
    """Pretty-print metrics comparison table."""
    df = pd.DataFrame(metrics_list).set_index("model")
    print("\n" + "=" * 55)
    print("           MODEL PERFORMANCE COMPARISON")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55)


def vessel_procurement_recommendation(predicted_rate: float, cargo_volume_mt: float) -> dict:
    """
    Given predicted freight rate and cargo volume,
    recommend optimal vessel class and estimated cost.
    """
    # Vessel classes with capacity and typical day rates
    vessels = [
        {"class": "Handysize",  "capacity_mt": 28_000,  "daily_rate_usd": 12_000},
        {"class": "Supramax",   "capacity_mt": 55_000,  "daily_rate_usd": 16_000},
        {"class": "Panamax",    "capacity_mt": 75_000,  "daily_rate_usd": 20_000},
        {"class": "Capesize",   "capacity_mt": 150_000, "daily_rate_usd": 28_000},
    ]

    # Find the smallest vessel that can carry the cargo
    suitable = [v for v in vessels if v["capacity_mt"] >= cargo_volume_mt]
    recommended = suitable[0] if suitable else vessels[-1]

    total_freight_cost = predicted_rate * cargo_volume_mt
    num_vessels = max(1, int(np.ceil(cargo_volume_mt / recommended["capacity_mt"])))
    efficiency_pct = round(cargo_volume_mt / (recommended["capacity_mt"] * num_vessels) * 100, 1)

    return {
        "recommended_vessel":  recommended["class"],
        "vessel_capacity_mt":  recommended["capacity_mt"],
        "vessels_required":    num_vessels,
        "daily_rate_usd":      recommended["daily_rate_usd"],
        "total_freight_usd":   round(total_freight_cost, 2),
        "total_freight_inr":   round(total_freight_cost * 83.5, 2),  # approx ?/$ rate
        "vessel_utilisation":  efficiency_pct,
        "decision":            _procurement_decision(predicted_rate),
    }


def _procurement_decision(rate: float) -> str:
    if rate < 10:
        return "? FAVORABLE ? Spot market rate is low. Procure now."
    elif rate < 18:
        return "? MODERATE ? Consider short-term contracts (3?6 months)."
    elif rate < 28:
        return "? ELEVATED ? Explore COA (Contract of Affreightment) options."
    else:
        return "? HIGH ? Delay procurement or negotiate long-term charter rates."
