"""
SIH26006 ? Bulk Freight Rate Forecasting
Data Generator: Synthetic but realistic freight shipping dataset
Ministry of Steel, India
"""

import numpy as np
import pandas as pd
import os

# ??????????????????????????? Route Definitions ????????????????????????????
ROUTES = {
    "Paradip->Visakhapatnam":   {"distance_nm": 480,  "base_rate": 12.0},
    "Kandla->Mundra":           {"distance_nm": 90,   "base_rate": 5.0},
    "Mumbai->Haldia":           {"distance_nm": 1300, "base_rate": 18.0},
    "Paradip->Mundra":          {"distance_nm": 1750, "base_rate": 22.0},
    "Haldia->Visakhapatnam":    {"distance_nm": 760,  "base_rate": 14.0},
    "Ennore->Mumbai":           {"distance_nm": 1050, "base_rate": 16.5},
    "Paradip->Ennore":          {"distance_nm": 1100, "base_rate": 17.0},
    "Kandla->Mumbai":           {"distance_nm": 540,  "base_rate": 11.0},
    "Haldia->Ennore":           {"distance_nm": 1250, "base_rate": 19.5},
    "Visakhapatnam->Mundra":    {"distance_nm": 1600, "base_rate": 21.0},
}

VESSEL_TYPES = {
    "Handysize":  {"capacity_mt": 28000,  "speed_knots": 14, "fuel_efficiency": 0.28},
    "Supramax":   {"capacity_mt": 55000,  "speed_knots": 13, "fuel_efficiency": 0.35},
    "Panamax":    {"capacity_mt": 75000,  "speed_knots": 13, "fuel_efficiency": 0.40},
    "Capesize":   {"capacity_mt": 150000, "speed_knots": 15, "fuel_efficiency": 0.50},
}

CARGO_TYPES = ["Iron Ore", "Coal", "Limestone", "Steel Coils", "Pig Iron"]

def seasonal_demand_index(month: int) -> float:
    """
    Models seasonal demand for steel shipping in India.
    - High demand: Oct?Mar (post-monsoon construction boom)
    - Low demand: Jun?Aug (monsoon disruptions)
    """
    base = 1.0 + 0.4 * np.sin((month - 3) * np.pi / 6)
    return np.clip(base, 0.5, 1.5)

def port_congestion_factor(route: str, month: int) -> float:
    """Port congestion varies by season and route."""
    base_congestion = {
        "Paradip->Visakhapatnam": 0.3, "Kandla->Mundra": 0.5,
        "Mumbai->Haldia": 0.4, "Paradip->Mundra": 0.2,
        "Haldia->Visakhapatnam": 0.35, "Ennore->Mumbai": 0.45,
        "Paradip->Ennore": 0.25, "Kandla->Mumbai": 0.55,
        "Haldia->Ennore": 0.3, "Visakhapatnam->Mundra": 0.2,
    }
    monsoon_bump = 0.2 if 6 <= month <= 9 else 0.0
    return min(base_congestion.get(route, 0.3) + monsoon_bump + np.random.uniform(-0.05, 0.05), 1.0)

def calculate_freight_rate(
    route: str, vessel_type: str, cargo_volume_mt: float,
    bunker_fuel_cost: float, month: int, year: int,
    port_congestion: float, cargo_type: str
) -> float:
    """
    Realistic freight rate formula ($/tonne):
    Rate = (BaseRate ? Distance ? FuelFactor ? VesselFactor ? SeasonFactor ? CongestionFactor) / Volume_Efficiency
    """
    route_info   = ROUTES[route]
    vessel_info  = VESSEL_TYPES[vessel_type]

    base_rate    = route_info["base_rate"]
    distance     = route_info["distance_nm"]
    capacity     = vessel_info["capacity_mt"]
    fuel_eff     = vessel_info["fuel_efficiency"]

    # Utilisation factor (fuller vessel = cheaper per tonne)
    utilisation  = min(cargo_volume_mt / capacity, 1.0)
    volume_factor = 1.0 / (0.5 + 0.5 * utilisation)

    # Fuel cost factor (bunker price drives rates significantly)
    fuel_factor  = 0.3 + (bunker_fuel_cost / 600) * fuel_eff

    # Seasonal demand
    season_factor = seasonal_demand_index(month)

    # Congestion adds cost
    congestion_factor = 1.0 + 0.3 * port_congestion

    # Year-over-year slight inflation
    year_factor = 1.0 + 0.03 * (year - 2020)

    # Cargo type premium
    cargo_premium = {"Iron Ore": 1.0, "Coal": 1.05, "Limestone": 0.95,
                     "Steel Coils": 1.15, "Pig Iron": 1.10}.get(cargo_type, 1.0)

    rate = (base_rate * (distance / 500) * fuel_factor * volume_factor *
            season_factor * congestion_factor * year_factor * cargo_premium)

    # Add realistic noise
    noise = np.random.normal(0, rate * 0.05)
    return max(round(rate + noise, 2), 1.0)


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic freight rate dataset."""
    np.random.seed(random_state)

    route_names  = list(ROUTES.keys())
    vessel_names = list(VESSEL_TYPES.keys())
    records = []

    for _ in range(n_samples):
        route       = np.random.choice(route_names)
        vessel_type = np.random.choice(vessel_names)
        cargo_type  = np.random.choice(CARGO_TYPES)
        year        = np.random.randint(2020, 2026)
        month       = np.random.randint(1, 13)
        day         = np.random.randint(1, 28)

        capacity    = VESSEL_TYPES[vessel_type]["capacity_mt"]
        cargo_vol   = np.random.uniform(0.4 * capacity, 0.98 * capacity)
        bunker_fuel = np.random.uniform(380, 920)  # $/MT IFO380 range
        port_cong   = port_congestion_factor(route, month)

        rate = calculate_freight_rate(
            route, vessel_type, cargo_vol, bunker_fuel,
            month, year, port_cong, cargo_type
        )

        records.append({
            "date":               f"{year}-{month:02d}-{day:02d}",
            "year":               year,
            "month":              month,
            "day":                day,
            "route":              route,
            "origin":             route.split("?")[0],
            "destination":        route.split("?")[1],
            "vessel_type":        vessel_type,
            "cargo_type":         cargo_type,
            "cargo_volume_mt":    round(cargo_vol, 0),
            "bunker_fuel_usd_mt": round(bunker_fuel, 2),
            "distance_nm":        ROUTES[route]["distance_nm"],
            "vessel_capacity_mt": capacity,
            "utilisation_pct":    round(cargo_vol / capacity * 100, 2),
            "port_congestion_idx":round(port_cong, 3),
            "seasonal_demand_idx":round(seasonal_demand_index(month), 3),
            "freight_rate_usd_t": rate,   # TARGET
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str = "data/raw/freight_data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"? Dataset saved ? {path}  |  Rows: {len(df)}  |  Cols: {len(df.columns)}")


if __name__ == "__main__":
    df = generate_dataset(n_samples=5000)
    save_dataset(df, "data/raw/freight_data.csv")
    print("\nSample records:")
    print(df[["route", "vessel_type", "cargo_volume_mt",
              "bunker_fuel_usd_mt", "seasonal_demand_idx",
              "freight_rate_usd_t"]].head(10).to_string(index=False))
    print(f"\nFreight Rate Stats:\n{df['freight_rate_usd_t'].describe().round(2)}")
