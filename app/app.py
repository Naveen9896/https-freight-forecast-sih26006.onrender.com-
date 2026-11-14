"""
SIH26006 ? Bulk Freight Rate Forecasting
Flask Web Application
"""

import os
import sys
import json
import io
import csv

from flask import (Flask, render_template, request,
                   jsonify, send_from_directory)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict, build_input_df
from src.feature_engineering import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.data_generator import ROUTES, VESSEL_TYPES, CARGO_TYPES

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")


# ??????????????????????? Route Helpers ????????????????????????????????????
@app.route("/")
def index():
    routes      = sorted(ROUTES.keys())
    vessels     = sorted(VESSEL_TYPES.keys())
    cargo_types = sorted(CARGO_TYPES)
    return render_template("index.html",
                           routes=routes,
                           vessels=vessels,
                           cargo_types=cargo_types)


# ??????????????????????? Predict API ??????????????????????????????????????
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()
        result = predict(
            route=data["route"],
            vessel_type=data["vessel_type"],
            cargo_type=data.get("cargo_type", "Iron Ore"),
            cargo_volume_mt=float(data["cargo_volume_mt"]),
            bunker_fuel_usd_mt=float(data["bunker_fuel_usd_mt"]),
            month=int(data["month"]),
            year=int(data.get("year", 2025)),
            port_congestion_idx=float(data.get("port_congestion_idx", 0.3)),
            model_name=data.get("model", "xgboost"),
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ??????????????????????? Feature Importance API ???????????????????????????
@app.route("/api/feature_importance")
def feature_importance():
    path = os.path.join(MODEL_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"labels": [], "values": []})


# ??????????????????????? Historical Rates API ?????????????????????????????
@app.route("/api/historical")
def historical_rates():
    path = os.path.join(MODEL_DIR, "historical_rates.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify([])


# ??????????????????????? Model Metrics API ????????????????????????????????
@app.route("/api/metrics")
def model_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify([])


# ??????????????????????? Route Info API ???????????????????????????????????
@app.route("/api/routes")
def get_routes():
    return jsonify([
        {"name": k, "distance_nm": v["distance_nm"], "base_rate": v["base_rate"]}
        for k, v in ROUTES.items()
    ])


# ??????????????????????? Batch Predict API ????????????????????????????????
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        content = file.read().decode("utf-8")
        reader  = csv.DictReader(io.StringIO(content))
        results = []
        for row in reader:
            try:
                res = predict(
                    route=row["route"],
                    vessel_type=row["vessel_type"],
                    cargo_type=row.get("cargo_type", "Iron Ore"),
                    cargo_volume_mt=float(row["cargo_volume_mt"]),
                    bunker_fuel_usd_mt=float(row["bunker_fuel_usd_mt"]),
                    month=int(row["month"]),
                    year=int(row.get("year", 2025)),
                )
                results.append({
                    "route":          res["route"],
                    "predicted_rate": res["predicted_rate_usd_t"],
                    "decision":       res["procurement"]["decision"],
                })
            except Exception as ex:
                results.append({"error": str(ex), "row": str(row)})

        return jsonify({"status": "success", "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
