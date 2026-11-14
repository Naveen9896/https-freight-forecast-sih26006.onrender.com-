# SIH26006 – Bulk Freight Rate Forecasting & Vessel Procurement

> **Ministry of Steel | Smart India Hackathon 2026 | Problem ID: SIH26006**

An end-to-end Machine Learning system that predicts bulk steel shipping freight rates and supports vessel procurement decisions using **Linear Regression** and **XGBoost**.

---

## 🎯 Problem Statement

The Ministry of Steel requires accurate bulk freight rate forecasting to optimize vessel procurement decisions for steel raw materials across Indian ports.

**Predict**: Freight rate ($/tonne) — continuous regression target  
**Features**: Route, cargo volume, fuel cost, seasonal demand, vessel type, port congestion  
**Output**: Rate prediction + vessel class recommendation + procurement decision

---

## 🏗️ Project Structure

```
freight-forecast/
├── data/
│   ├── raw/freight_data.csv          ← 5000-sample synthetic dataset
│   └── processed/                    ← Feature-engineered data
├── models/
│   ├── linear_regression.pkl         ← Ridge Regression + Polynomial Features
│   ├── xgboost_model.pkl             ← XGBoost Regressor (GridSearchCV tuned)
│   ├── metrics.json                  ← Evaluation results
│   ├── feature_importance.json       ← Feature importance for dashboard
│   └── historical_rates.json         ← Monthly trend data
├── src/
│   ├── data_generator.py             ← Synthetic data with Indian steel routes
│   ├── feature_engineering.py        ← Feature pipeline + ColumnTransformer
│   ├── train.py                      ← Training pipeline
│   ├── predict.py                    ← Inference module
│   └── utils.py                      ← Metrics + procurement recommendation
├── app/
│   ├── app.py                        ← Flask REST API
│   ├── templates/index.html          ← Dark-theme dashboard UI
│   └── static/
│       ├── style.css                 ← Premium CSS design
│       ├── main.js                   ← Chart.js + API integration
│       └── plots/                    ← Matplotlib plots (auto-generated)
├── requirements.txt
├── run.py                            ← 🚀 Single entry point
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run (auto-trains + starts dashboard)
```bash
python run.py
```
> Models are auto-trained on first run. Opens at **http://localhost:5000**

### 3. Train Only (optional)
```bash
python src/train.py
```

### 4. Test Inference
```bash
python src/predict.py
```

---

## 🤖 ML Models

### Linear Regression (Ridge + Polynomial)
- **Preprocessing**: StandardScaler + OneHotEncoder via ColumnTransformer
- **Features**: Degree-2 polynomial interaction features
- **Regularization**: Ridge (α=10) to prevent overfitting
- **Use case**: Interpretable baseline, regulatory transparency

### XGBoost Regressor
- **Tuning**: GridSearchCV (3-fold CV) over n_estimators, max_depth, learning_rate
- **Regularization**: L1 (reg_alpha) + L2 (reg_lambda)
- **Use case**: Primary production model (higher accuracy)

---

## 🛳️ Shipping Routes (India-centric)

| Route | Distance (NM) | Base Rate ($/t) |
|-------|--------------|-----------------|
| Paradip → Visakhapatnam | 480 | 12.0 |
| Kandla → Mundra | 90 | 5.0 |
| Mumbai → Haldia | 1,300 | 18.0 |
| Paradip → Mundra | 1,750 | 22.0 |
| Haldia → Visakhapatnam | 760 | 14.0 |
| Ennore → Mumbai | 1,050 | 16.5 |
| Paradip → Ennore | 1,100 | 17.0 |
| Kandla → Mumbai | 540 | 11.0 |
| Haldia → Ennore | 1,250 | 19.5 |
| Visakhapatnam → Mundra | 1,600 | 21.0 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/predict` | Single freight rate prediction |
| POST | `/batch_predict` | Bulk CSV upload prediction |
| GET | `/api/feature_importance` | XGBoost feature importance |
| GET | `/api/historical` | Monthly trend data |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/routes` | Available shipping routes |

### Predict Payload Example
```json
{
  "route": "Paradip→Visakhapatnam",
  "vessel_type": "Supramax",
  "cargo_type": "Iron Ore",
  "cargo_volume_mt": 42000,
  "bunker_fuel_usd_mt": 620,
  "month": 11,
  "year": 2025,
  "port_congestion_idx": 0.3,
  "model": "xgboost"
}
```

---

## 📊 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| route | Categorical | Origin→Destination port |
| vessel_type | Categorical | Handysize / Supramax / Panamax / Capesize |
| cargo_type | Categorical | Iron Ore, Coal, Limestone, Steel Coils, Pig Iron |
| cargo_volume_mt | Numerical | Cargo weight in metric tonnes |
| bunker_fuel_usd_mt | Numerical | Bunker fuel price ($/MT IFO380) |
| distance_nm | Numerical | Route distance in nautical miles |
| utilisation_pct | Engineered | Cargo ÷ vessel capacity × 100 |
| seasonal_demand_idx | Numerical | Cyclical demand index (0.5–1.5) |
| port_congestion_idx | Numerical | Port congestion factor (0–1) |
| fuel_x_distance | Engineered | Fuel cost × distance interaction |
| sin_month / cos_month | Engineered | Cyclical month encoding |
| fuel_per_nm | Engineered | Fuel cost density |

---

## ⚓ Vessel Procurement Decision Logic

| Predicted Rate | Decision |
|----------------|----------|
| < $10/t | 🟢 FAVORABLE – Spot market, procure now |
| $10–18/t | 🟡 MODERATE – Short-term contracts (3–6 months) |
| $18–28/t | 🟠 ELEVATED – Contract of Affreightment (COA) |
| > $28/t | 🔴 HIGH – Delay or negotiate long-term charter |

---

## 🔧 Tech Stack

- **Python 3.11** – Core language
- **scikit-learn** – Linear Regression, preprocessing, CV
- **XGBoost** – Gradient boosting regressor
- **Flask** – Web API and dashboard server
- **Chart.js** – Interactive dashboard charts
- **pandas / numpy** – Data manipulation
- **matplotlib / seaborn** – Training plots
