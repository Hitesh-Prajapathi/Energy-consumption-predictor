# ⚡ Energy Consumption Analysis — Time Series Forecasting with XGBoost

## 📘 Overview
This project focuses on **predicting hourly energy consumption** using advanced time series modeling techniques.  
The dataset used is the **AEP Hourly Energy Consumption dataset**, which contains hourly power usage data from 2004 to 2018.  

Through a step-by-step analysis and feature engineering approach, this project demonstrates how adding **temporal features** and **lagged features** can drastically improve model performance.  

---

## 🎯 Objective
The goal of this project is to:
- Analyze patterns and seasonal trends in hourly energy consumption data.
- Engineer time-based and lag-based features to capture temporal dependencies.
- Build a robust regression model using **XGBoost** to predict energy demand.
- Evaluate the model using statistical performance metrics.

---

## 🧩 Dataset
**Dataset name:** `AEP_hourly.csv`  
**Source:** [PJMW Hourly Energy Consumption Data (AEP)](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) or Kaggle equivalent  

| Column | Description |
|--------|-------------|
| `Datetime` | Timestamp of each energy reading (hourly frequency) |
| `AEP_MW` | Energy consumption in megawatts (MW) |

---

## 🔍 Exploratory Data Analysis (EDA)
- Visualized **overall hourly energy trends**.
- Examined **daily**, **weekly**, and **monthly** consumption patterns.
- Observed that:
  - **Energy usage peaks in evenings (17:00–21:00).**
  - **Higher consumption occurs during summer and winter seasons** due to AC and heating usage.

---

## ⚙️ Feature Engineering

### ✅ Temporal Features
Extracted from the datetime index:
- Hour of the day
- Day of the week
- Month, Quarter, Year
- Week of the year
- Day of month, Day of year  

These features help the model learn **seasonality and time-based patterns**.

### ✅ Lagged Features
To incorporate temporal dependencies, lag features were created:
- Previous 1, 2, and 3 hours (`lag_1`, `lag_2`, `lag_3`)
- Same hour previous day (`lag_24`)
- Same hour one week ago (`lag_168`)

These help the model understand **short-term and long-term dependencies** in consumption.

---

## 🤖 Model — XGBoost Regressor
The model used is **XGBoost**, a powerful gradient boosting algorithm known for handling structured/tabular data effectively.

### Model Parameters:
```python
xgb.XGBRegressor(
    n_estimators=10000,
    learning_rate=0.01,
    gamma=5,
    min_child_weight=8,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
