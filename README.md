# 🚚 DoorDash Delivery Duration Prediction

## 🎯 Executive Summary
Built an end-to-end machine learning pipeline in **R** to predict delivery duration using structured SQL data engineering and gradient boosting.  
Engineered congestion-based features that emerged as the strongest predictive drivers.  
Improved **MAE by ~7%** over a linear baseline while maintaining strict leakage prevention and production-style workflow discipline.

---

## 🔗 Live Dashboard
👉 **View Interactive Model Diagnostics:**  
**[Open Dashboard](https://divrajs-15.github.io/doordash-delivery-duration/)** 

---

## 🛠 Tech Stack
![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)
![Quarto](https://img.shields.io/badge/Quarto-39729E?style=flat-square)

---

## 📌 Project Overview
This project builds a production-style machine learning pipeline to predict DoorDash delivery duration (in seconds) from order creation to final delivery.

The workflow includes:
- **DuckDB (SQL)** for structured data processing  
- Feature engineering in **SQL + R**  
- **Linear Regression & Elastic Net**  
- **XGBoost (Gradient Boosting)**  
- Quarto dashboard for diagnostics  

The final XGBoost model reduced **Mean Absolute Error by ~7%** compared to the linear baseline.

---

## 🏗 Data Engineering (DuckDB)
Instead of manipulating the dataset directly in R, I used DuckDB as a lightweight analytical database layer.

**Why this approach?**
- Mimics production workflows  
- Keeps transformation logic structured  
- Separates data engineering from modeling  

**Key steps**
- Parsed timestamps  
- Created target variable (`delivery_duration_seconds`)  
- Removed invalid durations  
- Investigated percentiles  
- Trimmed extreme tail values (> 8000 seconds) based on empirical inspection  

---

## 🧠 Feature Engineering
Features were designed around real marketplace dynamics.

### Time-Based
- `order_hour`
- `order_dow`
- `is_weekend`

### Marketplace Congestion
- `busy_ratio`
- `orders_per_dasher`  
These ratios capture supply–demand imbalance more effectively than raw counts.

### Basket Complexity
- `avg_item_price`
- `item_price_range`
- `distinct_item_ratio`  
These proxy for order complexity and preparation workload.

---

## 📊 Modeling Progression

### 1️⃣ Linear Regression
**MAE ≈ 676 sec**  
Captured strong drivers: driving duration, congestion metrics, time-of-day, and order size.

### 2️⃣ Elastic Net
Performance nearly identical to the linear baseline.  
**Conclusion:** multicollinearity was not the main limitation.

### 3️⃣ XGBoost
**MAE ≈ 628 sec (~7% improvement)**  
Captured nonlinear tipping points, interactions, and threshold behavior.

Top predictive drivers:
- `orders_per_dasher`
- `estimated_store_to_consumer_driving_duration`
- `order_hour`
- `subtotal`
- `store_id`

---

## 🔍 Key Insights
- Driving duration is a dominant structural component.
- Marketplace congestion is the strongest nonlinear driver.
- Feature engineering reduced the performance gap between linear and boosted models.
- Regularization had minimal impact, suggesting bias dominated variance.
- Boosting improved performance by modeling nonlinear system stress.

---

## 🚀 Future Improvements
- Hyperparameter tuning via cross-validation  
- Log-transforming the target to reduce heavy-tail effects  
- Target encoding for `store_id`  
- LightGBM / CatBoost comparison  
- Deployment as a lightweight prediction API  

---

## 🧠 What This Project Demonstrates
- Structured SQL-based data engineering  
- Strict train/test discipline  
- Leakage prevention awareness  
- Bias–variance reasoning  
- Model comparison & interpretation  
- Production-oriented thinking  

