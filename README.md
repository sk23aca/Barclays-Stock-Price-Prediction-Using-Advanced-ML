## Project Overview
This repository provides an end-to-end framework for forecasting the **Open** and **Close** prices of Barclays PLC (ticker **BARC.L**) using three complementary models:

- **ARIMA** – A classical linear time-series model serving as a robust baseline.  
- **LSTM** – A deep-learning approach capturing non-linear dependencies in sequential data.  
- **Prophet** – Facebook’s decomposable model, excelling at trend, seasonality, and holiday effects.

We source ten years of historical data (2014–2023) via the **yfinance** library, engineer technical indicators (moving averages, returns, volatility), and rigorously compare model performance under various market regimes. Forecast accuracy is measured via MSE, RMSE, MAE, MAPE, and R²—offering actionable insights for traders and analysts.

---

## Features
- **Data Ingestion**  
  Fetch historical OHLCV data from Yahoo Finance using `yfinance`.  
- **Preprocessing**  
  - Rolling feature computation with Pandas  
  - Missing-value handling  
  - Chronological train/test split  
- **Modeling**  
  - **ARIMA**: Order selection via `pmdarima.auto_arima`  
  - **LSTM**: Sequence modeling with Keras (two LSTM layers)  
  - **Prophet**: Trend and seasonality decomposition  
- **Evaluation**  
  Standardized error metrics and visualizations of actual vs. predicted prices.  
- **Forecasting**  
  30-day ahead projections for all models, plus a simple ensemble.

---


## Evaluation & Results

We assess models via the following metrics:

| **Model**  | **Open RMSE** | **Close RMSE** | **Open MAPE** | **Close MAPE** | **Open R²** | **Close R²** |
|------------|---------------|----------------|---------------|----------------|-------------|--------------|
| **ARIMA**  | 54.6          | 53.6           | 20.7%         | 21.1%          | 0.89        | 0.83         |
| **LSTM**   | 7.4           | 6.1            | 30.98%        | 30.49%         | 0.98        | 0.99         |
| **Prophet**| 14.80         | 17.84          | 6.18%         | 7.88%          | 0.73        | 0.83         |



##Key finding:
LSTM excels in capturing non-linear patterns and achieving high R², though MAPE is inflated by low actual values; Prophet offers a compelling trade-off between accuracy and interpretability; ARIMA remains a fast, transparent baseline.

