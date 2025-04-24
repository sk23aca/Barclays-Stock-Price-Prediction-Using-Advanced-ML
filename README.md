Project Overview
This repository provides an end-to-end framework for forecasting the Open and Close prices of Barclays PLC (ticker BARC.L) using three complementary models:

ARIMA – A classical linear time-series model serving as a robust baseline.

LSTM – A deep‐learning approach capturing non-linear dependencies in sequential data.

Prophet – Facebook’s decomposable model, excelling at seasonality and holiday effects.

We source ten years of historical data (2014–2023) via the yfinance library, engineer technical indicators (moving averages, returns, volatility), and rigorously compare model performance under various market regimes. Forecast accuracy is measured through multiple metrics (MSE, RMSE, MAE, MAPE, R²), offering actionable insights for traders and analysts.

Features
Data Ingestion from Yahoo Finance using yfinance.

Preprocessing: rolling feature computation with Pandas, missing-value handling, and train/test split by date.

Modeling:

ARIMA order selection via pmdarima.auto_arima

Sequence modeling with Keras LSTM

Trend/seasonality decomposition with Prophet

Evaluation: standardized error metrics and visualizations of actual vs. predicted prices.

Forecasting: 30-day ahead projections for all models, plus a simple ensemble.

Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/barclays-stock-forecast.git
cd barclays-stock-forecast
Create a virtual environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Data Acquisition & Preprocessing
Download data

python
Copy
Edit
import yfinance as yf
df = yf.download("BARC.L", "2014-01-01", "2023-12-31", interval="1d")
df.sort_index(inplace=True)
Feature engineering

5, 10, 50-day moving averages (.rolling().mean().shift(1)) to avoid data leakage 
Gist

Daily returns (.pct_change().shift(1) * 100) and 10-day volatility

Train/Test split
Chronological split at January 1, 2023 to prevent look-ahead bias .

Modeling Approaches
ARIMA
Order selection with pmdarima.auto_arima (stepwise AIC minimization) 
HackerNoon

Fitting via statsmodels.tsa.arima.model.ARIMA

Forecasting with the .forecast() method for out-of-sample predictions

LSTM
Normalization using sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))

Sliding window dataset (time_step=100) via a custom generator

Keras Sequential model: two LSTM layers (50 units each), dense layers, trained with Adam optimizer 
Medium

Iterative forecasting for 30-day horizon

Prophet
Data format: DataFrame with ds (date) and y (value) columns 
Reddit

Model: Prophet(changepoint_prior_scale=0.1, multiplicative seasonality)

Holidays: UK public holidays injected to capture trading anomalies

Forecast: make_future_dataframe(periods=30)

Evaluation & Results
We assess models via:


Model	Open RMSE	Close RMSE	Open MAPE	Close MAPE	Open R²	Close R²
ARIMA	~54.6	~53.6	~20.7%	~21.1%	0.89	0.83
LSTM	~7.4	~6.1	~30.98%	~30.49%	0.98	0.99
Prophet	~14.80	~17.84	~6.18%	~7.88%	0.73	0.83
Key finding: LSTM excels in capturing non-linear patterns and achieving high R², though MAPE is inflated by low actual values; Prophet offers a compelling trade-off between accuracy and interpretability; ARIMA remains a fast, transparent baseline.

