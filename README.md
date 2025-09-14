📦 Demand Forecasting for Retail Supply Chain

This project demonstrates how to combine time series forecasting and optimization modeling to improve retail supply chain decision-making. It predicts store-level product demand using both statistical and deep learning models, then recommends optimized stock allocation to minimize shortages.

🔑 Key Features

Hybrid Forecasting:

ARIMA/SARIMAX for capturing seasonality & trends.

LSTM (Keras/TensorFlow) for nonlinear patterns.

Optimization Module:

Linear programming (via PuLP) to allocate limited stock across multiple stores.

Balances demand satisfaction with capacity constraints.

Synthetic Data Generator:

Creates realistic multi-store, multi-SKU sales history for experimentation.

Reproducible & Offline:

Runs fully on a local machine with Python libraries.

No external APIs or paid services needed.

🛠️ Tech Stack

Data Handling: pandas, numpy

Time Series Models: statsmodels (SARIMAX), Keras/TensorFlow (LSTM)

Optimization: PuLP (Linear Programming)

Visualization (optional): matplotlib

🚀 ##Run (quick demo with synthetic data)

```bash
pip install -r requirements.txt
python src/generate_synth_data.py --out data/demand.csv
python src/train_arima.py --input data/demand.csv --out models/arima.pkl
python src/train_lstm.py --input data/demand.csv --out models/lstm.keras
python src/hybrid_forecast.py --input data/demand.csv --arima models/arima.pkl --lstm models/lstm.keras --out data/forecast.csv
python src/plot_forecast.py --input data/forecast.csv
```
