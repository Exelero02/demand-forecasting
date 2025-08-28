# Demand Forecasting (Hybrid ARIMA + LSTM)

Hybridize classical and deep learning methods for store-level demand forecasting.

## Run (quick demo with synthetic data)
```bash
pip install -r requirements.txt
python src/generate_synth_data.py --out data/demand.csv
python src/train_arima.py --input data/demand.csv --out models/arima.pkl
python src/train_lstm.py --input data/demand.csv --out models/lstm.keras
python src/hybrid_forecast.py --input data/demand.csv --arima models/arima.pkl --lstm models/lstm.keras --out data/forecast.csv
python src/plot_forecast.py --input data/forecast.csv
```
