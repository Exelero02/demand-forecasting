import argparse, pickle, numpy as np, pandas as pd, tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/demand.csv")
    ap.add_argument("--arima", type=str, default="models/arima.pkl")
    ap.add_argument("--lstm", type=str, default="models/lstm.keras")
    ap.add_argument("--out", type=str, default="data/forecast.csv")
    ap.add_argument("--store_id", type=int, default=0)
    ap.add_argument("--h", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    s = df[df.store_id==args.store_id].sort_values("date")["demand"].reset_index(drop=True)

    with open(args.arima, "rb") as f:
        arima_res = pickle.load(f)
    arima_fc = arima_res.forecast(steps=args.h).values

    model = tf.keras.models.load_model(args.lstm)
    lookback = model.input_shape[1]
    hist = s.values.astype("float32")

    model = tf.keras.models.load_model(args.lstm)
    lookback = model.input_shape[1]
    hist = s.values.astype("float32")

    last_window = hist[-lookback:].tolist()
    lstm_fc = []
    for _ in range(args.h):
        Xstep = np.array(last_window, dtype="float32").reshape(1, lookback, 1)
        yhat = float(model.predict(Xstep, verbose=0).ravel()[0])  # scalar
        lstm_fc.append(yhat)
        last_window = last_window[1:] + [yhat]

    lstm_fc = np.asarray(lstm_fc, dtype=np.float32)
    arima_fc = np.asarray(arima_fc, dtype=np.float32)

    fc = 0.8 * arima_fc + 0.2 * lstm_fc

    out = pd.DataFrame({"t": np.arange(len(s), len(s)+args.h), "forecast": fc})
    out.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
