import argparse, os, numpy as np, pandas as pd, tensorflow as tf

def make_xy(series, lookback=12):
    X, y = [], []
    for i in range(len(series)-lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/demand.csv")
    ap.add_argument("--out", type=str, default="models/lstm.keras")
    ap.add_argument("--store_id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    s = df[df.store_id==args.store_id].sort_values("date")["demand"].values.astype("float32")
    X, y = make_xy(s)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=args.epochs, verbose=1)
    os.makedirs("models", exist_ok=True)
    model.save(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
