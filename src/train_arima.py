import argparse, pickle, pandas as pd, os
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/demand.csv")
    ap.add_argument("--out", type=str, default="models/arima.pkl")
    ap.add_argument("--store_id", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    s = df[df.store_id==args.store_id].sort_values("date")["demand"]
    model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12))
    res = model.fit(disp=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(res, f)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
