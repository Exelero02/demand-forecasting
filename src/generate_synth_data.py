import argparse, os, numpy as np, pandas as pd
rng = np.random.default_rng(42)

def make_series(T=200, season=12, noise=5.0, trend=0.2):
    t = np.arange(T)
    seasonal = 10 * np.sin(2*np.pi*t/season)
    y = 50 + trend*t + seasonal + rng.normal(0, noise, size=T)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/demand.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for store in range(3):
        y = make_series()
        for t, val in enumerate(y):
            rows.append({"date": t, "store_id": store, "demand": float(val)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
