import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/forecast.csv")
    args = ap.parse_args()

    df = pd.read_csv("data/demand.csv")
    fc = pd.read_csv(args.input)
    s = df[df.store_id==0].sort_values("date")[["date","demand"]]
    plt.figure()
    plt.plot(s["date"], s["demand"], label="history")
    plt.plot(fc["t"], fc["forecast"], label="forecast")
    plt.legend()
    plt.title("Hybrid Forecast")
    plt.show()

if __name__ == "__main__":
    main()
