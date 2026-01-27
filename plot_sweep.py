# plot_sweep.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")
    df = pd.read_csv(path)
    # 確保 value 是數值、排序用
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("value")
    return df

def plot_errorbar(df: pd.DataFrame, title: str, xlabel: str, out_png: str):
    x = df["value"].to_numpy()
    y = df["mean_sr"].to_numpy()
    yerr = df["std_sr"].to_numpy()

    plt.figure(figsize=(7.5, 4.8), dpi=140)
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, linewidth=2, markersize=6)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Sum-rate (prelog applied)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"Saved figure -> {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_csv", type=str, default="results/sweep_Q.csv")
    ap.add_argument("--taup_csv", type=str, default="results/sweep_tau_p.csv")
    ap.add_argument("--m_csv", type=str, default="results/sweep_M.csv")
    ap.add_argument("--out_dir", type=str, default="results/figs")
    args = ap.parse_args()

    # Q sweep
    df_q = load_csv(args.q_csv)
    plot_errorbar(
        df_q,
        title="Sweep: RB Reuse Capacity Q",
        xlabel="Q (users per RB)",
        out_png=os.path.join(args.out_dir, "sweep_Q.png")
    )

    # tau_p sweep
    df_t = load_csv(args.taup_csv)
    plot_errorbar(
        df_t,
        title="Sweep: Pilot Length τp (Pilot Reuse Severity)",
        xlabel="τp",
        out_png=os.path.join(args.out_dir, "sweep_tau_p.png")
    )

    # M sweep
    df_m = load_csv(args.m_csv)
    plot_errorbar(
        df_m,
        title="Sweep: # Antennas per AP (M)",
        xlabel="M",
        out_png=os.path.join(args.out_dir, "sweep_M.png")
    )

if __name__ == "__main__":
    main()

