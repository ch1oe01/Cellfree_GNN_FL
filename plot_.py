# plot_sweep.py (mean + 95% CI)
import os
import csv
import argparse
import math
import matplotlib.pyplot as plt

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    data = []
    for r in rows:
        try:
            v = float(r["value"])
            mean = float(r["mean_sr"])
            std = float(r["std_sr"])
            drops = int(float(r.get("drops", 0)))
            data.append((v, mean, std, drops))
        except Exception:
            pass

    data.sort(key=lambda x: x[0])
    return data

def plot_mean_ci(data, title, xlabel, out_png):
    x = [d[0] for d in data]
    mean = [d[1] for d in data]
    std = [d[2] for d in data]
    n = [max(1, d[3]) for d in data]

    # 95% CI = 1.96 * std / sqrt(n)
    ci = [1.96 * s / math.sqrt(nn) for s, nn in zip(std, n)]
    lower = [m - c for m, c in zip(mean, ci)]
    upper = [m + c for m, c in zip(mean, ci)]

    plt.figure(figsize=(7.5, 4.8), dpi=140)
    plt.plot(x, mean, "o-", linewidth=2, markersize=6, label="Mean")
    plt.fill_between(x, lower, upper, alpha=0.2, label="95% CI")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Sum-rate (prelog applied)")
    plt.legend()
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
    ap.add_argument("--out_dir", type=str, default="results/figs_paper")
    args = ap.parse_args()

    plot_mean_ci(load_csv(args.q_csv), "Sweep: RB Reuse Capacity Q", "Q (users per RB)",
                 os.path.join(args.out_dir, "sweep_Q_mean_95ci.png"))
    plot_mean_ci(load_csv(args.taup_csv), "Sweep: Pilot Length τp", "τp",
                 os.path.join(args.out_dir, "sweep_tau_p_mean_95ci.png"))
    plot_mean_ci(load_csv(args.m_csv), "Sweep: # Antennas per AP (M)", "M",
                 os.path.join(args.out_dir, "sweep_M_mean_95ci.png"))

if __name__ == "__main__":
    main()
