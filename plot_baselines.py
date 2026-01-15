# plot_baselines.py
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def read_summary_csv(csv_path: str):
    """
    讀 outputs/baseline_summary.csv
    期待欄位：
      scheduler, C, L, avg_sum_rate_bps, jain_fairness, service_prob (其餘欄位可有可無)
    回傳 list[dict]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV：{csv_path}\n請先跑：python eval_baselines.py 產生 outputs/baseline_summary.csv")

    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # 轉型
            row2 = dict(row)
            row2["C"] = int(float(row2["C"]))
            row2["L"] = int(float(row2["L"]))
            row2["avg_sum_rate_bps"] = float(row2["avg_sum_rate_bps"])
            row2["jain_fairness"] = float(row2["jain_fairness"])
            row2["service_prob"] = float(row2.get("service_prob", 0.0))
            rows.append(row2)
    return rows


def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def plot_metric_vs_L(rows, C_target: int, metric_key: str, ylabel: str, out_path_png: str, out_path_pdf: str):
    """
    rows: list of dict
    metric_key: "avg_sum_rate_bps" or "jain_fairness"
    """
    # collect: scheduler -> (L -> metric)
    data = defaultdict(dict)
    L_set = set()
    sched_set = set()

    for r in rows:
        if r["C"] != C_target:
            continue
        sch = r["scheduler"].strip()
        L = r["L"]
        data[sch][L] = r[metric_key]
        L_set.add(L)
        sched_set.add(sch)

    if not data:
        raise ValueError(f"CSV 裡找不到 C={C_target} 的資料。你目前有的 C 值：{sorted(set([rr['C'] for rr in rows]))}")

    L_list = sorted(L_set)

    # plot order & style
    sched_order = ["random", "gain", "pf"]
    colors = {"random": "#1f77b4", "gain": "#ff7f0e", "pf": "#2ca02c"}
    markers = {"random": "o", "gain": "s", "pf": "D"}

    plt.figure(figsize=(6.6, 4.2))

    for sch in sched_order:
        if sch not in sched_set:
            continue
        y = [data[sch].get(L, np.nan) for L in L_list]
        plt.plot(
            L_list, y,
            label=sch.upper(),
            color=colors.get(sch, None),
            marker=markers.get(sch, "o"),
            linewidth=2.0,
            markersize=6,
        )

    plt.xlabel("L (max UEs per RB)")   
    plt.ylabel(ylabel)
    plt.title(f"C={C_target}:{ylabel} vs L")

    # nice y-limits for Jain
    if metric_key == "jain_fairness":
        plt.ylim(0.0, 1.02)

    plt.xticks(L_list)
    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png)
    plt.savefig(out_path_pdf)
    plt.close()


def main():
    # ---- paths ----
    csv_path = os.path.join("outputs", "baseline_summary.csv")
    fig_dir = os.path.join("outputs", "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- choose C to plot ----
    # 你目前實驗是 C=3，所以預設畫 C=3
    C_target = 3

    set_paper_style()
    rows = read_summary_csv(csv_path)

    # 1) Sum-rate vs L
    plot_metric_vs_L(
        rows,
        C_target=C_target,
        metric_key="avg_sum_rate_bps",
        ylabel="Average Sum-Rate (bps)",
        out_path_png=os.path.join(fig_dir, f"sr_vs_L_C{C_target}.png"),
        out_path_pdf=os.path.join(fig_dir, f"sr_vs_L_C{C_target}.pdf"),
    )

    # 2) Jain fairness vs L
    plot_metric_vs_L(
        rows,
        C_target=C_target,
        metric_key="jain_fairness",
        ylabel="Jain's Fairness Index",
        out_path_png=os.path.join(fig_dir, f"jain_vs_L_C{C_target}.png"),
        out_path_pdf=os.path.join(fig_dir, f"jain_vs_L_C{C_target}.pdf"),
    )

    print("✅ Plots saved to:", fig_dir)
    print(" -", f"sr_vs_L_C{C_target}.png/.pdf")
    print(" -", f"jain_vs_L_C{C_target}.png/.pdf")


if __name__ == "__main__":
    main()
