# plot_vs_C.py
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def read_summary_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 CSV：{csv_path}\n請先跑：python eval_baselines.py"
        )

    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rr = dict(row)
            rr["C"] = int(float(rr["C"]))
            rr["L"] = int(float(rr["L"]))
            rr["avg_sum_rate_bps"] = float(rr["avg_sum_rate_bps"])
            rr["jain_fairness"] = float(rr["jain_fairness"])
            rr["service_prob"] = float(rr.get("service_prob", 0.0))
            rr["scheduler"] = rr["scheduler"].strip()
            rows.append(rr)
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
        "axes.unicode_minus": False,
    })


def plot_metric_vs_C(rows, L_target: int, metric_key: str, ylabel: str, out_path_png: str, out_path_pdf: str):
    """
    固定 L，畫 metric vs C（依 scheduler 分線）
    """
    # scheduler -> C -> metric
    data = defaultdict(dict)
    C_set = set()
    sched_set = set()

    for r in rows:
        if r["L"] != L_target:
            continue
        sch = r["scheduler"]
        C = r["C"]
        data[sch][C] = r[metric_key]
        C_set.add(C)
        sched_set.add(sch)

    if not data:
        raise ValueError(f"CSV 裡找不到 L={L_target} 的資料。你目前有的 L 值：{sorted(set([rr['L'] for rr in rows]))}")

    C_list = sorted(C_set)

    # order & style
    sched_order = ["random", "gain", "pf"]
    colors = {"random": "#1f77b4", "gain": "#ff7f0e", "pf": "#2ca02c"}
    markers = {"random": "o", "gain": "s", "pf": "D"}

    plt.figure(figsize=(6.6, 4.2))
    for sch in sched_order:
        if sch not in sched_set:
            continue
        y = [data[sch].get(C, np.nan) for C in C_list]
        plt.plot(
            C_list, y,
            label=sch.upper(),
            color=colors.get(sch, None),
            marker=markers.get(sch, "o"),
            linewidth=2.0,
            markersize=6,
        )

    plt.xlabel("C (APs per UE)")
    plt.ylabel(ylabel)
    plt.title(f"L={L_target}: {ylabel} vs C")
    plt.xticks(C_list)

    if metric_key == "jain_fairness":
        plt.ylim(0.0, 1.02)

    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png)
    plt.savefig(out_path_pdf)
    plt.close()


def main():
    csv_path = os.path.join("outputs", "baseline_summary.csv")
    fig_dir = os.path.join("outputs", "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # ✅ 最推薦：固定 L=1 看 cell-free 協作增益（C=1 是 no cooperation）
    L_target = 1

    set_paper_style()
    rows = read_summary_csv(csv_path)

    # Sum-rate vs C
    plot_metric_vs_C(
        rows,
        L_target=L_target,
        metric_key="avg_sum_rate_bps",
        ylabel="Average Sum-Rate (bps)",
        out_path_png=os.path.join(fig_dir, f"sr_vs_C_L{L_target}.png"),
        out_path_pdf=os.path.join(fig_dir, f"sr_vs_C_L{L_target}.pdf"),
    )

    # Jain vs C
    plot_metric_vs_C(
        rows,
        L_target=L_target,
        metric_key="jain_fairness",
        ylabel="Jain's Fairness Index",
        out_path_png=os.path.join(fig_dir, f"jain_vs_C_L{L_target}.png"),
        out_path_pdf=os.path.join(fig_dir, f"jain_vs_C_L{L_target}.pdf"),
    )

    print("✅ Plots saved to:", fig_dir)
    print(" -", f"sr_vs_C_L{L_target}.png/.pdf")
    print(" -", f"jain_vs_C_L{L_target}.png/.pdf")


if __name__ == "__main__":
    main()
