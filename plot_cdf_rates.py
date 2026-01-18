# plot_cdf_rates.py
import os
import numpy as np
import matplotlib.pyplot as plt


def set_style():
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


def cdf_xy(x: np.ndarray):
    x = np.maximum(x, 0.0)
    x = np.sort(x)
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y


def main():
    # 你要看的設定（先固定 L=1）
    L = 1
    C_list = [1, 2, 3, 4, 5]
    scheduler = "pf"   # 可改 "gain" / "random"

    detail_dir = os.path.join("outputs", "details")
    out_dir = os.path.join("outputs", "figs")
    os.makedirs(out_dir, exist_ok=True)

    set_style()

    plt.figure(figsize=(6.6, 4.6))
    for C in C_list:
        path = os.path.join(detail_dir, f"ue_avg_rate_C{C}_L{L}_{scheduler}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到：{path}\n請先跑 python eval_baselines.py")

        ue_avg_rate = np.load(path)  # [U]
        x, y = cdf_xy(ue_avg_rate)

        plt.plot(x, y, linewidth=2.0, label=f"C={C}")

    plt.xlabel("UE average rate (bps)")
    plt.ylabel("CDF")
    plt.title(f"UE Rate CDF (L={L}, scheduler={scheduler.upper()})")
    plt.legend(loc="lower right")
    plt.tight_layout()

    png = os.path.join(out_dir, f"cdf_ue_rate_L{L}_{scheduler}.png")
    pdf = os.path.join(out_dir, f"cdf_ue_rate_L{L}_{scheduler}.pdf")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()

    print("✅ Saved:", png)
    print("✅ Saved:", pdf)


if __name__ == "__main__":
    main()
