# eval_baselines_v2.py
import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from channel_env_v2 import SionnaCellFreeUplinkEnvV2


def jain_fairness(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jain's fairness index for non-negative vector x:
    J = (sum x)^2 / (n * sum x^2)
    """
    x = np.maximum(x, 0.0)
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    n = x.size
    return float((s1 * s1) / (n * s2 + eps))


def safe_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
    )


def run_one_setting(env: SionnaCellFreeUplinkEnvV2,
                    batch_size: int,
                    steps: int,
                    C: int,
                    L: int,
                    scheduler: str,
                    enable_power_control: bool):
    sum_rates = []
    nan_count = 0

    # UE-average rate accumulation for Jain fairness
    # We'll collect mean rate per UE across steps
    ue_rate_acc = np.zeros(env.U, dtype=np.float64)

    for _ in range(steps):
        out = env.step(
            batch_size=batch_size,
            C=C,
            L=L,
            scheduler=scheduler,
            enable_power_control=enable_power_control
        )

        sr = out["sum_rate"].numpy()
        if not np.isfinite(sr):
            nan_count += 1
            # still append 0 so stats don't crash
            sum_rates.append(0.0)
        else:
            sum_rates.append(float(sr))

        # out["rate"]: [B,U,K]
        rate = out["rate"].numpy()
        # mean across batch, sum across K -> per UE throughput for this step
        # rate_unit: (Hz * log2(1+sinr)) => bps (since multiplied by W)
        per_ue = np.mean(np.sum(rate, axis=2), axis=0)  # [U]
        per_ue = np.nan_to_num(per_ue, nan=0.0, posinf=0.0, neginf=0.0)
        ue_rate_acc += per_ue

    ue_rate_avg = ue_rate_acc / max(steps, 1)
    fairness = jain_fairness(ue_rate_avg)

    st = safe_stats(np.array(sum_rates))
    return {
        "sum_rate_mean": st["mean"],
        "sum_rate_p50": st["p50"],
        "sum_rate_p95": st["p95"],
        "nan_count": int(nan_count),
        "fairness": float(fairness),
    }


def main():
    # -------------------------
    # Recommended baseline config
    # -------------------------
    env = SionnaCellFreeUplinkEnvV2(
        
        num_ap=32,
        num_ue=16,
        num_rb=16,
        area_side_m=200.0,
        deployment="street",
        street_block_m=50.0,
        carrier_freq_hz=7.0e9,      # 想改回 3.5e9 也行
        num_ap_rx_ant=16,
        num_ue_tx_ant=1,
        rb_bw_hz=180e3,

        noise_figure_db=7.0,
        pmax_watt=0.2,

        indoor_prob=0.2,
        ue_speed_mps=1.0,

        enable_3gpp=True,
        debug=False,
        seed=7
    )

    # -------------------------
    # Sweep settings (recommended)
    # -------------------------
    batch_size = 1
    steps = 200  # 你 GPU 慢就降到 50~100；要穩就 200~500

    C_list = [2, 4, 6, 8]
    L_list = [1, 2]
    schedulers = ["random", "gain", "optimal", "pf"]
    power_list = [False, True]

    out_csv = "results.csv"
    rows = []

    t0 = time.time()
    for C in C_list:
        for L in L_list:
            for scheduler in schedulers:
                for pc in power_list:
                    # 每個 setting 讓 PF 狀態重新開始比較公平
                    env.reset_pf_state()

                    r = run_one_setting(
                        env=env,
                        batch_size=batch_size,
                        steps=steps,
                        C=C,
                        L=L,
                        scheduler=scheduler,
                        enable_power_control=pc
                    )

                    row = {
                        "C": C,
                        "L": L,
                        "scheduler": scheduler,
                        "power_control": int(pc),
                        "steps": steps,
                        **r
                    }
                    rows.append(row)
                    print(row)

    print(f"\nDone. Total time: {time.time()-t0:.1f}s")

    # -------------------------
    # Save CSV
    # -------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    # -------------------------
    # Plot (sum_rate_mean vs C) for key configs
    # -------------------------
    # Example: plot PF with/without power_control, L=1
    def filt(L, scheduler, pc):
        xs = []
        ys = []
        fs = []
        for rr in rows:
            if rr["L"] == L and rr["scheduler"] == scheduler and rr["power_control"] == int(pc):
                xs.append(rr["C"])
                ys.append(rr["sum_rate_mean"])
                fs.append(rr["fairness"])
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order], np.array(fs)[order]

    plt.figure()
    for pc in [False, True]:
        x, y, _ = filt(L=1, scheduler="pf", pc=pc)
        plt.plot(x, y, marker="o", label=f"PF, L=1, PC={pc}")
    plt.xlabel("C (APs per UE)")
    plt.ylabel("Mean Sum-Rate (bps)")
    plt.title("Mean Sum-Rate vs C (PF baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sumrate_vs_C.png", dpi=200)
    print("Saved: sumrate_vs_C.png")

    plt.figure()
    for pc in [False, True]:
        x, _, f = filt(L=1, scheduler="pf", pc=pc)
        plt.plot(x, f, marker="o", label=f"PF, L=1, PC={pc}")
    plt.xlabel("C (APs per UE)")
    plt.ylabel("Jain Fairness")
    plt.title("Fairness vs C (PF baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fairness_vs_C.png", dpi=200)
    print("Saved: fairness_vs_C.png")


if __name__ == "__main__":
    # Optional: make TF a bit quieter
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
