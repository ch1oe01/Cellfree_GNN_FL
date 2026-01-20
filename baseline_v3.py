# eval_paper_baselines_v3.py
import os
import csv
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 假設 channel_env_v2.py 與此檔案在同一目錄下
from channel_env_v2 import SionnaCellFreeUplinkEnvV2


# -------------------------
# Utils
# -------------------------
def jain_fairness(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.maximum(x, 0.0)
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    n = x.size
    return float((s1 * s1) / (n * s2 + eps))


def ecdf(data: np.ndarray):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0]), np.array([0.0])
    x = np.sort(data)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_readme(out_dir: str, text: str):
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(text)


# -------------------------
# Core runner
# -------------------------
def run_collect(env: SionnaCellFreeUplinkEnvV2,
                batch_size: int,
                steps: int,
                C: int,
                L: int,
                scheduler: str,
                enable_power_control: bool,
                record_pf_curve: bool = False):
    """
    Collect raw samples for paper plots:
    - sum_rate per step: [steps]
    - per_ue_rate per step (bps per UE): [steps, U]
    - optional PF avg_rate curve: [steps, U] (EMA state)
    """
    sum_rates = np.zeros(steps, dtype=np.float64)
    per_ue_rates_all = np.zeros((steps, env.U), dtype=np.float64)
    nan_count = 0

    pf_curve = None
    if record_pf_curve:
        pf_curve = np.zeros((steps, env.U), dtype=np.float64)

    for t in range(steps):
        out = env.step(
            batch_size=batch_size,
            C=C,
            L=L,
            scheduler=scheduler,
            enable_power_control=enable_power_control
        )

        sr = float(out["sum_rate"].numpy())
        if not np.isfinite(sr):
            nan_count += 1
            sr = 0.0
        sum_rates[t] = sr

        rate = out["rate"].numpy()  # [B,U,K]
        per_ue = np.mean(np.sum(rate, axis=2), axis=0)  # [U]
        per_ue = np.nan_to_num(per_ue, nan=0.0, posinf=0.0, neginf=0.0)
        per_ue_rates_all[t] = per_ue

        if record_pf_curve:
            pf_curve[t] = env.avg_rate.numpy()

    ue_rate_avg = np.mean(per_ue_rates_all, axis=0)  # [U]
    fairness = jain_fairness(ue_rate_avg)

    st = safe_stats(sum_rates)

    return {
        "sum_rates": sum_rates,
        "per_ue_rates_all": per_ue_rates_all,
        "nan_count": int(nan_count),
        "fairness": float(fairness),
        "sum_rate_mean": st["mean"],
        "sum_rate_p50": st["p50"],
        "sum_rate_p95": st["p95"],
        "pf_curve": pf_curve
    }


def build_env(fc_hz: float, seed: int = 7):
    # make TF/NP reproducible per-env
    tf.random.set_seed(seed)
    np.random.seed(seed)

    return SionnaCellFreeUplinkEnvV2(
        num_ap=32,
        # ---------------------------------------------------------
        # 🔥 修改點 1: 用戶數提升至 24 (製造資源稀缺)
        #    RB=16, UE=24 => Overloaded (1.5倍負載)
        # ---------------------------------------------------------
        num_ue=24,
        num_rb=16,
        area_side_m=200.0,
        deployment="street",
        street_block_m=50.0,

        carrier_freq_hz=fc_hz,
        rb_bw_hz=180e3,

        num_ap_rx_ant=16,
        num_ue_tx_ant=1,

        noise_figure_db=7.0,
        pmax_watt=0.2,

        indoor_prob=0.2,
        ue_speed_mps=1.0,

        enable_3gpp=True,
        debug=False,
        seed=seed
    )


# -------------------------
# Paper-style experiments
# -------------------------
def main():
    # Make TF a bit quieter
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # -------------------------
    # Output folder
    # -------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join("paper_runs", f"{run_tag}_umicf_v3_overloaded")) # 加個標籤識別
    figs_dir = ensure_dir(os.path.join(out_dir, "figs"))
    data_dir = ensure_dir(os.path.join(out_dir, "data"))

    # ✅ Three carrier frequencies
    fc_list = [3.5e9, 7.0e9, 28.0e9]
    fc_name = {3.5e9: "3.5GHz", 7.0e9: "7GHz", 28.0e9: "28GHz"}

    # ---------------------------------------------------------
    # 🔥 修改點 2: Focus Settings 策略調整
    #    目標：展示 L=1 時的擁塞 (PF vs Optimal 差異)
    #          展示 L=2 時的容量提升 (MU-MIMO Gain)
    # ---------------------------------------------------------
    focus_settings = [
        # (C, L, scheduler, power_control)
        
        # --- Group A: Overloaded Baseline (L=1) ---
        # 在 U=24, K=16 下，這組會顯示 "公平性(PF)" 與 "最大速率(Optimal)" 的劇烈權衡
        (4, 1, "pf", True),       
        (4, 1, "gain", False),    
        (4, 1, "optimal", False), 

        # --- Group B: MU-MIMO Capacity (L=2) ---
        # 每個 RB 可塞 2 人，理論容量變 32，足以容納 24 人。
        # 這組用來展示 "如果排程器/PC夠強，我們可以救回所有用戶"
        (4, 2, "pf", True),
    ]

    # ---------------------------------------------------------
    # 🔥 修改點 3: 繪圖代表組 (Reps)
    #    挑選最能 "說故事" 的幾條線畫在 CDF 上
    # ---------------------------------------------------------
    reps = [
        (4, 1, "pf", 1),       # 基線：公平但速率受限
        (4, 1, "optimal", 0),  # 上界：速率高但會餓死 1/3 用戶 (CDF起點會有 33% 是 0)
        (4, 2, "pf", 1),       # 進階：開啟 MU-MIMO 後的潛力 (應比 L=1 PF 好)
    ]

    batch_size = 1
    steps = 300          
    pf_curve_steps = 400 

    out_csv = os.path.join(out_dir, "paper_results.csv")

    # store raw outputs for plots
    raw_store = {}
    rows = []

    t0 = time.time()

    # -------------------------
    # Phase 1) Run focus settings for each fc
    # -------------------------
    for fc in fc_list:
        env = build_env(fc_hz=fc, seed=7)

        for (C, L, scheduler, pc) in focus_settings:
            env.reset_pf_state()  # fair comparison

            r = run_collect(
                env=env,
                batch_size=batch_size,
                steps=steps,
                C=C,
                L=L,
                scheduler=scheduler,
                enable_power_control=pc,
                record_pf_curve=False
            )

            key = (fc, C, L, scheduler, int(pc))
            raw_store[key] = r

            row = {
                "fc_hz": fc,
                "fc": fc_name[fc],
                "C": C,
                "L": L,
                "scheduler": scheduler,
                "power_control": int(pc),
                "steps": steps,
                "sum_rate_mean": r["sum_rate_mean"],
                "sum_rate_p50": r["sum_rate_p50"],
                "sum_rate_p95": r["sum_rate_p95"],
                "fairness": r["fairness"],
                "nan_count": r["nan_count"],
            }
            rows.append(row)
            print(row)

    print(f"\n[Phase-1 Done] time={time.time()-t0:.1f}s")

    # -------------------------
    # Save CSV
    # -------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    # -------------------------
    # Save raw samples (npz)
    # -------------------------
    npz_payload = {}
    for k, v in raw_store.items():
        fc, C, L, scheduler, pc = k
        tag = f"{fc_name[fc]}_C{C}_L{L}_{scheduler}_PC{pc}"
        npz_payload[f"{tag}__sum_rates"] = v["sum_rates"]
        npz_payload[f"{tag}__per_ue_rates_all"] = v["per_ue_rates_all"]
    raw_npz = os.path.join(data_dir, "raw_samples.npz")
    np.savez_compressed(raw_npz, **npz_payload)
    print(f"Saved: {raw_npz}")

    # -------------------------
    # Phase 2) CDF plots
    # -------------------------
    def get_key(fc, C, L, scheduler, pc):
        key = (fc, C, L, scheduler, pc)
        if key not in raw_store:
            raise KeyError(f"Missing raw_store key={key}. Ensure it exists in focus_settings.")
        return key

    # --- Sum-rate CDF ---
    plt.figure()
    for fc in fc_list:
        for (C, L, scheduler, pc) in reps:
            key = get_key(fc, C, L, scheduler, pc)
            data = raw_store[key]["sum_rates"]
            x, y = ecdf(data)
            # 調整 Label 顯示 L，方便辨識
            plt.plot(x, y, label=f"{fc_name[fc]} | {scheduler} L={L}")
    plt.xlabel("Sum-Rate (bps)")
    plt.ylabel("CDF")
    plt.title("CDF of Sum-Rate (UMi, U=24 Overloaded)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fpath = os.path.join(figs_dir, "cdf_sumrate_fc.png")
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"Saved: {fpath}")

    # --- Per-UE rate CDF ---
    plt.figure()
    for fc in fc_list:
        for (C, L, scheduler, pc) in reps:
            key = get_key(fc, C, L, scheduler, pc)
            per_ue_all = raw_store[key]["per_ue_rates_all"].reshape(-1)
            x, y = ecdf(per_ue_all)
            plt.plot(x, y, label=f"{fc_name[fc]} | {scheduler} L={L}")
    plt.xlabel("Per-UE Throughput (bps)")
    plt.ylabel("CDF")
    plt.title("CDF of Per-UE Throughput (UMi, U=24 Overloaded)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fpath = os.path.join(figs_dir, "cdf_perue_fc.png")
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"Saved: {fpath}")

    # -------------------------
    # Phase 3) PF Convergence curve
    # -------------------------
    plt.figure()
    pf_curve_store = {}
    for fc in fc_list:
        env = build_env(fc_hz=fc, seed=7)
        env.reset_pf_state()

        # 這裡也用 U=24, L=1 做收斂測試
        r_pf = run_collect(
            env=env,
            batch_size=batch_size,
            steps=pf_curve_steps,
            C=4,
            L=1,
            scheduler="pf",
            enable_power_control=True,
            record_pf_curve=True
        )

        mean_avg_rate = np.mean(r_pf["pf_curve"], axis=1)
        plt.plot(mean_avg_rate, label=f"{fc_name[fc]} mean")
        pf_curve_store[f"{fc_name[fc]}__pf_curve"] = r_pf["pf_curve"]

    plt.xlabel("Step")
    plt.ylabel("Mean PF avg_rate (bps)")
    plt.title("PF Convergence: mean(avg_rate) vs step (U=24, L=1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fpath = os.path.join(figs_dir, "pf_convergence_fc.png")
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"Saved: {fpath}")

    # Save PF curves
    pf_npz = os.path.join(data_dir, "pf_curves.npz")
    np.savez_compressed(pf_npz, **pf_curve_store)
    print(f"Saved: {pf_npz}")

    # -------------------------
    # Optional: fairness vs fc for PF
    # -------------------------
    plt.figure()
    fairs = []
    fcs = []
    for fc in fc_list:
        key = get_key(fc, 4, 1, "pf", 1)
        fairs.append(raw_store[key]["fairness"])
        fcs.append(fc_name[fc])
    plt.plot(fcs, fairs, marker="o")
    plt.xlabel("Carrier Frequency")
    plt.ylabel("Jain Fairness")
    plt.title("PF Fairness vs Carrier Frequency (U=24, L=1)")
    plt.grid(True)
    plt.tight_layout()
    fpath = os.path.join(figs_dir, "pf_fairness_fc.png")
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"Saved: {fpath}")

    # -------------------------
    # README
    # -------------------------
    readme = f"""Paper-style baseline run (UMi cell-free uplink) - OVERLOADED
=================================================

Output folder: {out_dir}

Config:
- U=24 (Overloaded), K=16, A=32
- Scenarios:
  1. L=1 (Baseline): 24 users compete for 16 RBs. Expect high contention.
  2. L=2 (MU-MIMO): 24 users share 32 slots (16*2). Expect higher capacity.

Focus:
- Compare PF fairness vs Optimal sum-rate under scarcity.
- Evaluate MU-MIMO gain (L=2 vs L=1).
"""
    save_readme(out_dir, readme)

    print("\n✅ All outputs saved into ONE folder.")
    print(f"📁 {out_dir}")


if __name__ == "__main__":
    main()