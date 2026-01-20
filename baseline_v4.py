# eval_paper_baselines_v4_no_optimal.py
import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


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
    - sum_rate per step
    - per_ue_rate per step (bps per UE)
    - optional PF avg_rate curve (env.avg_rate)
    """
    sum_rates = np.zeros(steps, dtype=np.float64)
    per_ue_rates_all = []  # list of [U] per step
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
        per_ue_rates_all.append(per_ue)

        if record_pf_curve:
            pf_curve[t] = env.avg_rate.numpy()

    per_ue_rates_all = np.stack(per_ue_rates_all, axis=0)  # [steps, U]
    ue_rate_avg = np.mean(per_ue_rates_all, axis=0)        # [U]
    fairness = jain_fairness(ue_rate_avg)

    st = safe_stats(sum_rates)

    return {
        "sum_rates": sum_rates,                          # [steps]
        "per_ue_rates_all": per_ue_rates_all,            # [steps, U]
        "nan_count": int(nan_count),
        "fairness": float(fairness),
        "sum_rate_mean": st["mean"],
        "sum_rate_p50": st["p50"],
        "sum_rate_p95": st["p95"],
        "pf_curve": pf_curve                             # [steps, U] or None
    }


def build_env(fc_hz: float, seed: int = 7):
    return SionnaCellFreeUplinkEnvV2(
        num_ap=32,
        num_ue=24,
        num_rb=16,
        area_side_m=200.0,
        deployment="street",
        street_block_m=50.0,
        carrier_freq_hz=fc_hz,
        num_ap_rx_ant=16,
        num_ue_tx_ant=1,
        rb_bw_hz=180e3,

        noise_figure_db=7.0,
        pmax_watt=0.2,

        indoor_prob=0.2,
        ue_speed_mps=1.0,

        enable_3gpp=True,
        debug=False,
        seed=seed
    )


# -------------------------
# Plot helpers
# -------------------------
def plot_cdf_sumrate(raw_store, fc_list, fc_name, reps, out_png, title):
    plt.figure()
    for fc in fc_list:
        for (C, L, scheduler, pc) in reps:
            key = (fc, C, L, scheduler, int(pc))
            data = raw_store[key]["sum_rates"]
            x, y = ecdf(data)
            plt.plot(x, y, label=f"{fc_name[fc]} | {scheduler} PC={pc} (C={C},L={L})")
    plt.xlabel("Sum-Rate (bps)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved: {out_png}")


def plot_cdf_perue(raw_store, fc_list, fc_name, reps, out_png, title):
    plt.figure()
    for fc in fc_list:
        for (C, L, scheduler, pc) in reps:
            key = (fc, C, L, scheduler, int(pc))
            per_ue_all = raw_store[key]["per_ue_rates_all"].reshape(-1)  # [steps*U]
            x, y = ecdf(per_ue_all)
            plt.plot(x, y, label=f"{fc_name[fc]} | {scheduler} PC={pc} (C={C},L={L})")
    plt.xlabel("Per-UE Throughput (bps)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved: {out_png}")


def plot_pf_convergence_compareL(fc_list, fc_name, batch_size, steps, C, out_png):
    plt.figure()
    for fc in fc_list:
        for L in [1, 2]:
            env = build_env(fc_hz=fc, seed=7)
            env.reset_pf_state()
            r_pf = run_collect(
                env=env,
                batch_size=batch_size,
                steps=steps,
                C=C,
                L=L,
                scheduler="pf",
                enable_power_control=True,
                record_pf_curve=True
            )
            mean_avg_rate = np.mean(r_pf["pf_curve"], axis=1)
            plt.plot(mean_avg_rate, label=f"{fc_name[fc]} | L={L}")
    plt.xlabel("Step")
    plt.ylabel("Mean PF avg_rate (bps)")
    plt.title(f"PF Convergence (C={C}, PF, PC=True): compare L=1 vs L=2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved: {out_png}")


def plot_fairness_vs_fc_compareL(raw_store, fc_list, fc_name, C, out_png):
    plt.figure()
    for L in [1, 2]:
        fairs = []
        labels = []
        for fc in fc_list:
            key = (fc, C, L, "pf", 1)  # PF + PC=True
            fairs.append(raw_store[key]["fairness"])
            labels.append(fc_name[fc])
        plt.plot(labels, fairs, marker="o", label=f"L={L}")
    plt.xlabel("Carrier Frequency")
    plt.ylabel("Jain Fairness")
    plt.title(f"Fairness vs Carrier Frequency (C={C}, PF, PC=True): L=1 vs L=2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved: {out_png}")


# -------------------------
# Main
# -------------------------
def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Output folder: paper_runs/<timestamp>/{figs,data}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join("paper_runs", ts))
    fig_dir = ensure_dir(os.path.join(run_dir, "figs"))
    data_dir = ensure_dir(os.path.join(run_dir, "data"))

    print(f"\n📁 Run folder created: {run_dir}")
    print(f"   - figs: {fig_dir}")
    print(f"   - data: {data_dir}\n")

    # Paper-like carrier frequencies (FR1 / FR3 / mmWave)
    fc_list = [3.5e9, 7.0e9, 28.0e9]
    fc_name = {3.5e9: "3.5GHz", 7.0e9: "7GHz", 28.0e9: "28GHz"}

    # Compare both C=4 and C=8
    C_compare_list = [4, 8]

    # Experiment length
    batch_size = 1
    steps_cdf = 300
    steps_pf = 400

    # ✅ Focus settings: PF + GAIN only (optimal removed)
    focus_settings = []
    for C in C_compare_list:
        focus_settings += [
            # --- L=1 baseline ---
            (C, 1, "pf", True),
            (C, 1, "gain", False),

            # --- L=2 RB-sharing ---
            (C, 2, "pf", True),
            (C, 2, "gain", True),
        ]

    raw_store = {}
    rows = []

    t0 = time.time()

    # Phase-1: run experiments
    for fc in fc_list:
        env = build_env(fc_hz=fc, seed=7)

        for (C, L, scheduler, pc) in focus_settings:
            env.reset_pf_state()

            r = run_collect(
                env=env,
                batch_size=batch_size,
                steps=steps_cdf,
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
                "steps": steps_cdf,
                "sum_rate_mean": r["sum_rate_mean"],
                "sum_rate_p50": r["sum_rate_p50"],
                "sum_rate_p95": r["sum_rate_p95"],
                "fairness": r["fairness"],
                "nan_count": r["nan_count"],
            }
            rows.append(row)
            print(row)

    print(f"\n[Phase-1 Done] time={time.time()-t0:.1f}s")

    # Save CSV summary
    out_csv = os.path.join(data_dir, "paper_results_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    # Save raw_store to NPZ
    npz_dict = {}
    for (fc, C, L, sch, pc), r in raw_store.items():
        prefix = f"fc={fc_name[fc]}_C={C}_L={L}_sch={sch}_pc={pc}"
        npz_dict[prefix + "__sum_rates"] = r["sum_rates"]
        npz_dict[prefix + "__per_ue_rates_all"] = r["per_ue_rates_all"]

    out_npz = os.path.join(data_dir, "raw_store_cdf.npz")
    np.savez_compressed(out_npz, **npz_dict)
    print(f"Saved: {out_npz}")

    # Phase-2: CDF plots (only PF + GAIN)
    for C in C_compare_list:
        reps = [
            # --- L=1 ---
            (C, 1, "pf", 1),
            (C, 1, "gain", 0),

            # --- L=2 ---
            (C, 2, "pf", 1),
            (C, 2, "gain", 1),
        ]

        plot_cdf_sumrate(
            raw_store=raw_store,
            fc_list=fc_list,
            fc_name=fc_name,
            reps=reps,
            out_png=os.path.join(fig_dir, f"CDF_sumrate_C{C}_PF_GAIN_Lcompare.png"),
            title=f"CDF of Sum-Rate (UMi street) | PF vs GAIN | Compare L=1 vs L=2 | C={C}"
        )

        plot_cdf_perue(
            raw_store=raw_store,
            fc_list=fc_list,
            fc_name=fc_name,
            reps=reps,
            out_png=os.path.join(fig_dir, f"CDF_perUE_C{C}_PF_GAIN_Lcompare.png"),
            title=f"CDF of Per-UE Throughput (UMi street) | PF vs GAIN | Compare L=1 vs L=2 | C={C}"
        )

    # Phase-3: PF convergence curve (C=4 and C=8)
    for C in C_compare_list:
        out_png = os.path.join(fig_dir, f"PF_convergence_C{C}_Lcompare.png")
        plot_pf_convergence_compareL(
            fc_list=fc_list,
            fc_name=fc_name,
            batch_size=batch_size,
            steps=steps_pf,
            C=C,
            out_png=out_png
        )

    # Phase-4: Fairness vs fc (C=4 and C=8)
    for C in C_compare_list:
        out_png = os.path.join(fig_dir, f"PF_fairness_vs_fc_C{C}_Lcompare.png")
        plot_fairness_vs_fc_compareL(
            raw_store=raw_store,
            fc_list=fc_list,
            fc_name=fc_name,
            C=C,
            out_png=out_png
        )

    print("\n✅ All paper-style results saved in:")
    print(f"   {run_dir}")
    print("\nFiles generated:")
    print(f" - {out_csv}")
    print(f" - {out_npz}")
    print(f" - figs/*.png (CDF + PF convergence + fairness comparisons)\n")


if __name__ == "__main__":
    main()
