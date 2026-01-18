# eval_baselines.py
import os
import csv
import time
import numpy as np
import tensorflow as tf

from channel_env import SionnaCellFreeUplinkEnv


def jain_fairness(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.maximum(x, 0.0)
    s1 = float(np.sum(x))
    s2 = float(np.sum(x * x))
    n = x.size
    return (s1 * s1) / (n * s2 + eps)


def run_eval(
    scheduler: str,
    num_slots: int,
    batch_size: int,
    C: int,
    L: int,
    force_all: bool,
    seed: int = 0,
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ✅ fixed_topology=True：同 run 內拓樸固定
    env = SionnaCellFreeUplinkEnv(num_ap=8, num_ue=64, num_rb=16, debug=False, fixed_topology=True)
    env.reset_pf_state()
    env.reset_topology()  # ✅ 明確固定一次 UE 位置/距離

    U = env.U
    K = env.K

    service_counts = np.zeros(U, dtype=np.int64)
    ue_rate_sum = np.zeros(U, dtype=np.float64)
    sum_rate_list = []

    if force_all and (K * L < U):
        raise ValueError(f"force_all=True 但容量不足：K*L={K*L} < U={U}")

    t0 = time.time()
    for _ in range(num_slots):
        out = env.step(
            batch_size=batch_size,
            C=C,
            L=L,
            scheduler=scheduler,
            force_all=force_all,
            p_uk=None
        )

        sr = float(out["sum_rate"].numpy())
        sum_rate_list.append(sr)

        x = out["x"].numpy()       # [B,U,K]
        rate = out["rate"].numpy() # [B,U,K]

        active = (np.sum(x, axis=2) > 0).astype(np.int32)  # [B,U]
        service_counts += np.sum(active, axis=0)

        ue_rate_sum += np.sum(np.sum(rate, axis=2), axis=0) / float(batch_size)

    elapsed = time.time() - t0

    avg_sum_rate = float(np.mean(sum_rate_list))
    ue_avg_rate = ue_rate_sum / float(num_slots)  # [U] bps

    fairness = jain_fairness(ue_avg_rate)
    service_prob = float(np.mean(service_counts / float(num_slots)))

    # ✅ 新增：分位數（論文常用）
    p5 = float(np.percentile(ue_avg_rate, 5))
    p50 = float(np.percentile(ue_avg_rate, 50))
    p95 = float(np.percentile(ue_avg_rate, 95))

    return {
        "scheduler": scheduler,
        "num_slots": num_slots,
        "batch_size": batch_size,
        "C": C,
        "L": L,
        "force_all": int(force_all),
        "avg_sum_rate_bps": avg_sum_rate,
        "jain_fairness": fairness,
        "service_prob": service_prob,
        "p5_rate_bps": p5,
        "p50_rate_bps": p50,
        "p95_rate_bps": p95,
        "elapsed_sec": elapsed,
        "service_counts": service_counts,
        "ue_avg_rate": ue_avg_rate,
    }


def save_csv_row(csv_path: str, row: dict, fieldnames: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in fieldnames})


def main():
    num_slots = 1000
    batch_size = 1

    C_list = [1, 2, 3, 4, 5]
    L_list = [1]  # ✅ 先固定 L=1 做 cell-free 協作增益（最推薦）
    schedulers = ["random", "gain", "pf"]
    force_all = False

    out_dir = "outputs"
    csv_path = os.path.join(out_dir, "baseline_summary.csv")
    detail_dir = os.path.join(out_dir, "details")

    summary_fields = [
        "scheduler", "num_slots", "batch_size", "C", "L", "force_all",
        "avg_sum_rate_bps", "jain_fairness", "service_prob",
        "p5_rate_bps", "p50_rate_bps", "p95_rate_bps",
        "elapsed_sec"
    ]

    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print("=== Baseline Evaluation (fixed topology) ===")
    print(f"slots={num_slots}, batch={batch_size}, force_all={force_all}")
    print(f"C_list={C_list}, L_list={L_list}, schedulers={schedulers}")
    print("----------------------------------------")

    for C in C_list:
        for L in L_list:
            for sch in schedulers:
                res = run_eval(
                    scheduler=sch,
                    num_slots=num_slots,
                    batch_size=batch_size,
                    C=C,
                    L=L,
                    force_all=force_all,
                    seed=1234,
                )

                print(
                    f"[C={C} L={L} {sch:6s}] "
                    f"SR={res['avg_sum_rate_bps']:.3f} | "
                    f"Jain={res['jain_fairness']:.4f} | "
                    f"p5={res['p5_rate_bps']:.3f} | "
                    f"ServProb={res['service_prob']:.4f} | "
                    f"time={res['elapsed_sec']:.1f}s"
                )

                save_csv_row(csv_path, res, summary_fields)

                os.makedirs(detail_dir, exist_ok=True)
                np.save(os.path.join(detail_dir, f"service_counts_C{C}_L{L}_{sch}.npy"), res["service_counts"])
                np.save(os.path.join(detail_dir, f"ue_avg_rate_C{C}_L{L}_{sch}.npy"), res["ue_avg_rate"])

    print("----------------------------------------")
    print(f"✅ Summary saved to: {csv_path}")
    print(f"✅ Details saved to: {detail_dir}/")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
