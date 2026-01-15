# eval_baselines.py
import os
import csv
import time
import numpy as np
import tensorflow as tf

from channel_env import SionnaCellFreeUplinkEnv


def jain_fairness(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jain's fairness index for non-negative vector x
    J = (sum x)^2 / (n * sum x^2)
    """
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
    debug_print_every: int = 0,
):
    """
    回傳：
      avg_sum_rate_bps
      fairness_jain (用 UE 長期平均 throughput 算)
      service_prob (UE 被排到的比例的平均)
      service_counts (每個 UE 被排到幾次)
      ue_rate_sum (每個 UE 累積總速率 bps-slot)
    """
    # 固定一些 randomness（可重現）
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = SionnaCellFreeUplinkEnv(num_ap=8, num_ue=64, num_rb=16, debug=False)
    env.reset_pf_state()  # PF 狀態重置

    U = env.U
    K = env.K

    service_counts = np.zeros(U, dtype=np.int64)
    ue_rate_sum = np.zeros(U, dtype=np.float64)
    sum_rate_list = []

    # 若 force_all=True，容量必須夠：K*L >= U
    if force_all and (K * L < U):
        raise ValueError(f"force_all=True 但容量不足：K*L={K*L} < U={U}")

    t0 = time.time()
    for t in range(num_slots):
        out = env.step(
            batch_size=batch_size,
            C=C,
            L=L,
            scheduler=scheduler,
            force_all=force_all,
            p_uk=None
        )

        # sum-rate：scalar tf
        sr = float(out["sum_rate"].numpy())
        sum_rate_list.append(sr)

        # x: [B,U,K], rate: [B,U,K]
        x = out["x"].numpy()
        rate = out["rate"].numpy()

        # UE 是否被排到（在任一 RB 有傳）
        active = (np.sum(x, axis=2) > 0).astype(np.int32)  # [B,U]
        service_counts += np.sum(active, axis=0)

        # UE 的 slot throughput：sum_k rate[b,u,k]
        ue_rate_sum += np.sum(np.sum(rate, axis=2), axis=0) / float(batch_size)

        if debug_print_every and ((t + 1) % debug_print_every == 0):
            print(f"[{scheduler}] slot {t+1}/{num_slots} | SR={sr:.3f} bps")

    elapsed = time.time() - t0

    avg_sum_rate = float(np.mean(sum_rate_list))
    ue_avg_rate = ue_rate_sum / float(num_slots)  # UE 長期平均 throughput（bps）
    fairness = jain_fairness(ue_avg_rate)
    service_prob = float(np.mean(service_counts / float(num_slots)))

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
    # ---- 實驗參數 ----
    num_slots = 1000
    batch_size = 1

    # ✅ 這裡改成 sweep C：包含 no cooperation (C=1)
    C_list = [1, 2, 3, 4, 5]

    # 你原本的 L sweep 保留
    L_list = [1, 2, 3, 4, 5]

    # baseline schedulers
    schedulers = ["random", "gain", "pf"]

    # 現實設定：不強迫每個 UE 每 slot 都要傳
    force_all = False

    # 輸出位置
    out_dir = "outputs"
    csv_path = os.path.join(out_dir, "baseline_summary.csv")
    detail_dir = os.path.join(out_dir, "details")

    summary_fields = [
        "scheduler", "num_slots", "batch_size", "C", "L", "force_all",
        "avg_sum_rate_bps", "jain_fairness", "service_prob", "elapsed_sec"
    ]

    # ✅ 避免重跑一直 append：每次 main() 先清掉舊 CSV
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print("=== Baseline Evaluation ===")
    print(f"slots={num_slots}, batch={batch_size}, force_all={force_all}")
    print(f"C_list={C_list}, L_list={L_list}, schedulers={schedulers}")
    print("----------------------------------------")

    # 用 env 讀 K、U，避免寫死
    tmp_env = SionnaCellFreeUplinkEnv(num_ap=8, num_ue=64, num_rb=16, debug=False)
    U = tmp_env.U
    K = tmp_env.K
    del tmp_env

    for C in C_list:
        for L in L_list:
            for sch in schedulers:
                # 若 force_all=True，必須 K*L >= U
                if force_all and (K * L < U):
                    print(f"Skip {sch} at C={C},L={L} because force_all requires K*L >= U")
                    continue

                res = run_eval(
                    scheduler=sch,
                    num_slots=num_slots,
                    batch_size=batch_size,
                    C=C,
                    L=L,
                    force_all=force_all,
                    seed=1234,
                    debug_print_every=0
                )

                print(
                    f"[C={C} L={L} {sch:6s}] "
                    f"SR={res['avg_sum_rate_bps']:.3f} bps | "
                    f"Jain={res['jain_fairness']:.4f} | "
                    f"ServProb={res['service_prob']:.4f} | "
                    f"time={res['elapsed_sec']:.1f}s"
                )

                # summary csv
                save_csv_row(csv_path, res, summary_fields)

                # 每個 UE 詳細：service_counts / ue_avg_rate 存成 npy
                os.makedirs(detail_dir, exist_ok=True)
                np.save(os.path.join(detail_dir, f"service_counts_C{C}_L{L}_{sch}.npy"), res["service_counts"])
                np.save(os.path.join(detail_dir, f"ue_avg_rate_C{C}_L{L}_{sch}.npy"), res["ue_avg_rate"])

    print("----------------------------------------")
    print(f"✅ Summary saved to: {csv_path}")
    print(f"✅ Details saved to: {detail_dir}/ (npy files)")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
