# baseline_v2.py
import os
import csv
import time
import numpy as np
import tensorflow as tf

from channel_env_v2 import SionnaCellFreeUplinkEnvV2


def jain_fairness(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.maximum(x, 0.0)
    s1 = float(np.sum(x))
    s2 = float(np.sum(x * x))
    return (s1 * s1) / (x.size * s2 + eps)


def run_eval(env, scheduler: str, num_slots: int, C: int, L: int, enable_power_control: bool):
    env.reset_pf_state()

    sum_rate_list = []
    ue_rate_accum = np.zeros(env.U, dtype=np.float64)

    t0 = time.time()
    for _ in range(num_slots):
        out = env.step(
            batch_size=1,
            C=C,
            L=L,
            scheduler=scheduler,
            enable_power_control=enable_power_control
        )

        sum_rate_list.append(float(out["sum_rate"].numpy()))

        # rate: [1,U,K] -> sum over K -> [U]
        r_inst = np.sum(out["rate"].numpy(), axis=2)[0]
        ue_rate_accum += r_inst

    elapsed = time.time() - t0

    avg_sum_rate = np.mean(sum_rate_list)
    ue_avg = ue_rate_accum / num_slots

    rate_5ile = np.percentile(ue_avg, 5)
    rate_50ile = np.percentile(ue_avg, 50)

    fairness = jain_fairness(ue_avg)

    return {
        "scheduler": scheduler,
        "C": C,
        "L": L,
        "PC": int(enable_power_control),
        "avg_sum_rate_Mbps": avg_sum_rate / 1e6,
        "rate_5ile_Mbps": rate_5ile / 1e6,
        "rate_50ile_Mbps": rate_50ile / 1e6,
        "jain_fairness": fairness,
        "elapsed_sec": elapsed,
        "ue_rates": ue_avg,
    }


def main():
    # ✅ 你 GTX1060 + 3GPP 建議先小規模跑通再放大
    num_slots = 200          # 3GPP 慢，先 200 slots 確認能跑
    C = 3
    L = 2

    schedulers = ["random", "gain", "pf", "optimal"]
    pc_modes = [False, True]

    # ✅ 3GPP 開關
    use_3gpp = True

    out_dir = "outputs_v2"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    fields = ["scheduler", "C", "L", "PC",
              "avg_sum_rate_Mbps", "rate_5ile_Mbps", "rate_50ile_Mbps",
              "jain_fairness", "elapsed_sec"]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"Initializing Env (3GPP={use_3gpp})...")

    env = SionnaCellFreeUplinkEnvV2(
        num_ap=8,
        num_ue=32,
        num_rb=8,
        enable_3gpp=use_3gpp,
        debug=False
    )

    print(f"{'Sch':10s} | {'PC':3s} | {'SumRate(Mbps)':>13s} | {'5%-tile(Mbps)':>13s} | {'Jain':>6s} | {'Time':>6s}")
    print("-" * 78)

    for sch in schedulers:
        for pc in pc_modes:
            res = run_eval(env, sch, num_slots, C=C, L=L, enable_power_control=pc)

            print(f"{sch:10s} | {str(pc):3s} | "
                  f"{res['avg_sum_rate_Mbps']:13.2f} | "
                  f"{res['rate_5ile_Mbps']:13.2f} | "
                  f"{res['jain_fairness']:.4f} | "
                  f"{res['elapsed_sec']:.1f}s")

            row = {k: res[k] for k in fields}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            np.save(os.path.join(out_dir, f"rates_{sch}_PC{int(pc)}.npy"), res["ue_rates"])

    print(f"\nDone! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
