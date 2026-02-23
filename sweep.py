# sweep.py
import argparse
import os
import csv
from dataclasses import replace

from utils import SimConfig, assert_feasible
from runner import run_baseline

# 修改 1: 在 ALLOWED 集合中加入 "n_ap"
ALLOWED = {"Q", "M", "tau_p", "TopL", "Ca", "K", "ao_iters", "wmmse_iters", "n_ap"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vary", type=str, required=True, help=f"要 sweep 的參數（支援：{sorted(ALLOWED)}）")
    p.add_argument("--values", type=float, nargs="+", required=True, help="要 sweep 的值（空白分隔）")
    p.add_argument("--drops", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default="results/sweep.csv")
    p.add_argument("--no_progress", action="store_true")
    args = p.parse_args()

    vary = args.vary
    if vary not in ALLOWED:
        raise ValueError(f"--vary={vary} 不支援。可用：{sorted(ALLOWED)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base_cfg = SimConfig(seed=args.seed)

    rows = []
    for i, val in enumerate(args.values):
        # 修改 2: 確保 n_ap 也會被正確轉為整數 (int)
        if vary in {"Q", "M", "tau_p", "TopL", "Ca", "K", "ao_iters", "wmmse_iters", "n_ap"}:
            val = int(val)

        cfg = replace(base_cfg, **{vary: val})

        # 若你想每個 sweep value 用不同 seed（避免完全相同 drop），可改成：
        # cfg = replace(cfg, seed=args.seed + 1000*i)
        assert_feasible(cfg)

        print("\n==============================================")
        print(f"[SWEEP] {vary} = {val} | drops={args.drops} | seed={cfg.seed}")
        print(f" A={cfg.n_ap}, U={cfg.n_ue}, M={cfg.M}, K={cfg.K}, Q={cfg.Q}")
        print(f" tau_p={cfg.tau_p}, TopL={cfg.TopL}, Ca={cfg.Ca}")
        print("==============================================")

        mean_sr, std_sr, elapsed, _ = run_baseline(cfg, args.drops, show_progress=(not args.no_progress))

        row = {
            "vary": vary,
            "value": val,
            "drops": args.drops,
            "seed": cfg.seed,
            "A": cfg.n_ap,
            "U": cfg.n_ue,
            "M": cfg.M,
            "K": cfg.K,
            "Q": cfg.Q,
            "TopL": cfg.TopL,
            "Ca": cfg.Ca,
            "tau_c": cfg.tau_c,
            "tau_p": cfg.tau_p,
            "ao_iters": cfg.ao_iters,
            "wmmse_iters": cfg.wmmse_iters,
            "mean_sr": mean_sr,
            "std_sr": std_sr,
            "elapsed_sec": elapsed,
        }
        rows.append(row)

        print(f"[DONE] mean_sr={mean_sr:.3f}, std_sr={std_sr:.3f}, time={elapsed:.1f}s")

    # write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n================= SWEEP FINISHED =================")
    print(f"Saved CSV -> {args.out}")
    print("==================================================\n")


if __name__ == "__main__":
    main()