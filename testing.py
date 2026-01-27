# testing.py
import argparse
from dataclasses import replace

from utils import SimConfig, set_seed, assert_feasible
from runner import run_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drops", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)

    # 允許覆寫常用參數
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--Q", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--tau_p", type=int, default=None)
    p.add_argument("--TopL", type=int, default=None)
    p.add_argument("--Ca", type=int, default=None)

    # AO/WMMSE
    p.add_argument("--ao_iters", type=int, default=None)
    p.add_argument("--wmmse_iters", type=int, default=None)

    args = p.parse_args()

    cfg = SimConfig(seed=args.seed)

    # 覆寫
    kw = {}
    for name in ["M", "Q", "K", "tau_p", "TopL", "Ca", "ao_iters", "wmmse_iters"]:
        v = getattr(args, name)
        if v is not None:
            kw[name] = v
    if kw:
        cfg = replace(cfg, **kw)

    assert_feasible(cfg)
    set_seed(cfg.seed)

    print("=== Baseline V1 (6G-realistic) Settings ===")
    print(f"A={cfg.n_ap}, U={cfg.n_ue}, M={cfg.M}, K={cfg.K}, Q={cfg.Q}")
    print(f"TopL={cfg.TopL}, Ca={cfg.Ca}")
    print(f"tau_c={cfg.tau_c}, tau_p={cfg.tau_p} (pilot reuse enabled)")
    print(f"AO iters={cfg.ao_iters}, WMMSE iters={cfg.wmmse_iters}")
    print("==========================================")

    mean_sr, std_sr, elapsed, _ = run_baseline(cfg, args.drops, show_progress=True)

    print("\n========== Results ==========")
    print(f"drops={args.drops}")
    print(f"Sum-rate (prelog applied): {mean_sr:.3f} ± {std_sr:.3f}")
    print(f"elapsed: {elapsed:.1f}s")
    print("================================\n")


if __name__ == "__main__":
    main()
