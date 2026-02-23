# testing.py
import argparse
from dataclasses import replace

from utils import SimConfig, set_seed, assert_feasible
from runner import run_baseline

def main():
    p = argparse.ArgumentParser(description="Cell-Free Network Simulation")
    p.add_argument("--drops", type=int, default=30, help="Number of Monte-Carlo drops")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")

    # [建議新增] 讓測試更方便的參數
    p.add_argument("--n_ap", type=int, default=None, help="Number of APs")
    p.add_argument("--n_ue", type=int, default=None, help="Number of UEs")

    # 其他參數覆寫
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

    # 覆寫參數
    # 加入 n_ap 和 n_ue 到覆寫清單
    override_keys = [
        "n_ap", "n_ue", "M", "Q", "K", "tau_p", "TopL", "Ca", 
        "ao_iters", "wmmse_iters"
    ]
    
    kw = {}
    for name in override_keys:
        if hasattr(args, name):
            v = getattr(args, name)
            if v is not None:
                kw[name] = v
    if kw:
        cfg = replace(cfg, **kw)

    assert_feasible(cfg)
    set_seed(cfg.seed)

    print("=== Advanced Baseline (AO+WMMSE) Settings ===")
    print(f"AP={cfg.n_ap}, UE={cfg.n_ue}, M={cfg.M}")
    print(f"RBs(K)={cfg.K}, Reuse(Q)={cfg.Q}")
    print(f"Cluster(TopL)={cfg.TopL}, LoadLimit(Ca)={cfg.Ca}")
    print(f"Pilots(tau_p)={cfg.tau_p}")
    print(f"AO iters={cfg.ao_iters}, WMMSE iters={cfg.wmmse_iters}")
    print("=============================================")

    mean_sr, std_sr, elapsed, _ = run_baseline(cfg, args.drops, show_progress=True)

    print("\n========== Results ==========")
    print(f"Drops: {args.drops}")
    print(f"Sum-rate (Prelog applied): {mean_sr:.3f} ± {std_sr:.3f} bits/s/Hz")
    print(f"Total Time: {elapsed:.1f} s")
    print("=============================\n")


if __name__ == "__main__":
    main()