# sweep_comparison.py
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import replace

# 引用您現有的模組
from utils import SimConfig, assert_feasible
from wirelessNetwork import CellFreeNetwork
from baseline_solver import teacher_ao

def run_comparison_drop(cfg: SimConfig, drops: int, method_type: str):
    """
    執行指定模式的模擬。
    method_type 支援: "greedy", "random", "perfect"
    """
    if method_type == "perfect":
        gen_pilot_method = "greedy"
    else:
        gen_pilot_method = method_type

    # assert_feasible(cfg) # 可註解掉以減少輸出干擾
    
    net = CellFreeNetwork(cfg)
    srs = []
    
    for i in range(drops):
        _ue_pos, _ap_pos, beta, pilot_id, H, Hhat = net.generate_drop(pilot_method=gen_pilot_method)
        
        if method_type == "perfect":
            H_input = H 
        else:
            H_input = Hhat

        _A, _S, _p, _C, _n, sr = teacher_ao(beta, pilot_id, H, H_input, cfg)
        srs.append(float(sr))
        
    return np.mean(srs), np.std(srs)

def main():
    parser = argparse.ArgumentParser()
    # 掃描參數設定
    parser.add_argument("--vary", type=str, default="n_ap", 
                        choices=["n_ap", "tau_p", "M", "n_ue"], help="要掃描的參數")
    parser.add_argument("--values", type=float, nargs="+", 
                        default=[16, 32, 64, 100], help="參數值列表")
    
    # 實驗設定
    parser.add_argument("--drops", type=int, default=30, help="每個點跑幾次 drop")
    parser.add_argument("--seed", type=int, default=1234, help="隨機種子")
    
    # 輸出檔案
    parser.add_argument("--out_csv", type=str, default="results/comparison_3way.csv")
    parser.add_argument("--out_png", type=str, default="results/comparison_3way.png")

    # 允許覆寫其他固定參數
    parser.add_argument("--tau_p", type=int, default=None, help="覆寫導頻長度")
    parser.add_argument("--n_ue", type=int, default=None, help="覆寫用戶數量")
    parser.add_argument("--M", type=int, default=None, help="覆寫天線數量")
    parser.add_argument("--K", type=int, default=None, help="覆寫 RB 數量")

    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    base_cfg = SimConfig(seed=args.seed)

    if args.tau_p is not None: base_cfg = replace(base_cfg, tau_p=args.tau_p)
    if args.n_ue is not None: base_cfg = replace(base_cfg, n_ue=args.n_ue)
    if args.M is not None: base_cfg = replace(base_cfg, M=args.M)
    if args.K is not None: base_cfg = replace(base_cfg, K=args.K)
    
    results = {
        args.vary: [],
        "Random_Mean": [], "Random_Std": [],
        "Greedy_Mean": [], "Greedy_Std": [],
        "Perfect_Mean": [], "Perfect_Std": []
    }

    print(f"=== 3-Way Comparison: Random vs Greedy vs Perfect CSI (Varying {args.vary}) ===")
    print(f"Drops: {args.drops} | tau_p: {base_cfg.tau_p} | n_ue: {base_cfg.n_ue}")

    for val in args.values:
        if args.vary in ["n_ap", "tau_p", "M", "n_ue"]:
            val = int(val)
            
        cfg = replace(base_cfg, **{args.vary: val})
        print(f"\n[Testing] {args.vary} = {val}")
        
        # 1. Random
        r_mean, r_std = run_comparison_drop(cfg, args.drops, "random")
        results["Random_Mean"].append(r_mean)
        results["Random_Std"].append(r_std)
        print(f"    [Gray]  Random : {r_mean:.2f}")

        # 2. Greedy
        g_mean, g_std = run_comparison_drop(cfg, args.drops, "greedy")
        results["Greedy_Mean"].append(g_mean)
        results["Greedy_Std"].append(g_std)
        print(f"    [Blue]  Greedy : {g_mean:.2f}")

        # 3. Perfect
        p_mean, p_std = run_comparison_drop(cfg, args.drops, "perfect")
        results["Perfect_Mean"].append(p_mean)
        results["Perfect_Std"].append(p_std)
        print(f"    [Green] Perfect: {p_mean:.2f}")
        
        results[args.vary].append(val)

    # 存檔
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nData saved to {args.out_csv}")

    # --- 繪圖 ---
    plt.figure(figsize=(10, 7))
    x = df[args.vary]
    
    # 畫線
    plt.errorbar(x, df["Perfect_Mean"], yerr=df["Perfect_Std"],
                 fmt='-^', color='forestgreen', linewidth=2, capsize=5, 
                 label='Perfect CSI (Upper Bound)')
    
    plt.errorbar(x, df["Greedy_Mean"], yerr=df["Greedy_Std"],
                 fmt='-o', color='royalblue', linewidth=2, capsize=5, 
                 label='Greedy Pilot (Proposed)')
    
    plt.errorbar(x, df["Random_Mean"], yerr=df["Random_Std"],
                 fmt='--s', color='gray', linewidth=2, capsize=5, alpha=0.8,
                 label='Random Pilot (Weak Baseline)')

    plt.xlabel(f"Parameter: {args.vary}", fontsize=12)
    plt.ylabel("Sum Spectral Efficiency (bps/Hz)", fontsize=12)
    plt.title(f"Performance Comparison: {args.vary} sweep\n(tau_p={base_cfg.tau_p}, U={base_cfg.n_ue})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # === [修改處] 標註所有數值 ===
    
    # 1. Perfect (綠色): 標註在上方
    for i, val in enumerate(df["Perfect_Mean"]):
        plt.annotate(f"{val:.1f}", (x[i], val), xytext=(0, 8), 
                     textcoords='offset points', ha='center', va='bottom',
                     color='forestgreen', fontweight='bold', fontsize=9)

    # 2. Greedy (藍色): 標註在上方 (與綠色稍微錯開，或依賴數值差距)
    for i, val in enumerate(df["Greedy_Mean"]):
        plt.annotate(f"{val:.1f}", (x[i], val), xytext=(0, 5), 
                     textcoords='offset points', ha='center', va='bottom',
                     color='royalblue', fontweight='bold', fontsize=9)

    # 3. Random (灰色): 標註在下方 (避免與藍色重疊)
    for i, val in enumerate(df["Random_Mean"]):
        plt.annotate(f"{val:.1f}", (x[i], val), xytext=(0, -15), 
                     textcoords='offset points', ha='center', va='top',
                     color='gray', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Plot saved to {args.out_png}")

if __name__ == "__main__":
    main()