# runner.py
import time
import numpy as np
from tqdm import tqdm

from utils import SimConfig, set_seed, assert_feasible
from wirelessNetwork import CellFreeNetwork
from baseline_solver import teacher_ao

def run_baseline(cfg: SimConfig, drops: int, show_progress: bool = True):
    assert_feasible(cfg)
    set_seed(cfg.seed)

    net = CellFreeNetwork(cfg)

    srs = []
    t0 = time.time()
    
    it = range(drops)
    if show_progress:
        # 顯示更詳細的進度條資訊
        it = tqdm(it, desc=f"Simulating drops (seed={cfg.seed})", unit="drop")

    for _ in it:
        # 1. 產生 Drop (物理層是 OFDM)
        # H, Hhat shape: [A, U, M, F] (F=72)
        _ue_pos, _ap_pos, beta, pilot_id, H, Hhat = net.generate_drop(pilot_method="greedy")
        
        # 2. [關鍵修正] 維度適配
        # Solver (teacher_ao) 只吃 [A, U, M] (Narrowband/Flat Fading)。
        # 我們取出「中央子載波」來做資源分配優化。
        F = H.shape[-1]
        center_f = F // 2
        
        H_solver = H[:, :, :, center_f]        # [A, U, M]
        Hhat_solver = Hhat[:, :, :, center_f]  # [A, U, M]

        # 3. 執行 Solver (AO + WMMSE)
        # 這裡傳入的是 3維矩陣，不會報錯
        _A, _S, _p, _C, _n, sr = teacher_ao(beta, pilot_id, H_solver, Hhat_solver, cfg)
        
        srs.append(float(sr))

    elapsed = time.time() - t0
    srs = np.array(srs, dtype=np.float32)
    
    # 回傳平均值、標準差、總時間、所有數據
    return float(srs.mean()), float(srs.std()), float(elapsed), srs.tolist()