# data_generation.py
import os
import numpy as np
from tqdm import tqdm
from utils import SimConfig, set_seed, assert_feasible
from wirelessNetwork import CellFreeNetwork
from baseline_solver import teacher_ao

def generate_dataset(out_path: str, drops: int = 200, cfg: SimConfig = SimConfig()):
    assert_feasible(cfg)
    set_seed(cfg.seed)
    
    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 初始化網路
    net = CellFreeNetwork(cfg)

    # 準備 list 儲存數據
    betas, pilots, As, Ss, ps, srs = [], [], [], [], [], []
    pos_ues, pos_aps = [], [] 

    print(f"Generating dataset with {drops} drops...")
    print(f"Config: {cfg.n_ap} APs, {cfg.n_ue} UEs, {cfg.M} Antennas, {cfg.fft_size} Subcarriers")
    
    # 使用 tqdm 顯示進度
    for _ in tqdm(range(drops), desc="Generating teacher dataset"):
        # 1. 產生物理層數據 (含 OFDM 通道)
        # H, Hhat shape: [A, U, M, F]
        ue_pos, ap_pos, beta, pilot_id, H, Hhat = net.generate_drop(pilot_method="greedy")
        
        # 2. [關鍵修正] 取出中央子載波給 Solver
        F = H.shape[-1]
        center_f = F // 2
        
        H_solver = H[:, :, :, center_f]        # [A, U, M]
        Hhat_solver = Hhat[:, :, :, center_f]  # [A, U, M]

        # 3. 執行 Teacher Algorithm (AO + WMMSE)
        # 這裡傳入 3維矩陣，才不會報錯
        A_mat, S, p, _C, _n_eff, sr = teacher_ao(beta, pilot_id, H_solver, Hhat_solver, cfg)

        # 4. 收集數據
        betas.append(beta.astype(np.float32))
        pilots.append(pilot_id.astype(np.int32))
        As.append(A_mat.astype(np.int32))
        Ss.append(S.astype(np.int32))
        ps.append(p.astype(np.float32))
        srs.append(float(sr))
        
        # 儲存座標 (對 GNN 理解空間關係很有幫助)
        pos_ues.append(ue_pos.astype(np.float32))
        pos_aps.append(ap_pos.astype(np.float32))

    # 5. 轉成 Numpy Array 並存檔
    print(f"Stacking and saving to {out_path}...")
    np.savez_compressed(out_path,
             beta=np.stack(betas),         # [N,A,U] - Channel Large Scale Fading (GNN Input)
             pilot=np.stack(pilots),       # [N,U]   - Pilot Indices
             A=np.stack(As),               # [N,U,A] - Clustering Matrix
             S=np.stack(Ss),               # [N,U,K] - Scheduling Matrix
             p=np.stack(ps),               # [N,U]   - Optimal Power (GNN Label)
             sum_rate=np.array(srs, dtype=np.float32),
             ue_pos=np.stack(pos_ues),     # [N,U,3] - UE Positions
             ap_pos=np.stack(pos_aps)      # [N,A,3] - AP Positions
    )
             
    print(f"Saved successfully!")
    print(f"Avg SR: {np.mean(srs):.3f} ± {np.std(srs):.3f} bits/s/Hz")

if __name__ == "__main__":
    # 測試產生數據
    # 建議先產生個 100 筆 train data 和 20 筆 test data
    cfg = SimConfig()
    
    # 1. 產生訓練集
    generate_dataset("./cf_data/data_train.npz", drops=100, cfg=cfg)
    
    # 2. 產生測試集 (用不同的 seed 或繼續跑)
    # cfg.seed = 9999
    # generate_dataset("./cf_data/data_test.npz", drops=20, cfg=cfg)