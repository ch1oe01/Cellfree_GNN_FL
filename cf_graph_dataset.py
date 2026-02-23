import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_and_build_graphs(npz_path):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    beta = data['beta']       # [N, A, U]
    p = data['p']             # [N, U] Optimal Power
    ue_pos = data['ue_pos']   # [N, U, 3]
    ap_pos = data['ap_pos']   # [N, A, 3]

    N, A, U = beta.shape
    dataset = []

    for i in range(N):
        num_nodes = A + U
        
        # 1. 建立節點特徵矩陣 X: [Is_UE_indicator, Pos_X, Pos_Y, Pos_Z]
        x = np.zeros((num_nodes, 4), dtype=np.float32)
        x[:A, 0] = 0.0 # 0 代表 AP
        x[:A, 1:] = ap_pos[i] / 1000.0 # 座標除以 1000 進行粗略正規化
        x[A:, 0] = 1.0 # 1 代表 UE
        x[A:, 1:] = ue_pos[i] / 1000.0

        # 2. 建立邊 (Edge Index) 與權重 (Edge Weight)
        edge_indices = []
        edge_weights = []
        
        # 將 beta 轉為 dB 並進行 Min-Max 正規化 (0~1 之間)，對 GNN 訓練至關重要
        beta_db = 10 * np.log10(beta[i] + 1e-12)
        beta_norm = (beta_db - np.min(beta_db)) / (np.max(beta_db) - np.min(beta_db) + 1e-12)

        for a in range(A):
            for u in range(U):
                ue_idx = A + u
                # 無向圖：建立 AP到UE 以及 UE到AP 的雙向邊
                edge_indices.append([a, ue_idx])
                edge_indices.append([ue_idx, a])
                w = beta_norm[a, u]
                edge_weights.extend([w, w])
                
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        # 3. 建立標籤 Y (最佳功率分配 p)
        y = np.zeros((num_nodes, 1), dtype=np.float32)
        y[A:, 0] = p[i] # 只有 UE 節點有功率標籤，AP 補 0
        y_tensor = torch.tensor(y)
        
        # 4. 建立訓練遮罩 (只計算 UE 節點的 Loss)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[A:] = True
        
        graph = Data(x=torch.tensor(x), edge_index=edge_index, 
                     edge_weight=edge_weight, y=y_tensor, train_mask=train_mask)
        dataset.append(graph)
        
    return dataset

# 測試用
if __name__ == "__main__":
    ds = load_and_build_graphs("./cf_data/data_train.npz")
    print(f"Successfully built {len(ds)} graph data objects.")