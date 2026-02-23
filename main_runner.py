import torch
from torch_geometric.data import Data
from fl_engine import run_fl_simulation
# 假設你原本的資料生成函數如下：
# from data_generation import generate_csi_data
# from baseline_solver import get_optimal_allocation

def create_graph_from_csi(csi_matrix, optimal_labels):
    """
    這是一個概念性的轉換函數。
    你需要根據你 data_generation.py 的實際輸出張量來修改這裡。
    """
    # 建立虛擬的節點特徵和邊 (這裡以隨機資料示範)
    num_nodes = 20 # 例如 UE + AP 的總數
    x = torch.rand((num_nodes, 4)) # 每個節點 4 個特徵 (例如位置、最大功率等)
    
    # 定義圖的連接 (Edge index)，2 x 邊數
    edge_index = torch.randint(0, num_nodes, (2, 50), dtype=torch.long)
    
    # 邊的權重 (這裡帶入你的 Sionna CSI 值)
    edge_weight = torch.rand(50) 
    
    # 目標標籤 (來自 baseline_solver 的最佳分配)
    y = torch.rand((num_nodes, 1)) 
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

if __name__ == "__main__":
    print("正在生成 CSI 數據與建構圖結構...")
    # 模擬 5 個 AP (Client) 的本地資料
    num_clients = 5
    clients_data = [create_graph_from_csi(None, None) for _ in range(num_clients)]
    
    print("啟動聯邦學習訓練...")
    final_model = run_fl_simulation(clients_data, num_rounds=20)
    print("訓練完成！模型已可進行推論測試。")