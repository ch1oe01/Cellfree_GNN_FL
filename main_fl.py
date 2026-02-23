import torch
from torch_geometric.loader import DataLoader
from cf_graph_dataset import load_and_build_graphs
from gnn_model import CellFreeGNN
from fl_engine import local_client_train, federated_averaging
import copy

def main():
    # 1. 讀取數據
    print(">>> 步驟 1: 讀取資料並建立圖結構...")
    dataset = load_and_build_graphs("./cf_data/data_train.npz")
    
    # 2. 劃分資料給多個 FL Clients
    num_clients = 5
    data_per_client = len(dataset) // num_clients
    client_dataloaders = []
    
    for i in range(num_clients):
        # 切割 Dataset 給不同的 Client 模擬分散式資料
        client_data = dataset[i * data_per_client : (i + 1) * data_per_client]
        loader = DataLoader(client_data, batch_size=8, shuffle=True)
        client_dataloaders.append(loader)
    
    print(f"成功劃分資料給 {num_clients} 個 Clients, 每個 Client 分配 {data_per_client} 個拓撲圖。")

    # 3. 初始化全域模型
    print(">>> 步驟 2: 初始化全域 GNN 模型...")
    global_model = CellFreeGNN(in_channels=4, hidden_channels=32)
    
    # 4. 開始聯邦學習訓練迴圈
    num_fl_rounds = 20
    local_epochs = 3
    
    print(">>> 步驟 3: 開始聯邦學習 (Federated Learning) 訓練...")
    for round_idx in range(num_fl_rounds):
        client_weights = []
        round_losses = []
        
        # 模擬每個 Client 進行本地訓練
        for client_id in range(num_clients):
            # 複製當前最新的 Global Model 給 Client
            local_model = copy.deepcopy(global_model)
            
            weights, loss = local_client_train(
                model=local_model, 
                dataloader=client_dataloaders[client_id], 
                epochs=local_epochs
            )
            client_weights.append(weights)
            round_losses.append(loss)
            
        # 中央 CPU (Server) 聚合模型
        global_model = federated_averaging(global_model, client_weights)
        
        avg_round_loss = sum(round_losses) / num_clients
        print(f"[FL Round {round_idx+1:02d}/{num_fl_rounds}] | Average Client MSE Loss: {avg_round_loss:.6f}")
        
    print(">>> 聯邦學習訓練完成！你可以儲存模型進行後續推論：")
    torch.save(global_model.state_dict(), "./cf_data/global_gnn_model.pth")
    print("模型已儲存至 ./cf_data/global_gnn_model.pth")

if __name__ == "__main__":
    main()