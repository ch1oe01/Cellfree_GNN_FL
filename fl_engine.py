import torch
import copy
from torch_geometric.loader import DataLoader

def local_client_train(model, dataloader, epochs=5, lr=0.005):
    """
    單一 Client (邊緣節點) 進行本地 GNN 訓練
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss() # 預測功率，使用 MSE 迴歸損失
    
    total_loss = 0
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            
            # 關鍵：利用 train_mask 只對 UE 節點計算功率預測的 Loss
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    avg_loss = total_loss / (epochs * len(dataloader))
    return model.state_dict(), avg_loss

def federated_averaging(global_model, client_weights_list):
    """
    CPU 進行 FedAvg 聚合所有 Client 上傳的權重
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights_list[i][k].float() for i in range(len(client_weights_list))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model