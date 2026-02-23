import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CellFreeGNN(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=32):
        super(CellFreeGNN, self).__init__()
        # 圖卷積層：學習通道特徵與拓撲
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # 輸出層：將隱藏特徵映射到 1 維的發射功率 (Power)
        self.out_layer = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        
        out = self.out_layer(x)
        # 功率 p 通常在 0 到 1 之間或大於 0，這裡用 Sigmoid 將預測限縮在 (0, 1)
        return torch.sigmoid(out)