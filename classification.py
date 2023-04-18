import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class QuboGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(QuboGNN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        x = self.fc(x)
        return x
