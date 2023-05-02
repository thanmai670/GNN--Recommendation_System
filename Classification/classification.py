import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class QUBOGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(QUBOGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.mean(x, dim=0)
        x = self.fc(x)

        return x


self.fc = torch.nn.Linear(32, 2)


def forward(self, data):
    x, edge_index = data.x, data.edge_index

    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)

    x = self.conv2(x, edge_index)
    x = F.relu(x)

    x = torch.mean(x, dim=0)
    x = self.fc(x)

    return F.relu(
        x
    )  # The ReLU activation function ensures the output is non-negative, since you can't have a negative number of nodes or edges.


# Initialize the model, loss function, and optimizer
model = QUBOGNN(num_node_features, 2)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
for epoch in range(100):  # Loop over the dataset multiple times
    total_loss = 0
    for (
        data
    ) in train_loader:  # train_loader is a DataLoader wrapping your training dataset
        # Forward pass
        outputs = model(data)
        labels = torch.tensor(
            [data.num_nodes, data.num_edges]
        )  # You'll need to define this based on your data
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("Epoch: {}, Loss: {:.4f}".format(epoch, total_loss))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:  # test_loader is a DataLoader wrapping your test dataset
        outputs = model(data)
        labels = torch.tensor([data.num_nodes, data.num_edges])
        total += labels.size(0)
        correct += (
            ((outputs - labels).abs() < 0.5).sum().item()
        )  # A prediction is considered correct if it's within 0.5 of the true value
    print("Test Accuracy: {:.2f}%".format(100 * correct / total))
