import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch.nn.functional as F
import torch

# Load dataset
file_path = "D:\\Programs\\Student Dropout\\dataSet.xlsx"
df = pd.read_excel(file_path)

# Factors and target
selected_factors = [
    "Financial and domestic constraints",
    "Poor teaching quality",
    "Excessive workload",
    "Academic performance and other institutional factors",
]
target = "Cluster"

# DEMATEL scores
dematel_scores = np.array(
    [[0, 0.8, 0.7, 0.6], [0.5, 0, 0.7, 0.6], [0.4, 0.5, 0, 0.8], [0.6, 0.4, 0.7, 0]]
)

dematel_scores = (dematel_scores - dematel_scores.min()) / (
    dematel_scores.max() - dematel_scores.min()
)

# Data preprocessing
scaler = StandardScaler()
df[selected_factors] = scaler.fit_transform(df[selected_factors])


def create_student_graph(features, label):
    num_factors = len(selected_factors)

    edge_index = []
    edge_attr = []
    for i in range(num_factors):
        for j in range(num_factors):
            if dematel_scores[i][j] > 0.2:
                edge_index.append([i, j])
                edge_attr.append([dematel_scores[i][j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    node_features = []
    for i, value in enumerate(features):
        position_encoding = [np.sin(i), np.cos(i)]
        node_features.append([value] + position_encoding)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# Graph creation
node_features = df[selected_factors].values
labels = df[target].values
graphs = [create_student_graph(node_features[i], labels[i]) for i in range(len(labels))]

train_idx, test_idx = train_test_split(
    range(len(graphs)), test_size=0.2, stratify=labels, random_state=42
)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
)

train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=32, shuffle=True)
val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=32)
test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=32)


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout
        )
        self.gat3 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout
        )
        self.gcn = GCNConv(hidden_dim * heads, hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x1 = self.gat1(x, edge_index, edge_attr)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.gat3(x2, edge_index, edge_attr)
        x3 = F.elu(x3)
        x = self.gcn(x3, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


model = GATModel(
    input_dim=3,
    hidden_dim=32,
    output_dim=2,
    heads=4,
    dropout=0.3,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        total_loss += loss.item()

    return total_loss / len(train_loader), correct / total


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total


# Training loop
best_val_acc = 0
patience = 10
patience_counter = 0
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train()
    val_acc = evaluate(val_loader)

    # Learning Rate schedule
    scheduler.step(val_acc)

    # Early stop
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    print(
        f"Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
    )

model.load_state_dict(torch.load("best_model.pt"))
test_acc = evaluate(test_loader)
print(f"Final Test Accuracy: {test_acc*100:.4f}")


# Save the model
def save_model(model, filepath="model.pt"):
    """
    Save the trained model
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_architecture": type(model).__name__,
            "input_dim": model.gat1.in_channels,
            "hidden_dim": model.gat1.out_channels,
            "output_dim": model.lin2.out_features,
            "heads": model.gat1.heads,
        },
        filepath,
    )
    print(f"Model saved to {filepath}")


save_model(model)
