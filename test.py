import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch.nn.functional as F
import torch


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


def load_model(filepath="D:\Programs\Student Dropout\Proper\model.pt"):
    checkpoint = torch.load(filepath)
    model = GATModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        heads=checkpoint["heads"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def visualize_graph(graph_data, factor_values, factor_names):
    G = nx.DiGraph()

    for i in range(len(factor_names)):
        G.add_node(i, value=factor_values[i], name=factor_names[i])

    edge_index = graph_data.edge_index.numpy()
    edge_attr = graph_data.edge_attr.numpy()

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        weight = float(edge_attr[i])
        G.add_edge(src, dst, weight=weight)

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G)

    node_colors = [G.nodes[node]["value"] for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=2000, cmap=plt.cm.YlOrRd
    )

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        width=2,
        edge_vmin=min(weights),
        edge_vmax=max(weights),
        arrowsize=20,
    )

    labels = {
        i: f"{factor_names[i]}\n({factor_values[i]:.2f})"
        for i in range(len(factor_names))
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    plt.colorbar(nodes, label="Factor Values")

    plt.title("Graph Structure of Student Factors")
    plt.axis("off")
    plt.tight_layout()

    return plt.gcf()


def predict_dropout_risk(
    model_path="D:\Programs\Student Dropout\Proper\model.pt",
    scaler_path="scaler.joblib",
):
    model = load_model(model_path)
    model.eval()

    factors = [
        "Financial and domestic constraints",
        "Poor teaching quality",
        "Excessive workload",
        "Academic performance and other institutional factors",
    ]

    dematel_scores = np.array(
        [[0, 0.8, 0.7, 0.6], [0.5, 0, 0.7, 0.6], [0.4, 0.5, 0, 0.8], [0.6, 0.4, 0.7, 0]]
    )

    dematel_scores = (dematel_scores - dematel_scores.min()) / (
        dematel_scores.max() - dematel_scores.min()
    )

    print("\nPlease enter values between 0 and 1 for each factor:")
    user_values = []
    for factor in factors:
        while True:
            try:
                value = float(input(f"{factor}: "))
                if 0 <= value <= 1:
                    user_values.append(value)
                    break
                else:
                    print("Please enter a value between 0 and 1")
            except ValueError:
                print("Please enter a valid number")

    user_df = pd.DataFrame([user_values], columns=factors)

    try:
        scaler = joblib.load(scaler_path)
    except:
        print("Warning: Scaler not found. Using raw values.")
        scaler = None

    if scaler:
        user_values_scaled = scaler.transform(user_df)[0]
    else:
        user_values_scaled = user_values

    def create_single_graph(features):
        num_factors = len(factors)

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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Create custom graph and visualize
    user_graph = create_single_graph(user_values_scaled)
    plt.figure(figsize=(10, 8))
    fig = visualize_graph(user_graph, user_values, factors)
    plt.show()

    # Make predictions
    with torch.no_grad():
        model.eval()
        out = model(user_graph)
        probabilities = torch.exp(out)
        prediction = out.argmax(dim=1)

    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Dropout Risk: {'High' if prediction.item() == 1 else 'Low'}")
    print(f"Confidence Scores:")
    print(f"No Dropout Probability: {probabilities[0][0]:.2%}")
    print(f"Dropout Probability: {probabilities[0][1]:.2%}")

    # Provide factor-specific insights
    print("\nFactor Analysis:")
    print("-" * 50)
    for factor, value in zip(factors, user_values):
        risk_level = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
        print(f"{factor}: {risk_level} ({value:.2%})")

    return prediction.item(), probabilities.numpy()


def save_scaler(scaler, filepath="scaler.joblib"):
    """
    Save the StandardScaler used during training
    """
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")


if __name__ == "__main__":
    while True:
        prediction, probabilities = predict_dropout_risk()

        again = input("\nWould you like to make another prediction? (yes/no): ")
        if again.lower() != ["yes", "Yes"]:
            break
