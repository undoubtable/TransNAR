# models/nar.py

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

class NARGNN(nn.Module):
    """
    简单 GNN 作为 NAR:
    - 输入: x, edge_index, batch (PyG Data)
    - 输出: node_logits, node_emb
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            ) for _ in range(num_layers)
        ])

        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes]
        """
        h = self.embed(x)
        for conv in self.convs:
            h = conv(h, edge_index)

        node_logits = self.readout(h)
        return node_logits, h   # h 给 TransNAR 用
