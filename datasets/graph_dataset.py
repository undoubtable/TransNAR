# datasets/graph_dataset.py

from torch.utils.data import Dataset
from salsaclrs import SALSACLRSDataset
import torch

class GraphDataset(Dataset):
    """
    提供 index -> PyG Data 映射，并确保 data.x 不为 None
    """
    def __init__(self, root="data/raw", algorithm="bfs", split="train"):
        self.inner = SALSACLRSDataset(
            root=root,
            split=split,
            algorithm=algorithm,
            num_samples=10000,
            graph_generator="er",
            graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)},
            hints=True
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        data = self.inner[idx]  # PyG Data

        # 如果没有节点特征，就造一份常数特征
        if getattr(data, "x", None) is None:
            num_nodes = data.num_nodes
            data.x = torch.ones(num_nodes, 1, dtype=torch.float)

        return data
