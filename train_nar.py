# train_nar.py

import torch

# --- 解决 torch 2.6 的 weights_only 问题 ---
_orig_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
# -----------------------------------------

import torch.nn as nn
from salsaclrs import SALSACLRSDataset, SALSACLRSDataLoader
from models.nar import NARGNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1. 官方 SALSACLRSDataset + 官方 DataLoader
    train_ds = SALSACLRSDataset(
        root="data/raw",
        split="train",
        algorithm="bfs",
        num_samples=10000,
        graph_generator="er",
        graph_generator_kwargs={"n": 16, "p": 0.1},
        hints=True,
    )

    train_loader = SALSACLRSDataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
    )

    # 2. 看一个 batch 结构
    first_batch = next(iter(train_loader))
    print("Batch contents:\n", first_batch)

    # 这里没有 batch.x，所以我们自己定义 in_dim=1（每个节点一个标量特征）
    in_dim = 1

    model = NARGNN(
        in_dim=in_dim,
        hidden_dim=128,
        out_dim=1,   # 预测每个节点一个标量（比如是否可达）
        num_layers=4,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        num_steps = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)

            # 3. 自己给节点造特征 x: [num_nodes, 1]，这里用全 1
            num_nodes = batch.pos.size(0)  # pos=[总节点数]
            x = torch.ones(num_nodes, 1, device=DEVICE)

            # 4. 前向：用 edge_index + batch（图归属）+ 我们的 x
            pred, _ = model(x, batch.edge_index, batch.batch)  # pred: [num_nodes, 1]

            # 5. 选择目标：用 reach_h 的最后一个时间步当目标
            #    reach_h: [num_nodes, T]，比如 T=10
            target = batch.reach_h[:, -1].float()  # [num_nodes]

            loss = criterion(pred.squeeze(-1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if num_steps % 50 == 0:
                print(f"Epoch {epoch} Step {num_steps} Loss {loss.item():.4f}")

        print(f"Epoch {epoch} Avg Loss {total_loss / max(1, num_steps):.4f}")

    # 6. 保存权重（后面 TransNAR 会用到 node embedding）
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/nar_bfs.pt")
    print("Saved NAR model to checkpoints/nar_bfs.pt")


if __name__ == "__main__":
    main()
