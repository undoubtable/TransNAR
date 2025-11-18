# train_nar.py
import os
import torch
import torch.nn as nn
from salsaclrs import SALSACLRSDataset, SALSACLRSDataLoader
from models.nar import NARGNN

# ---- 解决 PyTorch 2.6 torch.load(weights_only=True) 问题 ----
_orig_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
# -----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Using device:", DEVICE)

    # 1. 构建 SALSA-CLRS 的 BFS 数据集（图数据） 如果有的话就不用再构建了
    train_ds = SALSACLRSDataset(
        root="data/raw",
        split="train",
        algorithm="bfs",
        num_samples=10000,              # 可以改大，比如 20000
        graph_generator="er",
        graph_generator_kwargs={"n": 16, "p": 0.1},
        hints=True,
    )

    train_loader = SALSACLRSDataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
    )

    # 2. 看一个 batch 的结构（只打印一次，确认没问题）
    first_batch = next(iter(train_loader))
    print("Sample batch:\n", first_batch)
    print("  edge_index:", first_batch.edge_index.shape)
    print("  pos      :", first_batch.pos.shape)
    print("  reach_h  :", first_batch.reach_h.shape)

    # 我们自己给节点造特征 x: [num_nodes, 1]，这里用全 1
    in_dim = 1

    # 3. 定义 NAR GNN 模型
    model = NARGNN(
        in_dim=in_dim,
        hidden_dim=128,
        out_dim=1,       # 为每个节点输出一个标量（可达概率）
        num_layers=4,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # reach_h 是 0/1，我们可以用 MSE 做一个回归式训练（简单稳定）
    criterion = nn.MSELoss()

    num_epochs = 10   # 可以先用 3 快速跑通，之后改成 10 / 20

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_steps = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)

            # 节点个数（所有图合并后的总节点数）
            num_nodes = batch.pos.size(0)         # pos = [N]

            # 节点特征：目前简单用全 1，后面可以尝试加入更多图信息
            x = torch.ones(num_nodes, 1, device=DEVICE)

            # 前向：NAR 接收 x, edge_index, batch（图归属向量）
            # 注意：SALSACLRSDataLoader 返回的是 CLRSDataBatch，自带 batch 属性
            pred, _ = model(x, batch.edge_index, batch.batch)  # pred: [N, 1]

            # 目标：用 reach_h 的最后一个时间步当作监督
            # reach_h: [N, T]
            target = batch.reach_h[:, -1].float()  # [N]

            loss = criterion(pred.squeeze(-1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if num_steps % 10 == 0:
                print(f"[Epoch {epoch}] Step {num_steps} | Loss = {loss.item():.4f}")

        avg_loss = total_loss / max(1, num_steps)
        print(f"==> Epoch {epoch} | Avg Loss = {avg_loss:.6f}")

    # 4. 保存权重，TransNAR 会用到 node embedding
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/nar_bfs.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved NAR model to {save_path}")


if __name__ == "__main__":
    main()
