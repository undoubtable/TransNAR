# scripts/make_text_dataset.py

import os
import json
import torch

# ---- 解决 torch 2.6 weights_only=True 问题（跟 train_nar.py 一样）----
_orig_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
# -------------------------------------------------------------------

from salsaclrs import SALSACLRSDataset

ROOT = "data/raw"
ALGO = "bfs"
NUM_SAMPLES = 5000  # 先生成 5k 条，当训练集


def bfs_instance_to_text(g):
    """
    g: 一个 CLRSData 样本
    目标：构造 (text_in, text_out)
    """
    # 1. 图结构
    edge_index = g.edge_index          # [2, E]
    edges = list(zip(
        edge_index[0].tolist(),
        edge_index[1].tolist()
    ))
    n = g.pos.size(0)                  # 节点数

    # 2. 起点 s: 一般是 one-hot / 掩码
    s = g.s
    if s.ndim > 1:
        # 比如 [num_nodes, T]，取第 0 列
        s_vec = s[:, 0]
    else:
        s_vec = s
    start_node = int(torch.argmax(s_vec).item())

    # 3. 可达性：reach_h 最后一列
    reach = g.reach_h[:, -1]           # [num_nodes]
    reachable_list = reach.long().tolist()

    # 4. 组装文本
    text_in = f"bfs: n={n}, edges={edges}, start={start_node}"
    text_out = f"reachable: {reachable_list}"

    return text_in, text_out


def main():
    os.makedirs("data/text", exist_ok=True)

    # ⚠️ 这里只用 SALSACLRSDataset，不要改写 SALSACLRSDataLoader 变量名
    dataset = SALSACLRSDataset(
        root=ROOT,
        split="train",
        algorithm=ALGO,
        num_samples=NUM_SAMPLES,
        graph_generator="er",
        graph_generator_kwargs={"n": 16, "p": 0.1},
        hints=True,
    )

    # 先打印一个样本看看结构，调试用
    g0 = dataset[0]
    print("Sample g0:", g0)
    print("  pos shape:", g0.pos.shape)
    print("  s shape:", g0.s.shape)
    print("  reach_h shape:", g0.reach_h.shape)

    out_path = "data/text/bfs_text_train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            g = dataset[idx]
            text_in, text_out = bfs_instance_to_text(g)

            rec = {
                "text_in": text_in,
                "text_out": text_out,
                "graph_idx": idx,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
