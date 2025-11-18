# train_transnar.py
import os
import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- 解决 PyTorch 2.6 torch.load(weights_only=True) 问题 ----
_orig_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
# -----------------------------------------------------------

from salsaclrs import SALSACLRSDataset, SALSACLRSDataLoader
from models.nar import NARGNN
from utils.tokenizer import SimpleTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# 1. 配置
# =====================
@dataclass
class Config:
    text_path: str = "data/text/bfs_text_train.jsonl"
    raw_graph_root: str = "data/raw"
    algorithm: str = "bfs"
    num_graph_samples: int = 10000
    graph_n: int = 16
    graph_p: float = 0.1

    max_len: int = 512       # 文本最大长度（你之前统计最长 482，512 正好）
    batch_size: int = 2      # RTX 5060 建议先用 2 或 4
    num_epochs: int = 3      # 先小跑 3 轮看看
    lr: float = 1e-4

    nar_in_dim: int = 1
    nar_hidden_dim: int = 128
    nar_out_dim: int = 1
    nar_num_layers: int = 4

    d_model: int = 256
    nhead: int = 4
    num_layers_text: int = 2

    save_dir: str = "checkpoints"
    nar_ckpt: str = "checkpoints/nar_bfs.pt"
    transnar_ckpt: str = "checkpoints/transnar_bfs.pt"


CFG = Config()


# =====================
# 2. 文本 Dataset & collate
# =====================
class BFSTextSeqDataset(Dataset):
    """
    每条记录结构：
    {
      "text_in": "...",
      "text_out": "...",
      "graph_idx": int
    }
    """
    def __init__(self, path, tokenizer: SimpleTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        in_ids = self.tokenizer.encode(
            obj["text_in"], add_special_tokens=True, max_len=self.max_len
        )
        out_ids = self.tokenizer.encode(
            obj["text_out"], add_special_tokens=True, max_len=self.max_len
        )
        return {
            "input_ids": in_ids,
            "target_ids": out_ids,
            "graph_idx": obj["graph_idx"],
        }


def collate_seq_batch(batch, pad_id: int):
    """
    把一批样本 pad 对齐：
    input_ids: [B,L]
    target_ids: [B,L]
    graph_idx: [B]
    attention_mask: [B,L] (True=有效)
    """
    import torch

    B = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    target_ids = []
    graph_idx = []

    for item in batch:
        ids_in = item["input_ids"]
        ids_out = item["target_ids"]
        pad_len_in = max_len - len(ids_in)
        pad_len_out = max_len - len(ids_out)

        input_ids.append(ids_in + [pad_id] * pad_len_in)
        target_ids.append(ids_out + [pad_id] * pad_len_out)
        graph_idx.append(item["graph_idx"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)
    graph_idx = torch.tensor(graph_idx, dtype=torch.long)
    attention_mask = (input_ids != pad_id)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "graph_idx": graph_idx,
        "attention_mask": attention_mask,
    }


# =====================
# 3. GraphIndexDataset：按 graph_idx 抽子图
# =====================
class GraphIndexDataset(Dataset):
    """
    从基础 SALSACLRSDataSet 中按给定 indices 抽取图，
    这样我们可以用 SALSACLRSDataLoader 得到带 batch 向量的 CLRSDataBatch。
    """
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = list(int(i) for i in indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_ds[self.indices[idx]]


def make_graph_batch(base_ds, graph_indices):
    """
    给定 graph_idx 列表（来自文本 batch），构造对应的图 batch。
    返回：CLRSDataBatch，包含 edge_index, pos, reach_h, batch 等字段
    """
    sub_ds = GraphIndexDataset(base_ds, graph_indices)
    loader = SALSACLRSDataLoader(sub_ds, batch_size=len(sub_ds), shuffle=False)
    graph_batch = next(iter(loader))
    return graph_batch


# =====================
# 4. 文本 Encoder + CrossAttention + Head
# =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B,L,D]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, max_len=512, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B,L]
        attention_mask: [B,L]，True=有效，False=pad
        返回 text_h: [B,L,D]
        """
        x = self.embed(input_ids)   # [B,L,D]
        x = self.pos(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # True = 要mask掉

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h


class TextToGraphCrossAttention(nn.Module):
    """
    Text token 作为 Query，图节点 embedding 作为 Key/Value。
    """
    def __init__(self, d_model=256, d_nar=128, nhead=4):
        super().__init__()
        self.nar_proj = nn.Linear(d_nar, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, text_h, node_emb_list):
        """
        text_h: [B,L,D]
        node_emb_list: list，长度 B，第 i 个元素为 [N_i, d_nar]
        返回 cross_out: [B,L,D]
        """
        B, L, D = text_h.size()
        outs = []
        for i in range(B):
            h_i = text_h[i:i+1]            # [1,L,D]
            nodes_i = node_emb_list[i]     # [N_i, d_nar]
            nodes_i = nodes_i.to(h_i.device)

            kv_i = self.nar_proj(nodes_i)  # [N_i,D]
            kv_i = kv_i.unsqueeze(0)       # [1,N_i,D]

            out_i, _ = self.cross_attn(
                query=h_i,
                key=kv_i,
                value=kv_i,
                need_weights=False,
            )   # out_i: [1,L,D]

            outs.append(out_i)

        return torch.cat(outs, dim=0)      # [B,L,D]


class MiniTransNARHead(nn.Module):
    """
    简单的融合头：text_h + cross_out → LN → linear → vocab logits
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, text_h, cross_out):
        fused = text_h + cross_out
        fused = self.ln(fused)
        logits = self.lm_head(fused)  # [B,L,V]
        return logits


# =====================
# 5. 训练主函数
# =====================
def main():
    cfg = CFG
    print("Using device:", DEVICE)

    # ----- 5.1 构建 tokenizer -----
    print("Building tokenizer from:", cfg.text_path)
    texts = []
    with open(cfg.text_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text_in"])
            texts.append(obj["text_out"])

    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(texts)
    pad_id = tokenizer.token2id[tokenizer.pad_token]
    print("vocab_size:", tokenizer.vocab_size, "pad_id:", pad_id)

    # ----- 5.2 构建文本 Dataset & DataLoader -----
    text_ds = BFSTextSeqDataset(cfg.text_path, tokenizer, max_len=cfg.max_len)
    text_loader = DataLoader(
        text_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_seq_batch(b, pad_id),
    )
    print("Text dataset size:", len(text_ds))

    # ----- 5.3 构建图 Dataset -----
    graph_ds = SALSACLSDataset = SALSACLRSDataset(
        root=cfg.raw_graph_root,
        split="train",
        algorithm=cfg.algorithm,
        num_samples=cfg.num_graph_samples,
        graph_generator="er",
        graph_generator_kwargs={"n": cfg.graph_n, "p": cfg.graph_p},
        hints=True,
    )
    print("Graph dataset size:", len(graph_ds))

    # ----- 5.4 加载已训练好的 NAR，并冻结 -----
    nar = NARGNN(
        in_dim=cfg.nar_in_dim,
        hidden_dim=cfg.nar_hidden_dim,
        out_dim=cfg.nar_out_dim,
        num_layers=cfg.nar_num_layers,
    ).to(DEVICE)

    print("Loading NAR checkpoint from:", cfg.nar_ckpt)
    nar_state = torch.load(cfg.nar_ckpt, map_location=DEVICE)
    nar.load_state_dict(nar_state)
    nar.eval()
    for p in nar.parameters():
        p.requires_grad = False

    # ----- 5.5 构建 TextEncoder + CrossAttention + Head -----
    vocab_size = tokenizer.vocab_size
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers_text,
        max_len=cfg.max_len,
        pad_id=pad_id,
    ).to(DEVICE)

    cross_attn_module = TextToGraphCrossAttention(
        d_model=cfg.d_model,
        d_nar=cfg.nar_hidden_dim,
        nhead=cfg.nhead,
    ).to(DEVICE)

    head = MiniTransNARHead(d_model=cfg.d_model, vocab_size=vocab_size).to(DEVICE)

    # ----- 5.6 优化器 & 损失 -----
    params = list(text_encoder.parameters()) + \
             list(cross_attn_module.parameters()) + \
             list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ----- 5.7 训练循环 -----
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        text_encoder.train()
        cross_attn_module.train()
        head.train()

        total_loss = 0.0
        num_steps = 0

        for batch in text_loader:
            input_ids = batch["input_ids"].to(DEVICE)            # [B,L]
            target_ids = batch["target_ids"].to(DEVICE)          # [B,L]
            attention_mask = batch["attention_mask"].to(DEVICE)  # [B,L]
            graph_idx = batch["graph_idx"]                       # [B]

            B, L = input_ids.shape

            # 1) 根据 graph_idx 构造 graph_batch
            graph_batch = make_graph_batch(graph_ds, graph_idx.tolist()).to(DEVICE)

            # 2) 用 NAR 得到节点 embedding（不求梯度）
            num_nodes = graph_batch.pos.size(0)
            x_nodes = torch.ones(num_nodes, 1, device=DEVICE)

            with torch.no_grad():
                _, node_emb = nar(x_nodes, graph_batch.edge_index, graph_batch.batch)  # [N_total, d_nar]

            # 3) 按 batch 拆分 node_emb
            node_emb_list = []
            for i in range(B):
                mask = (graph_batch.batch == i)
                emb_i = node_emb[mask]          # [N_i, d_nar]
                node_emb_list.append(emb_i)

            # 4) 文本编码
            text_h = text_encoder(input_ids, attention_mask=attention_mask)  # [B,L,D]

            # 5) Cross Attention: text_h as Q, node_emb_list as K/V
            cross_out = cross_attn_module(text_h, node_emb_list)            # [B,L,D]

            # 6) Head 输出 logits
            logits = head(text_h, cross_out)  # [B,L,V]

            # 7) 计算 loss（简单版：position-wise token 分类；忽略 pad）
            loss = criterion(
                logits.view(-1, vocab_size),   # [B*L, V]
                target_ids.view(-1),           # [B*L]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if num_steps % 50 == 0:
                print(f"[Epoch {epoch}] Step {num_steps} | Loss = {loss.item():.4f}")

        avg_loss = total_loss / max(1, num_steps)
        print(f"==> Epoch {epoch} | Avg Loss = {avg_loss:.6f}")

        # 每个 epoch 存一次模型
        save_path = os.path.join(cfg.save_dir, f"transnar_bfs_epoch{epoch}.pt")
        torch.save({
            "config": cfg.__dict__,
            "text_encoder": text_encoder.state_dict(),
            "cross_attn": cross_attn_module.state_dict(),
            "head": head.state_dict(),
        }, save_path)
        print("Saved TransNAR checkpoint to:", save_path)

    # 保存最终版本
    final_path = cfg.transnar_ckpt
    torch.save({
        "config": cfg.__dict__,
        "text_encoder": text_encoder.state_dict(),
        "cross_attn": cross_attn_module.state_dict(),
        "head": head.state_dict(),
    }, final_path)
    print("Saved final TransNAR to:", final_path)


if __name__ == "__main__":
    main()
