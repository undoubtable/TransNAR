# train_transnar.py

import os
import json
import torch

# --- 解决 torch 2.6 的 weights_only 问题（跟 train_nar.py 一样）---
_orig_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
# -----------------------------------------------------------------

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from salsaclrs import SALSACLRSDataset
from models.nar import NARGNN
from models.transnar import TransNARModel
from utils.tokenizer import SimpleTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_PATH = "data/text/bfs_text_train.jsonl"
ROOT = "data/raw"     # 和前面 SALSACLRS 一致
ALGO = "bfs"


# ===== 1. 文本 Dataset：把 text_in + text_out 拼成一个序列 =====

class BFSTextSeqDataset(Dataset):
    """
    每条样本：
      -读取 jsonl 的 text_in, text_out
      -拼成一个完整的序列： seq = text_in + " " + text_out
      -encode 成 token id 列表
    只需要 input_ids 和 graph_idx，目标序列用 LM 的方式: 预测下一个 token。
    """
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                text_in = obj["text_in"]
                text_out = obj["text_out"]
                graph_idx = obj["graph_idx"]

                full_text = text_in + " " + text_out
                ids = tokenizer.encode(full_text, add_special_tokens=True, max_len=max_len)

                self.samples.append({
                    "input_ids": ids,
                    "graph_idx": graph_idx,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_seq_batch(batch, pad_id):
    """
    把若干样本拼成 batch：
      - 对 input_ids 做 padding
      - 构造 target_ids：下一个 token 预测（shift 一位）
    """
    input_ids_list = []
    target_ids_list = []
    graph_idx_list = []

    for b in batch:
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        # target: 预测下一个 token，最后一个 token 的 target 设成 pad
        tgt = torch.cat([ids[1:], torch.tensor([pad_id])])

        input_ids_list.append(ids)
        target_ids_list.append(tgt)
        graph_idx_list.append(b["graph_idx"])

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "graph_idx": torch.tensor(graph_idx_list, dtype=torch.long)
    }


# ===== 2. 构建 tokenizer，读取文本并建 vocab =====

def build_tokenizer_from_text(jsonl_path):
    all_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            all_texts.append(obj["text_in"])
            all_texts.append(obj["text_out"])

    tok = SimpleTokenizer(min_freq=1)
    tok.build_vocab(all_texts)
    print("Vocab size:", tok.vocab_size)
    return tok


def main():
    # 1) tokenizer
    tokenizer = build_tokenizer_from_text(TEXT_PATH)
    pad_id = tokenizer.token2id[tokenizer.pad_token]

    # 2) 文本 Dataset + DataLoader
    text_ds = BFSTextSeqDataset(TEXT_PATH, tokenizer, max_len=256)
    text_loader = DataLoader(
        text_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_seq_batch(b, pad_id)
    )

    # 3) 图 Dataset：跟 make_text_dataset 配置一致
    graph_ds = SALSACLRSDataset(
        root=ROOT,
        split="train",
        algorithm=ALGO,
        num_samples=len(text_ds),  # 保持同样数量和索引对应
        graph_generator="er",
        graph_generator_kwargs={"n": 16, "p": 0.1},
        hints=True,
    )

    # 4) 加载已训练的 NAR 并冻结
    nar_hidden_dim = 128
    nar_in_dim = 1  # 当时 NARGNN 用的是 x=[ones(num_nodes,1)]

    nar_model = NARGNN(
        in_dim=nar_in_dim,
        hidden_dim=nar_hidden_dim,
        out_dim=1,
        num_layers=4
    )
    nar_model.load_state_dict(torch.load("checkpoints/nar_bfs.pt", map_location="cpu"))
    nar_model.to(DEVICE)
    nar_model.eval()
    for p in nar_model.parameters():
        p.requires_grad = False

    # 5) 构建 TransNAR 模型
    model = TransNARModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        d_nar=nar_hidden_dim,
        n_heads=4,
        num_layers=4,
        max_len=256
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 6) 训练循环
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        steps = 0

        for batch in text_loader:
            input_ids = batch["input_ids"].to(DEVICE)      # [B,L]
            target_ids = batch["target_ids"].to(DEVICE)    # [B,L]
            graph_idx = batch["graph_idx"].tolist()        # list[int], 长度 B

            # === 用 NAR 生成节点 embedding ===
            nar_emb_list = []
            nar_mask_list = []

            for idx in graph_idx:
                g = graph_ds[idx]          # CLRSData
                # 把图放到设备上
                g = g.to(DEVICE)

                num_nodes = g.pos.size(0)
                x = torch.ones(num_nodes, 1, device=DEVICE)
                batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=DEVICE)

                with torch.no_grad():
                    _, node_emb = nar_model(x, g.edge_index, batch_vec)  # [num_nodes, d_nar]

                nar_emb_list.append(node_emb)
                nar_mask_list.append(torch.ones(num_nodes, dtype=torch.long, device=DEVICE))

            # padding 节点数，形成 [B, N_max, d_nar]
            max_n = max(e.size(0) for e in nar_emb_list)
            padded_emb = []
            padded_mask = []
            for emb, mask in zip(nar_emb_list, nar_mask_list):
                pad_len = max_n - emb.size(0)
                if pad_len > 0:
                    pad_emb = torch.zeros(pad_len, emb.size(1), device=DEVICE)
                    emb = torch.cat([emb, pad_emb], dim=0)
                    pad_mask = torch.zeros(pad_len, dtype=torch.long, device=DEVICE)
                    mask = torch.cat([mask, pad_mask], dim=0)
                padded_emb.append(emb)
                padded_mask.append(mask)

            nar_h = torch.stack(padded_emb, dim=0)     # [B, N_max, d_nar]
            nar_mask = torch.stack(padded_mask, dim=0) # [B, N_max]

            # === 前向 & loss ===
            logits = model(input_ids, nar_h, nar_mask=nar_mask)  # [B,L,V]
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 50 == 0:
                print(f"Epoch {epoch} Step {steps} Loss {loss.item():.4f}")

        print(f"Epoch {epoch} Avg Loss {total_loss / max(1, steps):.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/transnar_epoch{epoch}.pt")
        print(f"Saved checkpoints/transnar_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
