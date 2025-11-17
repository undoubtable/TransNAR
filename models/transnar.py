# models/transnar.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    text_h: [B, L, d_model]
    nar_h:  [B, N, d_nar]
    """
    def __init__(self, d_model, d_nar, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.d_nar = d_nar
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_nar, d_model)
        self.v_proj = nn.Linear(d_nar, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # gate: 控制 NAR 信息注入强度
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_h, nar_h, nar_mask=None):
        B, L, _ = text_h.shape
        _, N, _ = nar_h.shape

        q = self.q_proj(text_h)  # [B,L,d_model]
        k = self.k_proj(nar_h)   # [B,N,d_model]
        v = self.v_proj(nar_h)

        def split_heads(x):
            return x.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)  # [B,H,L,Dh]
        k = split_heads(k)  # [B,H,N,Dh]
        v = split_heads(v)  # [B,H,N,Dh]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,L,N]

        if nar_mask is not None:
            nar_mask = nar_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            attn_scores = attn_scores.masked_fill(nar_mask == 0, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,L,Dh]

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        g = torch.sigmoid(self.gate)
        return text_h + g * out


class TransNARLayer(nn.Module):
    """
    一层 TransNAR:
    - TransformerEncoderLayer 做 self-attn + FFN
    - 再做 CrossAttentionBlock 注入 NAR 信息
    """
    def __init__(self, d_model, d_nar, n_heads=4):
        super().__init__()
        self.self_attn_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.cross_block = CrossAttentionBlock(d_model, d_nar, n_heads)

    def forward(self, text_h, nar_h, nar_mask=None):
        text_h = self.self_attn_block(text_h)
        text_h = self.cross_block(text_h, nar_h, nar_mask)
        return text_h


class TransNARModel(nn.Module):
    """
    完整 TransNAR:
    - token + pos embedding
    - 多层 TransNARLayer
    - lm_head 输出 logits
    """
    def __init__(self, vocab_size, d_model=256, d_nar=128,
                 n_heads=4, num_layers=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            TransNARLayer(d_model, d_nar, n_heads)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, nar_h, nar_mask=None):
        """
        input_ids: [B,L]
        nar_h: [B,N,d_nar]
        nar_mask: [B,N]
        """
        device = input_ids.device
        B, L = input_ids.shape

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)  # [B,L,d_model]

        for layer in self.layers:
            h = layer(h, nar_h, nar_mask)

        logits = self.lm_head(h)
        return logits
