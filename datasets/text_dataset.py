# datasets/text_dataset.py

import json
from torch.utils.data import Dataset

class TextGraphDataset(Dataset):
    """
    每条样本: {"text_in", "text_out", "graph_idx"}
    """
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text_in = s["text_in"]
        text_out = s["text_out"]
        graph_idx = s["graph_idx"]

        input_ids = self.tokenizer.encode(text_in, add_special_tokens=True, max_len=self.max_len)
        target_ids = self.tokenizer.encode(text_out, add_special_tokens=True, max_len=self.max_len)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "graph_idx": graph_idx
        }
