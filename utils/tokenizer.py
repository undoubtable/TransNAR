# utils/tokenizer.py

import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.token2id = {}
        self.id2token = []
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

    def tokenize(self, text: str):
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))

        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for tok, freq in counter.items():
            if freq >= self.min_freq:
                vocab.append(tok)

        self.id2token = vocab
        self.token2id = {t: i for i, t in enumerate(vocab)}

    @property
    def vocab_size(self):
        return len(self.id2token)

    def encode(self, text, add_special_tokens=True, max_len=None):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        tokens = [self.id2token[i] for i in ids if i < len(self.id2token)]
        tokens = [t for t in tokens if t not in {self.bos_token, self.eos_token, self.pad_token}]
        return " ".join(tokens)
