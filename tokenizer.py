# Simple whitespace tokenizer and vocabulary. Replaceable with SentencePiece/subword later.
import re
from collections import Counter, defaultdict
import json
import os

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

class SimpleTokenizer:
    def __init__(self, vocab=None, min_freq=1):
        if vocab is None:
            self.token_to_id = {PAD:0, UNK:1, BOS:2, EOS:3}
            self.id_to_token = {v:k for k,v in self.token_to_id.items()}
            self._counter = Counter()
            self.frozen = False
        else:
            self.token_to_id = vocab
            self.id_to_token = {v:k for k,v in vocab.items()}
            self._counter = Counter()
            self.frozen = True
        self.min_freq = min_freq

    def normalize(self, text):
        # simple normalization
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        text = self.normalize(text)
        # naive tokenization (words and punctuation)
        tokens = re.findall(r"\w+|[^\s\w]", text)
        return tokens

    def build_vocab_from_texts(self, texts, max_size=10000):
        for t in texts:
            self._counter.update(self.tokenize(t))
        most = [tok for tok, cnt in self._counter.most_common(max_size) if cnt >= self.min_freq]
        for tok in most:
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        self.frozen = True

    def encode(self, text, add_bos_eos=True, max_len=None):
        tokens = self.tokenize(text)
        ids = []
        if add_bos_eos:
            ids.append(self.token_to_id[BOS])
        for t in tokens:
            ids.append(self.token_to_id.get(t, self.token_to_id[UNK]))
        if add_bos_eos:
            ids.append(self.token_to_id[EOS])
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        toks = [self.id_to_token.get(i, UNK) for i in ids]
        # remove bos/eos/pad
        toks = [t for t in toks if t not in (BOS, EOS, PAD)]
        return " ".join(toks)

    def size(self):
        return len(self.token_to_id)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)