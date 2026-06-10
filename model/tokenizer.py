"""
Simple character-level tokenizer for item names.
"""
import json
import os


class CharTokenizer:
    def __init__(self, max_len=32):
        self.max_len = max_len
        self.pad_token = 0
        self.unk_token = 1
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2char = {0: '<PAD>', 1: '<UNK>'}
        self._next_idx = 2

    def fit(self, texts):
        for text in texts:
            for ch in text.lower():
                if ch not in self.char2idx:
                    self.char2idx[ch] = self._next_idx
                    self.idx2char[self._next_idx] = ch
                    self._next_idx += 1

    @property
    def vocab_size(self):
        return self._next_idx

    def encode(self, text):
        tokens = []
        for ch in text.lower()[:self.max_len]:
            tokens.append(self.char2idx.get(ch, self.unk_token))
        while len(tokens) < self.max_len:
            tokens.append(self.pad_token)
        return tokens

    def get_mask(self, text):
        length = min(len(text), self.max_len)
        mask = [False] * length + [True] * (self.max_len - length)
        return mask

    def save(self, path):
        data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()},
            'max_len': self.max_len,
            'next_idx': self._next_idx,
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char2idx = data['char2idx']
        self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        self.max_len = data['max_len']
        self._next_idx = data['next_idx']
