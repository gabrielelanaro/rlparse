from typing import Dict, Iterable, NamedTuple, Set
from itertools import zip_longest
import torch


class Tokenizer:
    max_sent_length: int
    vocab_size: int

    def __init__(self, words: Set[str], max_sent_length: int) -> None:
        self._word_map: Dict[str, int] = {}
        self.max_sent_length = max_sent_length
        # reserved tokens
        self.unk_ix = 0
        self.pad_ix = 1
        self.cls_ix = 2
        self.spec1_ix = 3
        n_reserved = 4

        # assign the words, and we sort them for determinism
        for ix, word in enumerate(sorted(words)):
            self._word_map[word] = ix + n_reserved

        # invert the word map for fun and profit
        self._inverse_word_map = {
            self.unk_ix: "[UNK]",
            self.pad_ix: "[PAD]",
            self.cls_ix: "[CLS]",
            self.spec1_ix: "[SPEC1]",
        }

        for w, ix in self._word_map.items():
            self._inverse_word_map[ix] = w

        self.vocab_size = len(self._word_map) + n_reserved

    def encode_tokens(self, sent: Iterable[str]):
        sentl = list(sent)
        encoded_sent = []
        pad_mask = []
        for i in range(self.max_sent_length):
            if i == 0:
                enc = self.cls_ix
            if i > len(sentl):
                enc = self.pad_ix
            else:
                enc = self._word_map.get(sentl[i - 1], self.unk_ix)
            encoded_sent.append(enc)
            pad_mask.append(enc == self.pad_ix)

        return torch.tensor(encoded_sent), torch.tensor(pad_mask)
