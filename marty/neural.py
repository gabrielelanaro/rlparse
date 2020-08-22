import string
from typing import List, NamedTuple
import unicodedata
import torch
from torch.autograd import Variable
from torch.nn import Module
import torch.nn as nn
from .actions import Context

all_letters = string.ascii_letters + " .,;'-"


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def _unicode_to_ascii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


class CharacterEncoder(Module):
    def __init__(self, pad: int, encoding_size):
        super().__init__()
        self._pad = pad
        self._encoding_size = encoding_size

        self.emb = nn.Embedding(
            num_embeddings=len(all_letters) + 2, embedding_dim=encoding_size
        )

    def forward(self, strings: List[str]):
        return self.emb(torch.stack([_encode_str(x) for x in strings]))


def _encode_str(inp: str, pad=10) -> torch.Tensor:
    inp = _unicode_to_ascii(inp)

    return torch.tensor([_letter_to_int(inp, i) for i in range(pad)], dtype=torch.long)


PAD_CODE = 0
UNK_CODE = 1


def _letter_to_int(inp: str, index: int):

    if index >= len(inp):
        return PAD_CODE

    else:
        ix = all_letters.index(inp[index])
        return ix + 2 if ix != -1 else UNK_CODE


class CNNSeqEncoder(Module):
    def __init__(self, inp_channels, inner_size):
        super().__init__()
        self.cnn1 = nn.Conv1d(
            in_channels=inp_channels, out_channels=inner_size, kernel_size=2
        )

        self.cnn2 = nn.Conv1d(
            in_channels=inner_size, out_channels=inner_size, kernel_size=2
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Because that's the format it expects
        x = self.cnn1(x)
        x = nn.ReLU()(x)
        x = self.cnn2(x)
        return torch.mean(x, axis=2)


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(-1)

