import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DotProductAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.scale = 1.0 / (embedding_dim) ** 0.5

    def forward(self, inp):
        # input is batch x seq_len x embedding_dim

        raw_weights = torch.bmm(inp, inp.transpose(1, 2))
        weights = F.softmax(raw_weights, dim=2)

        return torch.bmm(weights, inp)
