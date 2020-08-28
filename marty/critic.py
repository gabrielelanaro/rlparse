from marty.actions import Context
from marty.layers.context import ContextEncoder
import torch.nn as nn


class CriticNet(nn.Module):
    def __init__(self, ctx: ContextEncoder):
        super().__init__()
        self.ctx = ctx
        self.head = nn.Sequential(nn.Linear(ctx.embedding_dim, 1))

    def forward(self, x: Context):
        xtens = self.ctx(x)
        return self.head(xtens[0])

