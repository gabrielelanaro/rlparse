"""Context encoders"""
from marty.layers.positional import PositionalEncoding
from torch import nn
import torch

from marty.types import Context
from marty.tokenizer import Tokenizer


class ContextEncoder(nn.Module):

    tf_head = 2
    dim_feedfw = 128
    dropout = 0.0

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_dim: int,
        mem_slots: int,
        max_mem_size: int,
    ) -> None:
        super().__init__()
        self.mem_slots = mem_slots
        self.max_mem_size = max_mem_size
        self.embedding_dim = embedding_dim
        self.out_shape = tokenizer.max_sent_length + mem_slots, embedding_dim
        self._tokenizer = tokenizer

        self._n_heads = 2
        self.emb = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self._tokenizer.pad_ix,
        )

        self.tf_enc = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.tf_head,
            dim_feedforward=self.dim_feedfw,
            dropout=self.dropout,
        )

        self.tf_enc_mem = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.tf_head,
            dim_feedforward=self.dim_feedfw,
            dropout=self.dropout,
        )

        self.tf_comb_buf_mem = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.tf_head,
            dim_feedforward=self.dim_feedfw,
            dropout=self.dropout,
        )
        self.pos_enc = PositionalEncoding(embedding_dim, dropout=self.dropout)

    def forward(self, ctx: Context):
        buf, buf_mask = self._encode_buf(ctx.buffer)

        # Now we need to encode memory!
        # Memory slots
        state = [buf]

        mem_mask = []
        for mem in range(self.mem_slots):

            if not mem in ctx.memory:
                # We pad the memory with a special token
                pad = self.emb(torch.tensor([self._tokenizer.spec1_ix]))
                state.append(pad)
                mem_mask.append(False)
            else:
                state.append(self._encode_mem(ctx.memory.get(mem), buf))
                mem_mask.append(False)

        state = torch.cat(state).unsqueeze(1)
        state_mask = torch.cat([buf_mask, torch.tensor(mem_mask)]).unsqueeze(0)

        # Combine buffer and memory all in the same transformer.
        # Also residual connection
        state = self.tf_comb_buf_mem(state) + state
        return state.squeeze(1)

    def _encode_buf(self, buf):
        # output is: SENT, OUT
        enc, mask = self._tokenizer.encode_tokens(buf)
        # Batch size is dimension 1
        emb = self.emb(enc).unsqueeze(1)
        # Positional encoding
        emb = self.pos_enc(emb)
        return (
            (self.tf_enc(emb, src_key_padding_mask=mask.unsqueeze(0)) + emb).squeeze(),
            mask,
        )

    def _encode_mem(self, memory_cell, buffer_vec):
        if isinstance(memory_cell, str):
            encoded, _ = self._tokenizer.encode_tokens([memory_cell])
            out = self.emb(encoded[:1])
            # Encode only the first token
            return out.detach()

        if isinstance(memory_cell, int):
            # We need + 1 because at position 0 we have the CLS token.
            out = buffer_vec[memory_cell + 1, :].unsqueeze(0)
            return out.detach()
        if isinstance(memory_cell, tuple):
            # We stack everything, we also need to have a max_mem_size
            n_items = min(len(memory_cell), self.max_mem_size)

            pad = self.emb(
                torch.tensor(
                    [self._tokenizer.pad_ix] * (self.max_mem_size - n_items),
                    dtype=torch.long,
                )
            )

            res = torch.cat(
                [self._encode_mem(x, buffer_vec) for x in memory_cell] + [pad]
            ).unsqueeze(1)

            # Adding positional encodings
            res = self.pos_enc(res)
            res = self.tf_enc_mem(res) + res

            # We take only the first element to go ahead
            return res[0]
        raise ValueError(f"TYPE {type(memory_cell)} not supported")
