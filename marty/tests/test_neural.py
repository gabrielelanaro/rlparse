from marty.actions import Context
from marty.neural import CNNSeqEncoder, CharacterEncoder

from marty.layers.context import ContextEncoder
from marty.tokenizer import Tokenizer


def test_character_encoder():

    c = CharacterEncoder(pad=10, encoding_size=16)

    ret = c(["hello", "world"])

    assert ret.shape == (2, 10, 16)


def test_character_encoder_stack1():
    c = CharacterEncoder(pad=10, encoding_size=16)

    ret = c(["hello"])

    assert ret.shape == (1, 10, 16)


def test_cnn_encoder():

    ce = CharacterEncoder(pad=10, encoding_size=16)
    we = CNNSeqEncoder(inp_channels=16, inner_size=8)

    ret = we(ce(["hello", "world"]))

    assert ret.shape == (2, 8)


def test_context_encoder_empty():
    max_sent_length = 10
    mem_slots = 3
    embedding_dim = 16
    tok = Tokenizer({"hello", "world", "COPY"}, max_sent_length)

    ctxe = ContextEncoder(tok, embedding_dim=embedding_dim, mem_slots=3, max_mem_size=3)

    ctx = Context(buffer=["hello", "world"])

    res = ctxe(ctx)
    assert res.shape == (max_sent_length + mem_slots, embedding_dim)


def test_context_encoder():
    max_sent_length = 10
    mem_slots = 3
    embedding_dim = 16
    tok = Tokenizer({"hello", "world", "COPY"}, max_sent_length)

    ctxe = ContextEncoder(tok, embedding_dim=embedding_dim, mem_slots=3, max_mem_size=3)

    ctx = Context(buffer=["hello", "world"], memory={0: ("COPY", (0,), (1,))})

    res = ctxe(ctx)
    assert res.shape == (max_sent_length + mem_slots, embedding_dim)
