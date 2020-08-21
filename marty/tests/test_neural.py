from marty.actions import Context
from marty.neural import CNNSeqEncoder, CharacterEncoder, ContextEncoder


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

    ctxe = ContextEncoder(memory_slots=3)

    ctx = Context(buffer=["hello", "world"])

    res = ctxe(ctx)
    assert res.shape == (3, 8)


def test_context_encoder():

    ctxe = ContextEncoder(memory_slots=3)

    ctx = Context(buffer=["hello", "world"], memory={0: ("COPY", (0,), (1,))})

    res = ctxe(ctx)
    assert res.shape == (3, 8)
