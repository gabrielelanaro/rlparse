from marty.actions import Context, push, op


def test_push():
    c = Context(buffer=["1"])

    c2 = push(c, 0, 1)

    assert c != c2
    assert c2.memory[1] == (0,)


def test_op():
    c = Context(buffer=[], memory={1: (0,)})

    c2 = op(c, "CARD", 1)

    assert c != c2

    assert c2.memory[1] == ("CARD", 0)


def test_produce():
    c = Context(buffer=[], memory={1: (0,)})
