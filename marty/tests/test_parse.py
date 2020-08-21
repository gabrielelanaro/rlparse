from marty.actions import Context, push, op, join, produce


def test_one_egg():

    buffer = ["1", "egg"]

    c = Context(buffer=buffer)

    c = push(c, 0, 1)
    c = op(c, "CARD", 1)
    c = push(c, 1, 2)
    c = op(c, "ING", 2)
    c = join(c, "QTY", 1, 2)
    result = produce(c, 1)

    assert result == ("QTY", ("CARD", 0), ("ING", 1))
