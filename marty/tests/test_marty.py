from marty.actions import Context
from marty.marty import Marty


def test_marty():

    c = Context(buffer=["1"])

    m = Marty()
    result = m.parse(c)
    print(result)
