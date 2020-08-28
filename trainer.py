# %%
from marty.marty import ActionBudgetExceeded, Marty, ParseOutcome
from marty.actions import Context


DATA = [
    {"buffer": ["egg"], "expected": ("ING", 0)},
    {"buffer": ["1", "BLANK"], "expected": ("CARD", 0)},
    {"buffer": ["1", "egg"], "expected": ("QTY", ("CARD", 0), ("ING", 1))},
]


def train(m: Marty):

    # buffer = ["1", "egg"]
    # expected = ("QTY", ("CARD", 0), ("ING", 1))
    datum = DATA[0]

    buffer = datum["buffer"]
    expected = datum["expected"]
    print("PARSING", buffer)

    n_tries = 0

    sample_ix = 0
    n_learning = 1

    while True:
        c = Context(buffer=buffer)

        try:
            result, actions = m.parse(c)
        except ActionBudgetExceeded:
            n_tries += 1
            continue

        if result == expected:
            print(
                buffer,
                "Found",
                result,
                "after",
                n_tries,
                [a.action for a in actions],
                "*" * 100,
            )
            m.learn(actions, ParseOutcome.CORRECT)
            n_tries = 0
            n_learning = max(n_learning, sample_ix + 2)
        else:
            print(
                buffer,
                "Found instead",
                result,
                "actions",
                [a.action for a in actions],
                end="\r",
            )
            m.learn(actions, ParseOutcome.INCORRECT)
            n_tries += 1
            # input()

        # if n_tries > 10:
        #     sample_ix = (sample_ix + 1) % n_learning

        buffer = DATA[sample_ix]["buffer"]
        expected = DATA[sample_ix]["expected"]


# %%
m = Marty()

train(Marty())

# %%
