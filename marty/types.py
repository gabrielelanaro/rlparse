from enum import Enum, auto
from marty.actions import Action, Context

from typing import Any, Tuple, Callable, NamedTuple


class ParseOutcome(Enum):
    ERROR = auto()
    EXCEEDED = auto()
    CORRECT = auto()
    INCORRECT = auto()
    VALID = auto()


class ActionTrace(NamedTuple):
    action: Action
    ctx: Context
    action_ix: Any

