# This is marty a parser for reinforcement learning

# We want to do something kind of generic to read the text.

# ACTIONS:
# push(bucket_id) put the information in bucket bucket_id
# join(op, bucket_source, bucket_target)
from dataclasses import dataclass
from functools import partial
from copy import deepcopy
from marty.policy import ActionSpace
from marty.diagnostics import display_action_scores
import random
from marty.types import ActionTrace, ParseOutcome
from marty.engine import ACEngine

import torch
from typing import List, Tuple, Any
from .actions import (
    ActionParam,
    ActionParamSlot,
    Context,
    Action,
    InvalidAction,
    join,
    op,
    produce,
    push,
)
import torch.nn.functional as F


class ActionBudgetExceeded(Exception):
    pass


@dataclass
class Marty:

    max_actions = 10
    mem_slots = 2
    max_buffer_size = 2
    op_types = ["ING", "CARD"]
    join_types = ["QTY"]

    def __post_init__(self):
        self.action_space = ActionSpace(self.mem_slots, self.max_buffer_size)
        self._engine = ACEngine(self.mem_slots, self.action_space)

    def parse(self, c: Context):
        initial = c

        action_seq = []
        while len(action_seq) < self.max_actions:
            action, ix = self._choose_action(c)
            action_seq.append(ActionTrace(action, c, ix))
            try:
                result = action(c)
            except InvalidAction:
                self.learn(action_seq, ParseOutcome.ERROR)
                c = initial
                action_seq = []
                continue

            if action.name == "produce":
                return result, action_seq
            else:
                c = result

        if action_seq:
            self.learn(action_seq, ParseOutcome.EXCEEDED)
        raise ActionBudgetExceeded()

    def _choose_action(self, c: Context) -> Tuple[Action, Any]:
        log_p = self._engine.policy(c)
        scores = torch.exp(log_p)

        # print("SCORES", torch.exp(scores))
        if random.random() > 0.0:
            # selected = scores.argmax()
            selected = torch.multinomial(scores, 1)[0]

        else:
            selected = scores.argmax()

        # selected = random.randint(0, len(scores) - 1)
        selected_action = self.action_space.avail_actions[selected]
        torch.set_printoptions(8, sci_mode=False)

        display_action_scores(self.action_space.avail_actions, scores, c)

        return selected_action, selected

    def learn(self, action_seq: List[ActionTrace], res: ParseOutcome):
        self._engine.learn(action_seq, res)
