# This is marty a parser for reinforcement learning

# We want to do something kind of generic to read the text.

# ACTIONS:
# push(bucket_id) put the information in bucket bucket_id
# join(op, bucket_source, bucket_target)
from dataclasses import dataclass
from functools import partial
from copy import deepcopy
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
    mem_slots = 3
    max_buffer_size = 10
    op_types = ["ING", "CARD"]
    join_types = ["QTY"]

    def __post_init__(self):
        self._avail_actions = self._compute_avail_actions()
        self._engine = ACEngine(self.mem_slots, self._avail_actions)

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
        # TODO: this is dumb, we should learn also the action chooser.
        # if random.random() > 0.5:
        #     selected = scores.argmax()
        # else:

        # print("SCORES", torch.exp(scores))
        if random.random() > 0.0:
            # selected = scores.argmax()
            selected = torch.multinomial(scores, 1)[0]

        else:
            selected = scores.argmax()

            # selected = random.randint(0, len(scores) - 1)
        selected_action = self._avail_actions[selected]
        torch.set_printoptions(8, sci_mode=False)
        action_names = [
            str((a.name, [p.value for p in a.params])) for a in self._avail_actions
        ]

        # print("scores")
        # [print(f"{s.item():.8f}", a) for a, s in zip(action_names, scores)]

        # print("selected", selected_action, "with proba", scores[selected])

        # print(
        #     "Selected action",
        #     selected_action.name,
        #     selected_action.slots,
        #     "score",
        #     max_value,
        # )
        return selected_action, selected

    def _compute_avail_actions(self) -> List[Action]:
        actions = []

        for mem in range(self.mem_slots)[::-1]:
            for buf in range(self.max_buffer_size)[::-1]:
                actions.append(
                    Action(
                        params=[
                            ActionParam(ActionParamSlot.BUF, buf),
                            ActionParam(ActionParamSlot.MEM, mem),
                        ],
                        op=push,
                    )
                )

        for mem1 in range(self.mem_slots):
            for mem2 in range(mem1, self.mem_slots):
                for op_type in self.join_types:
                    actions.append(
                        Action(
                            op=join,
                            params=[
                                ActionParam(ActionParamSlot.BINARY_OP, op_type),
                                ActionParam(ActionParamSlot.MEM, mem1),
                                ActionParam(ActionParamSlot.MEM, mem2),
                            ],
                        )
                    )

        for mem1 in range(self.mem_slots):
            actions.append(
                Action(op=produce, params=[ActionParam(ActionParamSlot.MEM, mem1)])
            )

            for op_type in self.op_types:
                actions.append(
                    Action(
                        op=op,
                        params=[
                            ActionParam(ActionParamSlot.UNARY_OP, op_type),
                            ActionParam(ActionParamSlot.MEM, mem1),
                        ],
                    )
                )

        return actions

    def learn(self, action_seq: List[ActionTrace], res: ParseOutcome):
        self._engine.learn(action_seq, res)
