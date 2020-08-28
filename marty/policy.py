from marty.actions import Action, ActionParam, ActionParamSlot, join, produce, push, op
from typing import List
from torch import nn
import torch
import torch.nn.functional as F


class ActionSpace:

    action_types = ["push", "op", "join", "produce"]
    unary_ops = ["ING", "CARD"]
    binary_ops = ["QTY"]

    def __init__(self, mem_slots: int, max_buffer: int):
        self.mem_slots = mem_slots
        self.max_buffer_size = max_buffer

        self.avail_actions = self._compute_avail_actions()

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
                for op_type in self.binary_ops:
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

            for op_type in self.unary_ops:
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


class PolicyNetwork(nn.Module):
    def __init__(self, action_space: ActionSpace, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self._as = action_space

        # Those are just to convert from the categorical values to integers
        self._action_type_space = {k: i for i, k in enumerate(self._as.action_types)}
        self._unary_ops_space = {k: i for i, k in enumerate(self._as.unary_ops)}
        self._binary_ops_space = {k: i for i, k in enumerate(self._as.binary_ops)}
        self._mem_space = list(range(self._as.mem_slots))
        self._buf_space = list(range(self._as.max_buffer_size))

        # Those are the various heads for all possible actions and parameters
        self.decision_layer = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.Tanh(),
            nn.Linear(
                256,
                len(self._action_type_space)
                + len(self._unary_ops_space)
                + len(self._binary_ops_space)
                + len(self._buf_space)
                + len(self._mem_space)
                + len(self._mem_space),
            ),
        )

        # Those are the incides of the heads
        ix = 0
        self.action_head = ix, len(self._action_type_space)
        ix = self.action_head[1]

        self.unary_ops_head = ix, ix + len(self._unary_ops_space)
        ix = self.unary_ops_head[1]

        self.binary_ops_head = ix, ix + len(self._binary_ops_space)
        ix = self.binary_ops_head[1]

        self.buf_head = ix, ix + len(self._buf_space)
        ix = self.buf_head[1]

        self.mem1_head = ix, ix + len(self._mem_space)
        ix = self.mem1_head[1]

        self.mem2_head = ix, ix + len(self._mem_space)

    def forward(self, context_tensor):
        act = self.decision_layer(context_tensor)

        action_log_p = F.log_softmax(act[slice(*self.action_head)], dim=-1)
        # print(list(zip(self._as.action_types, action_log_p.exp().detach().numpy())))
        unary_op_log_p = F.log_softmax(act[slice(*self.unary_ops_head)], dim=-1)
        # print(list(zip(self._as.unary_ops, unary_op_log_p.exp().detach().numpy())))

        binary_op_log_p = F.log_softmax(act[slice(*self.binary_ops_head)], dim=-1)
        # print(list(zip(self._as.binary_ops, binary_op_log_p.exp().detach().numpy())))

        buf_log_p = F.log_softmax(act[slice(*self.buf_head)], dim=-1)
        # print(list(zip(self._buf_space, buf_log_p.exp().detach().numpy())))

        mem1_log_p = F.log_softmax(act[slice(*self.mem1_head)], dim=-1)
        # print(list(zip(self._mem_space, mem1_log_p.exp().detach().numpy())))

        mem2_log_p = F.log_softmax(act[slice(*self.mem2_head)], dim=-1)

        # Calculate prob for the provided actions
        activations = []
        for action in self._as.avail_actions:

            total_log_p = action_log_p[self._action_type_space[action.name]]
            # DOn't know but I want to scale logp by the space size

            mem_param = 0
            for p in action.params:
                if p.slot == ActionParamSlot.BINARY_OP:
                    total_log_p = (
                        total_log_p + binary_op_log_p[self._binary_ops_space[p.value]]
                    )
                elif p.slot == ActionParamSlot.UNARY_OP:
                    total_log_p = (
                        total_log_p + unary_op_log_p[self._unary_ops_space[p.value]]
                    )

                elif p.slot == ActionParamSlot.BUF:
                    total_log_p = total_log_p + buf_log_p[self._buf_space[p.value]]
                elif p.slot == ActionParamSlot.MEM and mem_param == 0:
                    total_log_p = total_log_p + mem1_log_p[self._mem_space[p.value]]
                    mem_param += 1
                elif p.slot == ActionParamSlot.MEM and mem_param == 1:
                    total_log_p = total_log_p + mem2_log_p[self._mem_space[p.value]]

            activations.append(total_log_p)

        activations_t = torch.stack(activations)
        # TODO: need to combine those logp in more logp
        return activations_t

