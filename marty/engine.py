"""Reinforcement learning engine"""
from torch.nn import parameter
from torch.utils.tensorboard import summary
from marty.attention import DotProductAttention
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from marty.neural import ContextEncoder, ContextTensors, Flatten
from typing import List, Tuple
from marty.types import ActionTrace, ParseOutcome
from .actions import ActionParamSlot, Context, Action
from torch import nn
import torch.nn.functional as F
import numpy as np
from .curiosity import ForwardDynamics, InverseDynamics, IntrinsicCuriosity
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
step = 0


class ACEngine:

    entropy_beta = 0.5
    clip_grad = 100.0
    bellman_updates = 1

    def __init__(self, mem_slots, avail_actions: List[Action]):
        self._avail_actions = avail_actions

        encoding_size = 128
        hidden_size = 256
        self._state_size = hidden_size

        self._action_type_space = sorted(set([a.name for a in avail_actions]))

        self._ctx_encoder = ContextEncoder(
            memory_slots=mem_slots, hidden_size=hidden_size, encoding_size=encoding_size
        )

        self._att = DotProductAttention(hidden_size)
        self._value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )

        # How many actions we have? TODO: this should be fixed beforehand
        action_space = 45
        self._curiosity = IntrinsicCuriosity(
            inverse_dynamics=InverseDynamics(hidden_size * 3, action_space),
            forward_dynamics=ForwardDynamics(hidden_size * 3, action_space),
        )

        self._policy_net = PolicyNetwork(
            mem_slots, max_buf=10, inp_size=hidden_size * 5
        )

        self._opt = Adam(
            list(self._value_head.parameters())
            + list(self._ctx_encoder.parameters())
            + list(self._curiosity.fwd.parameters())
            + list(self._curiosity.inv.parameters())
        )

        self._opt_actor = Adam(
            list(self._policy_net.parameters())
            + list(self._curiosity.fwd.parameters())
            + list(self._curiosity.inv.parameters())
        )

    def learn(self, actions: List[ActionTrace], outc: ParseOutcome):

        episode_r = {
            ParseOutcome.CORRECT: 100.0,
            ParseOutcome.INCORRECT: 0.00,
            ParseOutcome.ERROR: -10.0,
            ParseOutcome.EXCEEDED: -1.0,
        }.get(outc, 0.0)

        global step
        step += 1
        writer.add_scalar("reward", episode_r, step)

        bellman_updates = 1
        print_stuff = False
        if outc == ParseOutcome.INCORRECT:
            bellman_updates = 1
            print_stuff = False
        if outc == ParseOutcome.CORRECT:
            bellman_updates = 10
            print_stuff = False

        for _ in range(bellman_updates):
            losses = []
            R = episode_r
            # For the final state we use a constant
            next_state = torch.zeros(self._state_size * 3)

            for trace in actions[::-1]:
                # Should I take the origin one or the next one?
                log_p = self.policy(trace.ctx)
                value = self._value(trace.ctx)
                state_rep = self._ctx_encoder(trace.ctx)[1].flatten()
                p = torch.exp(log_p)

                curiosity_fwd = self._curiosity.fwd_loss(
                    state_rep, next_state, p.detach()
                )
                curiosity_inv = self._curiosity.inv_loss(
                    state_rep, next_state, p.detach()
                )

                # Logp and advantage.
                # note R is the "next" reward (we're looping backwards)
                # and we add the curiosity reward
                advantage = curiosity_fwd.detach() + R - value
                policy_loss = -log_p[trace.action_ix] * advantage.detach()

                #  This is a bit of a problem here because we don't want them to be
                # exactly like that
                entropy_loss = (self.entropy_beta * p * log_p).sum()
                value_loss = advantage ** 2

                if print_stuff:
                    print(
                        "value",
                        value.item(),
                        "R",
                        float(R),
                        "context",
                        trace.ctx,
                        "action",
                        trace.action,
                        "chosen with prob",
                        log_p[trace.action_ix].exp().item(),
                        "advantage was",
                        advantage.item(),
                        "policy loss was",
                        policy_loss.item(),
                        "entropy loss was",
                        entropy_loss.item(),
                        "curiosity loss was",
                        curiosity_fwd.item(),
                        curiosity_inv.item(),
                        "p was",
                        p.detach().numpy(),
                        "sum was",
                        torch.exp(log_p).sum().item(),
                        "len logp",
                        len(log_p),
                    )

                total_loss = (
                    policy_loss
                    + value_loss
                    + entropy_loss
                    + curiosity_fwd
                    + curiosity_inv
                )

                losses.append(total_loss)
                # self._opt.zero_grad()
                # total_loss.backward()
                # self._opt.step()

                R = value.detach()
                next_state = state_rep.detach()

            self._opt.zero_grad()
            self._opt_actor.zero_grad()
            total_loss = torch.cat(losses).mean()
            total_loss.backward()
            self._opt.step()
            self._opt_actor.step()
            # # nn.utils.clip_grad_norm(self._opt.param_groups, self.clip_grad)
            # self._opt.step()

    def policy(self, ctx: Context):
        ctx_t = self._ctx_encoder(ctx)
        activations = self._policy_net(ctx_t, self._avail_actions)
        return activations

    def _value(self, ctx: Context):
        # Return just the
        ctx_t = self._ctx_encoder(ctx)
        combined = torch.cat([ctx_t.buffer, ctx_t.memory])
        attn_batch = self._att(combined.view(1, combined.size(0), combined.size(1)))
        attn = attn_batch[0].sum(0)

        return self._value_head(attn)


class PolicyNetwork(nn.Module):

    action_types = ["push", "op", "join", "produce"]
    unary_ops = ["ING", "CARD"]
    binary_ops = ["QTY"]

    def __init__(self, avail_mem: int, max_buf: int, inp_size: int):
        super().__init__()
        # Those are just to convert from the categorical values to integers
        self._action_type_space = {k: i for i, k in enumerate(self.action_types)}
        self._unary_ops_space = {k: i for i, k in enumerate(self.unary_ops)}
        self._binary_ops_space = {k: i for i, k in enumerate(self.binary_ops)}
        self._mem_space = list(range(avail_mem))
        self._buf_space = list(range(max_buf))

        # Those are the various heads for all possible actions and parameters
        self.action_head = nn.Linear(inp_size, len(self._action_type_space))
        self.unary_ops_head = nn.Linear(inp_size, len(self._unary_ops_space))
        self.binary_ops_head = nn.Linear(inp_size, len(self._binary_ops_space))
        self.buf_head = nn.Linear(inp_size, len(self._buf_space))
        self.mem1_head = nn.Linear(inp_size, len(self._mem_space))
        self.mem2_head = nn.Linear(inp_size, len(self._mem_space))

        # Attention, it would be better if the attention was pametrized for every
        # head.
        self._att = DotProductAttention(inp_size)

    def forward(self, ctx_t: ContextTensors, avail_actions: List[Action]):
        combined = torch.cat([ctx_t.buffer, ctx_t.memory]).flatten().unsqueeze(0)
        # Ideally the attention mechanism should be different for each "action head"
        attn_batch = self._att(combined.view(1, combined.size(0), combined.size(1)))
        attn = attn_batch[0].sum(0)

        action_log_p = F.log_softmax(self.action_head(attn))
        unary_op_log_p = F.log_softmax(self.unary_ops_head(attn))
        binary_op_log_p = F.log_softmax(self.binary_ops_head(attn))
        buf_log_p = F.log_softmax(self.buf_head(attn))

        mem1_log_p = F.log_softmax(self.mem1_head(attn))
        mem2_log_p = F.log_softmax(self.mem2_head(attn))

        # Calculate prob for the provided actions
        activations = []
        for action in avail_actions:

            total_log_p = action_log_p[self._action_type_space[action.name]]
            # DOn't know but I want to scale logp by the space size

            space_size = 1

            mem_param = 0
            for p in action.params:
                if p.slot == ActionParamSlot.BINARY_OP:
                    total_log_p = (
                        total_log_p + binary_op_log_p[self._binary_ops_space[p.value]]
                    )
                    space_size *= len(self._binary_ops_space)
                elif p.slot == ActionParamSlot.UNARY_OP:
                    total_log_p = (
                        total_log_p + unary_op_log_p[self._unary_ops_space[p.value]]
                    )
                    space_size *= len(self._unary_ops_space)

                elif p.slot == ActionParamSlot.BUF:
                    total_log_p = total_log_p + buf_log_p[self._buf_space[p.value]]
                    space_size *= len(self._buf_space)
                elif p.slot == ActionParamSlot.MEM and mem_param == 0:
                    total_log_p = total_log_p + mem1_log_p[self._mem_space[p.value]]
                    space_size *= len(self._mem_space)
                elif p.slot == ActionParamSlot.MEM and mem_param == 1:
                    total_log_p = total_log_p + mem2_log_p[self._mem_space[p.value]]
                    space_size *= len(self._mem_space)

            activations.append(total_log_p)

        activations_t = torch.stack(activations)
        # TODO: need to combine those logp in more logp
        return activations_t
