"""Reinforcement learning engine"""
from torch.nn.modules.linear import Linear
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from marty.tokenizer import Tokenizer
import torch
from torch.optim.adam import Adam
from typing import List
from marty.types import ActionTrace, ParseOutcome
from .actions import ActionParamSlot, Context, Action
from torch import nn
import torch.nn.functional as F
from .curiosity import ForwardDynamics, InverseDynamics, IntrinsicCuriosity
from marty.layers.context import ContextEncoder
from marty.diagnostics import get_global_step, increase_global_step, writer

step = 0


class ACEngine:

    entropy_beta = 0.1
    clip_grad = 100.0
    bellman_updates = 1

    def __init__(self, mem_slots, avail_actions: List[Action]):
        self._avail_actions = avail_actions

        encoding_size = 128
        max_sent_len = 10
        self._state_size = encoding_size

        self._action_type_space = sorted(set([a.name for a in avail_actions]))

        self._tokenizer = Tokenizer(
            {"1", "egg", "BLANK", "ING", "CARD", "QTY"}, max_sent_length=max_sent_len
        )

        self._ctx_encoder = ContextEncoder(
            tokenizer=self._tokenizer,
            embedding_dim=encoding_size,
            mem_slots=mem_slots,
            max_mem_size=4,
        )

        self._value_head = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.LeakyReLU(),
            nn.Linear(encoding_size, 1),
        )

        # How many actions we have? TODO: this should be fixed beforehand
        action_space = 45
        self._curiosity = IntrinsicCuriosity(
            inverse_dynamics=InverseDynamics(encoding_size, action_space),
            forward_dynamics=ForwardDynamics(encoding_size, action_space),
        )

        self._policy_net = PolicyNetwork(
            encoding_size, buf_size=max_sent_len, mem_space=mem_slots
        )

        self._opt_ctx = Adam(list(self._ctx_encoder.parameters()))

        self._opt_value = Adam(list(self._value_head.parameters()))

        self._opt_actor = Adam(list(self._policy_net.parameters()))

    def learn(self, actions: List[ActionTrace], outc: ParseOutcome):

        episode_r = {
            ParseOutcome.CORRECT: 100.0,
            ParseOutcome.INCORRECT: 1.00,
            ParseOutcome.ERROR: -1.0,
            ParseOutcome.EXCEEDED: -1.0,
        }.get(outc, 0.0)
        writer.add_scalar("reward", episode_r, get_global_step())
        increase_global_step()

        print("got reward", outc)
        bellman_updates = 1
        print_stuff = False
        if outc == ParseOutcome.INCORRECT:
            bellman_updates = 1
            print_stuff = False
        if outc == ParseOutcome.CORRECT:
            bellman_updates = 10
            print_stuff = False

        for _ in range(bellman_updates):
            self._opt_ctx.zero_grad()
            self._opt_actor.zero_grad()
            self._opt_value.zero_grad()

            losses = []
            R = episode_r
            # For the final state we use a constant
            next_state = torch.zeros(self._state_size)

            for trace in actions[::-1]:
                # Should I take the origin one or the next one?
                ctx_tensor = self._ctx_encoder(trace.ctx)
                log_p = self._policy_net(ctx_tensor, self._avail_actions)
                value = self._value_head(ctx_tensor[0])
                state_rep = ctx_tensor[0]

                p = torch.exp(log_p)

                # curiosity_fwd = self._curiosity.fwd_loss(
                #     state_rep, next_state, p.detach()
                # )
                # curiosity_inv = self._curiosity.inv_loss(
                #     state_rep, next_state, p.detach()
                # )

                # R += curiosity_fwd.detach()
                # Logp and advantage.
                # note R is the "next" reward (we're looping backwards)
                # and we add the curiosity reward
                advantage = episode_r - value
                policy_loss = -log_p[trace.action_ix] * advantage.detach()

                #  This is a bit of a problem here because we don't want them to be
                # exactly like that
                entropy_loss = self.entropy_beta * (p * log_p).sum()
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
                        # "curiosity loss was",
                        # curiosity_fwd.item(),
                        # curiosity_inv.item(),
                        "p was",
                        p.detach().numpy(),
                        "sum was",
                        torch.exp(log_p).sum().item(),
                        "len logp",
                        len(log_p),
                    )
                    input()

                total_loss = policy_loss + value_loss + entropy_loss
                writer.add_scalar("policy_loss", policy_loss, get_global_step())
                writer.add_scalar("value_loss", value_loss, get_global_step())
                writer.add_scalar("entropy_loss", entropy_loss, get_global_step())
                writer.add_scalar("total_loss", total_loss, get_global_step())
                increase_global_step()
                losses.append(total_loss)

                R = value.detach()
                next_state = state_rep.detach()

            total_loss = torch.cat(losses).mean()
            total_loss.backward()
            self._opt_ctx.step()
            self._opt_actor.step()
            self._opt_value.step()

    def policy(self, ctx: Context):
        ctx_t = self._ctx_encoder(ctx)
        activations = self._policy_net(ctx_t, self._avail_actions)
        return activations

    def _value(self, ctx: Context):
        # Return just the
        ctx_tensor = self._ctx_encoder(ctx)
        return self._value_head(ctx_tensor[0])


class PolicyNetwork(nn.Module):

    action_types = ["push", "op", "join", "produce"]
    unary_ops = ["ING", "CARD"]
    binary_ops = ["QTY"]

    def __init__(self, embedding_size: int, buf_size: int, mem_space: int):
        super().__init__()
        self.embedding_size = embedding_size

        tf_lay = TransformerEncoderLayer(
            d_model=embedding_size, nhead=4, dim_feedforward=256
        )

        self.tf = TransformerEncoder(tf_lay, 3)

        # Those are just to convert from the categorical values to integers
        self._action_type_space = {k: i for i, k in enumerate(self.action_types)}
        self._unary_ops_space = {k: i for i, k in enumerate(self.unary_ops)}
        self._binary_ops_space = {k: i for i, k in enumerate(self.binary_ops)}
        self._mem_space = list(range(mem_space))
        self._buf_space = list(range(buf_size))

        # Those are the various heads for all possible actions and parameters
        self.decision_layer = nn.Linear(
            embedding_size,
            len(self._action_type_space)
            + len(self._unary_ops_space)
            + len(self._binary_ops_space)
            + len(self._buf_space)
            + len(self._mem_space)
            + len(self._mem_space),
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
        self.mem2_head = ix, ix + len(self._mem_space)

    def forward(self, context_tensor, avail_actions: List[Action]):
        attn = self.tf(context_tensor.unsqueeze(1))[0, 0, :]
        act = self.decision_layer(attn)

        action_log_p = F.log_softmax(act[slice(*self.action_head)])
        unary_op_log_p = F.log_softmax(act[slice(*self.unary_ops_head)])
        binary_op_log_p = F.log_softmax(act[slice(*self.binary_ops_head)])
        buf_log_p = F.log_softmax(act[slice(*self.buf_head)])

        mem1_log_p = F.log_softmax(act[slice(*self.mem1_head)])
        mem2_log_p = F.log_softmax(act[slice(*self.mem2_head)])

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

