"""Reinforcement learning engine"""
from marty.critic import CriticNet
from marty.policy import ActionSpace, PolicyNetwork
from marty.tokenizer import Tokenizer
import torch
from torch.optim.adam import Adam
from typing import List, NamedTuple
from marty.types import ActionTrace, ParseOutcome
from .actions import Context, Action
from torch import FloatStorage, nn
from .curiosity import ForwardDynamics, InverseDynamics, IntrinsicCuriosity
from marty.layers.context import ContextEncoder
from marty.diagnostics import (
    display_state,
    get_global_step,
    increase_global_step,
    writer,
)
from torch.nn.utils import clip_grad_value_
from copy import deepcopy

step = 0


class Losses(NamedTuple):
    actor: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor

    def total(self):
        return self.actor + self.critic + self.entropy


class ACEngine:

    entropy_beta = 0.5
    clip_grad = 100.0
    bellman_updates = 1

    def __init__(self, mem_slots, action_space: ActionSpace):
        encoding_size = 128
        self._state_size = encoding_size
        self._ac = action_space
        self._tokenizer = Tokenizer(
            {"1", "egg", "BLANK", "ING", "CARD", "QTY"},
            max_sent_length=action_space.max_buffer_size,
        )

        # We may need two completely separate network,
        # one for value, and one for policy??
        self._ctx_encoder = ContextEncoder(
            tokenizer=self._tokenizer,
            embedding_dim=encoding_size,
            mem_slots=mem_slots,
            max_mem_size=4,
        )

        self._value_net = CriticNet(deepcopy(self._ctx_encoder))

        self._curiosity = IntrinsicCuriosity(
            inverse_dynamics=InverseDynamics(
                encoding_size, len(action_space.avail_actions)
            ),
            forward_dynamics=ForwardDynamics(
                encoding_size, len(action_space.avail_actions)
            ),
        )

        self._policy_net = PolicyNetwork(action_space, encoding_size)

        self._opt_ctx = Adam(list(self._ctx_encoder.parameters()), lr=0.001)

        self._opt_value = Adam(
            list(self._value_net.parameters()) + list(self._curiosity.fwd.parameters()),
            lr=0.001,
        )
        self._opt_actor = Adam(
            list(self._policy_net.parameters())
            + list(self._curiosity.inv.parameters()),
            lr=0.01,
        )

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

        for i in range(bellman_updates):
            self._opt_ctx.zero_grad()
            self._opt_actor.zero_grad()
            self._opt_value.zero_grad()

            total = 0
            total_critic = 0
            total_actor = 0
            for l in self._calculate_losses(actions, episode_r):
                total += l.total()
                total_critic += l.critic
                total_actor += l.actor + l.entropy
                writer.add_scalar("policy_loss", l.actor, get_global_step())
                writer.add_scalar("value_loss", l.critic, get_global_step())
                writer.add_scalar("entropy_loss", l.entropy, get_global_step())
                writer.add_scalar("total_loss", l.total(), get_global_step())
                increase_global_step()

            total_critic.backward()
            self._opt_ctx.step()
            self._opt_actor.step()
            self._opt_value.step()

            self._opt_ctx.zero_grad()
            self._opt_actor.zero_grad()
            self._opt_value.zero_grad()

            total = 0
            total_critic = 0
            total_actor = 0
            for l in self._calculate_losses(actions, episode_r):
                total += l.total()
                total_critic += l.critic
                total_actor += l.actor + l.entropy
                writer.add_scalar("policy_loss", l.actor, get_global_step())
                writer.add_scalar("value_loss", l.critic, get_global_step())
                writer.add_scalar("entropy_loss", l.entropy, get_global_step())
                writer.add_scalar("total_loss", l.total(), get_global_step())
                increase_global_step()

            total_actor.backward()
            self._opt_ctx.step()
            self._opt_actor.step()
            self._opt_value.step()

    def _calculate_losses(self, actions, episode_r):
        R = episode_r
        # For the final state we use a constant
        next_state = torch.zeros(self._state_size)
        for trace in actions[::-1]:
            ctx_tensor = self._ctx_encoder(trace.ctx)
            state_rep = ctx_tensor[0]

            # We don't optimize the policy
            log_p = self._policy_net(state_rep)
            value = self._value_net(trace.ctx)
            display_state(state_rep, trace.ctx)
            p = torch.exp(log_p)

            # Logp and advantage.
            advantage = R - value
            policy_loss = -log_p[trace.action_ix] * advantage.detach()

            entropy_loss = self.entropy_beta * (p * log_p).sum()
            value_loss = advantage ** 2

            yield Losses(actor=policy_loss, entropy=entropy_loss, critic=value_loss)

            R = self._value_net(trace.ctx).detach()

    def policy(self, ctx: Context):
        print("ctx", ctx)
        ctx_t = self._ctx_encoder(ctx)
        activations = self._policy_net(ctx_t[0])
        return activations

    def _value(self, ctx: Context):
        # Return just the
        return self._value_net(ctx)

