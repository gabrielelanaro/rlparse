"""IMplementing a nice curiosity module"""
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import Adam


class RewardGenerator(nn.Module):
    pass


class ForwardDynamics(nn.Module):
    def __init__(self, state_rep_dim: int, action_dim: int) -> None:
        super().__init__()
        # The state dynamics needs to be represented and from this we output another
        # state representation

        # Simplest way we use a linear model
        self.fc1 = nn.Linear(state_rep_dim + action_dim, state_rep_dim)

    def forward(self, state, action):
        input_ = torch.cat([state, action], dim=-1)
        return self.fc1(input_)


class InverseDynamics(nn.Module):
    def __init__(self, state_rep_dim: int, action_dim: int):
        super().__init__()
        # Given two states, we predict the action that was taken to get there.
        self.fc1 = nn.Linear(state_rep_dim * 2, action_dim)

    def forward(self, state, next_state):
        assert state.shape == next_state.shape, f"{state.shape}, {next_state.shape}"
        input_ = torch.cat([state, next_state], dim=-1)
        return self.fc1(input_)


class IntrinsicCuriosity:
    def __init__(
        self, inverse_dynamics: InverseDynamics, forward_dynamics: ForwardDynamics
    ):
        super().__init__()
        self.inv = inverse_dynamics
        self.fwd = forward_dynamics

    def inv_loss(self, state, next_state, action):
        pred_action = self.inv(state, next_state)
        return F.binary_cross_entropy_with_logits(pred_action, action)

    def fwd_loss(self, state, next_state, action):
        pred_state = self.fwd(state, action)
        return F.mse_loss(pred_state, next_state)
