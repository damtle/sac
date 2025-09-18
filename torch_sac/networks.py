"""Neural network components for the PyTorch SAC implementation."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN_MAX = (-20.0, 2.0)
EPS = 1e-6


def _init_layer(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Configurable multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            linear = nn.Linear(in_dim, out_dim)
            _init_layer(linear)
            layers.append(linear)
            layers.append(activation())
        final_linear = nn.Linear(dims[-2], dims[-1])
        _init_layer(final_linear)
        layers.append(final_linear)
        if output_activation is not None:
            layers.append(output_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(inputs)


class GaussianPolicy(nn.Module):
    """Gaussian policy network with Tanh-squashed actions."""

    def __init__(
        self,
        observation_dim: int,
        action_space,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_bounds: Tuple[float, float] = LOG_STD_MIN_MAX,
    ) -> None:
        super().__init__()
        self.net = MLP(observation_dim, hidden_dims, output_dim=2 * action_space.shape[0])
        self.log_std_bounds = log_std_bounds
        high = np.asarray(action_space.high, dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        if high.shape != low.shape:
            raise ValueError("Action space high/low must have the same shape")
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def _get_mean_log_std(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.net(observation)
        mean, log_std = torch.chunk(output, chunks=2, dim=-1)
        log_std = torch.tanh(log_std)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1.0)
        return mean, log_std

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._get_mean_log_std(observation)
        std = torch.exp(log_std)
        return mean, std

    def sample(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample reparameterised actions and their log probabilities."""
        mean, log_std = self._get_mean_log_std(observation)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(torch.clamp(self.action_scale * (1 - y_t.pow(2)) + EPS, min=EPS))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self.forward(observation)
            return torch.tanh(mean) * self.action_scale + self.action_bias
        action, _, _ = self.sample(observation)
        return action


class QNetwork(nn.Module):
    """State-action value function parameterised by an MLP."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = MLP(observation_dim + action_dim, hidden_dims, output_dim=1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observation, action], dim=-1)
        return self.net(inputs)
