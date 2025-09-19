"""Soft Actor-Critic agent implemented with PyTorch."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import GaussianPolicy, QNetwork
from .utils import hard_update, soft_update, to_tensor


class SACAgent:
    """Encapsulates policy, critics and entropy tuning logic."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dims: tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        observation_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        self.policy = GaussianPolicy(observation_dim, action_space, hidden_dims).to(self.device)
        self.qf1 = QNetwork(observation_dim, action_dim, hidden_dims).to(self.device)
        self.qf2 = QNetwork(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_qf1 = QNetwork(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_qf2 = QNetwork(observation_dim, action_dim, hidden_dims).to(self.device)
        hard_update(self.qf1, self.target_qf1)
        hard_update(self.qf2, self.target_qf2)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        q_parameters = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        self.q_optimizer = Adam(q_parameters, lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            if target_entropy is None:
                target_entropy = -float(action_dim)
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
            self._alpha = None
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self._alpha = torch.tensor(alpha, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return self._alpha

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy.act(obs, deterministic=deterministic)
        self.policy.train()
        return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        observations = to_tensor(batch["observations"], self.device)
        actions = to_tensor(batch["actions"], self.device)
        rewards = to_tensor(batch["rewards"], self.device)
        next_observations = to_tensor(batch["next_observations"], self.device)
        dones = to_tensor(batch["dones"], self.device)

        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        next_observations = next_observations.view(batch_size, -1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_observations)
            target_q1 = self.target_qf1(next_observations, next_actions)
            target_q2 = self.target_qf2(next_observations, next_actions)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1.0 - dones) * self.gamma * target_v

        current_q1 = self.qf1(observations, actions)
        current_q2 = self.qf2(observations, actions)
        qf1_loss = F.mse_loss(current_q1, target_q)
        qf2_loss = F.mse_loss(current_q2, target_q)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_pi, _ = self.policy.sample(observations)
        min_q_new_actions = torch.min(self.qf1(observations, new_actions), self.qf2(observations, new_actions))
        policy_loss = (self.alpha.detach() * log_pi - min_q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss_value = float("nan")
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        soft_update(self.qf1, self.target_qf1, self.tau)
        soft_update(self.qf2, self.target_qf2, self.tau)

        return {
            "qf1_loss": float(qf1_loss.item()),
            "qf2_loss": float(qf2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(-log_pi.mean().item()),
            "alpha_loss": float(alpha_loss_value),
        }

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "target_qf1": self.target_qf1.state_dict(),
            "target_qf2": self.target_qf2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy() if self.log_alpha is not None else None,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
        }
        torch.save(checkpoint, Path(path))

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        self.target_qf1.load_state_dict(checkpoint["target_qf1"])
        self.target_qf2.load_state_dict(checkpoint["target_qf2"])
        if checkpoint.get("automatic_entropy_tuning", False) and self.log_alpha is not None:
            log_alpha = checkpoint.get("log_alpha")
            if log_alpha is not None:
                self.log_alpha.data = torch.as_tensor(log_alpha, device=self.device)

