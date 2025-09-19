"""Utility helpers for the PyTorch SAC implementation."""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Soft-update target network parameters."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * param.data)


def hard_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy parameters from ``source`` to ``target``."""
    target.load_state_dict(source.state_dict())


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a NumPy array to a float32 tensor on ``device``."""
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def env_reset(env: Any, seed: int | None = None) -> Any:
    """Gym/Gymnasium compatible environment reset."""
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None:
            try:
                env.seed(seed)
            except AttributeError:
                pass
        result = env.reset()
    if isinstance(result, tuple):
        observation, *_ = result
    else:
        observation = result
    return observation


def env_step(env: Any, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Step the environment with compatibility across Gym versions."""
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    else:
        observation, reward, done, info = result
        terminated, truncated = bool(done), False
    return observation, float(reward), bool(terminated), bool(truncated), info or {}
