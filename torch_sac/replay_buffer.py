"""Simple replay buffer for PyTorch SAC."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer for storing environment transitions."""

    def __init__(
        self, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...], capacity: int
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._observations = np.zeros((self._capacity, *observation_shape), dtype=np.float32)
        self._next_observations = np.zeros_like(self._observations)
        self._actions = np.zeros((self._capacity, *action_shape), dtype=np.float32)
        self._rewards = np.zeros((self._capacity, 1), dtype=np.float32)
        self._dones = np.zeros((self._capacity, 1), dtype=np.float32)
        self._size = 0
        self._index = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: float,
    ) -> None:
        """Add a transition to the buffer."""
        self._observations[self._index] = observation
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._next_observations[self._index] = next_observation
        self._dones[self._index] = done

        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniformly sample a batch of transitions."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "observations": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_observations[indices],
            "dones": self._dones[indices],
        }
        return batch

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity
