from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from torch_sac.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample() -> None:
    buffer = ReplayBuffer((3,), (2,), capacity=5)
    for idx in range(5):
        observation = np.full(3, idx, dtype=np.float32)
        action = np.full(2, idx, dtype=np.float32)
        reward = float(idx)
        next_observation = observation + 1
        done = float(idx % 2)
        buffer.add(observation, action, reward, next_observation, done)

    batch = buffer.sample(batch_size=3)
    assert batch["observations"].shape == (3, 3)
    assert batch["actions"].shape == (3, 2)
    assert batch["rewards"].shape == (3, 1)
    assert batch["next_observations"].shape == (3, 3)
    assert batch["dones"].shape == (3, 1)
    assert batch["observations"].dtype == np.float32
    assert buffer.capacity == 5
    assert len(buffer) == 5


def test_sampling_from_empty_buffer_raises() -> None:
    buffer = ReplayBuffer((1,), (1,), capacity=1)
    with pytest.raises(ValueError):
        buffer.sample(1)
