from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from torch_sac.agent import SACAgent


class DummyObservationSpace:
    def __init__(self, dim: int) -> None:
        self.shape = (dim,)


class DummyActionSpace:
    def __init__(self, dim: int, low: float = -1.0, high: float = 1.0) -> None:
        self.low = np.full(dim, low, dtype=np.float32)
        self.high = np.full(dim, high, dtype=np.float32)
        self.shape = (dim,)


def _make_batch(batch_size: int, obs_dim: int, act_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "observations": rng.standard_normal((batch_size, obs_dim), dtype=np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(batch_size, act_dim)).astype(np.float32),
        "rewards": rng.standard_normal((batch_size, 1), dtype=np.float32),
        "next_observations": rng.standard_normal((batch_size, obs_dim), dtype=np.float32),
        "dones": rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32),
    }


def test_sac_agent_update_runs() -> None:
    obs_dim, act_dim = 5, 3
    agent = SACAgent(
        observation_space=DummyObservationSpace(obs_dim),
        action_space=DummyActionSpace(act_dim),
        hidden_dims=(32, 32),
        lr=5e-4,
        automatic_entropy_tuning=True,
        device="cpu",
    )
    batch = _make_batch(batch_size=64, obs_dim=obs_dim, act_dim=act_dim)
    metrics = agent.update(batch)
    assert "policy_loss" in metrics
    assert "qf1_loss" in metrics
    action = agent.act(np.zeros(obs_dim, dtype=np.float32))
    assert action.shape == (act_dim,)
