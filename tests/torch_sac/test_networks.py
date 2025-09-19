from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch_sac.networks import GaussianPolicy, QNetwork


def _make_action_space(low: float, high: float, dim: int):
    class _DummySpace:
        def __init__(self) -> None:
            self.low = np.full(dim, low, dtype=np.float32)
            self.high = np.full(dim, high, dtype=np.float32)
            self.shape = (dim,)

    return _DummySpace()


def test_gaussian_policy_sample_shapes() -> None:
    action_space = _make_action_space(-1.0, 1.0, dim=2)
    policy = GaussianPolicy(observation_dim=3, action_space=action_space, hidden_dims=(32, 32))
    observation = torch.zeros(4, 3)
    action, log_prob, mean_action = policy.sample(observation)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4, 1)
    assert mean_action.shape == (4, 2)
    assert torch.all(action <= 1.0001)
    assert torch.all(action >= -1.0001)


def test_q_network_outputs_scalar() -> None:
    q_net = QNetwork(observation_dim=3, action_dim=2, hidden_dims=(16, 16))
    observation = torch.randn(5, 3)
    action = torch.randn(5, 2)
    q_values = q_net(observation, action)
    assert q_values.shape == (5, 1)
