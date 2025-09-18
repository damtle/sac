"""Configuration dataclasses for the PyTorch SAC training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class TrainConfig:
    """Runtime configuration for :func:`torch_sac.trainer.train`."""

    env_id: str = "Walker2d-v4"
    seed: int = 0
    total_steps: int = 1_000_000
    start_steps: int = 10_000
    update_after: int = 1_000
    update_every: int = 1
    gradient_steps: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden_dims: Tuple[int, ...] = (256, 256)
    auto_entropy_tuning: bool = True
    target_entropy: float | None = None
    eval_interval: int = 5_000
    eval_episodes: int = 5
    deterministic_eval: bool = True
    save_interval: int = 100_000
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    device: str | None = None
    max_episode_steps: int | None = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["log_dir"] = str(self.log_dir)
        return data
