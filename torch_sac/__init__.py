"""PyTorch implementation of Soft Actor-Critic tailored for modern Python."""

from .agent import SACAgent
from .analysis import (
    SummaryConfig,
    load_multiple_runs,
    load_progress_csv,
    plot_learning_curves,
    smooth_series,
    summarize_training_runs,
)
from .config import TrainConfig
from .trainer import train

__all__ = [
    "SACAgent",
    "TrainConfig",
    "train",
    "SummaryConfig",
    "load_multiple_runs",
    "load_progress_csv",
    "plot_learning_curves",
    "smooth_series",
    "summarize_training_runs",
]
