"""Tests for the torch_sac.analysis utilities."""

from __future__ import annotations

import math
from pathlib import Path

import pytest


pandas = pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from torch_sac.analysis import (  # noqa: E402  - imported after pytest.importorskip
    SummaryConfig,
    plot_learning_curves,
    smooth_series,
    summarize_training_runs,
)


def _make_dataframe() -> "pandas.DataFrame":
    data = {
        "step": [100, 200, 300, 400, 500, 600],
        "episodes": [1, 2, 2, 3, 3, 4],
        "episode_return": [100.0, 110.0, math.nan, 120.0, math.nan, 130.0],
        "episode_length": [10, 11, math.nan, 12, math.nan, 13],
        "eval_return": [math.nan, math.nan, 200.0, math.nan, 250.0, math.nan],
        "eval_length": [math.nan, math.nan, 100.0, math.nan, 110.0, math.nan],
        "policy_loss": [0.5, 0.4, 0.35, 0.3, 0.28, 0.25],
        "qf1_loss": [1.0, 0.8, 0.7, 0.6, 0.55, 0.5],
        "qf2_loss": [1.1, 0.9, 0.75, 0.65, 0.6, 0.55],
        "alpha": [0.2, 0.19, 0.18, 0.17, 0.16, 0.15],
        "entropy": [0.9, 0.85, 0.8, 0.78, 0.75, 0.72],
    }
    frame = pandas.DataFrame(data)
    frame["run"] = "seed-1"
    return frame


def test_smooth_series_applies_moving_average() -> None:
    series = pandas.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    smoothed = smooth_series(series, window=3)
    expected = pandas.Series([0.0, 0.5, 1.0, 2.0, 3.0])
    pandas.testing.assert_series_equal(smoothed.round(6), expected)


def test_summarize_training_runs_returns_key_metrics() -> None:
    frame = _make_dataframe()
    summary = summarize_training_runs(
        frame,
        config=SummaryConfig(episode_window=10, success_thresholds=(240.0,)),
    )
    assert summary.loc[0, "final_eval_return"] == pytest.approx(250.0)
    assert summary.loc[0, "best_eval_step"] == pytest.approx(500.0)
    assert summary.loc[0, "normalized_eval_auc"] == pytest.approx(225.0)
    assert summary.loc[0, "mean_episode_return_last_window"] == pytest.approx(115.0)
    assert summary.loc[0, "episode_return_trend_last_window"] == pytest.approx(10.0)
    assert summary.loc[0, "step_to_return_240"] == pytest.approx(500.0)


def test_plot_learning_curves_creates_figure(tmp_path: Path) -> None:
    frame = _make_dataframe()
    matplotlib.use("Agg", force=False)
    figure = plot_learning_curves(frame, smoothing_window=2)
    output_file = tmp_path / "curves.png"
    figure.savefig(output_file)
    assert output_file.exists()
    import matplotlib.pyplot as plt

    plt.close(figure)
