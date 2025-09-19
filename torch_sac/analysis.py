"""Utilities for turning SAC training logs into publication-ready figures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


try:  # pragma: no cover - optional dependency for analysis utilities
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAVE_PLOTTING = True
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade when plotting stack is absent
    plt = None  # type: ignore
    sns = None  # type: ignore
    _HAVE_PLOTTING = False


try:  # pragma: no cover - optional dependency for analysis utilities
    import pandas as pd
    _HAVE_PANDAS = True
except ModuleNotFoundError:  # pragma: no cover - allow importing the module without pandas installed
    pd = None  # type: ignore
    _HAVE_PANDAS = False


@dataclass(frozen=True)
class SummaryConfig:
    """Configuration controlling how summary tables are generated."""

    episode_window: int = 50
    success_thresholds: Sequence[float] = ()


def _require_pandas() -> None:
    if not _HAVE_PANDAS:  # pragma: no cover - triggered only when pandas is missing
        raise ImportError(
            "pandas is required for torch_sac.analysis; install it via "
            "`pip install pandas` or use requirements-windows.txt"
        )


def _require_plotting() -> None:
    if not _HAVE_PLOTTING:  # pragma: no cover - triggered only when plotting stack is missing
        raise ImportError(
            "matplotlib and seaborn are required for plotting; install them via "
            "`pip install matplotlib seaborn` or use requirements-windows.txt"
        )


def load_progress_csv(path: str | Path, run_label: str | None = None) -> "pd.DataFrame":
    """Load a ``progress.csv`` file produced by :mod:`torch_sac` training.

    Parameters
    ----------
    path:
        Path to the CSV file.
    run_label:
        Optional human-readable label that is stored in the ``run`` column for downstream
        aggregation. If omitted, the parent directory name is used.
    """

    _require_pandas()

    csv_path = Path(path)
    if csv_path.is_dir():
        csv_path = csv_path / "progress.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No progress.csv found at {csv_path}")

    frame = pd.read_csv(csv_path)
    if "step" not in frame.columns:
        raise ValueError("progress.csv is missing the required 'step' column")

    frame = frame.sort_values("step").reset_index(drop=True)
    frame["run"] = run_label or csv_path.parent.name
    return frame


def load_multiple_runs(paths: Iterable[str | Path], labels: Sequence[str] | None = None) -> "pd.DataFrame":
    """Load multiple progress logs and concatenate them with a ``run`` column."""

    _require_pandas()

    frames: List[pd.DataFrame] = []
    provided_labels = list(labels) if labels is not None else None

    for index, raw_path in enumerate(paths):
        label = provided_labels[index] if provided_labels else None
        frame = load_progress_csv(raw_path, run_label=label)
        frames.append(frame)

    if not frames:
        raise ValueError("No log files provided")

    return pd.concat(frames, ignore_index=True, sort=False)


def smooth_series(series: "pd.Series", window: int) -> "pd.Series":
    """Apply a simple moving average with ``window`` samples."""

    _require_pandas()

    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _split_training_eval(frame: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    _require_pandas()

    train_df = frame.dropna(subset=["episode_return"]) if "episode_return" in frame else pd.DataFrame()
    eval_df = frame.dropna(subset=["eval_return"]) if "eval_return" in frame else pd.DataFrame()
    return train_df, eval_df


def _format_threshold(threshold: float) -> str:
    formatted = f"{threshold:g}".replace("-", "neg")
    return formatted.replace(".", "p")


def summarize_training_runs(
    frame: "pd.DataFrame", config: SummaryConfig | None = None
) -> "pd.DataFrame":
    """Compute descriptive statistics for one or more SAC training runs."""

    _require_pandas()

    if "run" not in frame.columns:
        raise ValueError("Input frame must contain a 'run' column. Use load_progress_csv().")

    cfg = config or SummaryConfig()
    rows: List[dict[str, float | str]] = []

    for run, run_frame in frame.groupby("run", sort=False):
        train_df, eval_df = _split_training_eval(run_frame)
        metrics: dict[str, float | str] = {"run": run}

        if not eval_df.empty:
            eval_returns = eval_df["eval_return"].astype(float)
            eval_steps = eval_df["step"].astype(float)

            metrics["final_eval_return"] = float(eval_returns.iloc[-1])
            metrics["final_eval_length"] = float(eval_df["eval_length"].iloc[-1]) if "eval_length" in eval_df else math.nan
            best_idx = int(eval_returns.idxmax())
            metrics["best_eval_return"] = float(eval_returns.loc[best_idx])
            metrics["best_eval_step"] = float(eval_steps.loc[best_idx])
            if len(eval_df) >= 2:
                total_span = float(eval_steps.iloc[-1] - eval_steps.iloc[0])
                auc = float(np.trapz(eval_returns.values, eval_steps.values))
                metrics["normalized_eval_auc"] = auc / total_span if total_span > 0 else auc
            else:
                metrics["normalized_eval_auc"] = float(eval_returns.iloc[0])

            metrics["mean_eval_length"] = float(eval_df["eval_length"].mean()) if "eval_length" in eval_df else math.nan

            for threshold in cfg.success_thresholds:
                above = eval_df.loc[eval_returns >= threshold]
                key = f"step_to_return_{_format_threshold(threshold)}"
                metrics[key] = float(above["step"].iloc[0]) if not above.empty else math.nan
        else:
            metrics.update(
                {
                    "final_eval_return": math.nan,
                    "final_eval_length": math.nan,
                    "best_eval_return": math.nan,
                    "best_eval_step": math.nan,
                    "normalized_eval_auc": math.nan,
                    "mean_eval_length": math.nan,
                }
            )
            for threshold in cfg.success_thresholds:
                metrics[f"step_to_return_{_format_threshold(threshold)}"] = math.nan

        if not train_df.empty:
            last_window = train_df.tail(cfg.episode_window)
            episode_returns = last_window["episode_return"].astype(float)
            metrics["mean_episode_return_last_window"] = float(episode_returns.mean())
            metrics["median_episode_return_last_window"] = float(episode_returns.median())
            metrics["std_episode_return_last_window"] = float(episode_returns.std(ddof=0))
            metrics["episodes_recorded"] = float(len(train_df))
            if "episode_length" in last_window:
                metrics["mean_episode_length_last_window"] = float(last_window["episode_length"].mean())
            else:
                metrics["mean_episode_length_last_window"] = math.nan

            if len(last_window) >= 2:
                x = np.arange(len(last_window), dtype=np.float64)
                slope, _ = np.polyfit(x, episode_returns.values, deg=1)
                metrics["episode_return_trend_last_window"] = float(slope)
            else:
                metrics["episode_return_trend_last_window"] = math.nan
        else:
            metrics.update(
                {
                    "mean_episode_return_last_window": math.nan,
                    "median_episode_return_last_window": math.nan,
                    "std_episode_return_last_window": math.nan,
                    "episodes_recorded": 0.0,
                    "mean_episode_length_last_window": math.nan,
                    "episode_return_trend_last_window": math.nan,
                }
            )

        rows.append(metrics)

    return pd.DataFrame(rows)


def plot_learning_curves(
    frame: "pd.DataFrame",
    smoothing_window: int = 10,
    style: str = "whitegrid",
    palette: str | Sequence[str] = "deep",
    figsize: tuple[int, int] = (15, 9),
) -> "plt.Figure":
    """Create a multi-panel figure visualising SAC training dynamics."""

    _require_pandas()
    _require_plotting()

    if "run" not in frame.columns:
        raise ValueError("Input frame must contain a 'run' column. Use load_progress_csv().")

    sns.set_theme(style=style)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.reshape(2, 2)

    palette_iter = sns.color_palette(palette, n_colors=len(frame["run"].unique()))

    # Evaluation returns
    eval_ax = axes[0, 0]
    eval_ax.set_title("Evaluation return over training")
    eval_ax.set_xlabel("Environment steps")
    eval_ax.set_ylabel("Average return")

    for color, (run, run_frame) in zip(palette_iter, frame.groupby("run", sort=False)):
        _, eval_df = _split_training_eval(run_frame)
        if eval_df.empty:
            continue
        eval_steps = eval_df["step"].astype(float)
        eval_returns = eval_df["eval_return"].astype(float)
        eval_ax.plot(eval_steps, eval_returns, color=color, alpha=0.3, linewidth=1.0)
        smoothed = smooth_series(eval_returns, smoothing_window)
        eval_ax.plot(eval_steps, smoothed, color=color, linewidth=2.0, label=run)

    eval_ax.legend(loc="best", frameon=False)

    # Episode returns
    train_ax = axes[0, 1]
    train_ax.set_title("Training episode return")
    train_ax.set_xlabel("Environment steps")
    train_ax.set_ylabel("Return per episode")

    palette_iter = sns.color_palette(palette, n_colors=len(frame["run"].unique()))
    for color, (run, run_frame) in zip(palette_iter, frame.groupby("run", sort=False)):
        train_df, _ = _split_training_eval(run_frame)
        if train_df.empty:
            continue
        episode_steps = train_df["step"].astype(float)
        episode_returns = train_df["episode_return"].astype(float)
        train_ax.scatter(episode_steps, episode_returns, color=color, alpha=0.2, s=10)
        train_ax.plot(
            episode_steps,
            smooth_series(episode_returns, smoothing_window),
            color=color,
            linewidth=1.8,
            label=run,
        )

    if len(frame["run"].unique()) == 1:
        train_ax.legend(loc="best", frameon=False)

    # Critic and actor losses
    loss_ax = axes[1, 0]
    loss_ax.set_title("Optimization losses")
    loss_ax.set_xlabel("Environment steps")
    loss_ax.set_ylabel("Loss value")

    loss_columns = [col for col in ["policy_loss", "qf1_loss", "qf2_loss"] if col in frame.columns]
    if loss_columns:
        for column in loss_columns:
            grouped = frame.dropna(subset=[column]).groupby("step", as_index=False)[column].mean()
            if grouped.empty:
                continue
            loss_ax.plot(grouped["step"], grouped[column], label=column)
        loss_ax.legend(loc="best", frameon=False)
    else:
        loss_ax.text(0.5, 0.5, "No loss metrics recorded", ha="center", va="center")

    # Entropy temperature dynamics
    alpha_ax = axes[1, 1]
    alpha_ax.set_title("Entropy temperature and policy entropy")
    alpha_ax.set_xlabel("Environment steps")

    legend_handles: List = []
    legend_labels: List[str] = []

    if "alpha" in frame.columns:
        alpha_grouped = frame.dropna(subset=["alpha"]).groupby("step", as_index=False)["alpha"].mean()
        if not alpha_grouped.empty:
            (line,) = alpha_ax.plot(
                alpha_grouped["step"],
                alpha_grouped["alpha"],
                color="tab:blue",
                label="alpha",
            )
            legend_handles.append(line)
            legend_labels.append("alpha")
        alpha_ax.set_ylabel("Temperature (alpha)", color="tab:blue")
        alpha_ax.tick_params(axis="y", labelcolor="tab:blue")
    else:
        alpha_ax.set_ylabel("Temperature (alpha)")

    if "entropy" in frame.columns:
        entropy_ax = alpha_ax.twinx()
        entropy_grouped = frame.dropna(subset=["entropy"]).groupby("step", as_index=False)["entropy"].mean()
        if not entropy_grouped.empty:
            (entropy_line,) = entropy_ax.plot(
                entropy_grouped["step"],
                entropy_grouped["entropy"],
                color="tab:orange",
                linestyle="--",
                label="entropy",
            )
            legend_handles.append(entropy_line)
            legend_labels.append("entropy")
        entropy_ax.set_ylabel("Policy entropy", color="tab:orange")
        entropy_ax.tick_params(axis="y", labelcolor="tab:orange")
    if legend_handles:
        alpha_ax.legend(legend_handles, legend_labels, loc="best", frameon=False)
    else:
        alpha_ax.text(0.5, 0.5, "No entropy metrics recorded", transform=alpha_ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    sns.despine(fig)
    return fig


__all__ = [
    "SummaryConfig",
    "load_progress_csv",
    "load_multiple_runs",
    "smooth_series",
    "summarize_training_runs",
    "plot_learning_curves",
]

