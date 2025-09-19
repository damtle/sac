"""Generate publication-ready figures from torch_sac training logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - triggered when pandas is missing
    raise SystemExit(
        "pandas is required for torch_analyze; install it via `pip install pandas` or"
        " include it through requirements-windows.txt"
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to progress.csv files or directories that contain them.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for each run; must match the number of logs when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory where figures and tables will be written.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Moving average window used to smooth curves for visualisation.",
    )
    parser.add_argument(
        "--episode-window",
        type=int,
        default=50,
        help="How many of the most recent episodes to include in summary statistics.",
    )
    parser.add_argument(
        "--success-thresholds",
        type=float,
        nargs="*",
        default=(),
        help="Evaluation return targets (e.g. 3000) to compute time-to-threshold metrics.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Seaborn style to apply when rendering the figure.",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="deep",
        help="Color palette used when plotting multiple runs.",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for the learning curve figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in dots per inch.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title to add to the generated figure.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also export the summary table as LaTeX for inclusion in papers.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also export the summary table as GitHub-flavoured Markdown.",
    )
    return parser.parse_args()


def _resolve_paths(raw_paths: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            csv_candidate = path / "progress.csv"
            if not csv_candidate.exists():
                raise FileNotFoundError(f"Directory {path} does not contain a progress.csv file")
            path = csv_candidate
        elif path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file or directory, got: {path}")
        resolved.append(path)
    return resolved


def main() -> None:
    args = _parse_args()

    try:  # pragma: no cover - optional dependency for headless rendering
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=False)
    except ModuleNotFoundError:
        pass

    paths = _resolve_paths(args.logs)
    if args.labels and len(args.labels) != len(paths):
        raise ValueError("--labels must have the same length as the number of log files")

    from torch_sac.analysis import (
        SummaryConfig,
        load_multiple_runs,
        plot_learning_curves,
        summarize_training_runs,
    )

    combined = load_multiple_runs(paths, labels=args.labels)
    summary = summarize_training_runs(
        combined,
        config=SummaryConfig(
            episode_window=args.episode_window,
            success_thresholds=tuple(args.success_thresholds),
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_learning_curves(
        combined,
        smoothing_window=args.smoothing_window,
        style=args.style,
        palette=args.palette,
    )
    if args.title:
        figure.suptitle(args.title, fontsize=14, fontweight="bold")

    figure_path = output_dir / f"learning_curves.{args.figure_format}"
    figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(figure)

    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    if args.latex:
        summary.to_latex(output_dir / "summary_metrics.tex", index=False, float_format="{:.2f}".format)
    if args.markdown:
        summary.to_markdown(output_dir / "summary_metrics.md", index=False)

    # Echo a nicely formatted table to the terminal for quick inspection.
    print("\nSummary metrics:\n")
    with pd.option_context("display.float_format", "{:.2f}".format):  # type: ignore[attr-defined]
        print(summary.to_string(index=False))
    print(f"\nFigure saved to: {figure_path}")
    print(f"Summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()

