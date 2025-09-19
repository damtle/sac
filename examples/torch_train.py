"""Command line entry point for training the PyTorch SAC agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch_sac import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Soft Actor-Critic with PyTorch.")
    parser.add_argument("--env-id", default="Walker2d-v4", help="Gym environment id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total environment steps.")
    parser.add_argument("--start-steps", type=int, default=10_000, help="Steps of random exploration.")
    parser.add_argument("--update-after", type=int, default=1_000, help="Number of steps before updates start.")
    parser.add_argument("--update-every", type=int, default=1, help="Environment steps between gradient updates.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps to run when updating.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for critic updates.")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay buffer capacity.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target smoothing coefficient.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for all optimizers.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(256, 256),
        help="Hidden layer sizes of policy and critic networks.",
    )
    parser.add_argument("--eval-interval", type=int, default=5_000, help="How often to run evaluation (0 to disable).")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--save-interval", type=int, default=100_000, help="How often to checkpoint (0 to disable).")
    parser.add_argument("--log-dir", default="runs/torch_sac", help="Directory where run artifacts are stored.")
    parser.add_argument("--device", default=None, help="PyTorch device string (e.g. 'cpu' or 'cuda').")
    parser.add_argument("--target-entropy", type=float, default=None, help="Optional manual target entropy.")
    parser.add_argument(
        "--disable-auto-entropy",
        action="store_true",
        help="Disable automatic entropy coefficient tuning.",
    )
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Use stochastic actions during evaluation instead of deterministic ones.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None, help="Override environment time limit if provided."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        auto_entropy_tuning=not args.disable_auto_entropy,
        target_entropy=args.target_entropy,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        deterministic_eval=not args.stochastic_eval,
        save_interval=args.save_interval,
        log_dir=Path(args.log_dir),
        device=args.device,
        max_episode_steps=args.max_episode_steps,
    )
    run_dir = train(config)
    print(f"Training artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
