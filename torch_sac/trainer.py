"""Training loop for the PyTorch SAC implementation."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
except ImportError:  # pragma: no cover - fall back to classic gym
    import gym

from .agent import SACAgent
from .config import TrainConfig
from .logger import CSVLogger
from .replay_buffer import ReplayBuffer
from .utils import env_reset, env_step, set_seed_everywhere


def evaluate_policy(
    env: "gym.Env", agent: SACAgent, episodes: int, deterministic: bool, seed: int | None
) -> Tuple[float, float]:
    returns = []
    lengths = []
    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observation = env_reset(env, seed=episode_seed)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = agent.act(observation, deterministic=deterministic)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, _ = env_step(env, action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    return float(np.mean(returns)), float(np.mean(lengths))


def train(config: TrainConfig) -> Path:
    """Run SAC training according to ``config`` and return the output directory."""
    run_root = Path(config.log_dir).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{config.env_id}_{time.strftime('%Y%m%d-%H%M%S')}_seed{config.seed}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    if config.update_every <= 0:
        raise ValueError("config.update_every must be positive")
    if config.gradient_steps <= 0:
        raise ValueError("config.gradient_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("config.batch_size must be positive")

    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    if config.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
        eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=config.max_episode_steps)

    set_seed_everywhere(config.seed)
    try:
        env.action_space.seed(config.seed)
    except AttributeError:
        pass
    try:
        eval_env.action_space.seed(config.seed + 1000)
    except AttributeError:
        pass

    replay_buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, config.replay_size)
    agent = SACAgent(
        env.observation_space,
        env.action_space,
        hidden_dims=config.hidden_dims,
        lr=config.lr,
        gamma=config.gamma,
        tau=config.tau,
        automatic_entropy_tuning=config.auto_entropy_tuning,
        target_entropy=config.target_entropy,
        device=config.device,
    )

    observation = env_reset(env, seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    completed_episodes = 0
    last_update_metrics: Dict[str, float] = {}

    eval_seed = config.seed + 10_000

    with CSVLogger(run_dir / "progress.csv") as logger:
        for step in range(1, config.total_steps + 1):
            if step <= config.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(observation, deterministic=False)
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_observation, reward, terminated, truncated, info = env_step(env, action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            timeout = bool(info.get("TimeLimit.truncated", False))
            done_for_buffer = 1.0 if terminated or (truncated and not timeout) else 0.0
            replay_buffer.add(observation, action, reward, next_observation, done_for_buffer)

            observation = next_observation
            if done:
                log_entry = {
                    "step": step,
                    "episodes": completed_episodes + 1,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "eval_return": math.nan,
                    "eval_length": math.nan,
                    "replay_size": len(replay_buffer),
                    "alpha": last_update_metrics.get("alpha", math.nan),
                    "policy_loss": last_update_metrics.get("policy_loss", math.nan),
                    "qf1_loss": last_update_metrics.get("qf1_loss", math.nan),
                    "qf2_loss": last_update_metrics.get("qf2_loss", math.nan),
                    "alpha_loss": last_update_metrics.get("alpha_loss", math.nan),
                    "entropy": last_update_metrics.get("entropy", math.nan),
                }
                logger.log(log_entry)
                observation = env_reset(env)
                episode_return = 0.0
                episode_length = 0
                completed_episodes += 1

            if (
                step >= config.update_after
                and step % config.update_every == 0
                and len(replay_buffer) >= config.batch_size
            ):
                metrics_accumulator: Dict[str, float] = defaultdict(float)
                for _ in range(config.gradient_steps):
                    batch = replay_buffer.sample(config.batch_size)
                    metrics = agent.update(batch)
                    for key, value in metrics.items():
                        metrics_accumulator[key] += value
                last_update_metrics = {
                    key: value / config.gradient_steps for key, value in metrics_accumulator.items()
                }

            if config.eval_interval and step % config.eval_interval == 0:
                eval_return, eval_length = evaluate_policy(
                    eval_env, agent, config.eval_episodes, config.deterministic_eval, eval_seed
                )
                eval_seed += config.eval_episodes
                log_entry = {
                    "step": step,
                    "episodes": completed_episodes,
                    "episode_return": math.nan,
                    "episode_length": math.nan,
                    "eval_return": eval_return,
                    "eval_length": eval_length,
                    "replay_size": len(replay_buffer),
                    "alpha": last_update_metrics.get("alpha", math.nan),
                    "policy_loss": last_update_metrics.get("policy_loss", math.nan),
                    "qf1_loss": last_update_metrics.get("qf1_loss", math.nan),
                    "qf2_loss": last_update_metrics.get("qf2_loss", math.nan),
                    "alpha_loss": last_update_metrics.get("alpha_loss", math.nan),
                    "entropy": last_update_metrics.get("entropy", math.nan),
                }
                logger.log(log_entry)
                print(
                    f"Step {step:,} | Episodes {completed_episodes} | "
                    f"Eval return {eval_return:.2f} | Eval length {eval_length:.1f}"
                )

            if config.save_interval and step % config.save_interval == 0:
                checkpoint_path = run_dir / f"checkpoint_{step:08d}.pt"
                agent.save(checkpoint_path)

        # Final evaluation at the end of training
        eval_return, eval_length = evaluate_policy(
            eval_env,
            agent,
            config.eval_episodes,
            config.deterministic_eval,
            eval_seed if config.eval_interval else None,
        )
        logger.log(
            {
                "step": config.total_steps,
                "episodes": completed_episodes,
                "episode_return": math.nan,
                "episode_length": math.nan,
                "eval_return": eval_return,
                "eval_length": eval_length,
                "replay_size": len(replay_buffer),
                "alpha": last_update_metrics.get("alpha", math.nan),
                "policy_loss": last_update_metrics.get("policy_loss", math.nan),
                "qf1_loss": last_update_metrics.get("qf1_loss", math.nan),
                "qf2_loss": last_update_metrics.get("qf2_loss", math.nan),
                "alpha_loss": last_update_metrics.get("alpha_loss", math.nan),
                "entropy": last_update_metrics.get("entropy", math.nan),
            }
        )
        agent.save(run_dir / "final_policy.pt")

    env.close()
    eval_env.close()
    return run_dir
