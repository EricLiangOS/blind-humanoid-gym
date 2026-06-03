import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    DummyVecEnv = None


@dataclass
class EpisodeMetrics:
    episode: int
    survived_steps: int
    survival_time_sec: float
    reached_max_survival: bool
    fell: bool
    total_reward: float
    distance_traveled: float
    mean_forward_velocity: float
    velocity_tracking_error: float
    action_energy: float
    action_smoothness: float
    cost_of_transport_proxy: float
    limp_recovered: Optional[bool]


def get_dt(env) -> float:
    """
    Tries to infer simulation timestep.
    Falls back to 0.01 sec if unavailable.
    """
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "dt"):
        return float(unwrapped.dt)

    if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "opt"):
        return float(unwrapped.model.opt.timestep)

    return 0.01


def get_x_position(env) -> float:
    """
    MuJoCo humanoid usually stores root x-position in qpos[0].
    Adjust this if your env stores position differently.
    """
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
        return float(unwrapped.data.qpos[0])

    if hasattr(unwrapped, "get_body_com"):
        return float(unwrapped.get_body_com("torso")[0])

    return 0.0


def get_forward_velocity(env, prev_x: float, dt: float) -> float:
    x = get_x_position(env)
    return (x - prev_x) / max(dt, 1e-8)


def should_count_as_fall(done: bool, truncated: bool, info: Dict) -> bool:
    """
    Prefer explicit info fields if your env provides them.
    Otherwise assume done=True before max step means fall/failure.
    """
    for key in ["fell", "fall", "is_fallen", "terminated_due_to_fall"]:
        if key in info:
            return bool(info[key])

    if done and not truncated:
        return True

    return False


def apply_limp_to_action(
    action: np.ndarray,
    step: int,
    limp_start_step: int,
    limp_duration_steps: int,
    limp_action_indices: List[int],
    limp_scale: float,
) -> np.ndarray:
    """
    Simulates a limp by weakening selected action dimensions for a fixed time window.
    """
    if limp_duration_steps <= 0:
        return action

    in_limp_window = limp_start_step <= step < limp_start_step + limp_duration_steps

    if not in_limp_window:
        return action

    action = np.array(action, copy=True)

    for idx in limp_action_indices:
        if 0 <= idx < action.shape[0]:
            action[idx] *= limp_scale

    return action


def evaluate_one_episode(
    vec_env,
    model,
    episode_idx: int,
    max_steps: int,
    target_speed: Optional[float],
    enable_limp_test: bool,
    limp_start_step: int,
    limp_duration_steps: int,
    limp_action_indices: List[int],
    limp_scale: float,
    recovery_velocity_threshold: float,
    recovery_window_steps: int,
) -> EpisodeMetrics:
    # Stable-Baselines3 Vectorized Environment reset only returns obs
    obs = vec_env.reset()
    
    # Extract underlying un-vectorized environment for direct property extraction
    single_env = vec_env.envs[0]

    dt = get_dt(single_env)
    start_x = get_x_position(single_env)
    prev_x = start_x

    total_reward = 0.0
    forward_velocities = []
    actions = []
    action_diffs = []

    fell = False
    survived_steps = 0
    prev_action = None

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        # Convert action array shape from (1, action_dim) to (action_dim,)
        action = np.asarray(action)[0]

        if enable_limp_test:
            action = apply_limp_to_action(
                action=action,
                step=step,
                limp_start_step=limp_start_step,
                limp_duration_steps=limp_duration_steps,
                limp_action_indices=limp_action_indices,
                limp_scale=limp_scale,
            )

        # Feed back step with a batch dimension expected by DummyVecEnv
        obs, reward, done, infos = vec_env.step([action])
        
        # Unpack vectorized results
        reward = float(reward[0])
        info = infos[0]
        
        # Pull core evaluation statuses out safely
        terminated = info.get("terminated", done[0] and not info.get("TimeLimit.truncated", False))
        truncated = info.get("truncated", info.get("TimeLimit.truncated", False))

        x = get_x_position(single_env)
        vx = get_forward_velocity(single_env, prev_x, dt)
        prev_x = x

        forward_velocities.append(vx)
        actions.append(action)

        if prev_action is not None:
            action_diffs.append(np.linalg.norm(action - prev_action))
        prev_action = action.copy()

        total_reward += reward
        survived_steps = step + 1

        if terminated or truncated:
            fell = should_count_as_fall(terminated, truncated, info)
            break

    final_x = get_x_position(single_env)
    distance = final_x - start_x

    forward_velocities = np.array(forward_velocities, dtype=np.float64)
    actions = np.array(actions, dtype=np.float64)

    mean_forward_velocity = float(np.mean(forward_velocities)) if len(forward_velocities) else 0.0

    if target_speed is not None:
        velocity_tracking_error = float(np.mean(np.abs(forward_velocities - target_speed)))
    else:
        velocity_tracking_error = float("nan")

    action_energy = float(np.sum(np.square(actions))) if len(actions) else 0.0
    action_smoothness = float(np.mean(action_diffs)) if len(action_diffs) else 0.0
    cost_of_transport_proxy = action_energy / max(abs(distance), 1e-6)
    reached_max_survival = survived_steps >= max_steps

    # Robust Limp Recovery Check calculated at end of episode to avoid truncation issues
    limp_recovered = None
    if enable_limp_test:
        recovery_start = limp_start_step + limp_duration_steps
        recovery_end = recovery_start + recovery_window_steps
        
        if survived_steps >= recovery_end:
            post_limp_velocities = forward_velocities[recovery_start:recovery_end]
            if len(post_limp_velocities) > 0:
                mean_recovery_velocity = float(np.mean(post_limp_velocities))
                limp_recovered = mean_recovery_velocity >= recovery_velocity_threshold
            else:
                limp_recovered = False
        else:
            # Humanoid fell down before completing recovery window tracking
            limp_recovered = False

    return EpisodeMetrics(
        episode=episode_idx,
        survived_steps=survived_steps,
        survival_time_sec=survived_steps * dt,
        reached_max_survival=reached_max_survival,
        fell=fell,
        total_reward=total_reward,
        distance_traveled=distance,
        mean_forward_velocity=mean_forward_velocity,
        velocity_tracking_error=velocity_tracking_error,
        action_energy=action_energy,
        action_smoothness=action_smoothness,
        cost_of_transport_proxy=cost_of_transport_proxy,
        limp_recovered=limp_recovered,
    )


def summarize(metrics: List[EpisodeMetrics]) -> Dict:
    def mean_of(field):
        vals = [getattr(m, field) for m in metrics]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else None

    def std_of(field):
        vals = [getattr(m, field) for m in metrics]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.std(vals)) if vals else None

    n = len(metrics)

    reached_max = [m.reached_max_survival for m in metrics]
    falls = [m.fell for m in metrics]
    limp_values = [m.limp_recovered for m in metrics if m.limp_recovered is not None]

    summary = {
        "num_episodes": n,
        "mean_survival_time_sec": mean_of("survival_time_sec"),
        "std_survival_time_sec": std_of("survival_time_sec"),
        "mean_survived_steps": mean_of("survived_steps"),
        "percent_reaching_max_survival": 100.0 * np.mean(reached_max) if n else None,
        "fall_rate_percent": 100.0 * np.mean(falls) if n else None,
        "mean_total_reward": mean_of("total_reward"),
        "mean_distance_traveled": mean_of("distance_traveled"),
        "mean_forward_velocity": mean_of("mean_forward_velocity"),
        "mean_velocity_tracking_error": mean_of("velocity_tracking_error"),
        "mean_action_energy": mean_of("action_energy"),
        "mean_action_smoothness": mean_of("action_smoothness"),
        "mean_cost_of_transport_proxy": mean_of("cost_of_transport_proxy"),
    }

    if limp_values:
        summary["limp_recovery_rate_percent"] = 100.0 * np.mean(limp_values)
    else:
        summary["limp_recovery_rate_percent"] = None

    return summary


def save_episode_csv(metrics: List[EpisodeMetrics], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def save_summary_json(summary: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def load_model(checkpoint_path: str, env):
    if PPO is None:
        raise ImportError(
            "stable_baselines3 is not installed. Install it or replace load_model() "
            "with your own checkpoint-loading code."
        )
    return PPO.load(checkpoint_path, env=env, device="auto")


def parse_int_list(s: str) -> List[int]:
    if not s or s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="eval_results")

    # Limp recovery settings
    parser.add_argument("--enable-limp-test", action="store_true")
    parser.add_argument("--limp-start-step", type=int, default=200)
    parser.add_argument("--limp-duration-steps", type=int, default=100)
    parser.add_argument("--limp-action-indices", type=str, default="")
    parser.add_argument("--limp-scale", type=float, default=0.2)
    parser.add_argument("--recovery-velocity-threshold", type=float, default=0.3)
    parser.add_argument("--recovery-window-steps", type=int, default=100)

    args = parser.parse_args()

    if DummyVecEnv is None:
        raise ImportError("stable_baselines3 is missing. Please install dependencies.")

    # Instantiate single Gymnasium environment and safely wrap it into an SB3 Vectorized Env wrapper
    base_env = gym.make(args.env_id)
    env = DummyVecEnv([lambda: base_env])

    model = load_model(args.checkpoint, env)
    limp_action_indices = parse_int_list(args.limp_action_indices)
    all_metrics = []

    for ep in range(args.episodes):
        metrics = evaluate_one_episode(
            vec_env=env,
            model=model,
            episode_idx=ep,
            max_steps=args.max_steps,
            target_speed=args.target_speed,
            enable_limp_test=args.enable_limp_test,
            limp_start_step=args.limp_start_step,
            limp_duration_steps=args.limp_duration_steps,
            limp_action_indices=limp_action_indices,
            limp_scale=args.limp_scale,
            recovery_velocity_threshold=args.recovery_velocity_threshold,
            recovery_window_steps=args.recovery_window_steps,
        )
        all_metrics.append(metrics)

        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"survival={metrics.survival_time_sec:.2f}s, "
            f"distance={metrics.distance_traveled:.2f}m, "
            f"fell={metrics.fell}, "
            f"limp_recovered={metrics.limp_recovered}"
        )

    summary = summarize(all_metrics)

    checkpoint_name = os.path.basename(args.checkpoint).replace(".zip", "").replace(".pt", "")
    csv_path = os.path.join(args.output_dir, f"{checkpoint_name}_episodes.csv")
    json_path = os.path.join(args.output_dir, f"{checkpoint_name}_summary.json")

    save_episode_csv(all_metrics, csv_path)
    save_summary_json(summary, json_path)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-episode metrics to: {csv_path}")
    print(f"Saved summary metrics to: {json_path}")

    env.close()


if __name__ == "__main__":
    main()