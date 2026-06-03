from isaacgym import gymapi  # must import before torch

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


@dataclass
class EpisodeMetrics:
    episode: int
    env_id: int
    survived_steps: int
    survival_time_sec: float
    reached_max_survival: bool
    fell: bool

    total_reward: float
    mean_reward: float

    distance_traveled: float
    mean_forward_velocity: float

    action_energy: float
    action_smoothness: float

    mechanical_energy: Optional[float]
    energy_per_meter: Optional[float]
    cost_of_transport: Optional[float]

    limp_recovered: Optional[bool]


def parse_eval_args_and_clean_argv():
    """
    Parse eval-only args, then remove them from sys.argv before calling the repo's get_args().
    This prevents get_args() from crashing on unknown args like --checkpoint-path.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Direct path to checkpoint .pt file, e.g. /content/drive/MyDrive/.../model_3000.pt",
    )

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="eval_results")

    parser.add_argument("--enable-limp-test", action="store_true")
    parser.add_argument("--limp-start-step", type=int, default=500)
    parser.add_argument("--limp-duration-steps", type=int, default=100)
    parser.add_argument("--limp-action-indices", type=str, default="")
    parser.add_argument("--limp-scale", type=float, default=0.2)
    parser.add_argument("--recovery-window-steps", type=int, default=200)
    parser.add_argument("--recovery-velocity-threshold", type=float, default=0.2)

    eval_args, remaining_argv = parser.parse_known_args()

    # Keep only args that the original repo's get_args() understands.
    sys.argv = [sys.argv[0]] + remaining_argv

    return eval_args


def attach_eval_args(repo_args, eval_args):
    for k, v in vars(eval_args).items():
        setattr(repo_args, k, v)

    return repo_args


def parse_int_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []

    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_policy_dt(env) -> float:
    """
    In your config:
        sim.dt = 0.001
        control.decimation = 10

    Therefore policy dt is usually 0.001 * 10 = 0.01 sec.
    Many LeggedRobot envs already expose env.dt, so prefer that.
    """
    if hasattr(env, "dt"):
        return float(env.dt)

    return float(env.cfg.sim.dt) * float(env.cfg.control.decimation)


def get_max_episode_steps(env) -> int:
    """
    LeggedRobot usually has max_episode_length.
    Otherwise compute it from episode_length_s / policy_dt.
    """
    if hasattr(env, "max_episode_length"):
        return int(env.max_episode_length)

    policy_dt = get_policy_dt(env)
    return int(env.cfg.env.episode_length_s / policy_dt)


def get_base_forward_velocity(env) -> torch.Tensor:
    """
    Your reward code uses env.base_lin_vel[:, 0], so this is the best source.
    Fall back to root_states[:, 7] if needed.
    """
    if hasattr(env, "base_lin_vel"):
        return env.base_lin_vel[:, 0]

    return env.root_states[:, 7]


def get_robot_mass(env, env_id: int) -> Optional[float]:
    """
    Try to get robot mass for cost of transport:
        COT = energy / (mass * g * distance)

    If unavailable, return None.
    """
    if hasattr(env, "body_mass"):
        mass = env.body_mass

        if torch.is_tensor(mass):
            if mass.numel() == 1:
                return float(mass.item())
            return float(mass[env_id].item())

        return float(mass)

    return None


def apply_forced_limp_to_actions(
    actions: torch.Tensor,
    active_steps: torch.Tensor,
    limp_start_step: int,
    limp_duration_steps: int,
    limp_action_indices: List[int],
    limp_scale: float,
) -> torch.Tensor:
    """
    Forces limp behavior by scaling selected action dimensions during a fixed time window.

    This is useful because your XBotLRadarMaskEnv has limp timers, but the code shown earlier
    did not clearly show the timers actually weakening the action/torque. This guarantees the
    eval perturbation is real.
    """
    if limp_duration_steps <= 0 or len(limp_action_indices) == 0:
        return actions

    in_limp_window = (
        (active_steps >= limp_start_step)
        & (active_steps < limp_start_step + limp_duration_steps)
    )

    if not in_limp_window.any():
        return actions

    actions = actions.clone()
    env_ids = in_limp_window.nonzero(as_tuple=False).flatten()

    for idx in limp_action_indices:
        if 0 <= idx < actions.shape[1]:
            actions[env_ids, idx] *= limp_scale

    return actions


def mark_env_limp_timers_if_available(
    env,
    active_steps: torch.Tensor,
    limp_start_step: int,
    limp_duration_steps: int,
    limp_action_indices: List[int],
):
    """
    Optional: if the env has joint_limp_timer and joint_limp_idx, mark them too.
    The actual perturbation is still applied directly to actions above.
    """
    if not hasattr(env, "joint_limp_timer") or not hasattr(env, "joint_limp_idx"):
        return

    if len(limp_action_indices) == 0:
        return

    should_start = active_steps == limp_start_step

    if should_start.any():
        env_ids = should_start.nonzero(as_tuple=False).flatten()
        env.joint_limp_timer[env_ids] = limp_duration_steps
        env.joint_limp_idx[env_ids] = limp_action_indices[0]


def summarize(episodes: List[EpisodeMetrics]):
    def valid_values(field):
        vals = []

        for ep in episodes:
            v = getattr(ep, field)

            if v is None:
                continue

            if isinstance(v, float) and np.isnan(v):
                continue

            vals.append(v)

        return vals

    def mean(field):
        vals = valid_values(field)
        return float(np.mean(vals)) if vals else None

    def std(field):
        vals = valid_values(field)
        return float(np.std(vals)) if vals else None

    n = len(episodes)

    reached_max = [ep.reached_max_survival for ep in episodes]
    fell = [ep.fell for ep in episodes]

    limp_vals = [ep.limp_recovered for ep in episodes if ep.limp_recovered is not None]

    return {
        "num_episodes": n,

        "mean_survival_time_sec": mean("survival_time_sec"),
        "std_survival_time_sec": std("survival_time_sec"),

        "mean_survived_steps": mean("survived_steps"),
        "std_survived_steps": std("survived_steps"),

        "percent_reaching_max_survival": 100.0 * float(np.mean(reached_max)) if n else None,
        "fall_rate_percent": 100.0 * float(np.mean(fell)) if n else None,

        "mean_total_reward": mean("total_reward"),
        "std_total_reward": std("total_reward"),

        "mean_reward": mean("mean_reward"),
        "std_reward": std("mean_reward"),

        "mean_distance_traveled": mean("distance_traveled"),
        "std_distance_traveled": std("distance_traveled"),

        "mean_forward_velocity": mean("mean_forward_velocity"),
        "std_forward_velocity": std("mean_forward_velocity"),

        "mean_action_energy": mean("action_energy"),
        "std_action_energy": std("action_energy"),

        "mean_action_smoothness": mean("action_smoothness"),
        "std_action_smoothness": std("action_smoothness"),

        "mean_mechanical_energy": mean("mechanical_energy"),
        "std_mechanical_energy": std("mechanical_energy"),

        "mean_energy_per_meter": mean("energy_per_meter"),
        "std_energy_per_meter": std("energy_per_meter"),

        "mean_cost_of_transport": mean("cost_of_transport"),
        "std_cost_of_transport": std("cost_of_transport"),

        "limp_recovery_rate_percent": (
            100.0 * float(np.mean(limp_vals)) if len(limp_vals) > 0 else None
        ),
    }


def save_outputs(episodes: List[EpisodeMetrics], summary: dict, output_dir: str, checkpoint_path: Optional[str]):
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_path is not None:
        ckpt_name = os.path.basename(checkpoint_path).replace(".pt", "")
    else:
        ckpt_name = "loaded_checkpoint"

    csv_path = os.path.join(output_dir, f"{ckpt_name}_episode_metrics.csv")
    json_path = os.path.join(output_dir, f"{ckpt_name}_summary_metrics.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(episodes[0]).keys()))
        writer.writeheader()

        for ep in episodes:
            writer.writerow(asdict(ep))

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-episode metrics to: {csv_path}")
    print(f"Saved summary metrics to: {json_path}")


def load_policy_runner(env, args):
    """
    If checkpoint_path is provided, create the runner WITHOUT task_registry auto-resume,
    then directly load that exact checkpoint.

    If checkpoint_path is not provided, fall back to the repo's normal --resume /
    --load_run / --checkpoint behavior.
    """
    direct_checkpoint = args.checkpoint_path

    if direct_checkpoint is not None:
        if not os.path.exists(direct_checkpoint):
            raise FileNotFoundError(f"Checkpoint path does not exist: {direct_checkpoint}")

        # Prevent task_registry from trying to auto-load some run based on --load_run.
        args.resume = False

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
    )

    if direct_checkpoint is not None:
        print(f"Loading direct checkpoint: {direct_checkpoint}")
        ppo_runner.load(direct_checkpoint)

    policy = ppo_runner.get_inference_policy(device=env.device)

    return ppo_runner, train_cfg, policy


def evaluate(args):
    print(f"Creating environment for task: {args.task}")

    env, env_cfg = task_registry.make_env(
        name=args.task,
        args=args,
    )

    ppo_runner, train_cfg, policy = load_policy_runner(env, args)

    device = env.device
    num_envs = env.num_envs
    policy_dt = get_policy_dt(env)

    max_episode_steps = get_max_episode_steps(env)

    if args.max_eval_steps is not None:
        max_episode_steps = min(max_episode_steps, int(args.max_eval_steps))

    limp_indices = parse_int_list(args.limp_action_indices)

    print("\n=== Eval Setup ===")
    print(f"task: {args.task}")
    print(f"num_envs: {num_envs}")
    print(f"checkpoint_path: {args.checkpoint_path}")
    print(f"policy_dt: {policy_dt}")
    print(f"max_episode_steps: {max_episode_steps}")
    print(f"max_survival_time_sec: {max_episode_steps * policy_dt:.2f}")
    print(f"episodes_to_collect: {args.episodes}")
    print(f"enable_limp_test: {args.enable_limp_test}")
    print(f"limp_action_indices: {limp_indices}")

    obs = env.get_observations()

    completed_episodes: List[EpisodeMetrics] = []

    active_steps = torch.zeros(num_envs, device=device, dtype=torch.long)

    active_reward_sum = torch.zeros(num_envs, device=device)
    active_distance = torch.zeros(num_envs, device=device)
    active_forward_vel_sum = torch.zeros(num_envs, device=device)

    active_action_energy = torch.zeros(num_envs, device=device)
    active_action_smoothness_sum = torch.zeros(num_envs, device=device)
    active_mechanical_energy = torch.zeros(num_envs, device=device)

    prev_actions = torch.zeros(num_envs, env.num_actions, device=device)

    # For limp recovery: accumulate velocity only in post-limp recovery window.
    recovery_vel_sum = torch.zeros(num_envs, device=device)
    recovery_vel_count = torch.zeros(num_envs, device=device)
    recovery_checked = torch.zeros(num_envs, device=device, dtype=torch.bool)
    recovery_success = torch.zeros(num_envs, device=device, dtype=torch.bool)

    while len(completed_episodes) < args.episodes:
        with torch.no_grad():
            actions = policy(obs)

        if args.enable_limp_test:
            mark_env_limp_timers_if_available(
                env=env,
                active_steps=active_steps,
                limp_start_step=args.limp_start_step,
                limp_duration_steps=args.limp_duration_steps,
                limp_action_indices=limp_indices,
            )

            actions = apply_forced_limp_to_actions(
                actions=actions,
                active_steps=active_steps,
                limp_start_step=args.limp_start_step,
                limp_duration_steps=args.limp_duration_steps,
                limp_action_indices=limp_indices,
                limp_scale=args.limp_scale,
            )

        # Use velocity before step, because legged_gym may auto-reset done envs inside env.step().
        forward_vel = get_base_forward_velocity(env)
        active_forward_vel_sum += forward_vel
        active_distance += forward_vel * policy_dt

        action_diff = torch.norm(actions - prev_actions, dim=1)
        active_action_smoothness_sum += action_diff
        active_action_energy += torch.sum(actions ** 2, dim=1)

        prev_actions = actions.clone()

        # Step environment.
        step_result = env.step(actions)

        if len(step_result) == 5:
            obs, privileged_obs, rewards, dones, infos = step_result
        elif len(step_result) == 4:
            obs, rewards, dones, infos = step_result
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")

        active_reward_sum += rewards
        active_steps += 1

        # Mechanical energy from torque * joint velocity, if available.
        if hasattr(env, "torques") and hasattr(env, "dof_vel"):
            power = torch.sum(torch.abs(env.torques * env.dof_vel), dim=1)
            active_mechanical_energy += power * policy_dt

        # Limp recovery tracking.
        if args.enable_limp_test:
            recovery_start = args.limp_start_step + args.limp_duration_steps
            recovery_end = recovery_start + args.recovery_window_steps

            in_recovery_window = (
                (active_steps >= recovery_start)
                & (active_steps < recovery_end)
            )

            if in_recovery_window.any():
                recovery_vel_sum[in_recovery_window] += get_base_forward_velocity(env)[in_recovery_window]
                recovery_vel_count[in_recovery_window] += 1

            should_check_recovery = (
                (active_steps >= recovery_end)
                & (~recovery_checked)
            )

            if should_check_recovery.any():
                env_ids = should_check_recovery.nonzero(as_tuple=False).flatten()

                mean_recovery_vel = (
                    recovery_vel_sum[env_ids]
                    / torch.clamp(recovery_vel_count[env_ids], min=1.0)
                )

                recovery_success[env_ids] = (
                    mean_recovery_vel >= args.recovery_velocity_threshold
                )

                recovery_checked[env_ids] = True

        # Natural env termination.
        done_envs = dones.nonzero(as_tuple=False).flatten()

        # Forced completion at max eval length.
        timeout_envs = (active_steps >= max_episode_steps).nonzero(as_tuple=False).flatten()

        all_done_envs = torch.unique(torch.cat([done_envs, timeout_envs]))

        if all_done_envs.numel() == 0:
            continue

        for env_id_tensor in all_done_envs:
            if len(completed_episodes) >= args.episodes:
                break

            env_id = int(env_id_tensor.item())

            steps = int(active_steps[env_id].item())
            survival_time = steps * policy_dt

            reached_max = steps >= max_episode_steps
            fell = not reached_max

            total_reward = float(active_reward_sum[env_id].item())
            mean_reward = total_reward / max(steps, 1)

            distance = float(active_distance[env_id].item())
            mean_forward_velocity = float(active_forward_vel_sum[env_id].item() / max(steps, 1))

            action_energy = float(active_action_energy[env_id].item())
            action_smoothness = float(active_action_smoothness_sum[env_id].item() / max(steps, 1))

            mechanical_energy = None
            energy_per_meter = None
            cot = None

            if hasattr(env, "torques") and hasattr(env, "dof_vel"):
                mechanical_energy = float(active_mechanical_energy[env_id].item())
                energy_per_meter = mechanical_energy / max(abs(distance), 1e-6)

                mass = get_robot_mass(env, env_id)

                if mass is not None:
                    cot = mechanical_energy / max(mass * 9.81 * abs(distance), 1e-6)

            limp_recovered = None
            if args.enable_limp_test:
                limp_recovered = bool(recovery_success[env_id].item())

            ep = EpisodeMetrics(
                episode=len(completed_episodes),
                env_id=env_id,
                survived_steps=steps,
                survival_time_sec=survival_time,
                reached_max_survival=reached_max,
                fell=fell,

                total_reward=total_reward,
                mean_reward=mean_reward,

                distance_traveled=distance,
                mean_forward_velocity=mean_forward_velocity,

                action_energy=action_energy,
                action_smoothness=action_smoothness,

                mechanical_energy=mechanical_energy,
                energy_per_meter=energy_per_meter,
                cost_of_transport=cot,

                limp_recovered=limp_recovered,
            )

            completed_episodes.append(ep)

            print(
                f"Episode {len(completed_episodes)}/{args.episodes} | "
                f"env={env_id} | "
                f"survival={survival_time:.2f}s | "
                f"distance={distance:.2f}m | "
                f"v_x={mean_forward_velocity:.3f} | "
                f"mean_reward={mean_reward:.3f} | "
                f"smooth={action_smoothness:.3f} | "
                f"fell={fell} | "
                f"limp_recovered={limp_recovered}"
            )

        # Reset our metric accumulators for completed envs.
        active_steps[all_done_envs] = 0
        active_reward_sum[all_done_envs] = 0.0
        active_distance[all_done_envs] = 0.0
        active_forward_vel_sum[all_done_envs] = 0.0
        active_action_energy[all_done_envs] = 0.0
        active_action_smoothness_sum[all_done_envs] = 0.0
        active_mechanical_energy[all_done_envs] = 0.0
        prev_actions[all_done_envs, :] = 0.0

        recovery_vel_sum[all_done_envs] = 0.0
        recovery_vel_count[all_done_envs] = 0.0
        recovery_checked[all_done_envs] = False
        recovery_success[all_done_envs] = False

        # If we forced completion before the env naturally reset, manually reset those envs.
        timeout_only_envs = timeout_envs[
            ~torch.isin(timeout_envs, done_envs)
        ] if timeout_envs.numel() > 0 and done_envs.numel() > 0 else timeout_envs

        if timeout_only_envs.numel() > 0 and hasattr(env, "reset_idx"):
            env.reset_idx(timeout_only_envs)
            obs = env.get_observations()

    summary = summarize(completed_episodes)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    save_outputs(
        episodes=completed_episodes,
        summary=summary,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    eval_args = parse_eval_args_and_clean_argv()
    repo_args = get_args()
    args = attach_eval_args(repo_args, eval_args)

    # Default to your XBot-L radar-mask task if no task is supplied.
    if not hasattr(args, "task") or args.task is None:
        args.task = "humanoid_ppo_radar_mask"

    evaluate(args)