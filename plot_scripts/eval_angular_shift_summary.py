import argparse
import csv
import json
import types
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import yaml


def load_run_args(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from onpolicy.config import get_config

    parser = get_config()
    all_args = parser.parse_known_args([])[0]

    for k, v in cfg.items():
        setattr(all_args, k, v)

    all_args.cuda = False
    all_args.use_wandb = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.n_render_rollout_threads = 1
    all_args.model_dir = str((run_dir / "models").resolve())
    return all_args


def make_envs(all_args):
    from onpolicy.scripts.train_mpe import make_train_env
    return make_train_env(all_args)


def get_base_env_from_vecenv(envs):
    if hasattr(envs, "envs") and len(envs.envs) > 0:
        base = envs.envs[0]
    else:
        base = envs

    while hasattr(base, "env"):
        base = base.env
    return base


def get_world_from_vecenv(envs):
    base = get_base_env_from_vecenv(envs)
    if not hasattr(base, "world"):
        raise AttributeError("Could not access base env .world")
    return base.world


def _safe_norm(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(max(eps, float(np.sum(x * x)))))


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float32)


def patch_start_angle(
    envs,
    start_angle_deg: float,
    x0_std_override: Optional[float] = None,
    x0_rad_override: Optional[float] = None,
):
    """
    Override the scenario reset placement so the agents start at a chosen angle.
    """
    base_env = get_base_env_from_vecenv(envs)

    if not hasattr(base_env, "reset_callback") or base_env.reset_callback is None:
        raise AttributeError("Base env does not expose reset_callback.")

    scenario = getattr(base_env.reset_callback, "__self__", None)
    if scenario is None:
        raise AttributeError("Could not access scenario via base_env.reset_callback.__self__")

    start_angle_rad = np.deg2rad(float(start_angle_deg))
    original_x0_std = float(getattr(scenario, "x0_std", 0.0))
    original_x0_rad = float(getattr(scenario, "x0_rad", 0.0))

    def _place_agents_override(self, world):
        c = self.orbit_center
        N = self.num_agents

        x0_std = original_x0_std if x0_std_override is None else float(x0_std_override)
        x0_rad = original_x0_rad if x0_rad_override is None else float(x0_rad_override)

        base_angles = (
            np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).astype(np.float32)
            + start_angle_rad
        )

        for i, agent in enumerate(world.agents):
            anchor = c + x0_rad * np.array(
                [np.cos(base_angles[i]), np.sin(base_angles[i])],
                dtype=np.float32,
            )
            noise = np.random.normal(0.0, x0_std, size=(2,)).astype(np.float32)
            p = anchor + noise

            agent.state.p_pos = p

            r = p - c
            r_hat = r / _safe_norm(r)
            t_hat = self.orbit_dir * _rot90(r_hat)
            agent.state.p_vel = self.v_star * t_hat

            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)

    scenario._place_agents = types.MethodType(_place_agents_override, scenario)
    return scenario


def _t2n(x):
    return x.detach().cpu().numpy()


@torch.no_grad()
def rollout_episode(runner, envs, all_args, deterministic=True, max_steps=None):
    world = get_world_from_vecenv(envs)

    eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_disturbances = envs.reset()

    n_threads = eval_obs.shape[0]
    num_agents = int(all_args.num_agents)

    eval_rnn_states = np.zeros(
        (n_threads, *runner.buffer.rnn_states.shape[2:]),
        dtype=np.float32,
    )
    eval_masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

    if runner.buffer.lru_hidden_states is not None:
        eval_ssm_states = np.zeros(
            (n_threads, *runner.buffer.lru_hidden_states.shape[2:]),
            dtype=np.float32,
        )
    else:
        eval_ssm_states = None

    T = int(getattr(all_args, "episode_length", 200))
    if max_steps is not None:
        T = min(T, int(max_steps))

    infos_per_step = []
    rewards_per_step = []

    for _ in range(T):
        runner.trainer.prep_rollout()

        if eval_ssm_states is not None:
            eval_action, eval_rnn_states_out, ssm_states_out = runner.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                None,
                np.concatenate(eval_disturbances),
                np.concatenate(eval_masks),
                deterministic=deterministic,
            )

            ssm_states_np = torch.stack([ssm_states_out.real, ssm_states_out.imag], dim=-1)
            eval_ssm_states = np.array(np.split(_t2n(ssm_states_np), n_threads))
        else:
            eval_action, eval_rnn_states_out = runner.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=deterministic,
            )

        eval_actions = np.array(np.split(_t2n(eval_action), n_threads))
        eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_out), n_threads))

        (
            eval_obs,
            eval_agent_id,
            eval_node_obs,
            eval_adj,
            eval_disturbances,
            eval_rewards,
            eval_dones,
            eval_infos,
        ) = envs.step(eval_actions)

        rewards_per_step.append(float(np.asarray(eval_rewards).sum()))
        infos_per_step.append(eval_infos)

        done_mask = np.asarray(eval_dones).astype(bool)
        if done_mask.ndim == 3:
            done_mask = done_mask.squeeze(-1)

        if done_mask.any():
            break

        num_done = int(done_mask.sum())
        if num_done > 0:
            eval_rnn_states[done_mask] = np.zeros(
                (num_done, *eval_rnn_states.shape[2:]), dtype=np.float32
            )
            if eval_ssm_states is not None:
                eval_ssm_states[done_mask] = np.zeros(
                    (num_done, *eval_ssm_states.shape[2:]), dtype=np.float32
                )

        eval_masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
        if num_done > 0:
            eval_masks[done_mask] = 0.0

    return infos_per_step, rewards_per_step


def _extract_agent_infos(step_infos):
    """
    Tries to return the per-agent info dicts for the first environment thread.
    """
    if step_infos is None:
        return []

    env_info = step_infos
    if isinstance(step_infos, (list, tuple)) and len(step_infos) > 0:
        env_info = step_infos[0]

    if isinstance(env_info, dict):
        return [env_info]

    if isinstance(env_info, (list, tuple)):
        return [x for x in env_info if isinstance(x, dict)]

    return []


def summarize_episode(
    infos_per_step,
    rewards_per_step,
    num_agents: int,
    collision_key: str = "Num_obst_collisions",
) -> Dict[str, float]:
    episode_reward = float(np.sum(rewards_per_step))
    num_steps = int(len(rewards_per_step))
    denom_steps = max(num_steps, 1)

    total_obstacle_collisions = 0.0

    for step_infos in infos_per_step:
        agent_infos = _extract_agent_infos(step_infos)
        step_collision_total = 0.0
        for info in agent_infos:
            try:
                step_collision_total += float(info.get(collision_key, 0.0))
            except Exception:
                step_collision_total += 0.0
        total_obstacle_collisions += step_collision_total

    mean_cost_per_agent_per_step = (-episode_reward) / float(num_agents * denom_steps)
    obstacle_collisions_per_episode_per_agent = total_obstacle_collisions / float(num_agents)

    return {
        "episode_reward": episode_reward,
        "num_steps": float(num_steps),
        "mean_cost_per_agent_per_step": float(mean_cost_per_agent_per_step),
        "total_obstacle_collisions": float(total_obstacle_collisions),
        "obstacle_collisions_per_episode_per_agent": float(obstacle_collisions_per_episode_per_agent),
    }


def run_angle_condition(
    runner,
    envs,
    all_args,
    angle_deg: float,
    num_episodes: int,
    deterministic: bool,
    max_steps: Optional[int],
    x0_std_override: Optional[float],
    x0_rad_override: Optional[float],
    collision_key: str,
):
    patch_start_angle(
        envs,
        start_angle_deg=angle_deg,
        x0_std_override=x0_std_override,
        x0_rad_override=x0_rad_override,
    )

    rows = []
    num_agents = int(all_args.num_agents)

    for ep in range(num_episodes):
        infos_per_step, rewards_per_step = rollout_episode(
            runner,
            envs,
            all_args,
            deterministic=deterministic,
            max_steps=max_steps,
        )

        summary = summarize_episode(
            infos_per_step=infos_per_step,
            rewards_per_step=rewards_per_step,
            num_agents=num_agents,
            collision_key=collision_key,
        )

        row = {
            "angular_shift_degrees": float(angle_deg),
            "episode_index": int(ep),
            "episode_reward": summary["episode_reward"],
            "num_steps": int(summary["num_steps"]),
            "mean_cost_per_agent_per_step": summary["mean_cost_per_agent_per_step"],
            "total_obstacle_collisions": summary["total_obstacle_collisions"],
            "obstacle_collisions_per_episode_per_agent": summary["obstacle_collisions_per_episode_per_agent"],
        }
        rows.append(row)

        print(
            f"[angle={angle_deg:g}°, ep={ep}] "
            f"cost/agent/step={row['mean_cost_per_agent_per_step']:.6f}, "
            f"obst_collisions/ep/agent={row['obstacle_collisions_per_episode_per_agent']:.6f}"
        )

    mean_cost = float(np.mean([r["mean_cost_per_agent_per_step"] for r in rows])) if rows else float("nan")
    mean_obst = float(np.mean([r["obstacle_collisions_per_episode_per_agent"] for r in rows])) if rows else float("nan")
    std_cost = float(np.std([r["mean_cost_per_agent_per_step"] for r in rows])) if rows else float("nan")
    std_obst = float(np.std([r["obstacle_collisions_per_episode_per_agent"] for r in rows])) if rows else float("nan")

    return {
        "rows": rows,
        "summary": {
            "angular_shift_degrees": float(angle_deg),
            "mean_cost_per_agent_per_step": mean_cost,
            "mean_obstacle_collisions_per_episode_per_agent": mean_obst,
            "std_cost_per_agent_per_step": std_cost,
            "std_obstacle_collisions_per_episode_per_agent": std_obst,
            "num_episodes": int(num_episodes),
        },
    }


def save_csv(rows: List[Dict[str, Any]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_csv}")


def save_json(obj: Any, out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to .../experiment_name/runX")
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75])
    ap.add_argument("--x0_std_override", type=float, default=None)
    ap.add_argument("--x0_rad_override", type=float, default=None)
    ap.add_argument("--num_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--disable_disturbance", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--collision_key", type=str, default="Num_obst_collisions")
    ap.add_argument("--out_dir", type=str, default="generalization_start_angle_summary")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    all_args = load_run_args(run_dir)
    if args.disable_disturbance:
        all_args.use_disturbance = False

    envs = make_envs(all_args)

    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner

    device = torch.device("cpu")
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "device": device,
        "num_agents": int(all_args.num_agents),
        "run_dir": run_dir,
    }

    runner = Runner(config)
    runner.restore()

    deterministic = not args.stochastic

    all_episode_rows = []
    summary_rows = []

    angles = [float(a) for a in args.angles]

    for angle_deg in angles:
        result = run_angle_condition(
            runner=runner,
            envs=envs,
            all_args=all_args,
            angle_deg=angle_deg,
            num_episodes=args.num_episodes,
            deterministic=deterministic,
            max_steps=args.max_steps,
            x0_std_override=args.x0_std_override,
            x0_rad_override=args.x0_rad_override,
            collision_key=args.collision_key,
        )
        all_episode_rows.extend(result["rows"])
        summary_rows.append(result["summary"])

    summary_rows = sorted(summary_rows, key=lambda x: x["angular_shift_degrees"])

    baseline_row = None
    for row in summary_rows:
        if abs(row["angular_shift_degrees"]) < 1e-12:
            baseline_row = row
            break
    if baseline_row is None and len(summary_rows) > 0:
        baseline_row = summary_rows[0]

    baseline_cost = baseline_row["mean_cost_per_agent_per_step"] if baseline_row is not None else float("nan")

    final_summary_rows = []
    for row in summary_rows:
        cost = row["mean_cost_per_agent_per_step"]
        if np.isfinite(baseline_cost) and abs(baseline_cost) > 1e-12:
            pct_increase = 100.0 * (cost - baseline_cost) / baseline_cost
        else:
            pct_increase = float("nan")

        final_summary_rows.append({
            "angular_shift_degrees": row["angular_shift_degrees"],
            "mean_cost_per_agent_per_step": row["mean_cost_per_agent_per_step"],
            "pct_increase_relative_to_0deg": pct_increase,
            "mean_obstacle_collisions_per_episode_per_agent": row["mean_obstacle_collisions_per_episode_per_agent"],
            "std_cost_per_agent_per_step": row["std_cost_per_agent_per_step"],
            "std_obstacle_collisions_per_episode_per_agent": row["std_obstacle_collisions_per_episode_per_agent"],
            "num_episodes": row["num_episodes"],
        })

    save_csv(all_episode_rows, out_dir / "episode_level_results.csv")
    save_csv(final_summary_rows, out_dir / "angular_shift_summary.csv")

    save_json(
        {
            "run_dir": str(run_dir),
            "num_agents": int(all_args.num_agents),
            "angles": angles,
            "num_episodes_per_angle": int(args.num_episodes),
            "deterministic": bool(deterministic),
            "disturbance_enabled": bool(getattr(all_args, "use_disturbance", False)),
            "x0_std_override": args.x0_std_override,
            "x0_rad_override": args.x0_rad_override,
            "collision_key": args.collision_key,
            "baseline_cost_row": baseline_row,
            "summary_rows": final_summary_rows,
        },
        out_dir / "angular_shift_summary.json",
    )

    print("\nSummary table")
    print(
        f"{'Angle (deg)':>12} | "
        f"{'Mean cost / agent / step':>24} | "
        f"{'% increase vs 0°':>18} | "
        f"{'Mean obst. collisions / ep / agent':>34}"
    )
    print("-" * 96)
    for row in final_summary_rows:
        print(
            f"{row['angular_shift_degrees']:12.3f} | "
            f"{row['mean_cost_per_agent_per_step']:24.3f} | "
            f"{row['pct_increase_relative_to_0deg']:18.3f} | "
            f"{row['mean_obstacle_collisions_per_episode_per_agent']:34.3f}"
        )

    try:
        envs.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()