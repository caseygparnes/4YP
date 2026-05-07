import argparse
import csv
import json
import types
from pathlib import Path
from typing import Optional, Dict, List, Any

import matplotlib.pyplot as plt
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

    # force CPU / no wandb for evaluation
    all_args.cuda = False
    all_args.use_wandb = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.n_render_rollout_threads = 1

    # restore path
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

    positions = [
        np.array([ag.state.p_pos.copy() for ag in world.agents], dtype=np.float32)
    ]
    infos_per_step = []
    rewards_per_step = []

    for _step in range(T):
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

        positions.append(
            np.array([ag.state.p_pos.copy() for ag in world.agents], dtype=np.float32)
        )

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

    return np.stack(positions, axis=0), infos_per_step, rewards_per_step, world


def summarize_episode(positions, infos_per_step, rewards_per_step, world):
    summary = {
        "episode_reward": float(np.sum(rewards_per_step)),
        "num_steps": int(len(rewards_per_step)),
        "start_x": float(positions[0, 0, 0]),
        "start_y": float(positions[0, 0, 1]),
        "end_x": float(positions[-1, 0, 0]),
        "end_y": float(positions[-1, 0, 1]),
    }

    radial_errs = []
    align_errs = []
    rhos = []
    obst_cols = []
    agent_cols = []

    for step_infos in infos_per_step:
        if len(step_infos) == 0:
            continue
        env_info = step_infos[0]
        if len(env_info) == 0:
            continue
        agent_info = env_info[0]

        radial_errs.append(float(agent_info.get("Radial_error", np.nan)))
        align_errs.append(float(agent_info.get("Alignment_error", np.nan)))
        rhos.append(float(agent_info.get("Rho", np.nan)))
        obst_cols.append(float(agent_info.get("Num_obst_collisions", 0.0)))
        agent_cols.append(float(agent_info.get("Num_agent_collisions", 0.0)))

    def _safe_mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    final_rho = float(np.linalg.norm(positions[-1, 0] - np.asarray(world.orbit_center)))
    orbit_radius = float(getattr(world, "orbit_radius", 0.0))

    summary.update(
        {
            "avg_radial_error": _safe_mean(radial_errs),
            "avg_alignment_error": _safe_mean(align_errs),
            "avg_rho": _safe_mean(rhos),
            "final_rho": final_rho,
            "final_radial_error": abs(final_rho - orbit_radius),
            "total_obstacle_collisions": float(np.sum(obst_cols)),
            "total_agent_collisions": float(np.sum(agent_cols)),
        }
    )

    return summary


def plot_trajectory(positions, world, out_png: Path, title: str, show_plot: bool = False):
    center = np.array(getattr(world, "orbit_center", [0.0, 0.0]), dtype=np.float32)
    R = float(getattr(world, "orbit_radius", 1.0))

    _, N, _ = positions.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.get_cmap("tab10" if N <= 10 else "tab20")

    for i in range(N):
        color = cmap(i % cmap.N)
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            "-",
            linewidth=1.5,
            color=color,
            label=f"agent {i}",
        )
        ax.plot(positions[0, i, 0], positions[0, i, 1], "o", color=color)   # start
        ax.plot(positions[-1, i, 0], positions[-1, i, 1], "s", color=color) # end

    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        center[0] + (R+6) * np.cos(th),
        center[1] + (R+6) * np.sin(th),
        "k--",
        label=f"orbit R={R:g}",
    )

    if hasattr(world, "obstacles"):
        for obs in world.obstacles:
            p = obs.state.p_pos
            rad = float(getattr(obs, "size", 0.0))
            ax.add_patch(plt.Circle((p[0], p[1]), rad, color="r", alpha=0.35))

    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_side_by_side(
    positions_a,
    positions_b,
    world_a,
    world_b,
    label_a: str,
    label_b: str,
    out_png: Path,
    show_plot: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, positions, world, label in zip(
        axes,
        [positions_a, positions_b],
        [world_a, world_b],
        [label_a, label_b],
    ):
        center = np.array(getattr(world, "orbit_center", [0.0, 0.0]), dtype=np.float32)
        R = float(getattr(world, "orbit_radius", 1.0))

        for i in range(positions.shape[1]):
            ax.plot(positions[:, i, 0], positions[:, i, 1], linewidth=1.8, label=f"agent {i}")
            ax.plot(positions[0, i, 0], positions[0, i, 1], "o", label=f"start {i}")
            ax.plot(positions[-1, i, 0], positions[-1, i, 1], "s", label=f"end {i}")

        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(
            center[0] + R * np.cos(th),
            center[1] + R * np.sin(th),
            "k--",
            label=f"orbit R={R:g}",
        )

        if hasattr(world, "obstacles"):
            for obs in world.obstacles:
                p = obs.state.p_pos
                rad = float(getattr(obs, "size", 0.0))
                ax.add_patch(plt.Circle((p[0], p[1]), rad, alpha=0.30))

        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"[saved] {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


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


def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = {}
    if len(rows) == 0:
        return agg

    numeric_keys = [
        "episode_reward",
        "avg_radial_error",
        "avg_alignment_error",
        "avg_rho",
        "final_rho",
        "final_radial_error",
        "total_obstacle_collisions",
        "total_agent_collisions",
        "num_steps",
    ]

    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
        if len(vals) > 0:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return agg


def run_condition(
    runner,
    envs,
    all_args,
    label: str,
    angle_deg: float,
    num_episodes: int,
    deterministic: bool,
    max_steps: Optional[int],
    x0_std_override: Optional[float],
    x0_rad_override: Optional[float],
    out_dir: Path,
    show_plot: bool,
):
    patch_start_angle(
        envs,
        start_angle_deg=angle_deg,
        x0_std_override=x0_std_override,
        x0_rad_override=x0_rad_override,
    )

    condition_rows = []
    first_positions = None
    first_world = None

    for ep in range(num_episodes):
        positions, infos_per_step, rewards_per_step, world = rollout_episode(
            runner,
            envs,
            all_args,
            deterministic=deterministic,
            max_steps=max_steps,
        )

        if first_positions is None:
            first_positions = positions.copy()
            first_world = world

        summary = summarize_episode(positions, infos_per_step, rewards_per_step, world)
        summary["condition"] = label
        summary["episode_index"] = ep
        summary["start_angle_deg"] = float(angle_deg)
        summary["deterministic"] = bool(deterministic)
        summary["disturbance_enabled"] = bool(getattr(all_args, "use_disturbance", False))
        summary["x0_std_override"] = x0_std_override
        summary["x0_rad_override"] = x0_rad_override
        condition_rows.append(summary)

    

        print(
            f"[{label} | episode {ep}] "
            f"reward={summary['episode_reward']:.3f}, "
            f"avg_radial_error={summary['avg_radial_error']:.3f}, "
            f"avg_alignment_error={summary['avg_alignment_error']:.3f}, "
            f"total_obstacle_collisions={summary['total_obstacle_collisions']:.3f}"
        )

    return {
        "rows": condition_rows,
        "first_positions": first_positions,
        "first_world": first_world,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to .../experiment_name/runX")
    ap.add_argument("--original_angle_deg", type=float, default=0.0)
    ap.add_argument("--shifted_angle_deg", type=float, required=True)
    ap.add_argument("--x0_std_override", type=float, default=None)
    ap.add_argument("--x0_rad_override", type=float, default=None)
    ap.add_argument("--num_episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--disable_disturbance", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out_dir", type=str, default="generalization_start_angle_compare")
    ap.add_argument("--show_plot", action="store_true")
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

    original_result = run_condition(
        runner=runner,
        envs=envs,
        all_args=all_args,
        label="original",
        angle_deg=args.original_angle_deg,
        num_episodes=args.num_episodes,
        deterministic=deterministic,
        max_steps=args.max_steps,
        x0_std_override=args.x0_std_override,
        x0_rad_override=args.x0_rad_override,
        out_dir=out_dir / "original",
        show_plot=args.show_plot,
    )

    shifted_result = run_condition(
        runner=runner,
        envs=envs,
        all_args=all_args,
        label="shifted",
        angle_deg=args.shifted_angle_deg,
        num_episodes=args.num_episodes,
        deterministic=deterministic,
        max_steps=args.max_steps,
        x0_std_override=args.x0_std_override,
        x0_rad_override=args.x0_rad_override,
        out_dir=out_dir / "shifted",
        show_plot=args.show_plot,
    )

    all_rows = original_result["rows"] + shifted_result["rows"]
    save_csv(all_rows, out_dir / "summary_all.csv")
    save_json(all_rows, out_dir / "summary_all.json")

    original_agg = aggregate_results(original_result["rows"])
    shifted_agg = aggregate_results(shifted_result["rows"])

    aggregate = {
        "run_dir": str(run_dir),
        "num_episodes_per_condition": int(args.num_episodes),
        "deterministic": bool(deterministic),
        "disturbance_enabled": bool(getattr(all_args, "use_disturbance", False)),
        "x0_std_override": args.x0_std_override,
        "x0_rad_override": args.x0_rad_override,
        "original": {
            "angle_deg": float(args.original_angle_deg),
            "aggregate": original_agg,
        },
        "shifted": {
            "angle_deg": float(args.shifted_angle_deg),
            "aggregate": shifted_agg,
        },
    }

    save_json(aggregate, out_dir / "aggregate_compare.json")

    if shifted_result["first_positions"] is not None:
        plot_trajectory(
            shifted_result["first_positions"],
            shifted_result["first_world"],
            out_png=out_dir / "shifted_trajectory.png",
            title=f"Example Trajectory: Shifted by {args.shifted_angle_deg:g}°",
            show_plot=args.show_plot,
        )

    print(json.dumps(aggregate, indent=2))

    try:
        envs.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()