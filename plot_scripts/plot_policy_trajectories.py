import argparse
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch


def load_run_args(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from onpolicy.config import get_config
    parser = get_config()
    all_args = parser.parse_known_args([])[0]

    # Merge saved config into args namespace
    for k, v in cfg.items():
        setattr(all_args, k, v)

    # Force CPU + no wandb, and single-env rollout
    all_args.cuda = False
    all_args.use_wandb = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.n_render_rollout_threads = 1
    # Force correct checkpoint directory for restore()
    all_args.model_dir = str((run_dir / "models").resolve())

    return all_args


def make_envs(all_args):
    # Reuse the repo's official env factory (same as training)
    from onpolicy.scripts.train_mpe import make_train_env
    return make_train_env(all_args)


def get_world_from_vecenv(envs):
    # GraphDummyVecEnv keeps underlying envs in .envs
    if hasattr(envs, "envs") and len(envs.envs) > 0:
        base = envs.envs[0]
    else:
        base = envs

    # Unwrap common wrappers
    while hasattr(base, "env"):
        base = base.env

    if not hasattr(base, "world"):
        raise AttributeError("Could not access base env .world for plotting.")
    return base.world



def _t2n(x):
    return x.detach().cpu().numpy()

@torch.no_grad()
def rollout_episode(runner, envs, all_args, deterministic=True, max_steps=None):
    world = get_world_from_vecenv(envs)

    # Reset env (GraphMPE signature)
    eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_disturbances = envs.reset()

    n_threads = eval_obs.shape[0]  # should be 1
    num_agents = int(all_args.num_agents)

    # RNN states follow runner.eval() shape
    eval_rnn_states = np.zeros(
        (n_threads, *runner.buffer.rnn_states.shape[2:]),
        dtype=np.float32,
    )
    eval_masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

    # SSM state (if MAD policy uses it)
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

    positions = [np.array([ag.state.p_pos.copy() for ag in world.agents], dtype=np.float32)]

    for step in range(T):
        runner.trainer.prep_rollout()

        if eval_ssm_states is not None:
            # numpy -> torch.complex (matches runner.eval)
            ssm_states = torch.complex(
                torch.from_numpy(np.concatenate(eval_ssm_states)[..., 0]).float(),
                torch.from_numpy(np.concatenate(eval_ssm_states)[..., 1]).float(),
            ).to(runner.device)

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

            # torch.complex -> numpy [.., 2]
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

        # Continuous action space: pass actions directly (same as runner.eval)
        (eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_disturbances,
        eval_rewards, eval_dones, eval_infos) = envs.step(eval_actions)

        done_mask = np.asarray(eval_dones).astype(bool)
        if done_mask.ndim == 3:
            done_mask = done_mask.squeeze(-1)  # (threads, agents, 1) -> (threads, agents)

        # If episode ended, env likely auto-reset inside step(). Don't record that reset state.
        if done_mask.any():
            break

        # Only append if we are still in the same episode
        positions.append(np.array([ag.state.p_pos.copy() for ag in world.agents], dtype=np.float32))

        num_done = int(done_mask.sum())

        if num_done > 0:
            eval_rnn_states[done_mask] = np.zeros((num_done, *eval_rnn_states.shape[2:]), dtype=np.float32)
            if eval_ssm_states is not None:
                eval_ssm_states[done_mask] = np.zeros((num_done, *eval_ssm_states.shape[2:]), dtype=np.float32)

        eval_masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
        if num_done > 0:
            eval_masks[done_mask] = 0.0

    return np.stack(positions, axis=0), world



def plot_trajectories(positions, world, out_png: Path, title: str):
    _, N, _ = positions.shape

    center = np.array(getattr(world, "orbit_center", [0.0, 0.0]), dtype=np.float32)
    R = float(getattr(world, "orbit_radius", 1.0))

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
            label=f"agent {i}"
        )
        ax.plot(positions[0, i, 0], positions[0, i, 1], "o", color=color)   # start
        ax.plot(positions[-1, i, 0], positions[-1, i, 1], "s", color=color) # end

    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        center[0] + (R+6) * np.cos(th),
        center[1] + (R+6) * np.sin(th),
        "k--",
        label=f"orbit R={R:g}"
    )

    if hasattr(world, "obstacles"):
        for obs in world.obstacles:
            p = obs.state.p_pos
            rad = float(getattr(obs, "size", 0.0))
            ax.add_patch(plt.Circle((p[0], p[1]), rad, color="r", alpha=0.35))

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_png}")
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to .../experiment_name/runX")
    ap.add_argument("--out_png", type=str, default="orbit_trajectory.png")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    all_args = load_run_args(run_dir)

    # Construct env + runner using the repo's actual Runner class
    envs = make_envs(all_args)

    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
    device = torch.device("cpu")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "device": device,
        "num_agents": int(all_args.num_agents),
        "run_dir": run_dir,  # MUST be Path for BaseRunner
    }
    runner = Runner(config)

    # Restore checkpoint (runner.restore prints where it loads from)
    runner.restore()

    positions, world = rollout_episode(
        runner, envs, all_args,
        deterministic=args.deterministic,
        max_steps=args.max_steps
    )

    plot_trajectories(
        positions, world,
        out_png=Path(args.out_png),
        title= f"Example Trajectory, N={all_args.num_agents}",
        #title=f"Policy rollout: {run_dir.name}",
    )


if __name__ == "__main__":
    main()