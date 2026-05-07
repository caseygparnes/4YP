import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ensure repo root is on path
sys.path.append(os.path.abspath(os.getcwd()))


def load_run_args(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


def override_for_eval(all_args: argparse.Namespace, num_agents: int):
    """Keep everything the same as training except override N and force CPU/no-wandb + eval threads."""
    a = copy.deepcopy(all_args)
    a.num_agents = int(num_agents)

    # force CPU + no wandb (consistent with your repo's train_mpe device selection)
    a.cuda = False
    a.use_wandb = False

    # eval env threads
    a.n_eval_rollout_threads = 1

    # make_eval_env uses n_eval_rollout_threads, but some wrappers also look at n_rollout_threads
    if hasattr(a, "n_rollout_threads"):
        a.n_rollout_threads = 1
    if hasattr(a, "n_training_threads"):
        a.n_training_threads = 1

    # ensure eval enabled (not strictly necessary but consistent)
    a.use_eval = True
    return a


def make_eval_env_from_repo(all_args: argparse.Namespace):
    """Use the exact env constructor used in training code."""
    from onpolicy.scripts.train_mpe import make_eval_env
    return make_eval_env(all_args)  # :contentReference[oaicite:3]{index=3}


def unwrap_base_env(vec_env):
    """Return the underlying single env object."""
    e = vec_env
    if hasattr(e, "envs") and len(e.envs) > 0:
        e = e.envs[0]
    # try common wrappers
    for _ in range(6):
        if hasattr(e, "env"):
            e = e.env
            continue
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
            continue
        break
    return e


def get_world(vec_env):
    base = unwrap_base_env(vec_env)
    if hasattr(base, "world"):
        return base.world
    # sometimes nested deeper
    if hasattr(base, "env") and hasattr(base.env, "world"):
        return base.env.world
    raise RuntimeError("Could not locate .world on the environment for trajectory plotting.")


def get_spaces(vec_env):
    """Extract spaces the same way training does (spaces are lists indexed by agent)."""
    base = unwrap_base_env(vec_env)

    # action_space / observation_space in wrappers are typically lists length num_agents
    obs_space = vec_env.observation_space[0] if hasattr(vec_env, "observation_space") else base.observation_space[0]
    cent_obs_space = vec_env.share_observation_space[0] if hasattr(vec_env, "share_observation_space") else base.share_observation_space[0]

    # graph spaces come from MultiAgentGraphEnv.set_graph_obs_space() :contentReference[oaicite:4]{index=4}
    node_obs_space = None
    edge_obs_space = None
    if hasattr(vec_env, "node_observation_space"):
        node_obs_space = vec_env.node_observation_space[0]
    elif hasattr(base, "node_observation_space"):
        node_obs_space = base.node_observation_space[0]

    if hasattr(vec_env, "edge_observation_space"):
        edge_obs_space = vec_env.edge_observation_space[0]
    elif hasattr(base, "edge_observation_space"):
        edge_obs_space = base.edge_observation_space[0]

    act_space = vec_env.action_space[0] if hasattr(vec_env, "action_space") else base.action_space[0]

    if node_obs_space is None or edge_obs_space is None:
        raise RuntimeError(
            "Could not find node_observation_space / edge_observation_space on env. "
            "Your MultiAgentGraphEnv should define these in set_graph_obs_space()."
        )

    return obs_space, cent_obs_space, node_obs_space, edge_obs_space, act_space


def build_policy(all_args: argparse.Namespace, vec_env, device):
    """Instantiate MAD_MAPPOPolicy exactly as in your repo."""
    from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy
    obs_space, cent_obs_space, node_obs_space, edge_obs_space, act_space = get_spaces(vec_env)
    policy = MAD_MAPPOPolicy(
        all_args,
        obs_space,
        cent_obs_space,
        node_obs_space,
        edge_obs_space,
        act_space,
        device=device,
    )  # 
    # evaluation mode
    policy.actor.eval()
    policy.critic.eval()
    policy.actor.under_training = False
    return policy


def load_actor_critic(policy, run_dir: Path, device):
    models_dir = run_dir / "models"
    actor_path = models_dir / "actor.pt"
    critic_path = models_dir / "critic.pt"

    if not actor_path.exists():
        raise FileNotFoundError(f"Expected actor.pt in: {models_dir}")

    actor_sd = torch.load(actor_path, map_location=device)
    policy.actor.load_state_dict(actor_sd, strict=False)

    # IMPORTANT:
    # Do NOT load critic when changing N, because the centralized critic input dimension
    # depends on num_agents (share_obs is concatenated across agents).
    # This makes critic weight shapes incompatible across N.
    if critic_path.exists():
        print("[WARN] Skipping critic load because critic input dim changes with num_agents.")


def init_eval_states(all_args: argparse.Namespace, policy, num_agents: int, device):
    """Match GMPERunner.eval() state handling."""
    n_threads = 1
    recurrent_N = int(getattr(all_args, "recurrent_N", 1))
    hidden_size = int(getattr(all_args, "hidden_size", 1))

    eval_rnn_states = np.zeros((n_threads, num_agents, recurrent_N, hidden_size), dtype=np.float32)
    eval_masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

    # SSM hidden state: runner stores complex state as (.., hidden_dim, 2) then converts to torch.complex 
    # infer hidden_dim from actor.ssm.LRUR.state_features if available
    ssm_hidden_dim = None
    if hasattr(policy.actor, "ssm") and hasattr(policy.actor.ssm, "LRUR") and hasattr(policy.actor.ssm.LRUR, "state_features"):
        ssm_hidden_dim = int(policy.actor.ssm.LRUR.state_features)

    if ssm_hidden_dim is None:
        # fallback: try common arg name
        ssm_hidden_dim = int(getattr(all_args, "lru_state_features", 1))

    eval_ssm_states = np.zeros((n_threads, num_agents, ssm_hidden_dim, 2), dtype=np.float32)

    return eval_rnn_states, eval_masks, eval_ssm_states


@torch.no_grad()
def run_eval_for_N(run_dir: Path, base_args: argparse.Namespace, N: int, episodes: int, deterministic: bool):
    device = torch.device("cpu")

    argsN = override_for_eval(base_args, N)
    envs = make_eval_env_from_repo(argsN)

    policy = build_policy(argsN, envs, device)
    load_actor_critic(policy, run_dir, device)

    episode_length = int(getattr(argsN, "episode_length", 1000))

    # for plotting: capture trajectories from FIRST episode only
    world = get_world(envs)

    # metrics accumulators
    ep_returns = []          # list of (N,) total reward per agent per episode
    traj_positions = None    # (T, N, 2) for episode 0 only (T varies if we break early)

    for ep in range(episodes):
        # reset
        eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_disturbances = envs.reset()

        # init states
        eval_rnn_states, eval_masks, eval_ssm_states = init_eval_states(argsN, policy, N, device)

        # Start-of-episode position (ONLY for ep==0)
        if ep == 0:
            traj_positions = [
                np.stack([a.state.p_pos.copy() for a in world.agents], axis=0)
            ]

        # accumulate per-episode return per agent
        ret = np.zeros((N,), dtype=np.float64)

        for t in range(episode_length):
            # convert ssm state to torch.complex as runner does
            ssm_states = torch.complex(
                torch.from_numpy(np.concatenate(eval_ssm_states)[..., 0]),
                torch.from_numpy(np.concatenate(eval_ssm_states)[..., 1]),
            ).to(device)

            # call MAD_MAPPOPolicy.act with correct signature
            eval_action, eval_rnn_states_out, ssm_states_out = policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                ssm_states,
                np.concatenate(eval_disturbances),
                np.concatenate(eval_masks),
                deterministic=deterministic,
            )

            # convert back to numpy, and split back into (n_threads=1, N, ...)
            eval_actions = np.array(np.split(eval_action.detach().cpu().numpy(), 1))
            eval_rnn_states = np.array(np.split(eval_rnn_states_out.detach().cpu().numpy(), 1))

            # convert SSM back to numpy [.., hidden_dim, 2] as runner does
            ssm_states_np = torch.stack([ssm_states_out.real, ssm_states_out.imag], dim=-1)
            eval_ssm_states = np.array(np.split(ssm_states_np.detach().cpu().numpy(), 1))

            # action space is Box -> pass actions directly
            eval_actions_env = eval_actions

            # step
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_disturbances,
                eval_rewards,
                eval_dones,
                eval_infos,
            ) = envs.step(eval_actions_env)

            # rewards shape is (1, N, 1) -> accumulate per agent
            r = np.asarray(eval_rewards).reshape(1, N, 1)[0, :, 0]
            ret += r

            # dones shape usually (1, N) or (1, N, 1)
            done_mask = np.asarray(eval_dones).astype(bool)
            if done_mask.ndim == 3:
                done_mask = done_mask.squeeze(-1)  # (1, N, 1) -> (1, N)

            # IMPORTANT: if done happened, env likely auto-reset inside step().
            # Do NOT record positions after this, and stop this episode.
            if done_mask.any():
                break

            # Only append positions if we are still in the same episode (no reset)
            if ep == 0:
                pos = np.stack([a.state.p_pos.copy() for a in world.agents], axis=0)
                traj_positions.append(pos)

            # reset RNN/SSM on done (same as runner) - usually redundant since we break, but safe
            d = done_mask.reshape(1, N)[0]

            if d.any():
                eval_rnn_states[0, d] = 0.0
                eval_ssm_states[0, d] = 0.0

            eval_masks = np.ones((1, N, 1), dtype=np.float32)
            eval_masks[0, d, 0] = 0.0

        ep_returns.append(ret.copy())

    envs.close()

    ep_returns = np.stack(ep_returns, axis=0)  # (episodes, N)
    mean_ep_reward_per_agent = ep_returns.mean(axis=0)  # (N,)
    mean_ep_loss_per_agent = -mean_ep_reward_per_agent  # loss = -reward
    mean_step_loss_per_agent = mean_ep_loss_per_agent / float(episode_length)

    traj_positions = np.stack(traj_positions, axis=0) if traj_positions is not None else None

    return {
        "N": N,
        "mean_ep_reward_per_agent": mean_ep_reward_per_agent,
        "mean_ep_loss_per_agent": mean_ep_loss_per_agent,
        "mean_step_loss_per_agent": mean_step_loss_per_agent,
        "avg_loss_per_agent": float(mean_ep_loss_per_agent.mean()),
        "traj_positions": traj_positions,
        "args": argsN,
    }


def plot_trajectories(run_dir: Path, results_by_N):
    out_dir = run_dir / "generalization_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    Ns = sorted(results_by_N.keys())

    for N in Ns:
        res = results_by_N[N]
        positions = res["traj_positions"]  # (T, N, 2)
        argsN = res["args"]

        center = np.array(getattr(argsN, "orbit_center", [0.0, 0.0]), dtype=np.float32)
        R = float(getattr(argsN, "orbit_radius", 1.0))

        fig, ax = plt.subplots(figsize=(7, 7))
        cmap = plt.cm.get_cmap("tab10" if N <= 10 else "tab20")

        # agent trajectories + start/end markers
        for i in range(positions.shape[1]):
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

        # orbit circle
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(
            center[0] + (R+6) * np.cos(th),
            center[1] + (R+6) * np.sin(th),
            "k--",
            label=f"orbit R={R:g}"
        )

        # obstacles
        try:
            env_tmp = make_eval_env_from_repo(argsN)
            world_tmp = get_world(env_tmp)
            if hasattr(world_tmp, "obstacles"):
                for obs in world_tmp.obstacles:
                    p = obs.state.p_pos
                    rad = float(getattr(obs, "size", 0.0))
                    ax.add_patch(plt.Circle((p[0], p[1]), rad, color="r", alpha=0.35))
            env_tmp.close()
        except Exception:
            pass

        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Example Trajectory, N={N}")
        ax.legend(ncol=2, fontsize="small")
        fig.tight_layout()

        out_path = out_dir / f"trajectory_N_{N}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved trajectory plot: {out_path}")



def plot_loss_bar(run_dir: Path, results_by_N):
    out_dir = run_dir / "generalization_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "avg_loss_per_agent_vs_N.png"

    Ns = sorted(results_by_N.keys())
    ys = [results_by_N[N]["avg_loss_per_agent"] for N in Ns]

    plt.figure(figsize=(8, 5))
    plt.bar(Ns, ys)
    plt.xlabel("Number of agents (N)")
    plt.ylabel("Average episode loss per agent (=-reward)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved loss plot: {out_path}")

def save_loss_table(run_dir: Path, results_by_N):
    out_dir = run_dir / "generalization_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "avg_loss_per_agent_vs_N.csv"

    import csv

    rows = []
    for N in sorted(results_by_N.keys()):
        res = results_by_N[N]
        row = {
            "N": N,
            "avg_loss_per_agent_episode": res["avg_loss_per_agent"],
        }
        for i, v in enumerate(res["mean_ep_loss_per_agent"]):
            row[f"agent_{i}_mean_episode_loss"] = float(v)
        for i, v in enumerate(res["mean_step_loss_per_agent"]):
            row[f"agent_{i}_mean_step_loss"] = float(v)
        rows.append(row)

    # Build a header that includes all agent columns across all N
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Saved loss table: {csv_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--Ns", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_args = load_run_args(run_dir)

    results_by_N = {}

    for N in args.Ns:
        print(f"\n[INFO] Evaluating N={N}")
        res = run_eval_for_N(
            run_dir=run_dir,
            base_args=base_args,
            N=N,
            episodes=args.episodes,
            deterministic=args.deterministic,
        )
        results_by_N[N] = res

        print(f"[RESULT] N={N} avg episode loss/agent = {res['avg_loss_per_agent']:.6f}")
        print("         per-agent mean episode loss:", np.round(res["mean_ep_loss_per_agent"], 4).tolist())
        print("         per-agent mean step loss:   ", np.round(res["mean_step_loss_per_agent"], 6).tolist())

    plot_trajectories(run_dir, results_by_N)
    plot_loss_bar(run_dir, results_by_N)
    save_loss_table(run_dir, results_by_N)

    print("\n=== Summary (avg episode loss per agent) ===")
    for N in sorted(results_by_N.keys()):
        print(f"N={N}: {results_by_N[N]['avg_loss_per_agent']:.6f}")


if __name__ == "__main__":
    main()