"""
Script to generate figure 1 in the paper.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from multiagent.custom_scenarios.orbit_graph import Scenario
from multiagent.environment import MultiAgentGraphEnv
from onpolicy.config import graph_config, get_config


def parse_args(args, parser):
    from distutils.util import strtobool

    parser.add_argument("--scenario_name", type=str, default="orbit_graph")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--collaborative", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--max_speed", type=float, default=2)
    parser.add_argument("--collision_rew", type=float, default=5)
    parser.add_argument("--goal_rew", type=float, default=5)
    parser.add_argument("--min_dist_thresh", type=float, default=0.05)
    parser.add_argument("--use_dones", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--use_trained_policy", action="store_true", default=False)

    # Experiment parameters
    parser.add_argument("--max_time_steps", type=int, default=50)

    parser.add_argument("--model_path", type=str, default="./onpolicy/results/GraphMPE/orbit_graph/rmappo/stable_gnn_train/run1/models/actor.pt")
    parser.add_argument("--model_path_informarl", type=str, default="./onpolicy/results/GraphMPE/orbit_graph/rmappo/informarl/run1/models/actor.pt")

    all_args = parser.parse_known_args(args)[0]
    return all_args, parser


def run_episode(env, policy, max_steps):
    """
    Run episode and return state-goal error norms over time.
    """
    reset_output = env.reset()
    if len(reset_output) == 5:
        obs_n, agent_id_n, node_obs_n, adj_n, disturbance_n = reset_output
    else:
        obs_n, agent_id_n, node_obs_n, adj_n = reset_output
        disturbance_n = None

    # Initialize states
    rnn_states = np.zeros((env.n, policy.actor._recurrent_N, policy.actor.hidden_size), dtype=np.float32)
    masks = np.zeros((env.n, 1), dtype=np.float32)
    ssm_states = [None] * env.n  # For MAD policy

    state_norms = []

    for step in range(max_steps):
        # Extract state errors for each agent
        # Orbit stability metric:
        # - radial error: rho - R
        # - alignment error: (1 - cos(v_hat, t_star))^2
        state_errors = []
        center = env.world.orbit_center
        R = env.world.orbit_radius
        orbit_dir = env.world.orbit_dir

        for i in range(env.n):
            vel = obs_n[i][0:2]
            pos = obs_n[i][2:4]

            rho = np.linalg.norm(pos - center)
            e_r = rho - R

            r_hat = (pos - center) / max(1e-8, rho)
            t_star = orbit_dir * np.array([-r_hat[1], r_hat[0]])
            v_norm = np.linalg.norm(vel)
            if v_norm < 1e-8:
                cos_align = 1.0
            else:
                v_hat = vel / v_norm
                cos_align = np.clip(np.dot(v_hat, t_star), -1.0, 1.0)

            align_err = (1.0 - cos_align) ** 2

            # per-agent error vector (2 dims)
            state_errors.append(np.array([e_r, align_err], dtype=np.float32))

        stacked_errors = np.concatenate(state_errors)  # (2*num_agents,)
        error_norm = np.linalg.norm(stacked_errors)
        state_norms.append(error_norm)
        act_n = []
        for i in range(env.n):
            with torch.no_grad():
                # Agent ID must be 1D array with shape (1,) for batch size 1
                # CRITICAL: Keep as 2D to prevent squeezing issues
                agent_id = np.array([[i]], dtype=np.int64)  # Shape: (1, 1)

                # Forward pass (MAD vs InforMARL)
                if hasattr(policy.actor, 'ssm'):
                    # MAD policy - needs disturbances
                    action, _, rnn_states_out, *extra = policy.actor.forward(
                        obs=obs_n[i][None, :],
                        node_obs=node_obs_n[i][None, :, :],
                        adj=adj_n[i][None, :, :],
                        agent_id=agent_id,
                        rnn_states=rnn_states[i:i+1],
                        ssm_states=ssm_states[i],
                        disturbances=disturbance_n[i][None, :],  # Exactly like test_model.py
                        masks=masks[i:i+1],
                        deterministic=True
                    )
                else:
                    # InforMARL policy - no disturbances
                    action, _, rnn_states_out, *extra = policy.actor.forward(
                        obs=obs_n[i][None, :],
                        node_obs=node_obs_n[i][None, :, :],
                        adj=adj_n[i][None, :, :],
                        agent_id=agent_id,
                        rnn_states=rnn_states[i:i+1],
                        masks=masks[i:i+1],
                        deterministic=True
                    )

                rnn_states[i:i+1] = rnn_states_out
                if hasattr(policy.actor, 'ssm') and len(extra) > 0:
                    ssm_states[i] = extra[0]

            act_n.append(action[0].detach().cpu().numpy())

        step_output = env.step(act_n)
        if len(step_output) == 8:
            obs_n, agent_id_n, node_obs_n, adj_n, disturbance_n, reward_n, done_n, info_n = step_output
        else:
            obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = step_output
            disturbance_n = None
        masks = np.ones((env.n, 1), dtype=np.float32)

        if all(done_n):
            break

    return state_norms


def main():
    args = sys.argv[1:]

    # Setup config
    parser = get_config()
    _, parser = parse_args(args, parser)
    all_args, parser = graph_config(args, parser)

    print("=" * 80)
    print("Comparing MAD Policy vs InforMARL Policy Stability")
    print("=" * 80)
    print(f"Number of agents: 1 to 10")
    print(f"Number of obstacles: {all_args.num_obstacles}")
    print(f"Episode length: {all_args.max_time_steps}")
    print(f"Metric: ||[e^0, ..., e^N]|| where e^i = [rel_goal, velocity]")
    print(f"        Both should converge to [0, 0, 0, 0] at goal")
    print("=" * 80)

    # Run experiments for different numbers of agents
    episode_length = all_args.max_time_steps
    mad_norm_trajectories = []
    informarl_norm_trajectories = []
    agent_counts = []

    print("\nRunning stability experiments...")
    for num_agents in range(1, 11):
        print(f"\n--- Testing with {num_agents} agent(s) ---")

        # Update agent count in args
        all_args.num_agents = num_agents

        # Create scenario and environment
        scenario = Scenario()
        world = scenario.make_world(all_args)
        env = MultiAgentGraphEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            graph_observation_callback=scenario.graph_observation,
            info_callback=scenario.info_callback,
            done_callback=scenario.done,
            id_callback=scenario.get_id,
            update_graph=scenario.update_graph,
            shared_viewer=False,
            discrete_action=False
        )

        # Create untrained policies
        mad_policy = MAD_MAPPOPolicy(
            all_args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.node_observation_space[0],
            env.edge_observation_space[0],
            env.action_space[0]
        )

        informarl_policy = GR_MAPPOPolicy(
            all_args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.node_observation_space[0],
            env.edge_observation_space[0],
            env.action_space[0]
        )

        if all_args.use_trained_policy:
            mad_policy.actor.load_state_dict(torch.load(all_args.model_path, map_location=torch.device('cpu')))
            informarl_policy.actor.load_state_dict(torch.load(all_args.model_path_informarl, map_location=torch.device('cpu')))

        mad_policy.actor.eval()
        informarl_policy.actor.eval()

        # Run episode
        mad_norms = run_episode(env, mad_policy, episode_length)
        informarl_norms = run_episode(env, informarl_policy, episode_length)

        mad_norm_trajectories.append(mad_norms)
        informarl_norm_trajectories.append(informarl_norms)
        agent_counts.append(num_agents)

        print(f"Completed | MAD final error: {mad_norms[-1]:.4f} | "
              f"InforMARL final error: {informarl_norms[-1]:.4f}")

    # Pad trajectories to same length (in case episodes ended early)
    max_len = max(len(t) for t in mad_norm_trajectories + informarl_norm_trajectories)

    # Pad with the last value
    mad_norm_trajectories_padded = []
    for traj in mad_norm_trajectories:
        if len(traj) < max_len:
            padded = traj + [traj[-1]] * (max_len - len(traj))
        else:
            padded = traj
        mad_norm_trajectories_padded.append(padded)

    informarl_norm_trajectories_padded = []
    for traj in informarl_norm_trajectories:
        if len(traj) < max_len:
            padded = traj + [traj[-1]] * (max_len - len(traj))
        else:
            padded = traj
        informarl_norm_trajectories_padded.append(padded)

    # Convert to arrays
    mad_norm_trajectories = np.array(mad_norm_trajectories_padded)
    informarl_norm_trajectories = np.array(informarl_norm_trajectories_padded)

    # Plot results
    print("\nGenerating plot...")
    time_steps = np.arange(max_len)

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Create colormap for different agent counts with more contrast
    cmap_blue = plt.cm.Blues(np.linspace(0.3, 0.95, 10))
    cmap_red = plt.cm.Reds(np.linspace(0.3, 0.95, 10))

    plt.figure(figsize=(12, 4))

    # Plot individual MAD trajectories (one per agent count)
    for i, traj in enumerate(mad_norm_trajectories):
        plt.plot(time_steps, traj, '-', color=cmap_blue[i], linewidth=2.5, alpha=0.9)

    # Plot individual InforMARL trajectories (one per agent count)
    for i, traj in enumerate(informarl_norm_trajectories):
        plt.plot(time_steps, traj, '-', color=cmap_red[i], linewidth=2.5, alpha=0.9)

    # Add legend entries with darker colors for visibility
    plt.plot([], [], '-', color=cmap_blue[6], linewidth=2.5, alpha=0.9, label='Ours')
    plt.plot([], [], '-', color=cmap_red[6], linewidth=2.5, alpha=0.9, label='InforMARL')
    plt.plot([], [], ' ', label='Lighter → Darker: 1 → 10 agents')

    plt.xlabel('Time Step', fontsize=25)
    plt.ylabel(r"$|x_t|_2$ before training", fontsize=25)
   # plt.title('Convergence to Goal: MAD vs InforMARL (1-10 Agents, Untrained)', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', labelsize=20)  # Change tick label size
    plt.legend(fontsize=17, loc='upper center', ncol=3, framealpha=0.9, columnspacing=1.0)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if all_args.use_trained_policy:
        plt.savefig('plots/stability_train.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved to: stability_train.png")
    else:
        plt.savefig('plots/stability_no_train.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved to: stability_no_train.png")

    # Print summary statistics
    print("\nSummary Statistics (by agent count):")
    print(f"{'N_agents':<10} {'MAD Final Error':<18} {'InforMARL Final Error':<18}")
    print("-" * 50)
    for i, n_agents in enumerate(agent_counts):
        mad_final = mad_norm_trajectories[i][-1]
        informarl_final = informarl_norm_trajectories[i][-1]
        print(f"{n_agents:<10} {mad_final:<18.4f} {informarl_final:<18.4f}")

    print("\nOverall Statistics:")
    mad_final_errors = [traj[-1] for traj in mad_norm_trajectories]
    informarl_final_errors = [traj[-1] for traj in informarl_norm_trajectories]
    print(f"MAD Policy - Mean: {np.mean(mad_final_errors):.4f}, Std: {np.std(mad_final_errors):.4f}")
    print(f"InforMARL Policy - Mean: {np.mean(informarl_final_errors):.4f}, Std: {np.std(informarl_final_errors):.4f}")

    plt.show()

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
