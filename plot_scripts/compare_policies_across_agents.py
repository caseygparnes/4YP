"""
Script to generate figure 3 in the paper.
Simulates X random environment initializations for each agent count from 1 to 10,
collects rewards, calculates mean rewards, and plots the results.
"""

import argparse
import numpy as np
import torch
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.envs.env_wrappers import GraphDummyVecEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Model paths
    parser.add_argument("--model_path", type=str,
                        default="./onpolicy/results/GraphMPE/navigation_graph/rmappo/stable_gnn_train/run1/models/actor.pt")
    parser.add_argument("--model_path_informarl", type=str,
                        default="./onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run1/models/actor.pt")

    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of random environment initializations per agent count")
    parser.add_argument("--min_agents", type=int, default=1,
                        help="Minimum number of agents to test")
    parser.add_argument("--max_agents", type=int, default=10,
                        help="Maximum number of agents to test")
    parser.add_argument("--episode_length", type=int, default=25,
                        help="Length of each episode")
    parser.add_argument("--num_obstacles", type=int, default=3,
                        help="Number of obstacles in environment")

    # Environment parameters
    parser.add_argument("--scenario_name", type=str, default="orbit_graph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])

    # Output parameters
    parser.add_argument("--save_plot", type=str, default="./plots/policy_comparison_across_agents.png",
                        help="Path to save the comparison plot")
    parser.add_argument("--save_results", type=str, default="./results/policy_comparison_data.npz",
                        help="Path to save raw results data")

    return parser.parse_args()


def load_model_config(model_path):
    """Load configuration from the model directory."""
    import yaml

    model_dir = Path(model_path).parent.parent
    config_path = model_dir / "config.yaml"

    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}")
        return None

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def create_args_for_env(num_agents, num_obstacles, episode_length, scenario_name, seed, base_config=None):
    """Create arguments for environment initialization."""
    parser = get_config()

    # Parse minimal args
    args = parser.parse_args([])

    # Set required environment parameters
    args.env_name = "GraphMPE"
    args.scenario_name = scenario_name
    args.num_agents = num_agents
    args.num_obstacles = num_obstacles
    args.num_landmarks = num_agents  
    args.episode_length = episode_length
    args.seed = seed

    # Environment settings
    args.collaborative = True
    args.max_edge_dist = 1.0
    args.graph_feat_type = "relative"
    args.collision_rew = 5.0
    args.goal_rew = 5.0
    args.min_dist_thresh = 0.05
    args.max_speed = 2.0
    args.use_dones = False
    args.world_size = 2
    args.n_rollout_threads = 1

    # Copy network architecture settings from base config if available
    if base_config is not None:
        # GNN settings
        args.gnn_hidden_size = base_config.get('gnn_hidden_size', 16)
        args.gnn_layer_N = base_config.get('gnn_layer_N', 2)
        args.gnn_num_heads = base_config.get('gnn_num_heads', 3)
        args.gnn_use_ReLU = base_config.get('gnn_use_ReLU', True)
        args.gnn_concat_heads = base_config.get('gnn_concat_heads', False)

        # Embedding settings
        args.embed_hidden_size = base_config.get('embed_hidden_size', 16)
        args.embed_layer_N = base_config.get('embed_layer_N', 1)
        args.embed_use_ReLU = base_config.get('embed_use_ReLU', True)
        args.embedding_size = base_config.get('embedding_size', 2)
        args.num_embeddings = base_config.get('num_embeddings', 3)
        args.embed_add_self_loop = base_config.get('embed_add_self_loop', False)

        # Policy settings
        args.hidden_size = base_config.get('hidden_size', 64)
        args.layer_N = base_config.get('layer_N', 1)
        args.use_ReLU = base_config.get('use_ReLU', False)
        args.recurrent_N = base_config.get('recurrent_N', 1)
        args.use_recurrent_policy = base_config.get('use_recurrent_policy', True)
        args.use_naive_recurrent_policy = base_config.get('use_naive_recurrent_policy', False)

        # MAD-specific settings
        args.use_mad_policy = base_config.get('use_mad_policy', False)
        args.ssm_hidden_dim = base_config.get('ssm_hidden_dim', 64)
        args.learnable_kp = base_config.get('learnable_kp', False)
        args.kp_val = base_config.get('kp_val', 1.0)

        # Aggregation settings
        args.actor_graph_aggr = base_config.get('actor_graph_aggr', 'node')
        args.critic_graph_aggr = base_config.get('critic_graph_aggr', 'global')
        args.global_aggr_type = base_config.get('global_aggr_type', 'mean')

        # Other settings
        args.gain = base_config.get('gain', 0.01)
        args.use_orthogonal = base_config.get('use_orthogonal', True)
        args.use_feature_normalization = base_config.get('use_feature_normalization', True)
    else:
        # Default settings
        args.gnn_hidden_size = 16
        args.gnn_layer_N = 2
        args.gnn_num_heads = 3
        args.gnn_use_ReLU = True
        args.gnn_concat_heads = False
        args.embed_hidden_size = 16
        args.embed_layer_N = 1
        args.embed_use_ReLU = True
        args.embedding_size = 2
        args.num_embeddings = 3
        args.embed_add_self_loop = False
        args.hidden_size = 64
        args.layer_N = 1
        args.use_ReLU = False
        args.recurrent_N = 1
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
        args.use_mad_policy = False
        args.ssm_hidden_dim = 64
        args.learnable_kp = False
        args.kp_val = 1.0
        args.actor_graph_aggr = 'node'
        args.critic_graph_aggr = 'global'
        args.global_aggr_type = 'mean'
        args.gain = 0.01
        args.use_orthogonal = True
        args.use_feature_normalization = True

    # Additional required settings
    args.use_cent_obs = False
    args.use_centralized_V = True
    args.share_policy = True
    args.discrete_action = False

    # orbit_graph params (match training defaults unless overridden)
    args.use_orbit_base_controller = True
    args.orbit_center_x = 0.0
    args.orbit_center_y = 0.0
    args.orbit_radius = 50.0
    args.orbit_dir = +1

    args.v_star = 10.0
    args.a_n_max = 10.0
    args.dt = 0.1
    args.u_range = 5.0

    args.guidance_k = 0.1
    args.delta_bl = 5.0
    args.d_shift = -1.0  # compute default

    args.x0_rad = 200.0
    args.x0_std = 5.0

    args.obstacle_layout = "rings3"
    args.ring1_count = 15
    args.ring2_count = 15
    args.ring3_count = 15
    args.ring1_dist = 80.0
    args.ring2_dist = 120.0
    args.ring3_dist = 160.0
    args.obs1_rad = 12.0
    args.obs2_rad = 14.0
    args.obs3_rad = 16.0
    args.ring2_offset = 0.7 * 3.0 * np.pi / 5.0
    args.ring3_offset = np.pi / 4.0
    args.obs_rad = 12.0
    args.obstacle_safety_margin = 2.5

    args.agent_radius = 1.0
    args.agent_safety_margin = 1.5

    args.w_orbit = 0.1
    args.w_align = 0.1
    args.w_obstacle = 100.0
    args.w_agent_coll = 10.0
    args.w_control = 0.0

    # Disturbance settings (required by navigation_graph scenario)
    args.use_disturbance = base_config.get('use_disturbance', True) if base_config else True
    args.disturbance_std = base_config.get('disturbance_std', 0.1) if base_config else 0.1
    args.disturbance_decay_rate = base_config.get('disturbance_decay_rate', 0.1) if base_config else 0.1

    # LRU-specific settings for MAD
    args.lru_hidden_dim = base_config.get('lru_hidden_dim', 64) if base_config else 64
    args.ssm_mlp_hidden = base_config.get('ssm_mlp_hidden', 64) if base_config else 64
    args.use_base_controller = base_config.get('use_base_controller', True) if base_config else True

    # LRU stability parameters (rmin, rmax control eigenvalue magnitudes)
    args.rmin = base_config.get('rmin', 0.85) if base_config else 0.85
    args.rmax = base_config.get('rmax', 0.9) if base_config else 0.9
    args.ifi = base_config.get('ifi', 0.1) if base_config else 0.1

    # MAD magnitude scheduling parameters
    args.m_max_start = base_config.get('m_max_start', 1.0) if base_config else 1.0
    args.m_max_final = base_config.get('m_max_final', 5.0) if base_config else 5.0
    args.m_max_step_episode = base_config.get('m_max_step_episode', 100) if base_config else 100
    args.m_max_warmup_episodes = base_config.get('m_max_warmup_episodes', 250) if base_config else 250
    args.m_schedule_type = base_config.get('m_schedule_type', 'none') if base_config else 'none'

    return args


def make_eval_env(args):
    """Create a single evaluation environment."""
    def get_env_fn():
        def init_env():
            env = GraphMPEEnv(args)
            env.seed(args.seed)
            return env
        return init_env

    return GraphDummyVecEnv([get_env_fn()])


def evaluate_policy(policy, args, num_episodes, episode_length, device, policy_name="Policy", is_mad=False):
    """
    Evaluate a policy over multiple episodes.

    Returns:
        mean_reward: Average reward across all episodes
        std_reward: Standard deviation of rewards
        all_rewards: List of episode rewards
    """
    episode_rewards = []

    # Create environment
    envs = make_eval_env(args)
    num_agents = args.num_agents

    # Set policy to eval mode
    policy.actor.eval()
    policy.actor.under_training = False

    for episode in range(num_episodes):
        # Reset environment - each reset will use random initialization
        obs, agent_ids, node_obs, adj, disturbances = envs.reset()

        # Initialize RNN states and masks
        rnn_states = np.zeros((num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.zeros((num_agents, 1), dtype=np.float32)

        # Initialize SSM states for MAD policy (will be initialized on first forward pass)
        if is_mad:
            ssm_states = [None] * num_agents

        # Track episode reward
        episode_reward = 0.0

        for step in range(episode_length):
            actions_list = []

            # Get actions for each agent
            for i in range(num_agents):
                with torch.no_grad():
                    if is_mad:
                        # MAD policy requires ssm_states and disturbances
                        # Returns: (actions, action_log_probs, rnn_states, ssm_states, y, [magnitude])
                        result = policy.actor.forward(
                            obs=obs[:, i, :],  # (1, obs_dim)
                            node_obs=node_obs[:, i, :, :],  # (1, num_nodes, node_feat_dim)
                            adj=adj[:, i, :, :],  # (1, num_nodes, num_nodes)
                            agent_id=agent_ids[:, i],  # (1,)
                            rnn_states=rnn_states[i:i+1],
                            ssm_states=ssm_states[i],
                            disturbances=disturbances[:, i, :],  # (1, disturbance_dim)
                            masks=masks[i:i+1],
                            deterministic=True
                        )
                        # Unpack: may be 5 or 6 values depending on debug mode
                        action = result[0]
                        rnn_states_out = result[2]
                        ssm_states_out = result[3]
                        # Update SSM state
                        ssm_states[i] = ssm_states_out
                    else:
                        # InforMARL policy (standard GR_MAPPO)
                        action, _, rnn_states_out = policy.actor.forward(
                            obs=obs[:, i, :],  # (1, obs_dim)
                            node_obs=node_obs[:, i, :, :],  # (1, num_nodes, node_feat_dim)
                            adj=adj[:, i, :, :],  # (1, num_nodes, num_nodes)
                            agent_id=agent_ids[:, i],  # (1,)
                            rnn_states=rnn_states[i:i+1],
                            masks=masks[i:i+1],
                            deterministic=True
                        )

                    # Update RNN state
                    rnn_states[i:i+1] = rnn_states_out.cpu().numpy()

                    # Extract action
                    action_array = action[0].detach().cpu().numpy()
                    actions_list.append(action_array)

            # Stack actions: shape (1, num_agents, action_dim)
            actions = np.array(actions_list)[np.newaxis, :, :]

            # Step environment
            obs, agent_ids, node_obs, adj, disturbances, rewards, dones, infos = envs.step(actions)

            # Accumulate reward (average across agents)
            episode_reward += rewards[0].mean()

            # Update masks (set to 1 after first step)
            masks = np.ones((num_agents, 1), dtype=np.float32)

        episode_rewards.append(episode_reward)

    # Close environment
    envs.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, episode_rewards


def main():
    args = parse_args()

    print("="*80)
    print("Policy Comparison Across Agent Counts")
    print("="*80)
    print(f"InforMARL model: {args.model_path_informarl}")
    print(f"MAD model: {args.model_path}")
    print(f"Agent range: {args.min_agents} to {args.max_agents}")
    print(f"Episodes per configuration: {args.num_episodes}")
    print(f"Episode length: {args.episode_length}")
    print(f"Number of obstacles: {args.num_obstacles}")
    print("="*80 + "\n")

    # Set device
    device = torch.device(args.device)

    # Load model configurations
    informarl_config = load_model_config(args.model_path_informarl)
    mad_config = load_model_config(args.model_path)

    # Results storage
    agent_counts = list(range(args.min_agents, args.max_agents + 1))
    informarl_means = []
    informarl_stds = []
    mad_means = []
    mad_stds = []

    # Evaluate for each agent count
    for num_agents in agent_counts:
        print(f"\nEvaluating with {num_agents} agents...")
        print("-" * 80)

        # ===== Evaluate InforMARL =====
        print(f"  InforMARL policy:")

        # Create args for environment
        informarl_args = create_args_for_env(
            num_agents=num_agents,
            num_obstacles=args.num_obstacles,
            episode_length=args.episode_length,
            scenario_name=args.scenario_name,
            seed=args.seed,
            base_config=informarl_config
        )
        informarl_args.use_mad_policy = False

        # Create environment to get observation spaces
        temp_env = make_eval_env(informarl_args)
        obs_space = temp_env.observation_space[0]
        share_obs_space = temp_env.share_observation_space[0]
        node_obs_space = temp_env.node_observation_space[0]
        edge_obs_space = temp_env.edge_observation_space[0]
        act_space = temp_env.action_space[0]
        temp_env.close()

        # Create InforMARL policy
        informarl_policy = GR_MAPPOPolicy(
            informarl_args,
            obs_space,
            share_obs_space,
            node_obs_space,
            edge_obs_space,
            act_space,
            device=device
        )

        # Load weights
        informarl_state_dict = torch.load(args.model_path_informarl, map_location=device)
        informarl_policy.actor.load_state_dict(informarl_state_dict)

        # Evaluate
        mean_reward, std_reward, _ = evaluate_policy(
            informarl_policy,
            informarl_args,
            args.num_episodes,
            args.episode_length,
            device,
            policy_name="InforMARL",
            is_mad=False
        )

        informarl_means.append(mean_reward)
        informarl_stds.append(std_reward)

        print(f"    Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

        # ===== Evaluate Stable GNN =====
        print(f"Stable GNN policy:")

        # Create args for environment
        mad_args = create_args_for_env(
            num_agents=num_agents,
            num_obstacles=args.num_obstacles,
            episode_length=args.episode_length,
            scenario_name=args.scenario_name,
            seed=args.seed,
            base_config=mad_config
        )
        mad_args.use_mad_policy = True

        # Create environment to get observation spaces
        temp_env = make_eval_env(mad_args)
        obs_space = temp_env.observation_space[0]
        share_obs_space = temp_env.share_observation_space[0]
        node_obs_space = temp_env.node_observation_space[0]
        edge_obs_space = temp_env.edge_observation_space[0]
        act_space = temp_env.action_space[0]
        temp_env.close()

        # Create MAD policy
        mad_policy = MAD_MAPPOPolicy(
            mad_args,
            obs_space,
            share_obs_space,
            node_obs_space,
            edge_obs_space,
            act_space,
            device=device
        )

        # Load weights
        mad_state_dict = torch.load(args.model_path, map_location=device)
        mad_policy.actor.load_state_dict(mad_state_dict)

        # Evaluate
        mean_reward, std_reward, _ = evaluate_policy(
            mad_policy,
            mad_args,
            args.num_episodes,
            args.episode_length,
            device,
            policy_name="MAD",
            is_mad=True
        )

        mad_means.append(mean_reward)
        mad_stds.append(std_reward)

        print(f"    Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

    # Save raw results
    print(f"\nSaving results to {args.save_results}...")
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    np.savez(
        args.save_results,
        agent_counts=agent_counts,
        informarl_means=informarl_means,
        informarl_stds=informarl_stds,
        mad_means=mad_means,
        mad_stds=mad_stds
    )

    cmap_blue = plt.cm.Blues(np.linspace(0.3, 0.95, 10))
    cmap_red = plt.cm.Reds(np.linspace(0.3, 0.95, 10))

    # Plot results
    print(f"Creating comparison plot...")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays
    agent_counts = np.array(agent_counts)
    informarl_means = np.array(informarl_means)
    informarl_stds = np.array(informarl_stds)
    mad_means = np.array(mad_means)
    mad_stds = np.array(mad_stds)

    # Plot MAD
    plt.plot(agent_counts, mad_means, '-', label='Ours', linewidth=2, markersize=8, color=cmap_blue[6])
    plt.fill_between(
        agent_counts,
        mad_means - mad_stds,
        mad_means + mad_stds,
        alpha=0.3,
        color=cmap_blue[6]
    )

    # Plot InforMARL
    plt.plot(agent_counts, informarl_means, '-', label='InforMARL', linewidth=2, markersize=8, color=cmap_red[6])
    plt.fill_between(
        agent_counts,
        informarl_means - informarl_stds,
        informarl_means + informarl_stds,
        alpha=0.3,
        color=cmap_red[6]
    )

    plt.xlabel('Number of Agents', fontsize=25)
    plt.ylabel('Mean Episode Reward', fontsize=25)
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=20)  # Change tick label size
    plt.xticks(agent_counts)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.save_plot}")

    # Show plot
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Agents':<10} {'InforMARL Mean':<20} {'MAD Mean':<20}")
    print("-"*80)
    for i, n in enumerate(agent_counts):
        print(f"{n:<10} {informarl_means[i]:>10.3f} ± {informarl_stds[i]:<6.3f} {mad_means[i]:>10.3f} ± {mad_stds[i]:<6.3f}")
    print("="*80)


if __name__ == '__main__':
    main()
