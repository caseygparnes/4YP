import argparse
from distutils.util import strtobool
from loguru import logger


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private
    hyperparameters only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including
            `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for
            some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting
            training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server.
            or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or
            else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network;
            set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode;
            or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal init for weights and 0 init for biases.
            or else, will use xavier uniform init.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time.
            If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate.
            If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation.
            If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render.
            NOTE: something, the environment has internal render process which
            is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument(
        "--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"]
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="test",
        help="project name to store logs under",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; " "or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. "
        "if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for "
        "simply collecting training data.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date "
        "to wandb server. or else will use tensorboard to log data.",
    )

    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default="MPE",
        choices=["MPE", "GraphMPE"],
        help="specify the name of environment",
    )
    parser.add_argument(
        "--discrete_action",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Whether to use discrete action space. If not specified, "
        "defaults to True (discrete) for standard training, False (continuous) for MAD policy. "
        "Note: MAD policy requires continuous actions",
    )
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    parser.add_argument(
        "--world_size",
        type=float,
        default=2,
        help="The world size of MPE; it will range from "
        "-world_size/2 to world_size/2",
    )
    parser.add_argument(
        "--num_scripted_agents",
        type=int,
        default=0,
        help="The number of scripted agents in MPE",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        choices=["local", "global", "nbd"],
        default="global",
        help="Whether to use local obs for navigation.py",
    )
    parser.add_argument(
        "--max_edge_dist",
        type=float,
        default=1,
        help="Maximum distance above which edges cannot be "
        "connected between the entities; used for `obs_type==ndb_obs` "
        "or for graph version",
    )

    parser.add_argument(
    "--orbit_max_edge_dist",
    type=float,
    default=250.0,
    help="Default max_edge_dist for orbit_graph if max_edge_dist is left at its generic default."
    )

    parser.add_argument(
        "--num_nbd_entities",
        type=int,
        default=3,
        help="Number of entities to be considered as neighbors "
        "for `obs_type==ndb_obs`",
    )
    parser.add_argument(
        "--use_comm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use communication " "channel for agent observation",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Max length for any episode"
    )

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU", action="store_false", default=True, help="Whether to use ReLU"
    )
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std " "to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for "
        "weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain", type=float, default=0.01, help="The gain # of last action layer"
    )
    parser.add_argument(
        "--split_batch",
        action="store_true",
        default=False,
        help="Whether to split batch for actor/critic networks",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=32,
        help="The max batch size for training. Only matters when " "--split_batch=True",
    )

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=True,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N", type=int, default=1, help="The number of recurrent layers."
    )
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument(
        "--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)"
    )
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. " "If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask " "useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask " "useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=10.0, help=" coefficience of huber loss."
    )

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, "
        "start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. "
        "If set, start render. Note: something, the environment "
        "has internal render process which is not controlled by "
        "this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )
    parser.add_argument(
        "--render_eval",
        action="store_true",
        default=False,
        help="by default, do not render while evaluating. If set, render video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    # misc parameters
    parser.add_argument(
        "--verbose",
        action="store_false",
        default=True,
        help="by default, print args and network at the begining.",
    )

    return parser


def graph_config(args, parser):
    entity_mapping = {"agent": 0, "landmark": 1, "obstacle": 2}
    num_entities = len(entity_mapping.keys())
    embedding_size = (num_entities) // 2 + 1
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=num_entities,
        help="Number of entity types in the env to have "
        "different embeddings for each entity type",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=embedding_size,
        help="Embedding layer output size for each category",
    )
    parser.add_argument(
        "--embed_hidden_size",
        type=int,
        default=16,
        help="Hidden layer dimension after the embedding layer",
    )
    parser.add_argument(
        "--embed_layer_N",
        type=int,
        default=1,
        help="Number of hidden linear layers after the " "embedding layer",
    )
    parser.add_argument(
        "--embed_use_ReLU",
        action="store_false",
        default=True,
        help="Whether to use ReLU in the linear layers after " "the embedding layer",
    )
    parser.add_argument(
        "--embed_add_self_loop",
        action="store_true",
        default=False,
        help="Whether to add self loops in adjacency matrix",
    )
    parser.add_argument(
        "--gnn_hidden_size",
        type=int,
        default=16,
        help="Hidden layer dimension in the GNN",
    )
    parser.add_argument(
        "--gnn_num_heads",
        type=int,
        default=3,
        help="Number of heads in the transformer conv layer (GNN)",
    )
    parser.add_argument(
        "--gnn_concat_heads",
        action="store_true",
        default=False,
        help="Whether to concatenate the head output or average",
    )
    parser.add_argument(
        "--gnn_layer_N", type=int, default=2, help="Number of GNN conv layers"
    )
    parser.add_argument(
        "--gnn_use_ReLU",
        action="store_false",
        default=True,
        help="Whether to use ReLU in GNN conv layers",
    )
    # parser.add_argument('--max_edge_dist', type=float, default=1,
    #                     help="Maximum distance above which edges cannot be "
    #                     "connected between the entities")
    parser.add_argument(
        "--graph_feat_type",
        type=str,
        default="global",
        choices=["global", "relative"],
        help="Whether to use " "'global' node/edge feats or 'relative'",
    )
    parser.add_argument(
        "--actor_graph_aggr",
        type=str,
        default="node",
        choices=["global", "node"],
        help="Whether we want to "
        "pull node specific features from the output or perform "
        "global_pool on all nodes. ",
    )
    parser.add_argument(
        "--critic_graph_aggr",
        type=str,
        default="global",
        choices=["global", "node"],
        help="Whether we want to "
        "pull node specific features from the output or perform "
        "global_pool on all nodes. ",
    )
    parser.add_argument(
        "--global_aggr_type",
        type=str,
        default="mean",
        choices=["mean", "max", "add"],
        help="The type of " "aggregation to perform if `graph_aggr` is `global`",
    )
    parser.add_argument(
        "--use_cent_obs",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Whether to use centralized observation " "for critic or not",
    )
    parser.add_argument(
        "--auto_mini_batch_size",
        action="store_true",
        default=False,
        help="Whether to automatically set mini batch size",
    )
    parser.add_argument(
        "--target_mini_batch_size",
        type=int,
        default=32,
        help="The target mini batch size to use",
    )

    parser.add_argument(
        "--kp_val",
        type=float,
        default=0.5,
        help="Initial value for learnable proportional gains in the base controller",
    )


    parser.add_argument(
        "--m_max_start",
        type=float,
        default=2,
        help="Starting value for magnitude annealing in the MAD policy",
    )

    parser.add_argument(
    "--m_max_final",
    type=float,
    default=None,
    help="Final value for m_max. If None, equals m_max_start."
)
    parser.add_argument(
        "--m_warmup_episodes",
        type=int,
        default=0,
        help="Linearly move m_max from m_max_start to m_max_final over this many episodes."
    )

    parser.add_argument(
        "--rmin",
        type=float,
        default=0.85,
        help="Minimum value for the randomization in the LRU",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=0.9,
        help="Maximum value for the randomization in the LRU",
    )

    parser.add_argument(
        "--use_mad_policy",
        action="store_true",
        default=False,
        help="Whether to use MAD (Magnitude And Direction) policy "
        "for stability-constrained RL. MAD decomposes control as u = u_base + |M(x0)| * D(neighbors)",
    )

    parser.add_argument(
        "--ssm_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for SSM (State Space Model) in MAD policy magnitude term",
    )

    parser.add_argument(
        "--ssm_mlp_hidden",
        type=int,
        default=64,
        help="Hidden dimension for MLP in SSM (State Space Model) in MAD policy magnitude term",
    )

        # ---------------------------
    # Orbit / fixed-wing scenario params (orbit_graph)
    # ---------------------------
    parser.add_argument("--use_orbit_base_controller", action="store_true", default=False,
                        help="If set, MAD base controller uses Gone-with-the-Wind orbit guidance instead of Kp*rel_goal.")

    parser.add_argument("--orbit_center_x", type=float, default=0.0)
    parser.add_argument("--orbit_center_y", type=float, default=0.0)
    parser.add_argument("--orbit_radius", type=float, default=50.0)
    parser.add_argument("--orbit_dir", type=int, default=+1, choices=[-1, +1],
                        help="+1 CCW, -1 CW")

    parser.add_argument("--v_star", type=float, default=10.0, help="Fixed-wing constant speed")
    parser.add_argument("--a_n_max", type=float, default=10.0,
                        help="Normal acceleration clamp. Set <=0 to disable clamp.")
    parser.add_argument(
    "--u_to_an_gain",
    type=float,
    default=1.0,
    help="Gain in fixed-wing mapping: a_n = u_to_an_gain * v_star^2 * (u dot N).",
)
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep")
    parser.add_argument("--u_range", type=float, default=5.0, help="Action range for continuous action space Box[-u_range, u_range]")

    # Gone-with-the-Wind guidance shaping parameters (Eq 13-15)
    parser.add_argument("--guidance_k", type=float, default=0.1, help="Gain k in u_base = k * L_hat")
    parser.add_argument("--delta_bl", type=float, default=5.0, help="Boundary layer delta_bl used in theta_L")
    parser.add_argument("--d_shift", type=float, default=-1.0,
                        help="If >=0 uses this constant; if <0 compute default shift from (k,R,delta_bl).")

    # initial condition distribution (agents on circle)
    parser.add_argument("--x0_rad", type=float, default=200.0)
    parser.add_argument("--x0_std", type=float, default=5.0)
    parser.add_argument(
    "--randomize_single_agent_start_angle",
    action="store_true",
    default=False,
    help="If set and num_agents==1, sample the initial agent angle uniformly in "
            "[single_agent_start_angle_min_deg, single_agent_start_angle_max_deg] "
            "instead of always starting at 0 degrees."
    )
    parser.add_argument(
        "--single_agent_start_angle_min_deg",
        type=float,
        default=0.0,
        help="Minimum initial angle in degrees for the optional single-agent random start-angle reset."
    )
    parser.add_argument(
        "--single_agent_start_angle_max_deg",
        type=float,
        default=90.0,
        help="Maximum initial angle in degrees for the optional single-agent random start-angle reset."
    )

    # obstacles
    parser.add_argument("--obstacle_layout", type=str, default="rings3", choices=["rings3", "random"])
    parser.add_argument("--obstacle_safety_margin", type=float, default=2.5)

    # rings3 params (must sum to num_obstacles)
    parser.add_argument("--ring1_count", type=int, default=15)
    parser.add_argument("--ring2_count", type=int, default=15)
    parser.add_argument("--ring3_count", type=int, default=15)
    parser.add_argument("--ring1_dist", type=float, default=80.0)
    parser.add_argument("--ring2_dist", type=float, default=120.0)
    parser.add_argument("--ring3_dist", type=float, default=160.0)
    parser.add_argument("--obs1_rad", type=float, default=12.0)
    parser.add_argument("--obs2_rad", type=float, default=14.0)
    parser.add_argument("--obs3_rad", type=float, default=16.0)
    parser.add_argument("--ring2_offset", type=float, default=0.7 * 3.0 * 3.141592653589793 / 5.0)
    parser.add_argument("--ring3_offset", type=float, default=3.141592653589793 / 4.0)

    # random obstacles
    parser.add_argument("--obs_rad", type=float, default=12.0)

    # agent geometry / collision penalty
    parser.add_argument("--agent_radius", type=float, default=1.0)
    parser.add_argument("--agent_safety_margin", type=float, default=1.5)
    parser.add_argument("--obstacle_smooth_margin", type=float, default=15.0)
    parser.add_argument("--w_obstacle_smooth", type=float, default=1.0)
    parser.add_argument(
    "--obstacle_cost_mode",
    type=str,
    default="mean",
    choices=["sum", "mean", "max"],
    help="How to aggregate obstacle hinge costs across obstacles."
)

    # reward weights (reward = -(weighted cost))
    parser.add_argument("--w_orbit", type=float, default=0.1)
    parser.add_argument("--w_align", type=float, default=0.1)
    parser.add_argument("--w_obstacle", type=float, default=100.0)
    parser.add_argument("--w_agent_coll", type=float, default=10.0)
    parser.add_argument("--w_control", type=float, default=0.0)

    parser.add_argument("--k_obs1", type=float, default=1.0,
                    help="Exponential obstacle cost scale.")
    parser.add_argument("--k_obs2", type=float, default=0.1,
                        help="Exponential obstacle cost decay rate (1/distance). Larger => shorter range.")
    parser.add_argument("--use_exp_obstacle_cost", action="store_true",
                        help="If set, obstacle proximity uses exponential cost instead of hinge^2.")








    # Disturbance parameters
    parser.add_argument(
        "--use_disturbance",
        action="store_true",
        default=False,
        help="Whether to add disturbance to the system states during rollout. "
        "Disturbance follows w_t = std * N(0,1) * exp(-decay_rate * t)",
    )

    parser.add_argument(
        "--disturbance_std",
        type=float,
        default=0.1,
        help="Standard deviation for the disturbance process",
    )
    parser.add_argument(
        "--disturbance_decay_rate",
        type=float,
        default=0.1,
        help="Exponential decay rate for the disturbance over time",
    )

    # Control penalty parameter
    parser.add_argument(
        "--control_penalty_weight",
        type=float,
        default=0.0,
        help="Weight for penalizing excessive control inputs in the reward function. "
        "Higher values encourage smoother, lower-magnitude controls.",
    )

    parser.add_argument(
        "--dist_weight",
        type=float,
        default=1.0,
        help="Weight for penalizing distance to goal in the reward function. "
        "Higher values encourage agents to reach the goal slower.",
    )

    all_args = parser.parse_known_args(args)[0]

    if all_args.auto_mini_batch_size:
        # for recurrent generator only
        num_mini_batch = (
            all_args.n_rollout_threads * all_args.episode_length * all_args.num_agents
        ) // (all_args.target_mini_batch_size)
        new_batch_size = (
            (all_args.n_rollout_threads * all_args.episode_length * all_args.num_agents)
            // num_mini_batch
            // all_args.data_chunk_length
            * all_args.data_chunk_length
        )
        setattr(all_args, "num_mini_batch", num_mini_batch)
        print("_" * 50)
        print(f"Overriding num_mini_batch to {num_mini_batch}")
        print(f"Batch size to be: {new_batch_size}")
        print("_" * 50)
        
    return all_args, parser
