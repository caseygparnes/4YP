import gymnasium as gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from onpolicy.algorithms.mad_actor_critic import MAD_Actor, MAD_Critic
from onpolicy.utils.util import update_linear_schedule, update_magnitude_schedule
from loguru import logger


class MAD_MAPPOPolicy:
    """
    Graph-based MAPPO Policy with SSM magnitude term and base controller.

    Args:
        args: Arguments containing relevant model and policy information
        obs_space: Observation space
        cent_obs_space: Centralized observation space (for critic)
        node_obs_space: Node observation space (graph features)
        edge_obs_space: Edge observation space
        act_space: Action space (must be continuous/Box)
        device: Device to run on (cpu/gpu)
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        act_space: gym.Space,
        device=torch.device("cpu"),
    ) -> None:
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.node_obs_space = node_obs_space
        self.edge_obs_space = edge_obs_space
        self.act_space = act_space
        self.split_batch = args.split_batch
        self.max_batch_size = args.max_batch_size

        self.actor = MAD_Actor(
            args,
            self.obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.act_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )
        self.critic = MAD_Critic(
            args,
            self.share_obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        # Magnitude cap schedule
        self.m_max_start = float(getattr(args, "m_max_start", 1.0))
        self.m_warmup_episodes = int(getattr(args, "m_warmup_episodes", 0))

        m_final = getattr(args, "m_max_final", None)
        if m_final is None:
            self.m_max_final = self.m_max_start
        else:
            self.m_max_final = float(m_final)

        # Initialize actor cap immediately
        self.actor.m_max = self.m_max_start

    def update_magnitude_lr_schedule(self, episode: int, episodes: int) -> None:
        # Update magnitude params
        update_magnitude_schedule(
            param_groups=[self.actor_optimizer.param_groups[1]],
            epoch=episode,
            total_num_epochs=episodes,
            warmup_epochs=self.magnitude_warmup_epochs,
            initial_lr=self.magnitude_initial_lr,
            max_lr=self.magnitude_max_lr,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        Decay the actor and critic learning rates.

        Args:
            episode: Current training episode
            episodes: Total number of training episodes
        """
        # Update direction params
        update_linear_schedule(
            param_groups=[self.actor_optimizer.param_groups[0]],
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.lr,
        )

        # Update critic params
        update_linear_schedule(
            param_groups=[self.critic_optimizer.param_groups],
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.critic_lr,
        )

    def update_m_max(self, episode: int, episodes: int) -> None:
        """
        Linearly ramps actor.m_max from m_max_start -> m_max_final over m_warmup_episodes.
        If m_warmup_episodes <= 0, uses m_max_final immediately.
        """
        if self.m_warmup_episodes <= 0:
            self.actor.m_max = float(self.m_max_final)
            return

        frac = min(1.0, float(episode) / float(self.m_warmup_episodes))
        self.actor.m_max = float(self.m_max_start + frac * (self.m_max_final - self.m_max_start))

    def get_actions(
        self,
        cent_obs,
        obs,
        node_obs,
        adj,
        agent_id,
        share_agent_id,
        rnn_states_actor,
        rnn_states_critic,
        ssm_states,
        disturbances,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions and value function predictions for the given inputs.

        Args:
            cent_obs: Centralized input to the critic
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix for the graph
            agent_id: Agent id for observations
            share_agent_id: Agent id for centralized observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            ssm_states: SSM hidden states for magnitude term
            masks: Reset masks
            available_actions: Available actions (if None, all available)
            deterministic: Whether to use deterministic actions

        Returns:
            values: Value function predictions
            actions: Actions to take
            action_log_probs: Log probabilities of chosen actions
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
            ssm_states: Updated SSM hidden states
            pre_tanh_value: Raw Gaussian sample (y) for policy gradient computation
        """
        out = self.actor.forward(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            ssm_states,
            disturbances,
            masks,
            available_actions,
            deterministic,
        )

        actions = out[0]
        action_log_probs = out[1] if len(out) > 1 else None
        rnn_states_actor = out[2] if len(out) > 2 else rnn_states_actor
        ssm_states = out[3] if len(out) > 3 else ssm_states
        pre_tanh_value = out[4] if len(out) > 4 else None

        values, rnn_states_critic = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, ssm_states, pre_tanh_value

    def get_values(
        self, cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
    ) -> Tensor:
        """
        Get value function predictions.

        Args:
            cent_obs: Centralized input to the critic
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            share_agent_id: Agent id for centralized observations
            rnn_states_critic: RNN states for critic
            masks: Reset masks

        Returns:
            values: Value function predictions
        """
        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        node_obs,
        adj,
        agent_id,
        share_agent_id,
        rnn_states_actor,
        rnn_states_critic,
        disturbances,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        lru_hidden_states=None,
        pre_tanh_value=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action logprobs / entropy and value function predictions for actor update.

        Args:
            cent_obs: Centralized input to the critic
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            agent_id: Agent id for observations
            share_agent_id: Agent id for shared observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            action: Actions to evaluate
            masks: Reset masks
            available_actions: Available actions
            active_masks: Active agent masks
            lru_hidden_states: SSM/LRU hidden states from buffer
            pre_tanh_value: Raw Gaussian sample (y) stored during rollout

        Returns:
            values: Value function predictions
            action_log_probs: Log probabilities of input actions
            dist_entropy: Action distribution entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        lru_hidden_states,
        disturbances,
        action,
        masks,
        available_actions,
        active_masks,
        pre_tanh_value,
    )

        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        ssm_states,
        disturbances,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions using the given inputs.

        Args:
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            agent_id: Agent id for nodes
            rnn_states_actor: RNN states for actor
            ssm_states: SSM hidden states
            masks: Reset masks
            available_actions: Available actions
            deterministic: Whether to use deterministic actions

        Returns:
            actions: Actions to take
            rnn_states_actor: Updated actor RNN states
            ssm_states: Updated SSM hidden states
        """
        out = self.actor.forward(
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        ssm_states,
        disturbances,
        masks,
        available_actions,
        deterministic,
        )

        # actor.forward may return 5 values or more; we only need these:
        actions = out[0]
        rnn_states_actor = out[2]
        ssm_states = out[3]
        return actions, rnn_states_actor, ssm_states
