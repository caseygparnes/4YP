import argparse
import math
from typing import Tuple

import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.gnn import GNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.ssm import SSM
from onpolicy.algorithms.utils.gnn import StableGNNBase

def minibatchGenerator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class MAD_Actor(nn.Module):
    """
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space (must be Box/continuous).
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(MAD_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.K_p = args.kp_val

        self.m_max = args.m_max_start
        # Orbit base controller toggle + params (for orbit_graph)
        self.use_orbit_base_controller = getattr(args, "use_orbit_base_controller", False)

        self.orbit_radius = float(getattr(args, "orbit_radius", 1.0))
        self.orbit_dir = int(getattr(args, "orbit_dir", +1))
        self.guidance_k = float(getattr(args, "guidance_k", 0.1))
        self.delta_bl = float(getattr(args, "delta_bl", 1.0))

        d_shift_arg = float(getattr(args, "d_shift", -1.0))
        if d_shift_arg >= 0.0:
            self.d_shift = d_shift_arg
        else:
            # default d_shift
            kR = self.guidance_k * self.orbit_radius
            if kR < 1.0:
                self.d_shift = 0.0
            else:
                phi = math.acos(1.0 / kR)
                rhs = 1.0 - (2.0 * phi / math.pi) ** 2
                rhs = max(0.0, min(rhs, 1.0))
                self.d_shift = self.delta_bl * rhs





        self.under_training = True  # Flag to control return values during training vs evaluation

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (edge_dim,)

        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        # Stable GNN for magnitude pathway (preserves L_p-stability)
        self.mag_gnn = StableGNNBase(
            args, node_obs_shape, edge_dim, args.actor_graph_aggr
        )
        mag_gnn_out_dim = self.mag_gnn.out_dim

        ssm_state_features = getattr(args, 'ssm_hidden_dim', 64)
        ssm_mlp_hidden = getattr(args, 'ssm_mlp_hidden', 64)

        # SSM now receives GNN output instead of just rel_goal
        self.ssm = SSM(
            input_size=mag_gnn_out_dim,  # GNN output dimension
            output_size=1,
            lru_output_size=self.hidden_size,
            state_features=ssm_state_features,
            scan=False,
            mlp_hidden_size=ssm_mlp_hidden,
            rmin=args.rmin,
            rmax=args.rmax,
        )

        self.to(device)
    
    def _rot90(self, v: Tensor) -> Tensor:
        return torch.stack([-v[..., 1], v[..., 0]], dim=-1)

    def _safe_norm(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        return torch.sqrt(torch.clamp((x * x).sum(dim=-1), min=eps))

    def _theta_L(self, d_mag: Tensor) -> Tensor:
        # theta_L(d) = (pi/2) * sqrt( max(0, 1 - d/delta_bl ) )
        r = torch.clamp(d_mag / max(1e-8, self.delta_bl), 0.0, 1.0)
        return (math.pi / 2.0) * torch.sqrt(torch.clamp(1.0 - r, min=0.0))

    def _orbit_u_base(self, pos: Tensor, rel_center: Tensor) -> Tensor:
        """
        Gone-with-the-Wind orbit look-ahead field:
        u_base = k * L_hat  (Eq 13)
        where L_hat is constructed from (Eq 14-15)-style blending.
        Inputs:
        pos: (B,2)
        rel_center: (B,2) = center - pos
        """
        # delta = pos - center = -rel_center
        delta = -rel_center
        rho = self._safe_norm(delta)
        N_P = delta / rho.unsqueeze(-1)

        # handle degenerate rho
        zero_mask = (rho < 1e-6)
        if zero_mask.any():
            N_P = N_P.clone()
            N_P[zero_mask] = torch.tensor([1.0, 0.0], device=pos.device, dtype=pos.dtype)

        T_P = self._rot90(N_P)
        e_r = rho - self.orbit_radius

        sign_e = torch.sign(e_r)
        sign_e = torch.where(sign_e == 0.0, torch.ones_like(sign_e), sign_e)

        d_mag = torch.abs(e_r) + float(self.d_shift)
        theta_L = self._theta_L(d_mag)

        L0 = -sign_e.unsqueeze(-1) * N_P
        cos_th = torch.cos(theta_L).unsqueeze(-1)
        sin_th = torch.sin(theta_L).unsqueeze(-1)

        L_hat = cos_th * L0 + sin_th * (float(self.orbit_dir) * T_P)

        # normalize
        L_hat = L_hat / self._safe_norm(L_hat).unsqueeze(-1)

        return float(self.guidance_k) * L_hat

        



    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        ssm_states,
        disturbances,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        adj (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        ssm_states: (torch.Tensor)
            SSM hidden states for this timestep.
        disturbances: (torch.Tensor)
            Disturbance vector for this timestep. (batch_size, 2*dim_p)
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        disturbances = check(disturbances).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Detect episode resets (mask == 0)
        reset_mask = (masks.squeeze(-1) == 0)
        reset_indices = reset_mask.nonzero(as_tuple=True)[0]

        # Initialize or reset SSM hidden states
        if ssm_states is None or ssm_states.shape[0] != batch_size:
            # Initialize for all environments
            ssm_states = torch.complex(
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
            )
        elif len(reset_indices) > 0:
            # Reset only environments that are resetting
            ssm_states = ssm_states.clone()  # Clone to avoid in-place modification
            for idx in reset_indices:
                ssm_states[idx] = torch.complex(
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                )

        mag_gnn_out = self.mag_gnn(disturbances, adj, agent_id)
        ssm_input = mag_gnn_out

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)         # Generate node embedding for the agent
            actor_features = torch.cat([obs, nbd_features], dim=1)        # Concatenate agent observation with node embedding
            actor_features = self.base(actor_features)                    # Pass through actor MLP (batch size, hidden size)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Sample from Gaussian distribution
        y, y_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        # Direction term: apply tanh to ensure |u_gnn| ≤ 1
        u_gnn = torch.tanh(y)

        # Observation structure: [vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]
        rel_goal = obs[:, 4:6]  # for orbit_graph: rel_goal is actually rel_center = center - pos
        if self.use_orbit_base_controller:
            pos = obs[:, 2:4]
            u_base = self._orbit_u_base(pos, rel_goal)
        else:
            u_base = self.K_p * rel_goal


        ssm_out_raw, ssm_states, _ = self.ssm.step(ssm_input, ssm_states)
        magnitude = torch.clamp(ssm_out_raw.abs(), max=self.m_max)

        actions = u_base + magnitude * u_gnn
        # store MAD components for logging (does not change outputs)
        self.last_u_base = u_base.detach()
        self.last_magnitude = magnitude.detach()
        self.last_u_gnn = u_gnn.detach()
        self.last_delta_u = (magnitude * u_gnn).detach()

        action_dim = u_gnn.shape[-1]
        log_jac_tanh = torch.log(1 - u_gnn.pow(2) + 1e-8).sum(-1, keepdim=True)
        log_jac_M = torch.log(magnitude + 1e-8).sum(-1, keepdim=True) * action_dim

        action_log_probs = y_log_probs - log_jac_M - log_jac_tanh

        if self.under_training:
            return (actions, action_log_probs, rnn_states, ssm_states, y)
        else:
            return (actions, action_log_probs, rnn_states, ssm_states, y, magnitude)

    def evaluate_actions(
    self,
    obs,
    node_obs,
    adj,
    agent_id,
    rnn_states,
    ssm_states,
    disturbances,
    action,
    masks,
    available_actions=None,
    active_masks=None,
    pre_tanh_value=None,
) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        node_obs (torch.Tensor):
            Local agent graph node features to the actor.
        adj (torch.Tensor):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        action: (torch.Tensor)
            Total actions (u_base + M*tanh(y)) stored in buffer.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        ssm_states: (torch.Tensor)
            SSM hidden states for this timestep.
        disturbances: (torch.Tensor)
            Disturbance vector for this timestep. (batch_size, 2*dim_p)
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities under current policy.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        disturbances = check(disturbances).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Detect episode resets (mask == 0)
        reset_mask = (masks.squeeze(-1) == 0)
        reset_indices = reset_mask.nonzero(as_tuple=True)[0]

        # Initialize or reset SSM hidden states if needed
        if ssm_states is None:
            # Initialize for all environments (states not in buffer)
            ssm_states = torch.complex(
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
            )
        else:
            # SSM states exist, convert from buffer format if needed
            # Buffer stores as numpy [batch, state_dim, 2] with [real, imag] components
            import numpy as np
            if isinstance(ssm_states, np.ndarray):
                # Convert from numpy to torch.complex
                ssm_states = check(ssm_states).to(**self.tpdv)
                ssm_states = torch.complex(ssm_states[..., 0], ssm_states[..., 1])
            elif not torch.is_complex(ssm_states):
                # Already torch tensor but not complex, convert
                ssm_states = ssm_states.to(**self.tpdv)
                ssm_states = torch.complex(ssm_states[..., 0], ssm_states[..., 1])

            # Check size and reset if needed
            if ssm_states.shape[0] != batch_size:
                ssm_states = torch.complex(
                    torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                    torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                )

        # Reset specific indices if episodes ended
        if len(reset_indices) > 0:
            # Reset only environments that are resetting
            ssm_states = ssm_states.clone()  # Clone to avoid in-place modification
            for idx in reset_indices:
                ssm_states[idx] = torch.complex(
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                )

        mag_gnn_out = self.mag_gnn(disturbances, adj, agent_id)
        ssm_input = mag_gnn_out
        # Observation structure: [vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]
        rel_goal = obs[:, 4:6]
        if self.use_orbit_base_controller:
            pos = obs[:, 2:4]
            u_base = self._orbit_u_base(pos, rel_goal)
        else:
            u_base = self.K_p * rel_goal

        ssm_out_raw, ssm_states, _ = self.ssm.step(ssm_input, ssm_states)

        magnitude = torch.clamp(ssm_out_raw.abs(), max=self.m_max)

        if pre_tanh_value is not None:
            y = check(pre_tanh_value).to(**self.tpdv)
            u_gnn_tanh = torch.tanh(y)
        else:
            mag_safe = torch.clamp(magnitude, min=1e-8)
            u_gnn_tanh = torch.clamp((action - u_base) / mag_safe, -0.999, 0.999)
            y = torch.atanh(u_gnn_tanh)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Evaluate log probs of the Gaussian sample y under current policy

        y_log_probs, dist_entropy_y = self.act.evaluate_actions(
            actor_features,
            y,  
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        action_dim = u_gnn_tanh.shape[-1]

        log_jac_tanh = torch.log(1 - u_gnn_tanh.pow(2) + 1e-8).sum(-1, keepdim=True)
        log_jac_M = torch.log(torch.clamp(magnitude, min=1e-8)).sum(-1, keepdim=True) * action_dim

        action_log_probs = y_log_probs - log_jac_M - log_jac_tanh

        log_det_jacobian = log_jac_M + log_jac_tanh
        dist_entropy = dist_entropy_y + log_det_jacobian.mean()

        return (action_log_probs, dist_entropy)


class MAD_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(MAD_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents
        mlp_base_in_dim = gnn_out_dim
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, cent_obs_shape, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self, cent_obs, node_obs, adj, agent_id, rnn_states, masks
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (cent_obs.shape[0] > self.max_batch_size):
            # print(f'Cent obs: {cent_obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                cent_obs, node_obs, adj, agent_id, self.max_batch_size
            )
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                critic_feats_batch = self.base(act_feats_batch)
                critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(
                node_obs, adj, agent_id
            )  
            if self.args.use_cent_obs:
                critic_features = torch.cat(
                    [cent_obs, nbd_features], dim=1
                )  
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)  # Cent obs here

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return (values, rnn_states)
