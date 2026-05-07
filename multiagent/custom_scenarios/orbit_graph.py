"""
Fixed-wing multi-agent orbiting (GraphMPE scenario).

Goal:
- N agents converge to and maintain a circular orbit of radius orbit_radius
  about (orbit_center_x, orbit_center_y), while avoiding obstacles in transient.

Dynamics:
- Fixed speed v_star
- Control is an auxiliary vector u_tot in R^2, interpreted by core.py fixed-wing integrator:
    a_n = v_star^2 * (u_tot · N_body)
  where N_body is the unit normal to the velocity direction.

Observations (per agent, same shape as navigation_graph):
  obs = [vel_x, vel_y, pos_x, pos_y, rel_center_x, rel_center_y]
  where rel_center = center - pos

Graph node features (same dimension pattern as navigation_graph):
  node_feat = [vel(2), pos(2), goal_pos(2), entity_type(1)]
  Here goal_pos for agents is the orbit center; landmarks/obstacles use their own pos.
"""

from typing import Tuple, List, Optional
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark, Entity
from multiagent.scenario import BaseScenario

entity_mapping = {"agent": 0, "landmark": 1, "obstacle": 2}


def _safe_norm(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(max(eps, float(np.sum(x * x)))))


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float32)

def _segment_point_dist(p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> float:
    v = p1 - p0
    vv = float(np.dot(v, v)) + 1e-8
    t = float(np.dot(c - p0, v)) / vv
    t = float(np.clip(t, 0.0, 1.0))
    proj = p0 + t * v
    return _safe_norm(proj - c)


class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace) -> World:
        # ---------- required / existing args (keep compatibility) ----------
        self.world_size = args.world_size
        self.num_agents = args.num_agents
        self.num_scripted_agents = args.num_scripted_agents
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.episode_length = args.episode_length
        self.max_edge_dist = getattr(args, "max_edge_dist", 1.0)
        # If user left it at the generic default (=1), override for orbit_graph scale
        if abs(self.max_edge_dist - 1.0) < 1e-6:
            self.max_edge_dist = float(getattr(args, "orbit_max_edge_dist", 250.0))



        self.graph_feat_type = args.graph_feat_type

        # ---------- orbit / fixed-wing params ----------
        self.orbit_center = np.array(
            [args.orbit_center_x, args.orbit_center_y], dtype=np.float32
        )
        self.orbit_radius = float(args.orbit_radius)
        self.orbit_dir = int(args.orbit_dir)  # +1 CCW, -1 CW

        self.v_star = float(args.v_star)
        self.a_n_max = None if args.a_n_max <= 0 else float(args.a_n_max)
        self.dt = float(args.dt)
                # ---------- base-controller params (needed for delta_u penalty) ----------
        self.use_orbit_base_controller = bool(getattr(args, "use_orbit_base_controller", False))
        self.guidance_k = float(getattr(args, "guidance_k", 0.1))
        self.delta_bl = float(getattr(args, "delta_bl", 5.0))
        self.d_shift = float(getattr(args, "d_shift", -1.0))
        self.K_p = float(getattr(args, "K_p", 1.0))  # fallback if not using orbit base

        # optional sharper obstacle aggregation
        self.obstacle_topk = int(getattr(args, "obstacle_topk", 1))
        # ---------- exponential obstacle cost params ----------
        self.use_exp_obstacle_cost = bool(getattr(args, "use_exp_obstacle_cost", False))
        self.k_obs1 = float(getattr(args, "k_obs1", 1.0))
        self.k_obs2 = float(getattr(args, "k_obs2", 0.1))

        # ---------- reset distribution ----------
        self.x0_rad = float(args.x0_rad)
        self.x0_std = float(args.x0_std)
        # optional single-agent random start-angle reset
        self.randomize_single_agent_start_angle = bool(
            getattr(args, "randomize_single_agent_start_angle", False)
        )
        self.single_agent_start_angle_min_deg = float(
            getattr(args, "single_agent_start_angle_min_deg", 0.0)
        )
        self.single_agent_start_angle_max_deg = float(
            getattr(args, "single_agent_start_angle_max_deg", 90.0)
        )

        if self.single_agent_start_angle_max_deg < self.single_agent_start_angle_min_deg:
            raise ValueError(
                "single_agent_start_angle_max_deg must be >= single_agent_start_angle_min_deg"
            )

        # ---------- obstacles layout ----------
        self.obstacle_layout = str(args.obstacle_layout)
        self.obstacle_safety_margin = float(args.obstacle_safety_margin)
        self.obstacle_smooth_margin = float(args.obstacle_smooth_margin)
        self.w_obstacle_smooth = float(args.w_obstacle_smooth)
        self.obstacle_cost_mode = str(getattr(args, "obstacle_cost_mode", "mean"))

        # rings3 layout
        self.ring1_count = int(args.ring1_count)
        self.ring2_count = int(args.ring2_count)
        self.ring3_count = int(args.ring3_count)
        self.ring1_dist = float(args.ring1_dist)
        self.ring2_dist = float(args.ring2_dist)
        self.ring3_dist = float(args.ring3_dist)
        self.obs1_rad = float(args.obs1_rad)
        self.obs2_rad = float(args.obs2_rad)
        self.obs3_rad = float(args.obs3_rad)
        self.ring2_offset = float(args.ring2_offset)
        self.ring3_offset = float(args.ring3_offset)

        # random layout
        self.obs_rad = float(args.obs_rad)

        # ---------- reward weights ----------
        self.w_orbit = float(args.w_orbit)
        self.w_align = float(args.w_align)
        self.w_obstacle = float(args.w_obstacle)
        self.w_agent_coll = float(args.w_agent_coll)
        self.w_control = float(args.w_control)

        # ---------- agent geometry ----------
        self.agent_radius = float(args.agent_radius)
        self.agent_safety_margin = float(args.agent_safety_margin)

        # ---------- world ----------
        world = World()
        world.cache_dists = True
        world.graph_mode = True
        world.graph_feat_type = self.graph_feat_type
        world.world_length = self.episode_length
        world.dim_c = 2

        # FIXED-WING switches (used by multiagent/core.py integrate_state)
        world.fixed_wing = True
        world.v_star = self.v_star
        world.a_n_max = self.a_n_max
        world.u_to_an_gain = float(getattr(args, "u_to_an_gain", 1.0))
        world.dt = self.dt

        # store orbit metadata (for plotting scripts)
        world.orbit_center = self.orbit_center.copy()
        world.orbit_radius = self.orbit_radius
        world.orbit_dir = self.orbit_dir

        # ---------- entities ----------
        global_id = 0

        # agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = f"agent {i}"
            agent.global_id = global_id
            global_id += 1

            agent.silent = True
            agent.collide = False  # disable physics collision forces; we do penalty in reward

            agent.size = self.agent_radius

            # IMPORTANT: avoid env._set_action scaling by 5.0 when agent.accel is None
            agent.accel = 1.0

            # allow larger action magnitudes if you want
            agent.u_range = float(args.u_range)

        # scripted agents (kept for compatibility; default 0)
        world.scripted_agents = [Agent() for _ in range(self.num_scripted_agents)]
        for j, agent in enumerate(world.scripted_agents):
            agent.id = self.num_agents + j
            agent.name = f"agent {self.num_agents + j}"
            agent.global_id = global_id
            global_id += 1
            agent.silent = True
            agent.collide = False
            agent.size = self.agent_radius
            agent.accel = 1.0
            agent.u_range = float(args.u_range)

        # landmarks (dummy goals): MUST be num_agents (runner assumes this)
        num_landmarks = self.num_agents
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, lm in enumerate(world.landmarks):
            lm.id = i
            lm.name = f"landmark {i}"
            lm.global_id = global_id
            global_id += 1
            lm.collide = False
            lm.movable = False

        # obstacles
        world.obstacles = [Landmark() for _ in range(self.num_obstacles)]
        for i, obs in enumerate(world.obstacles):
            obs.id = i
            obs.name = f"obstacle {i}"
            obs.global_id = global_id
            global_id += 1
            obs.collide = False
            obs.movable = False

        # disturbance settings (used by MultiAgentGraphEnv)
        world.use_disturbance = args.use_disturbance
        world.disturbance_std = args.disturbance_std
        world.disturbance_decay_rate = args.disturbance_decay_rate

        # initialize
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        world.current_time_step = 0

        # metrics for logging
        world.num_obstacle_collisions = np.zeros(self.num_agents, dtype=np.float32)
        world.num_agent_collisions = np.zeros(self.num_agents, dtype=np.float32)

        self._place_obstacles(world)
        self._place_landmarks(world)
        self._place_agents(world)

        world.calculate_distances()
        self.update_graph(world)

    def _place_landmarks(self, world: World) -> None:
        # Put all landmarks at the orbit center (keeps shapes identical to navigation_graph)
        for lm in world.landmarks:
            lm.state.p_pos = self.orbit_center.copy()
            lm.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

    def _place_obstacles(self, world: World) -> None:
        c = self.orbit_center

        if self.obstacle_layout == "rings3":
            total = self.ring1_count + self.ring2_count + self.ring3_count
            assert (
                total == self.num_obstacles
            ), f"rings3 requires ring1_count+ring2_count+ring3_count == num_obstacles. Got {total} vs {self.num_obstacles}"

            idx = 0

            def place_ring(count: int, ring_dist: float, obs_rad: float, offset: float):
                nonlocal idx
                if count <= 0:
                    return
                angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + offset
                for k in range(count):
                    obs = world.obstacles[idx]
                    obs.size = float(obs_rad)
                    obs.state.p_pos = c + ring_dist * np.array(
                        [np.cos(angles[k]), np.sin(angles[k])], dtype=np.float32
                    )
                    obs.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
                    idx += 1

            place_ring(self.ring1_count, self.ring1_dist, self.obs1_rad, 0.0)
            place_ring(self.ring2_count, self.ring2_dist, self.obs2_rad, self.ring2_offset)
            place_ring(self.ring3_count, self.ring3_dist, self.obs3_rad, self.ring3_offset)

        else:
            # random obstacles in a box
            for obs in world.obstacles:
                obs.size = float(self.obs_rad)
                obs.state.p_pos = 0.8 * np.random.uniform(
                    -self.world_size / 2, self.world_size / 2, world.dim_p
                ).astype(np.float32)
                obs.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

    def _place_agents(self, world: World) -> None:
        c = self.orbit_center
        N = self.num_agents

        # Default behavior: equally spaced anchor angles starting at 0
        base_angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).astype(np.float32)

        # Optional behavior for the single-agent case:
        # sample one anchor angle uniformly in the configured degree range.
        if N == 1 and self.randomize_single_agent_start_angle:
            angle_deg = np.random.uniform(
                self.single_agent_start_angle_min_deg,
                self.single_agent_start_angle_max_deg,
            )
            base_angles[0] = np.float32(np.deg2rad(angle_deg))

        for i, agent in enumerate(world.agents):
            # anchor circle + noise
            anchor = c + self.x0_rad * np.array(
                [np.cos(base_angles[i]), np.sin(base_angles[i])],
                dtype=np.float32,
            )
            noise = np.random.normal(0.0, self.x0_std, size=(2,)).astype(np.float32)
            p = anchor + noise

            agent.state.p_pos = p

            # tangential velocity direction (orbit_dir * rot90(r_hat))
            r = p - c
            r_hat = r / _safe_norm(r)
            t_hat = self.orbit_dir * _rot90(r_hat)
            agent.state.p_vel = self.v_star * t_hat

            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
    

    def _theta_L_np(self, d_mag: float) -> float:
        r = np.clip(d_mag / max(1e-8, self.delta_bl), 0.0, 1.0)
        return 0.5 * np.pi * np.sqrt(max(0.0, 1.0 - r))

    def _orbit_u_base_np(self, p: np.ndarray) -> np.ndarray:
        rel_center = self.orbit_center - p
        delta = -rel_center
        rho = _safe_norm(delta)

        if rho < 1e-6:
            N_P = np.array([1.0, 0.0], dtype=np.float32)
        else:
            N_P = delta / rho

        T_P = _rot90(N_P)
        e_r = rho - self.orbit_radius

        sign_e = 1.0 if e_r >= 0.0 else -1.0
        d_mag = abs(e_r) + self.d_shift
        theta_L = self._theta_L_np(d_mag)

        L0 = -sign_e * N_P
        L_hat = (
            np.cos(theta_L) * L0
            + np.sin(theta_L) * (float(self.orbit_dir) * T_P)
        )

        L_hat = L_hat / max(_safe_norm(L_hat), 1e-8)
        return (float(self.guidance_k) * L_hat).astype(np.float32)

    def _base_u_prev_np(self, p_prev: np.ndarray) -> np.ndarray:
        if self.use_orbit_base_controller:
            return self._orbit_u_base_np(p_prev)

        rel_center_prev = self.orbit_center - p_prev
        return (self.K_p * rel_center_prev).astype(np.float32)

    def _aggregate_obstacle_terms(self, terms):
        if len(terms) == 0:
            return 0.0

        mode = getattr(self, "obstacle_cost_mode", "mean")

        if mode == "sum":
            return float(np.sum(terms))
        if mode == "max":
            return float(np.max(terms))
        if mode == "topk_mean":
            k = max(1, min(self.obstacle_topk, len(terms)))
            terms = np.sort(np.asarray(terms, dtype=np.float32))[::-1][:k]
            return float(np.mean(terms))

        return float(np.mean(terms))  # default: "mean"

    # ---------- RL callbacks ----------
    def observation(self, agent: Agent, world: World) -> arr:
        # obs = [vel, pos, rel_center]
        rel_center = self.orbit_center - agent.state.p_pos
        return np.concatenate([agent.state.p_vel, agent.state.p_pos, rel_center]).astype(np.float32)

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id], dtype=np.float32)

    def done(self, agent: Agent, world: World) -> bool:
        return world.current_time_step >= world.world_length

    def reward(self, agent: Agent, world: World) -> float:
        c = self.orbit_center
        p = np.asarray(agent.state.p_pos, dtype=np.float32)
        v = np.asarray(agent.state.p_vel, dtype=np.float32)
        p_prev = np.asarray(getattr(agent.state, "p_pos_prev", p), dtype=np.float32)

        # ---------- orbit cost (normalized) ----------
        rho = _safe_norm(p - c)
        e_r = rho - self.orbit_radius
        orbit_cost = (e_r / max(1e-6, self.orbit_radius)) ** 2

        # ---------- tangent alignment ----------
        r_hat = (p - c) / max(rho, 1e-8)
        t_star = self.orbit_dir * _rot90(r_hat)
        v_hat = v / max(_safe_norm(v), 1e-8)
        cos_align = float(np.clip(np.dot(v_hat, t_star), -1.0, 1.0))
        align_cost = (1.0 - cos_align) ** 2

        # ---------- obstacle penalties based on swept segment ----------
        obs_terms = []
        obs_smooth_terms = []
        hard_collision = 0.0

        for obs in world.obstacles:
            d_seg = _segment_point_dist(p_prev, p, obs.state.p_pos)
            R_eff = float(agent.size + obs.size)
            gap = d_seg - R_eff

            # ----- HARD obstacle term -----
            if getattr(self, "use_exp_obstacle_cost", False):
                # Use signed distance to the buffered boundary (safety margin)
                # delta > 0: outside buffer, small penalty
                # delta < 0: inside buffer/collision, large penalty
                delta = d_seg - (R_eff + float(self.obstacle_safety_margin))

                # Exponential cost: k_obs1 * exp(-k_obs2 * delta)
                exponent = -float(self.k_obs2) * float(delta)

                # Clip for numerical stability (prevents overflow if deep inside obstacle)
                exponent = float(np.clip(exponent, -50.0, 50.0))

                obs_terms.append(float(self.k_obs1) * float(np.exp(exponent)))
            else:
                # Original hinge^2
                hinge = max(0.0, float(self.obstacle_safety_margin) - float(gap))
                obs_terms.append(hinge * hinge)

            # ----- SMOOTH (far) obstacle term -----
            # Keep your existing smooth hinge shaping (recommended initially).
            hinge_far = max(0.0, float(self.obstacle_smooth_margin) - float(gap))
            obs_smooth_terms.append(hinge_far * hinge_far)

            # collision indicator (penetration)
            if gap < 0.0:
                hard_collision = 1.0

        obs_cost = self._aggregate_obstacle_terms(obs_terms)
        obs_smooth_cost = self._aggregate_obstacle_terms(obs_smooth_terms)

        # ---------- agent-agent hinge cost ----------
        coll_cost = 0.0
        for other in world.agents:
            if other is agent:
                continue
            d = _safe_norm(p - other.state.p_pos)
            gap = d - (agent.size + other.size)
            hinge = max(0.0, self.agent_safety_margin - gap)
            coll_cost += hinge * hinge

        # ---------- control penalty on delta_u, not total u ----------
        u_tot = (
            np.asarray(agent.action.u, dtype=np.float32)
            if agent.action.u is not None
            else np.zeros(world.dim_p, dtype=np.float32)
        )
        u_base_prev = self._base_u_prev_np(p_prev)
        delta_u = u_tot - u_base_prev
        u_cost = float(np.dot(delta_u, delta_u))

        cost = (
            self.w_orbit * orbit_cost
            + self.w_align * align_cost
            + self.w_obstacle * obs_cost
            + self.w_agent_coll * coll_cost
            + self.w_obstacle_smooth * obs_smooth_cost
            + self.w_control * u_cost
        )

        # optional extra one-shot penalty for actual penetration
        if hard_collision > 0.0:
            cost += self.w_obstacle * 1.0

        return -float(cost)

    def info_callback(self, agent: Agent, world: World) -> Tuple:
        c = self.orbit_center
        p = agent.state.p_pos
        v = agent.state.p_vel

        rho = _safe_norm(p - c)
        radial_error = float(abs(rho - self.orbit_radius))

        r_hat = (p - c) / rho
        t_star = self.orbit_dir * _rot90(r_hat)
        v_hat = v / _safe_norm(v)
        cos_align = float(np.clip(np.dot(v_hat, t_star), -1.0, 1.0))
        alignment_error = float((1.0 - cos_align) ** 2)

        # collisions (for logging only)
        obst_cols = 0.0
        for obs in world.obstacles:
            p0 = getattr(agent.state, "p_pos_prev", p)
            d = _segment_point_dist(p0, p, obs.state.p_pos)
            if d < (agent.size + obs.size):
                    obst_cols += 1.0

        agent_cols = 0.0
        for other in world.agents:
            if other is agent:
                continue
            d = _safe_norm(p - other.state.p_pos)
            if d < (agent.size + other.size):
                agent_cols += 1.0

        world.num_obstacle_collisions[agent.id] = obst_cols
        world.num_agent_collisions[agent.id] = agent_cols

        return {
            "Radial_error": radial_error,
            "Alignment_error": alignment_error,
            "Rho": float(rho),
            "Num_agent_collisions": float(agent_cols),
            "Num_obst_collisions": float(obst_cols),
        }

    # ---------- graph observation ----------
    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        node_obs = []
        if world.graph_feat_type == "global":
            for ent in world.entities:
                node_obs.append(self._get_entity_feat_global(ent, world))
        else:
            for ent in world.entities:
                node_obs.append(self._get_entity_feat_relative(agent, ent, world))

        node_obs = np.array(node_obs, dtype=np.float32)
        adj = world.cached_dist_mag.astype(np.float32)
        return node_obs, adj

    def update_graph(self, world: World):
        dists = world.cached_dist_mag
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect).tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        world.edge_list = np.stack([row, col])
        world.edge_weight = dists[row, col]

    def _get_entity_feat_global(self, entity: Entity, world: World) -> arr:
        pos = entity.state.p_pos
        vel = entity.state.p_vel
        if "agent" in entity.name:
            goal_pos = self.orbit_center  # center acts as "goal"
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            goal_pos = pos
            entity_type = entity_mapping["landmark"]
        elif "obstacle" in entity.name:
            goal_pos = pos
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([vel, pos, goal_pos, entity_type]).astype(np.float32)

    def _get_entity_feat_relative(self, agent: Agent, entity: Entity, world: World) -> arr:
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel
        ent_pos = entity.state.p_pos
        ent_vel = entity.state.p_vel

        rel_pos = ent_pos - agent_pos
        rel_vel = ent_vel - agent_vel

        if "agent" in entity.name:
            goal_pos = self.orbit_center
            rel_goal = goal_pos - agent_pos
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            rel_goal = rel_pos
            entity_type = entity_mapping["landmark"]
        elif "obstacle" in entity.name:
            rel_goal = rel_pos
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([rel_vel, rel_pos, rel_goal, entity_type]).astype(np.float32)