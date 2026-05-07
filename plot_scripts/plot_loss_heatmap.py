import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def safe_norm(x, eps=1e-8):
    return np.sqrt(np.maximum(eps, np.sum(x * x, axis=-1)))


def build_rings3_obstacles(cfg):
    cx = float(cfg.get("orbit_center_x", 0.0))
    cy = float(cfg.get("orbit_center_y", 0.0))
    center = np.array([cx, cy], dtype=np.float32)

    ring1_count = int(cfg["ring1_count"])
    ring2_count = int(cfg["ring2_count"])
    ring3_count = int(cfg["ring3_count"])
    num_obstacles = int(cfg["num_obstacles"])

    total = ring1_count + ring2_count + ring3_count
    if total != num_obstacles:
        raise ValueError(
            f"rings3 requires ring1_count + ring2_count + ring3_count == num_obstacles, "
            f"but got {total} vs {num_obstacles}"
        )

    ring1_dist = float(cfg["ring1_dist"])
    ring2_dist = float(cfg["ring2_dist"])
    ring3_dist = float(cfg["ring3_dist"])

    obs1_rad = float(cfg["obs1_rad"])
    obs2_rad = float(cfg["obs2_rad"])
    obs3_rad = float(cfg["obs3_rad"])

    ring2_offset = float(cfg["ring2_offset"])
    ring3_offset = float(cfg["ring3_offset"])

    obstacles = []

    def place_ring(count, ring_dist, obs_rad, offset):
        if count <= 0:
            return
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + offset
        for ang in angles:
            pos = center + ring_dist * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            obstacles.append((pos, float(obs_rad)))

    place_ring(ring1_count, ring1_dist, obs1_rad, 0.0)
    place_ring(ring2_count, ring2_dist, obs2_rad, ring2_offset)
    place_ring(ring3_count, ring3_dist, obs3_rad, ring3_offset)

    return center, obstacles


def aggregate_terms(terms, mode):
    """
    terms: shape (M, H, W)
    returns: shape (H, W)
    Matches orbit_graph.py behavior exactly.
    """
    if terms.shape[0] == 0:
        return np.zeros(terms.shape[1:], dtype=np.float64)

    if mode == "sum":
        return np.sum(terms, axis=0)
    if mode == "max":
        return np.max(terms, axis=0)
    if mode == "mean":
        return np.mean(terms, axis=0)

    raise ValueError(f"Unknown obstacle_cost_mode: {mode}")

def compute_heatmaps(cfg, xlim, ylim, resolution):
    obstacle_layout = str(cfg["obstacle_layout"])
    if obstacle_layout != "rings3":
        raise NotImplementedError("This script currently supports obstacle_layout='rings3' only.")

    center, obstacles = build_rings3_obstacles(cfg)

    orbit_radius = float(cfg["orbit_radius"])
    w_orbit = float(cfg["w_orbit"])
    w_obstacle = float(cfg["w_obstacle"])
    w_obstacle_smooth = float(cfg.get("w_obstacle_smooth", 0.0))

    obstacle_safety_margin = float(cfg["obstacle_safety_margin"])
    obstacle_smooth_margin = float(cfg["obstacle_smooth_margin"])
    obstacle_cost_mode = str(cfg.get("obstacle_cost_mode", "mean"))

    # Fallback if not present in config
    agent_radius = float(cfg.get("agent_radius", 2.5))

    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], axis=-1)

    # Orbit term (normalized by orbit_radius, matching orbit_graph.py)
    rho = safe_norm(P - center[None, None, :])
    e_r = rho - orbit_radius
    raw_orbit_cost = (e_r / max(1e-6, orbit_radius)) ** 2
    orbit_term = w_orbit * raw_orbit_cost


    # Obstacle terms
    hard_terms = []
    smooth_terms = []

    use_exp = bool(cfg.get("use_exp_obstacle_cost", False))
    k_obs1 = float(cfg.get("k_obs1", 1.0))
    k_obs2 = float(cfg.get("k_obs2", 0.1))

    for obs_pos, obs_rad in obstacles:
        d = safe_norm(P - obs_pos[None, None, :])
        R_eff = agent_radius + obs_rad
        gap = d - R_eff  # same "gap" concept as environment (positive outside)

        if use_exp:
            # Match orbit_graph.py: delta = d - (R_eff + obstacle_safety_margin)
            delta = d - (R_eff + obstacle_safety_margin)

            # exponential: k1 * exp(-k2 * delta)
            exponent = -k_obs2 * delta

            # numerical safety, match your environment clipping
            exponent = np.clip(exponent, -50.0, 50.0)

            hard_terms.append(k_obs1 * np.exp(exponent))
        else:
            hinge = np.maximum(0.0, obstacle_safety_margin - gap)
            hard_terms.append(hinge ** 2)

        # Keep smooth hinge term as before (unless you also changed it in env)
        hinge_far = np.maximum(0.0, obstacle_smooth_margin - gap)
        smooth_terms.append(hinge_far ** 2)

    hard_terms = np.stack(hard_terms, axis=0) if len(hard_terms) > 0 else np.zeros((0, *X.shape))
    smooth_terms = np.stack(smooth_terms, axis=0) if len(smooth_terms) > 0 else np.zeros((0, *X.shape))

    raw_obs_cost = aggregate_terms(hard_terms, obstacle_cost_mode)
    raw_obs_smooth_cost = aggregate_terms(smooth_terms, obstacle_cost_mode)

    obstacle_term = w_obstacle * raw_obs_cost + w_obstacle_smooth * raw_obs_smooth_cost

    # TRUE combined term
    total_cost = orbit_term + obstacle_term

    return {
        "X": X,
        "Y": Y,
        "center": center,
        "obstacles": obstacles,
        "orbit_term": orbit_term,
        "obstacle_term": obstacle_term,
        "total_cost": total_cost,
        "raw_orbit_cost": raw_orbit_cost,
        "raw_obs_cost": raw_obs_cost,
        "raw_obs_smooth_cost": raw_obs_smooth_cost,
    }


def draw_overlay(ax, center, orbit_radius, obstacles):
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        center[0] + orbit_radius * np.cos(th),
        center[1] + orbit_radius * np.sin(th),
        "w--",
        linewidth=1.5,
    )

    for obs_pos, obs_rad in obstacles:
        circ = plt.Circle(
            (obs_pos[0], obs_pos[1]),
            obs_rad,
            edgecolor="white",
            facecolor="none",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.add_patch(circ)

    ax.plot(center[0], center[1], "w+", markersize=10, mew=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--xmin", type=float, default=-260.0)
    parser.add_argument("--xmax", type=float, default=260.0)
    parser.add_argument("--ymin", type=float, default=-260.0)
    parser.add_argument("--ymax", type=float, default=260.0)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Plot log10(1 + cost) instead of raw cost.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="orbit_obstacle_loss_heatmap.png",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = compute_heatmaps(
        cfg=cfg,
        xlim=(args.xmin, args.xmax),
        ylim=(args.ymin, args.ymax),
        resolution=args.resolution,
    )

    orbit_term = result["orbit_term"]
    obstacle_term = result["obstacle_term"]
    total_cost = result["total_cost"]

    # Sanity check prints
    print(f"orbit_term:    min={orbit_term.min():.6f}, max={orbit_term.max():.6f}")
    print(f"obstacle_term: min={obstacle_term.min():.6f}, max={obstacle_term.max():.6f}")
    print(f"total_cost:    min={total_cost.min():.6f}, max={total_cost.max():.6f}")
    print(f"max abs(total - (orbit+obstacle)) = {np.max(np.abs(total_cost - (orbit_term + obstacle_term))):.6e}")

    if args.log_scale:
        orbit_plot = np.log10(1.0 + orbit_term)
        obstacle_plot = np.log10(1.0 + obstacle_term)
        total_plot = np.log10(1.0 + total_cost)
        cbar_label = "log10(1 + cost)"
    else:
        orbit_plot = orbit_term
        obstacle_plot = obstacle_term
        total_plot = total_cost
        cbar_label = "cost"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    panels = [
        (orbit_plot, "Orbit term"),
        (obstacle_plot, "Obstacle term"),
        (total_plot, "Total cost = orbit + obstacle"),
    ]

    for ax, (Z, title) in zip(axes, panels):
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[args.xmin, args.xmax, args.ymin, args.ymax],
            aspect="equal",
            cmap="RdYlGn_r",   # low = green, high = red
        )
        draw_overlay(ax, result["center"], float(cfg["orbit_radius"]), result["obstacles"])
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    fig.suptitle("Heatmap of orbit + obstacle loss over the 2D plane", fontsize=14)
    fig.savefig(args.save_path, dpi=200)
    print(f"Saved figure to: {args.save_path}")
    plt.show()


if __name__ == "__main__":
    main()