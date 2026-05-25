import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys
import argparse

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from map_generator.RuleMapGenerator import MAP_TYPES, generate
from neural_controller.env.drone_env import MultiDroneEnv

OUTPUT_DIR = os.path.join(os.path.dirname(current_file), "..", "..", "data", "vis_v4")


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _plot_grid(ax, grid, title=None):
    ax.imshow(grid.T, cmap="gray_r", origin="lower", interpolation="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)


def _plot_waypoints(ax, waypoints, **kwargs):
    wp = np.array(waypoints)
    ax.plot(wp[:, 0], wp[:, 1], **kwargs)


def visualize_map_types(size=(100, 100), seed=42):
    _ensure_output()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (name, func) in enumerate(MAP_TYPES.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        grid, waypoints = generate(name, size, seed=seed)
        _plot_grid(ax, grid, title=name)
        _plot_waypoints(ax, waypoints, color="red", linewidth=1, alpha=0.7, label="waypoints")
        ax.legend(fontsize=8)
    axes[-1].text(0.5, 0.5, "Rule Map Types\nDronePathPlanning v4",
                  transform=axes[-1].transAxes, ha="center", va="center", fontsize=14)
    axes[-1].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_types.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_map_pool(map_type, size=(100, 100), pool_size=9, base_seed=0):
    _ensure_output()
    cols = min(3, int(np.ceil(pool_size ** 0.5)))
    rows = int(np.ceil(pool_size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(pool_size):
        ax = axes[i]
        grid, waypoints = generate(map_type, size, seed=base_seed + i)
        _plot_grid(ax, grid, title=f"{map_type} #{i}")
        _plot_waypoints(ax, waypoints, color="red", linewidth=0.8, alpha=0.6)
    for j in range(pool_size, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"pool_{map_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def sample_eval(
    model_path,
    map_type="maze",
    map_size=(100, 100),
    num_drones=6,
    max_steps=500,
    seed=42,
):
    _ensure_output()
    np.random.seed(seed)

    grid, waypoints = generate(map_type, map_size, seed=seed)

    config = {
        "num_drones": num_drones,
        "max_steps": max_steps,
        "dt": 0.1,
        "max_speed": 2.0,
        "collision_radius": 0.25,
        "num_dynamic_obs": 0,
        "w_track": 1.0,
        "w_formation": 0.0,
        "w_collision": 0.0,
        "map_size": map_size,
        "use_jps_rrt": False,
        "local_grid_size": 5,
        "use_map_pool": False,
        "grid": grid,
    }
    env = MultiDroneEnv(config)
    env.waypoints = waypoints
    env.grid = grid

    if model_path and os.path.isfile(model_path):
        import importlib

        try:
            actor_mod = importlib.import_module("neural_controller.actor_critic")
        except ModuleNotFoundError:
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            actor_mod = importlib.import_module("neural_controller.actor_critic")

        Actor = actor_mod.Actor
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(obs_dim, action_dim, hidden_dim=256).to(device)
        state = torch.load(model_path, map_location=device)
        actor.load_state_dict(state, strict=False)
        actor.eval()

    obs = env.reset()
    traj = {a: [] for a in env.agents}

    for _ in range(max_steps):
        if model_path:
            obs_t = torch.from_numpy(np.stack([obs[a] for a in env.agents], axis=0)).float().to(device)
            with torch.no_grad():
                mean, _ = actor(obs_t)
            actions_np = mean.cpu().numpy()
            actions = {a: actions_np[i] for i, a in enumerate(env.agents)}
        else:
            actions = {a: np.random.uniform(-2, 2, size=2) for a in env.agents}

        for a in env.agents:
            traj[a].append(env.drone_positions[env.agents.index(a)].copy())

        obs, rewards, dones, _ = env.step(actions)

        if dones["__all__"]:
            break

    steps_actual = len(traj["drone_0"])
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_drones))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    _plot_grid(ax1, grid, title=f"{map_type} — Drone Trajectories ({steps_actual} steps)")
    _plot_waypoints(ax1, waypoints, color="red", linewidth=1.5, alpha=0.5, label="waypoints")
    for i in range(num_drones):
        t = np.array(traj[f"drone_{i}"])
        ax1.plot(t[:, 0], t[:, 1], color=colors[i], linewidth=1.2, alpha=0.8, label=f"drone_{i}")
        ax1.scatter(t[0, 0], t[0, 1], color=colors[i], marker="o", s=40, zorder=5)
        ax1.scatter(t[-1, 0], t[-1, 1], color=colors[i], marker="x", s=60, zorder=5)
    ax1.legend(fontsize=7, loc="upper right")

    _plot_grid(ax2, grid, title=f"{map_type} — Terminal Positions")
    _plot_waypoints(ax2, waypoints, color="red", linewidth=1, alpha=0.3, label="waypoints")
    for i in range(num_drones):
        t = np.array(traj[f"drone_{i}"])
        ax2.plot(t[:, 0], t[:, 1], color=colors[i], linewidth=0.5, alpha=0.4)
        ax2.scatter(t[0, 0], t[0, 1], color=colors[i], marker="o", s=30, zorder=5, label=f"d{i} start")
        ax2.scatter(t[-1, 0], t[-1, 1], color=colors[i], marker="s", s=50, zorder=5, label=f"d{i} end")

    plt.tight_layout()
    fname = f"eval_{map_type}_{os.path.basename(model_path or 'random').replace('.pth', '')}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    env.close()


def visualize_all_maps_grid(size=(100, 100), seed=42):
    _ensure_output()
    fig, axes = plt.subplots(len(MAP_TYPES), 3, figsize=(15, 4 * len(MAP_TYPES)))
    for row, (name, func) in enumerate(MAP_TYPES.items()):
        grid, waypoints = generate(name, size, seed=seed)
        _plot_grid(axes[row, 0], grid, title=f"{name} — grid")
        _plot_waypoints(axes[row, 0], waypoints, color="red", linewidth=1, alpha=0.6)

        axes[row, 1].hist(grid.flatten(), bins=2, rwidth=0.8)
        axes[row, 1].set_xticks([0, 1])
        axes[row, 1].set_xticklabels(["free", "obstacle"])
        axes[row, 1].set_title(f"{name} — obstacle ratio: {grid.mean():.2f}")

        wp = np.array(waypoints)
        axes[row, 2].plot(wp[:, 0], wp[:, 1], "b-", alpha=0.5)
        axes[row, 2].scatter(wp[0, 0], wp[0, 1], c="green", s=30, label="start")
        axes[row, 2].scatter(wp[-1, 0], wp[-1, 1], c="red", s=30, label="goal")
        axes[row, 2].set_title(f"{name} — {len(waypoints)} waypoints")
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xlabel("x")
        axes[row, 2].set_ylabel("y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "all_maps_detail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Rule Map Visualization")
    parser.add_argument("--model", type=str, default=None, help="Path to actor .pth checkpoint")
    parser.add_argument("--map-type", type=str, default="maze", choices=list(MAP_TYPES.keys()) + ["all"],
                        help="Map type to visualize")
    parser.add_argument("--task", type=str, default="maps",
                        choices=["maps", "pool", "eval", "detail"],
                        help="Visualization task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=6)
    args = parser.parse_args()

    _ensure_output()

    if args.task == "maps":
        visualize_map_types(seed=args.seed)

    elif args.task == "detail":
        visualize_all_maps_grid(seed=args.seed)

    elif args.task == "pool":
        if args.map_type == "all":
            for mt in MAP_TYPES:
                visualize_map_pool(mt, pool_size=args.pool_size, base_seed=args.seed)
        else:
            visualize_map_pool(args.map_type, pool_size=args.pool_size, base_seed=args.seed)

    elif args.task == "eval":
        sample_eval(args.model, args.map_type, seed=args.seed)
