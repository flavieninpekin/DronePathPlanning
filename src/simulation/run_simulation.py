import argparse
import os
import time
import sys
import numpy as np
import torch

# 获取项目根目录 (DronePathPlanning/)
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import pybullet as p
except ImportError as exc:
    raise RuntimeError("pybullet is required. Install with: pip install pybullet") from exc

from neural_controller.env.drone_env import MultiDroneEnv
from neural_controller.train.actor_critic import Actor, try_load_actor_checkpoint


def build_env_config():
    """与训练配置保持一致"""
    return {
        "num_drones": 2,
        "max_steps": 1000,                # 与训练一致
        "dt": 0.1,
        "max_speed": 1.5,                 # 与训练一致
        "collision_radius": 0.25,
        "num_dynamic_obs": 2,
        "dynamic_obs_speed": 1.0,         # 可根据需要调整
        "w_track": 1.0,
        "w_formation": 0.05,
        "w_collision": 1.0,
        "map_size": (100.0, 100.0),       # 大地图
        "use_jps_rrt": True,              # 启用 JPS-RRT
        "obstacle_density": 0.15,
        "local_grid_size": 5,             # 必须与训练一致
    }


def _draw_obstacles(grid: np.ndarray):
    """在 PyBullet 中绘制栅格地图中的障碍物（灰色方块）"""
    if grid is None:
        return
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                x = i + 0.5
                y = j + 0.5
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.2]
                    ),
                    baseVisualShapeIndex=p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.2], rgbaColor=[0.5, 0.5, 0.5, 1.0]
                    ),
                    basePosition=[x, y, 0.2]
                )


def _draw_waypoints(waypoints):
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        p.addUserDebugLine([x1, y1, 0.03], [x2, y2, 0.03], [0.2, 0.6, 1.0], 1.0)


def _make_sphere(radius, color):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=[0, 0, radius])


def _resolve_model_path(model_path):
    if os.path.isabs(model_path):
        return model_path
    if os.path.isfile(model_path):
        return model_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(script_dir, ".."))
    project_root = os.path.abspath(os.path.join(src_root, ".."))
    candidates = [
        os.path.join(src_root, model_path),
        os.path.join(project_root, model_path),
        os.path.join(src_root, "models", os.path.basename(model_path)),
    ]
    for pth in candidates:
        if os.path.isfile(pth):
            return pth
    return model_path


def run_rollout(env, actor, loaded, steps=1000, episodes=3, real_time=True, gui=True, hold=False):
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(env.dt)

    map_x, map_y = env.map_size
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[map_x / 2, map_y / 2, 0.01]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[map_x / 2, map_y / 2, 0.01], rgbaColor=[0.9, 0.9, 0.9, 1.0]
        ),
        basePosition=[map_x / 2, map_y / 2, -0.01],
    )
    _draw_waypoints(env.waypoints)
    _draw_obstacles(env.grid)  # 绘制静态障碍物

    drone_ids = [_make_sphere(env.collision_radius, [0.1, 0.4, 0.95, 1.0]) for _ in env.agents]
    obs_ids = [_make_sphere(env.collision_radius, [0.95, 0.2, 0.2, 0.9]) for _ in range(env.num_dynamic_obs)]

    for ep in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        for step in range(steps):
            actions = {}
            for agent, ob in obs.items():
                if loaded:
                    with torch.no_grad():
                        obs_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
                        mean, _ = actor(obs_t)
                        action = mean.squeeze(0).numpy()
                else:
                    action = env.action_space.sample()
                actions[agent] = action

            obs, rewards, dones, _ = env.step(actions)
            episode_reward += float(sum(rewards.values()))

            for i in range(env.num_drones):
                x, y = env.drone_positions[i]
                color = [0.1, 0.4, 0.95, 1.0] if env.drone_active[i] else [0.4, 0.4, 0.4, 0.4]
                p.changeVisualShape(drone_ids[i], -1, rgbaColor=color)
                p.resetBasePositionAndOrientation(
                    drone_ids[i], [float(x), float(y), env.collision_radius], [0, 0, 0, 1]
                )

            for i, dyn_obs in enumerate(env.dynamic_obstacles):
                x, y = dyn_obs["pos"]
                p.resetBasePositionAndOrientation(obs_ids[i], [float(x), float(y), env.collision_radius], [0, 0, 0, 1])

            if step % 20 == 0:
                alive = int(np.sum(env.drone_active))
                print(
                    f"episode={ep + 1}/{episodes} step={step:04d} "
                    f"reward={episode_reward:8.2f} alive={alive}/{env.num_drones}"
                )

            p.stepSimulation()
            if real_time:
                time.sleep(env.dt)
            if dones["__all__"]:
                print(f"Episode {ep + 1} done at step={step}, total_reward={episode_reward:.2f}")
                break

    if gui and hold:
        print("Simulation finished. Close the PyBullet window to exit.")
        while p.isConnected(client):
            time.sleep(0.1)

    if p.isConnected(client):
        p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description="PyBullet policy rollout viewer")
    parser.add_argument("--model", default=os.path.join("models", "actor_best.pth"), help="Actor checkpoint path")
    parser.add_argument("--steps", type=int, default=1000, help="Max simulation steps")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep each step")
    parser.add_argument("--hold", action="store_true", help="Keep GUI open after rollout")
    args = parser.parse_args()

    env = MultiDroneEnv(build_env_config())
    actor = Actor(env.obs_dim, env.action_dim)
    model_path = _resolve_model_path(args.model)
    loaded = try_load_actor_checkpoint(actor, model_path, map_location="cpu")
    actor.eval()
    if loaded:
        print(f"Loaded trained model: {model_path}")
    else:
        print(f"No compatible model found at: {model_path}. Running random policy for diagnosis.")

    run_rollout(
        env,
        actor,
        loaded,
        steps=args.steps,
        episodes=args.episodes,
        real_time=not args.no_realtime,
        gui=not args.no_gui,
        hold=args.hold,
    )


if __name__ == "__main__":
    main()