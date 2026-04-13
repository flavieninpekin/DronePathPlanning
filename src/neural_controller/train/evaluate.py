import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.drone_env import MultiDroneEnv
from actor_critic import Actor, try_load_actor_checkpoint

def evaluate(actor_path, num_episodes=10):
    config = {
        'num_drones': 2,
        'max_steps': 300,
        'dt': 0.1,
        'max_speed': 2.0,
        'collision_radius': 0.5,
        'num_dynamic_obs': 3,
        'w_track': 1.0,
        'w_formation': 0.5,
        'w_collision': 5.0,
        'map_size': (30, 30)
    }
    env = MultiDroneEnv(config)
    actor = Actor(env.obs_dim, env.action_dim)
    if not try_load_actor_checkpoint(actor, actor_path, map_location="cpu"):
        raise SystemExit(f"Cannot load actor weights from {actor_path}")
    actor.eval()
    
    success_count = 0
    total_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < 300:
            actions = {}
            for agent, ob in obs.items():
                with torch.no_grad():
                    obs_t = torch.FloatTensor(ob).unsqueeze(0)
                    mean, _ = actor(obs_t)
                    action = mean.numpy()[0]
                actions[agent] = action
            obs, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards.values())
            done = dones["__all__"]
            step += 1
        total_rewards.append(episode_reward)
        # 检查是否所有无人机都到达终点（简单判断：所有waypoint索引到达末尾）
        if env.waypoint_indices is not None:
            all_reached = all([idx >= len(env.waypoints)-1 for idx in env.waypoint_indices])
        else:
            all_reached = False
        if all_reached:
            success_count += 1
        print(f"Episode {ep}: reward={episode_reward:.2f}, success={all_reached}")
    print(f"Success rate: {success_count/num_episodes:.2f}, Avg reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    evaluate("models/actor_latest.pth")