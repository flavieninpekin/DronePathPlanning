import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
from neural_controller.env.drone_env import MultiDroneEnv
from neural_controller.train.actor_critic import Actor, try_load_actor_checkpoint

def main():
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
    # 如果训练了模型，加载
    actor = Actor(env.obs_dim, env.action_dim)
    model_path = "models/actor_latest.pth"
    loaded = try_load_actor_checkpoint(actor, model_path, map_location="cpu")
    if loaded:
        actor.eval()
        print("Loaded trained model")
    elif not os.path.isfile(model_path):
        print("No trained model, using random actions")
    else:
        actor.eval()
        print("Checkpoint incompatible with env; using random policy")
    
    obs = env.reset()
    for step in range(500):
        actions = {}
        for agent, ob in obs.items():
            if loaded:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(ob).unsqueeze(0)
                    mean, _ = actor(obs_t)
                    action = mean.numpy()[0]
            else:
                action = env.action_space.sample()
            actions[agent] = action
        obs, rewards, dones, _ = env.step(actions)
        print(f"Step {step}, rewards: {rewards}")
        if dones["__all__"]:
            break

if __name__ == "__main__":
    main()