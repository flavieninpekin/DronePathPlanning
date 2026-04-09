import torch
import torch.optim as optim
import numpy as np
import os
import sys
from tensorboardX import SummaryWriter

# 添加 neural_controller 目录到 sys.path，使得可以导入 env 包
current_dir = os.path.dirname(__file__)                     # .../neural_controller/train
parent_dir = os.path.dirname(current_dir)                   # .../neural_controller
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)                          # 现在可以 import env.xxx

# 现在可以正确导入（env 是 neural_controller 下的子包）
from env.drone_env import MultiDroneEnv
# actor_critic 在同一目录下，直接导入
from actor_critic import Actor, Critic  # 注意文件名是 actor_critic.py，类名 Actor, Critic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MAPPO:
    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2, 
                 ppo_epochs=10, num_mini_batch=4, value_loss_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.num_agents = env.num_drones
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        
        self.global_state_dim = self.num_agents * self.obs_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.global_state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def get_actions(self, obs_dict, deterministic=False):
        actions = {}
        log_probs = {}
        for agent, obs in obs_dict.items():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mean, std = self.actor(obs_t)
            if deterministic:
                action = mean.detach().cpu().numpy()[0]
                log_prob = 0
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                action = action.detach().cpu().numpy()[0]
                log_prob = log_prob.detach().cpu().numpy()[0]
            actions[agent] = action
            log_probs[agent] = log_prob
        return actions, log_probs
    
    def collect_experience(self, env, steps):
        buffer = {
            'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []
        }
        obs = env.reset()
        for _ in range(steps):
            actions, log_probs = self.get_actions(obs)
            next_obs, rewards, dones, _ = env.step(actions)
            obs_list = [obs[agent] for agent in env.agents]
            action_list = [actions[agent] for agent in env.agents]
            log_prob_list = [log_probs[agent] for agent in env.agents]
            reward_list = [rewards[agent] for agent in env.agents]
            done_list = [dones[agent] for agent in env.agents]
            buffer['obs'].append(obs_list)
            buffer['actions'].append(action_list)
            buffer['log_probs'].append(log_prob_list)
            buffer['rewards'].append(reward_list)
            buffer['dones'].append(done_list)
            obs = next_obs
            if dones["__all__"]:
                obs = env.reset()
        for key in buffer:
            buffer[key] = np.array(buffer[key]).tolist()
        return buffer
    
    def update(self, buffer):
        obs_buf = torch.FloatTensor(np.array(buffer['obs'])).to(device)  # (steps, num_agents, obs_dim)
        actions_buf = torch.FloatTensor(np.array(buffer['actions'])).to(device)
        old_log_probs_buf = torch.FloatTensor(np.array(buffer['log_probs'])).to(device)
        rewards_buf = torch.FloatTensor(np.array(buffer['rewards'])).to(device)
        dones_buf = torch.FloatTensor(np.array(buffer['dones'])).to(device)
        
        # 展平所有智能体数据
        total_steps = obs_buf.shape[0] * self.num_agents
        obs_flat = obs_buf.view(-1, self.obs_dim)
        actions_flat = actions_buf.view(-1, self.action_dim)
        old_log_probs_flat = old_log_probs_buf.view(-1)
        rewards_flat = rewards_buf.view(-1)
        dones_flat = dones_buf.view(-1)
        
        # 全局状态（每个智能体对应的全局状态 = 所有智能体观测拼接）
        global_states = obs_buf.view(obs_buf.shape[0], -1)  # (steps, num_agents*obs_dim)
        global_states_flat = global_states.unsqueeze(1).expand(-1, self.num_agents, -1).reshape(-1, global_states.shape[-1])
        
        with torch.no_grad():
            values_flat = self.critic(global_states_flat).squeeze(-1)
            advantages_flat = rewards_flat - values_flat
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(total_steps)
            for start in range(0, total_steps, total_steps // self.num_mini_batch):
                end = start + total_steps // self.num_mini_batch
                batch_idx = indices[start:end]
                
                obs_batch = obs_flat[batch_idx]
                actions_batch = actions_flat[batch_idx]
                old_log_probs_batch = old_log_probs_flat[batch_idx]
                advantages_batch = advantages_flat[batch_idx]
                values_batch = values_flat[batch_idx]
                
                mean, std = self.actor(obs_batch)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                new_values = self.critic(global_states_flat[batch_idx]).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(new_values, values_batch)
                
                total_loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

def main():
    config = {
        'num_drones': 2,
        'max_steps': 300,
        'dt': 0.1,
        'max_speed': 2.0,
        'collision_radius': 0.25,
        'num_dynamic_obs': 0,
        'w_track': 1.0,
        'w_formation': 0.5,
        'w_collision': 5.0,
        'map_size': (30.0, 30.0),
    }
    env = MultiDroneEnv(config)
    agent = MAPPO(env, lr=3e-4)
    writer = SummaryWriter("runs/mappo")
    
    # 创建 models 文件夹（相对于项目根目录，即 src 的上一级）
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(model_dir, exist_ok=True)
    
    num_episodes = 2000
    steps_per_update = 200
    for ep in range(num_episodes):
        buffer = agent.collect_experience(env, steps_per_update)
        agent.update(buffer)
        if ep % 20 == 0:
            total_reward = 0
            obs = env.reset()
            for _ in range(200):
                actions, _ = agent.get_actions(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(actions)
                total_reward += sum(rewards.values())
                if dones["__all__"]:
                    break
            writer.add_scalar("eval_reward", total_reward, ep)
            print(f"Episode {ep}, Eval Reward: {total_reward:.2f}")
            # 保存模型到 models 文件夹
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, "actor_latest.pth"))
    writer.close()

if __name__ == "__main__":
    main()