import torch
import torch.optim as optim
import numpy as np
import os
import sys
from contextlib import nullcontext
from tensorboardX import SummaryWriter

from torch.cuda.amp import autocast, GradScaler

# 添加 neural_controller 目录到 sys.path，使得可以导入 env 包
current_dir = os.path.dirname(__file__)                     # .../neural_controller/train
parent_dir = os.path.dirname(current_dir)                   # .../neural_controller
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)                          # 现在可以 import env.xxx

# 现在可以正确导入（env 是 neural_controller 下的子包）
from env.drone_env import MultiDroneEnv
# actor_critic 在同一目录下，直接导入
from actor_critic import Actor, Critic, try_load_actor_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return autocast(enabled=True)


class MAPPO:
    def __init__(self, env, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2, 
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
        self._use_amp = self.device.type == "cuda"
        if self._use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def get_actions(self, obs_dict, deterministic=False):
        """单次前向处理所有智能体，减少 kernel 启动与 CPU/GPU 同步次数。"""
        agents = self.env.agents
        obs_stack = np.stack([obs_dict[a] for a in agents], axis=0)
        obs_t = torch.from_numpy(obs_stack).to(self.device, non_blocking=True)
        with torch.inference_mode():
            mean, std = self.actor(obs_t)
            if deterministic:
                actions_arr = mean.cpu().numpy()
                log_probs_arr = np.zeros(self.num_agents, dtype=np.float32)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                actions_arr = action.cpu().numpy()
                log_probs_arr = log_prob.cpu().numpy()
        actions = {a: actions_arr[i] for i, a in enumerate(agents)}
        log_probs = {a: float(log_probs_arr[i]) for i, a in enumerate(agents)}
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
        arr = lambda k: np.ascontiguousarray(np.array(buffer[k], dtype=np.float32))
        obs_buf = torch.from_numpy(arr("obs")).to(self.device, non_blocking=True)
        actions_buf = torch.from_numpy(arr("actions")).to(self.device, non_blocking=True)
        old_log_probs_buf = torch.from_numpy(arr("log_probs")).to(self.device, non_blocking=True)
        rewards_buf = torch.from_numpy(arr("rewards")).to(self.device, non_blocking=True)
        dones_buf = torch.from_numpy(arr("dones")).to(self.device, non_blocking=True)
        
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
            with _amp_autocast(self._use_amp):
                values_flat = self.critic(global_states_flat).squeeze(-1)
            returns = torch.zeros_like(rewards_buf)
            running_return = torch.zeros(self.num_agents, device=self.device)
            for t in reversed(range(obs_buf.shape[0])):
                running_return = rewards_buf[t] + self.gamma * running_return * (1.0 - dones_buf[t])
                returns[t] = running_return
            returns_flat = returns.view(-1)
            advantages_flat = returns_flat - values_flat
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        chunk = total_steps // self.num_mini_batch
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, chunk):
                batch_idx = indices[start : start + chunk]
                
                obs_batch = obs_flat[batch_idx]
                actions_batch = actions_flat[batch_idx]
                old_log_probs_batch = old_log_probs_flat[batch_idx]
                advantages_batch = advantages_flat[batch_idx]
                returns_batch = returns_flat[batch_idx]
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                with _amp_autocast(self._use_amp):
                    mean, std = self.actor(obs_batch)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    new_values = self.critic(global_states_flat[batch_idx]).squeeze(-1)
                    value_loss = torch.nn.functional.mse_loss(new_values, returns_batch.float())
                    total_loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

def main():
    config = {
        'num_drones': 2,
        'max_steps': 450,
        'dt': 0.1,
        'max_speed': 1.2,
        'collision_radius': 0.25,
        'num_dynamic_obs': 2,
        'w_track': 1.0,
        'w_formation': 0.05,
        'w_collision': 1.0,
        'map_size': (30.0, 30.0),
    }
    env = MultiDroneEnv(config)
    agent = MAPPO(env, lr=1e-4)
    writer = SummaryWriter("runs/mappo")
    
    # 创建 models 文件夹（相对于项目根目录，即 src 的上一级）
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(model_dir, exist_ok=True)
    # 在 agent = MAPPO(env, lr=3e-4) 之后
    model_path = os.path.join(model_dir, "actor_latest.pth")
    if not try_load_actor_checkpoint(agent.actor, model_path, map_location=agent.device):
        if not os.path.isfile(model_path):
            print("No existing model, starting from scratch")
    
    num_episodes = 2000
    steps_per_update = 512
    for ep in range(num_episodes):
        buffer = agent.collect_experience(env, steps_per_update)
        agent.update(buffer)
        if ep % 50 == 0:
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