import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time
import random
import logging
from contextlib import nullcontext
from tensorboardX import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# 路径修复
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 现在可以正确导入（env 是 neural_controller 下的子包）
from neural_controller.env.drone_env import MultiDroneEnv
# actor_critic 在同一目录下，直接导入
from actor_critic import Actor, Critic, try_load_actor_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    log_dir = os.path.join(project_root, "data", "log")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_mappo_{ts}.log")

    logger = logging.getLogger("mappo_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger, log_path

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return autocast('cuda', enabled=True)


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
            self.scaler = GradScaler('cuda')
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
        episode_count = 0
        episode_inactive_end_count = 0
        episode_lengths = []
        current_ep_len = 0
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
            current_ep_len += 1
            obs = next_obs
            if dones["__all__"]:
                episode_count += 1
                episode_lengths.append(current_ep_len)
                current_ep_len = 0
                if any(done_list):
                    episode_inactive_end_count += 1
                obs = env.reset()
        for key in buffer:
            buffer[key] = np.array(buffer[key]).tolist()
        if current_ep_len > 0:
            episode_lengths.append(current_ep_len)
        stats = {
            "episodes": episode_count,
            "inactive_end_episodes": episode_inactive_end_count,
            "mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else float(steps),
        }
        return buffer, stats
    
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
            values = values_flat.view(obs_buf.shape[0], self.num_agents)
            advantages = torch.zeros_like(rewards_buf)
            gae = torch.zeros(self.num_agents, device=self.device)
            next_values = torch.zeros(self.num_agents, device=self.device)
            for t in reversed(range(obs_buf.shape[0])):
                non_terminal = 1.0 - dones_buf[t]
                delta = rewards_buf[t] + self.gamma * next_values * non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
                advantages[t] = gae
                next_values = values[t]
            returns = advantages + values
            returns_flat = returns.view(-1)
            advantages_flat = advantages.view(-1)
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


def evaluate_policy(agent, env, episodes=10, max_steps=200, seed_base=12345):
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    else:
        cuda_states = None

    returns = []
    success_count = 0
    instant_fail_count = 0
    for i in range(episodes):
        seed = seed_base + i
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        total_reward = 0.0
        obs = env.reset()
        episode_success = False
        for _ in range(max_steps):
            actions, _ = agent.get_actions(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(actions)
            total_reward += sum(rewards.values())
            # 检查是否有无人机到达终点（最后一个航点）
            if np.any(env.waypoint_indices >= len(env.waypoints)):
                episode_success = True
            if dones["__all__"]:
                break
        returns.append(total_reward)
        if episode_success:
            success_count += 1

    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)

    returns_arr = np.array(returns, dtype=np.float32)
    success_rate = success_count / max(1, episodes)
    instant_fail_rate = instant_fail_count / max(1, episodes)
    return float(returns_arr.mean()), float(returns_arr.std()), success_rate, instant_fail_rate

def main():
    logger, log_path = setup_logger()
    logger.info("Using device: %s", device)
    logger.info("Log file: %s", log_path)

    # 确保数据和地图缓存目录存在
    data_dir = os.path.join(project_root, "data")
    map_cache_dir = os.path.join(data_dir, "map")
    os.makedirs(map_cache_dir, exist_ok=True)

    base_config = {
        'num_drones': 6,
        'max_steps': 2000,
        'dt': 0.1,
        'max_speed': 2.0,
        'collision_radius': 0.25,
        'num_dynamic_obs': 2,
        'dynamic_obs_speed': 0.0,
        'w_track': 1.0,
        'w_formation': 0.05,
        'w_collision': 0.0,
        'death_penalty': -40.0,
        'collision_alert_radius_factor': 4.0,
        'obstacle_alert_coef': 1.5,
        'stuck_max_steps': 35,
        'stuck_progress_eps': 0.015,
        'stuck_speed_eps': 0.08,
        'stuck_penalty': -45.0,
        'collision_grace_steps': 10,
        'min_spawn_dist_factor': 6.0,
        'min_obs_start_dist_factor': 10.0,
        'map_size': (100, 100),
        'use_jps_rrt': True,         # 启用 JPS-RRT 全局路径规划
        'obstacle_density': 0.15,    # 障碍物密度（0~1），仅在未传入 grid 时使用
        'local_grid_size': 5,
        'map_generation_attempts': 10,
        'use_map_pool': True,          # 启用地图池
        'map_pool_size': 150,          # 池大小，可根据需要调整
        'regenerate_map': False,       # 使用池时务必设为 False
        'formation_sight_range': 4.0,    # 视觉范围
        'w_formation_a': 0,
        'w_formation_b': 0,
        'w_collision': 0.5,              # 初始碰撞权重降低，由课程逐步增加
        'map_pool_cache_path': os.path.abspath(os.path.join(project_root, "data", "map", "map_pool_150.pkl")),
    }
    curriculum_schedule = [
        (0,     0.0, 0.0),    # Stage 1: 无动态障碍，仅学习沿路径飞行和集群保持
        (5000,  0.0, 0.1),    # Stage 2: 引入静态障碍物贴近惩罚的碰撞权重
        (9000,  0.3, 0.3),    # Stage 3: 动态障碍缓慢移动
        (12000, 0.6, 0.6),    # Stage 4: 加速
        (16000, 1.0, 1.0),    # Stage 5: 全速
    ]
    schedule_idx = 0
    env = MultiDroneEnv(base_config)
    agent = MAPPO(env, lr=3e-5, gamma=0.99, gae_lambda=0.95, clip_param=0.2, ppo_epochs=10, num_mini_batch=2, value_loss_coef=0.5, entropy_coef=0.02)      # 降低学习率，使训练更稳定，增大熵系数，鼓励探索
    writer = SummaryWriter("runs/mappo")
    
    # 创建 models 文件夹（相对于项目根目录，即 src 的上一级）
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(model_dir, exist_ok=True)
    # 在 agent = MAPPO(env, lr=3e-4) 之后
    model_path = os.path.join(model_dir, "actor_latest.pth")
    best_actor_path = os.path.join(model_dir, "actor_best.pth")
    best_critic_path = os.path.join(model_dir, "critic_best.pth")
    if not try_load_actor_checkpoint(agent.actor, model_path, map_location=agent.device):
        if not os.path.isfile(model_path):
            logger.info("No existing model, starting from scratch")
    else:
        logger.info("Loaded checkpoint: %s", model_path)
    
    num_episodes = 6000
    steps_per_update = 2048
    eval_episodes = 10
    best_eval_score = -float("inf")
    for ep in range(num_episodes):
        if schedule_idx + 1 < len(curriculum_schedule) and ep >= curriculum_schedule[schedule_idx + 1][0]:
            schedule_idx += 1
            _, speed, collision_weight = curriculum_schedule[schedule_idx]
            env.dynamic_obs_speed = speed
            env.w_collision = collision_weight
            logger.info(
                f"[Curriculum] stage={schedule_idx + 1}/{len(curriculum_schedule)} "
                f"dynamic_obs_speed={speed:.2f}, w_collision={collision_weight:.2f}"
            )

        t_collect_start = time.perf_counter()
        buffer, rollout_stats = agent.collect_experience(env, steps_per_update)
        collect_time = time.perf_counter() - t_collect_start

        t_update_start = time.perf_counter()
        agent.update(buffer)
        update_time = time.perf_counter() - t_update_start
        inactive_end_rate = rollout_stats["inactive_end_episodes"] / max(1, rollout_stats["episodes"])

        writer.add_scalar("train/collect_time_s", collect_time, ep)
        writer.add_scalar("train/update_time_s", update_time, ep)
        writer.add_scalar("train/inactive_end_rate", inactive_end_rate, ep)
        writer.add_scalar("train/mean_episode_length", rollout_stats["mean_ep_len"], ep)

        if ep % 10 == 0:
            logger.info(
                f"Episode {ep} | collect {collect_time:.2f}s | update {update_time:.2f}s | "
                f"inactive_end_rate {inactive_end_rate:.2f} | mean_ep_len {rollout_stats['mean_ep_len']:.1f}"
            )
        if ep % 50 == 0:
            eval_mean, eval_std, eval_success_rate, eval_instant_fail_rate = evaluate_policy(
                agent, env, episodes=eval_episodes, max_steps=1000, seed_base=20260
            )
            writer.add_scalar("eval/reward_mean", eval_mean, ep)
            writer.add_scalar("eval/reward_std", eval_std, ep)
            writer.add_scalar("eval/success_rate", eval_success_rate, ep)
            writer.add_scalar("eval/instant_fail_rate", eval_instant_fail_rate, ep)
            logger.info(
                f"Episode {ep}, Eval mean/std: {eval_mean:.2f} / {eval_std:.2f} | "
                f"success_rate {eval_success_rate:.2f} | instant_fail_rate {eval_instant_fail_rate:.2f}"
            )
            # 保存模型到 models 文件夹
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, "actor_latest.pth"))
            score = eval_mean + 200.0 * eval_success_rate - 80.0 * eval_instant_fail_rate
            if score > best_eval_score:
                best_eval_score = score
                torch.save(agent.actor.state_dict(), best_actor_path)
                torch.save(agent.critic.state_dict(), best_critic_path)
                logger.info(
                    f"New best checkpoint saved at episode {ep}: "
                    f"mean={eval_mean:.2f}, success_rate={eval_success_rate:.2f}, "
                    f"instant_fail_rate={eval_instant_fail_rate:.2f}"
                )
    writer.close()
    logger.info("Training finished. TensorBoard dir: runs/mappo")

if __name__ == "__main__":
    main()