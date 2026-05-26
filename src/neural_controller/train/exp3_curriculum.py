"""Exp3: Obstacle density curriculum learning.

Progressively increase obstacle density so the agent first learns
pure path-following on empty maps, then gradually learns to avoid
obstacles as they appear, eventually reaching Metropolis-like density.

Curriculum stages (map 200x200):
  Ep 0:      density=0.0   open field
  Ep 2000:   density=0.03  very sparse
  Ep 5000:   density=0.06  sparse
  Ep 10000:  density=0.10  moderate
  Ep 16000:  density=0.20  dense
  Ep 24000:  density=0.36  v.dense (≈Metropolis 36% free)
  Ep 34000:  density=0.50  extra dense
"""
import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time
import random
import logging
import multiprocessing as mp
from collections import deque
from contextlib import nullcontext

try:
    from torch.amp.autocast_mode import autocast
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast
    _NEW_AMP = False
try:
    from torch.amp.grad_scaler import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from neural_controller.env.drone_env_v2 import MultiDroneEnvV2
from actor_critic import Actor, Critic, try_load_actor_checkpoint
from map_generator.MapGenerator import generate_map_with_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Map helpers
# =============================================================================

def _interpolate_path(path, step=0.5):
    wps = []
    for i in range(len(path) - 1):
        a = np.array(path[i], dtype=np.float64)
        b = np.array(path[i + 1], dtype=np.float64)
        seg = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(seg / step)))
        for j in range(n):
            wps.append(tuple(a + (j / n) * (b - a)))
    wps.append(tuple(path[-1]))
    return wps


def _bfs_path(grid, start, goal):
    """BFS from start to goal on free cells (grid==0). Returns list of (x,y)."""
    h, w = grid.shape
    visited = {start}
    q = deque([(start, [start])])
    while q:
        pos, path = q.popleft()
        if pos == goal:
            return path
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append(((nx, ny), path + [(nx, ny)]))
    return None


def _make_pool(size, density, n, seed=0):
    pool = []
    rng = random.Random(seed)
    for i in range(n):
        s = rng.randint(0, 2 ** 31)
        np.random.seed(s)
        random.seed(s)
        grid = generate_map_with_path(size, density, dim=2, step_factor=2.5, p=0.7, max_attempts=100, channel_expansion=1)
        start, goal = (0, 0), (size[0] - 1, size[1] - 1)
        p = _bfs_path(grid, start, goal)
        if p is None:
            continue
        wps = _interpolate_path(p)
        pool.append({"grid": grid, "waypoints": wps})
    if len(pool) < n:
        raise RuntimeError(f"Only generated {len(pool)}/{n} maps at density={density}")
    return pool


# =============================================================================
# Curriculum
# =============================================================================

STAGES = [
    (0,     0.0,  5,  "open field"),
    (2000,  0.03, 10, "very sparse"),
    (5000,  0.06, 10, "sparse"),
    (10000, 0.10, 10, "moderate"),
    (16000, 0.20, 10, "dense"),
    (24000, 0.36, 10, "v.dense ≈ Metropolis"),
    (34000, 0.50, 10, "extra dense"),
]


def _stage_for_ep(ep):
    for i in range(len(STAGES) - 1, -1, -1):
        if ep >= STAGES[i][0]:
            return i
    return 0


# =============================================================================
# PATCH: fix speed reward direction + stuck counter
# =============================================================================

def _patched_step(self, actions):
    action_matrix = np.array([actions[agent] for agent in self.agents], dtype=np.float32)
    action_matrix = np.clip(action_matrix, -self.max_speed, self.max_speed)
    new_positions = self.drone_positions + action_matrix * self.dt
    new_positions = np.clip(new_positions, 0, self.map_size[0])
    self.drone_velocities = action_matrix
    self.drone_positions = new_positions

    for obs in self.dynamic_obstacles:
        obs['pos'] += obs['vel'] * self.dt
        for dim in range(2):
            if obs['pos'][dim] < 0 or obs['pos'][dim] > self.map_size[0]:
                obs['vel'][dim] *= -1
                obs['pos'][dim] = np.clip(obs['pos'][dim], 0, self.map_size[0])

    old_waypoint_indices = self.waypoint_indices.copy()
    old_dist_to_target = np.array([
        np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[self.waypoint_indices[i]]))
        if self.drone_active[i] and self.waypoint_indices[i] < len(self.waypoints)
        else 0.0
        for i in range(self.num_drones)
    ], dtype=np.float32)

    self.prev_dist_to_target = self.prev_dist_to_target.copy()

    for i in range(self.num_drones):
        if not self.drone_active[i]:
            continue
        wp_idx = self.waypoint_indices[i]
        if wp_idx < len(self.waypoints) - 1:
            dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
            if dist_to_wp < self.goal_tolerance:
                self.waypoint_indices[i] = wp_idx + 1
        elif wp_idx == len(self.waypoints) - 1:
            dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
            if dist_to_wp < self.goal_tolerance:
                self.waypoint_indices[i] = len(self.waypoints)
        if self.waypoint_indices[i] < len(self.waypoints):
            self.prev_dist_to_target[i] = np.linalg.norm(
                self.drone_positions[i] - np.array(self.waypoints[self.waypoint_indices[i]]))
        else:
            self.prev_dist_to_target[i] = 0.0

    prev_active = self.drone_active.copy()
    collision_occurred = np.zeros(self.num_drones, dtype=bool)

    if self.current_step >= self.collision_grace_steps:
        for i in range(self.num_drones):
            if not self.drone_active[i]:
                continue
            for j in range(i + 1, self.num_drones):
                if not self.drone_active[j]:
                    continue
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 2 * self.collision_radius:
                    collision_occurred[i] = True
                    collision_occurred[j] = True
            for obs in self.dynamic_obstacles:
                if np.linalg.norm(self.drone_positions[i] - obs['pos']) < 2 * self.collision_radius:
                    collision_occurred[i] = True
                    break
            if self._is_in_obstacle(self.drone_positions[i]):
                collision_occurred[i] = True

    self.drone_active = np.logical_and(self.drone_active, ~collision_occurred)
    newly_inactive_collision = np.logical_and(prev_active, collision_occurred)
    newly_inactive_stuck = np.zeros(self.num_drones, dtype=bool)

    for i in range(self.num_drones):
        if not self.drone_active[i]:
            continue
        if self.waypoint_indices[i] > old_waypoint_indices[i]:
            self.stuck_counters[i] = 0
        else:
            curr_dist = self.prev_dist_to_target[i]
            if curr_dist < old_dist_to_target[i] - 0.02:
                self.stuck_counters[i] = 0
            elif curr_dist > old_dist_to_target[i] + 0.1:
                self.stuck_counters[i] += 2
            else:
                self.stuck_counters[i] += 1

    rewards = {}
    for i, agent in enumerate(self.agents):
        if not self.drone_active[i]:
            if newly_inactive_collision[i]:
                rewards[agent] = self.death_penalty
            elif newly_inactive_stuck[i]:
                rewards[agent] = self.stuck_penalty
            else:
                rewards[agent] = 0.0
            continue

        reward = 0.0

        if self.waypoint_indices[i] >= len(self.waypoints):
            reward += self.goal_bonus
            self.drone_active[i] = False
            self.stuck_counters[i] = 0
            rewards[agent] = reward
            continue

        if self.waypoint_indices[i] > old_waypoint_indices[i]:
            wp_reward = self.wp_base + self.wp_scale * self.waypoint_indices[i]
            reward += wp_reward
            self.stuck_counters[i] = 0

        curr_wp = np.array(self.waypoints[self.waypoint_indices[i]])
        target_dir = curr_wp - self.drone_positions[i]
        norm = np.linalg.norm(target_dir) + 1e-8
        target_dir_norm = target_dir / norm
        speed_along = np.dot(self.drone_velocities[i], target_dir_norm)
        reward += self.speed_reward_coef * max(0, speed_along)

        if self.stuck_counters[i] >= self.stuck_max_steps:
            self.drone_active[i] = False
            newly_inactive_stuck[i] = True
            reward += self.stuck_penalty
            rewards[agent] = reward
            self.stuck_counters[i] = 0
            continue

        rewards[agent] = reward

    self.current_step += 1
    done_all = (not np.any(self.drone_active)) or (self.current_step >= self.max_steps)
    dones = {agent: (not self.drone_active[i]) for i, agent in enumerate(self.agents)}
    dones["__all__"] = done_all
    obs = self._get_obs()
    infos = {agent: {} for agent in self.agents}
    return obs, rewards, dones, infos


MultiDroneEnvV2.step = _patched_step


# =============================================================================
# Parallel runner (same as exp2)
# =============================================================================

def _env_worker(env_config, conn, seed):
    np.random.seed(seed)
    random.seed(seed)
    env = MultiDroneEnvV2(env_config)
    map_pool = conn.recv()
    env.map_pool = map_pool
    env.use_map_pool = True
    obs = env.reset()
    conn.send(obs)
    while True:
        cmd = conn.recv()
        if cmd == "step":
            actions = conn.recv()
            obs, rewards, dones, _ = env.step(actions)
            conn.send((obs, rewards, dones))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "set_params":
            params = conn.recv()
            for k, v in params.items():
                setattr(env, k, v)
            conn.send("ok")
        elif cmd == "get_success":
            success = all(i >= len(env.waypoints) for i in env.waypoint_indices)
            conn.send(success)
        elif cmd == "get_debug":
            info = {
                "wp_indices": [int(i) for i in env.waypoint_indices],
                "active": [bool(a) for a in env.drone_active],
                "step": int(env.current_step),
                "num_wp": int(len(env.waypoints)),
            }
            conn.send(info)
        elif cmd == "close":
            break


class ParallelEnvRunner:
    def __init__(self, env_config, num_envs, initial_pool):
        self.num_envs = num_envs
        self.num_agents = env_config['num_drones']
        self.obs_dim = 0
        self.action_dim = 2
        self.parent_conns = []
        self.workers = []
        for i in range(num_envs):
            parent, child = mp.Pipe()
            p = mp.Process(target=_env_worker, args=(env_config, child, 42 + i), daemon=True)
            p.start()
            self.parent_conns.append(parent)
            self.workers.append(p)
            child.close()
            parent.send(initial_pool)
        first_obs = self.parent_conns[0].recv()
        self.obs_dim = len(first_obs[list(first_obs.keys())[0]])
        for conn in self.parent_conns[1:]:
            conn.recv()

    def _batch_obs(self, all_obs):
        batch = np.zeros((self.num_envs, self.num_agents, self.obs_dim), dtype=np.float32)
        for e in range(self.num_envs):
            obs_dict = all_obs[e]
            for i, agent in enumerate(sorted(obs_dict.keys())):
                batch[e, i] = obs_dict[agent]
        return batch

    def get_actions_batched(self, all_obs, actor, deterministic=False):
        batch = self._batch_obs(all_obs)
        obs_t = torch.from_numpy(batch).to(device, non_blocking=True)
        obs_flat = obs_t.view(-1, self.obs_dim)
        with torch.inference_mode():
            with autocast('cuda', enabled=True) if _NEW_AMP else autocast(enabled=True):
                mean, std = actor(obs_flat)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
        actions_np = action.cpu().numpy()
        if deterministic:
            log_probs_np = np.zeros(self.num_envs * self.num_agents, dtype=np.float32)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
            log_probs_np = log_prob.cpu().numpy()
        all_actions, all_log_probs = [], []
        for e in range(self.num_envs):
            start = e * self.num_agents
            all_actions.append({f"drone_{i}": actions_np[start + i] for i in range(self.num_agents)})
            all_log_probs.append({f"drone_{i}": float(log_probs_np[start + i]) for i in range(self.num_agents)})
        return all_actions, all_log_probs

    def step(self, all_actions):
        for e in range(self.num_envs):
            self.parent_conns[e].send("step")
            self.parent_conns[e].send(all_actions[e])
        all_obs, all_rewards, all_dones = [], [], []
        for e in range(self.num_envs):
            obs, rewards, dones = self.parent_conns[e].recv()
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_dones.append(dones)
        return all_obs, all_rewards, all_dones

    def reset(self):
        for e in range(self.num_envs):
            self.parent_conns[e].send("reset")
        return [conn.recv() for conn in self.parent_conns]

    def set_params(self, params):
        for conn in self.parent_conns:
            conn.send("set_params")
            conn.send(params)
        for conn in self.parent_conns:
            conn.recv()

    def get_success(self, env_idx=0):
        self.parent_conns[env_idx].send("get_success")
        return self.parent_conns[env_idx].recv()

    def get_debug(self, env_idx=0):
        self.parent_conns[env_idx].send("get_debug")
        return self.parent_conns[env_idx].recv()

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send("close")
                conn.close()
            except Exception:
                pass
        for p in self.workers:
            p.terminate()
            p.join()


class MAPPO:
    def __init__(self, env_runner, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 ppo_epochs=20, num_mini_batch=8, value_loss_coef=0.5, entropy_coef=0.03,
                 actor_hidden=256, critic_hidden=512):
        self.runner = env_runner
        self.num_envs = env_runner.num_envs
        self.num_agents = env_runner.num_agents
        self.obs_dim = env_runner.obs_dim
        self.action_dim = env_runner.action_dim
        self.global_state_dim = self.num_agents * self.obs_dim
        self.device = device
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dim=actor_hidden).to(self.device)
        self.critic = Critic(self.global_state_dim, hidden_dim=critic_hidden).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self._use_amp = self.device.type == "cuda"
        if self._use_amp:
            self.scaler = GradScaler('cuda') if _NEW_AMP else GradScaler()
        else:
            self.scaler = None
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def collect_experience(self, steps):
        num_envs, num_agents = self.num_envs, self.num_agents
        buffer = {
            'obs': np.zeros((steps, num_envs, num_agents, self.obs_dim), dtype=np.float32),
            'actions': np.zeros((steps, num_envs, num_agents, self.action_dim), dtype=np.float32),
            'log_probs': np.zeros((steps, num_envs, num_agents), dtype=np.float32),
            'rewards': np.zeros((steps, num_envs, num_agents), dtype=np.float32),
            'dones': np.zeros((steps, num_envs, num_agents), dtype=np.float32),
        }
        all_obs = self.runner.reset()
        for step in range(steps):
            all_actions, all_log_probs = self.runner.get_actions_batched(all_obs, self.actor)
            next_obs, rewards, dones = self.runner.step(all_actions)
            for e in range(num_envs):
                for i in range(num_agents):
                    agent = f"drone_{i}"
                    buffer['obs'][step, e, i] = all_obs[e][agent]
                    buffer['actions'][step, e, i] = all_actions[e][agent]
                    buffer['log_probs'][step, e, i] = all_log_probs[e][agent]
                    buffer['rewards'][step, e, i] = rewards[e][agent]
                    buffer['dones'][step, e, i] = dones[e][agent]
            all_obs = next_obs
        return buffer

    def update(self, buffer):
        steps = buffer['obs'].shape[0]
        num_envs = self.num_envs
        total = steps * num_envs * self.num_agents
        obs_buf = torch.from_numpy(buffer['obs']).to(self.device, non_blocking=True)
        act_buf = torch.from_numpy(buffer['actions']).to(self.device, non_blocking=True)
        old_lp_buf = torch.from_numpy(buffer['log_probs']).to(self.device, non_blocking=True)
        rew_buf = torch.from_numpy(buffer['rewards']).to(self.device, non_blocking=True)
        don_buf = torch.from_numpy(buffer['dones']).to(self.device, non_blocking=True)
        obs_f = obs_buf.view(-1, self.obs_dim)
        act_f = act_buf.view(-1, self.action_dim)
        old_lp_f = old_lp_buf.view(-1)
        gs = obs_buf.view(steps, num_envs, -1)
        gs_f = gs.unsqueeze(-2).expand(-1, -1, self.num_agents, -1).reshape(-1, self.global_state_dim)

        with torch.no_grad():
            with autocast('cuda', enabled=True) if _NEW_AMP else autocast(enabled=True):
                values = self.critic(gs_f)
            values = values.view(steps, -1).cpu().numpy()
            rew = rew_buf.cpu().numpy().reshape(steps, -1)
            don = don_buf.cpu().numpy().reshape(steps, -1)
            advantages = np.zeros((steps, num_envs * self.num_agents), dtype=np.float32)
            last_gae = np.zeros(num_envs * self.num_agents, dtype=np.float32)
            for t in reversed(range(steps)):
                mask = 1.0 - don[t]
                delta = rew[t] + self.gamma * values[min(t + 1, steps - 1)] * mask - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
                advantages[t] = last_gae
            returns = advantages + values
            advantages = advantages.ravel()
            returns = returns.ravel()

        adv_t = torch.from_numpy(advantages).to(self.device, non_blocking=True)
        ret_t = torch.from_numpy(returns).to(self.device, non_blocking=True)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(total)
            for start in range(0, total, self.num_mini_batch):
                idx = perm[start:start + self.num_mini_batch]
                with autocast('cuda', enabled=True) if _NEW_AMP else autocast(enabled=True):
                    mean, std = self.actor(obs_f[idx])
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(act_f[idx]).sum(dim=-1)
                    ratio = torch.exp(log_prob - old_lp_f[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_t[idx]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = torch.nn.functional.mse_loss(self.critic(gs_f[idx]).view(-1), ret_t[idx])
                    loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist.entropy().mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(actor, runner, episodes=5, max_steps=4000, logger=None):
    accum_rewards, successes = [], []
    for ep in range(episodes):
        obs = runner.reset()
        ep_rew = 0.0
        for step in range(max_steps):
            all_actions, _ = runner.get_actions_batched(obs, actor, deterministic=True)
            obs, rewards, dones = runner.step(all_actions)
            ep_rew += sum(rewards[0].values())
            if all(dones[0].values()):
                break
        success = runner.get_success(0)
        accum_rewards.append(ep_rew)
        successes.append(success)
        if logger and ep == 0:
            info = runner.get_debug(0)
            logger.info("[eval] step=%d wp=%s active=%s nwp=%d ok=%s",
                        info['step'], info['wp_indices'], info['active'], info['num_wp'], success)
    return float(np.mean(accum_rewards)), float(np.std(accum_rewards)), float(np.mean(successes))


# =============================================================================
# Main
# =============================================================================

def setup_logger():
    log_dir = os.path.join(project_root, "data", "log")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"exp3_curriculum_{ts}.log")
    logger = logging.getLogger("exp3_curriculum")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger, log_path


def main():
    logger, _ = setup_logger()
    logger.info("Device: %s", device)

    # Map config
    map_size = (200, 200)

    # Generate initial pool (stage 0: empty)
    initial_density = STAGES[0][1]
    initial_n = STAGES[0][2]
    logger.info("Generating initial map pool: density=%.2f n=%d ...", initial_density, initial_n)
    current_pool = _make_pool(map_size, initial_density, initial_n, seed=42)
    logger.info("Map pool ready (%d maps)", len(current_pool))
    wps = current_pool[0]['waypoints']
    path_len = sum(np.linalg.norm(np.array(wps[i+1]) - np.array(wps[i])) for i in range(len(wps) - 1))
    logger.info("Path length: %.0f units, %d waypoints", path_len, len(wps))

    base_config = {
        'num_drones': 1,
        'max_steps': 4000,
        'dt': 0.1,
        'max_speed': 2.0,
        'collision_radius': 0.25,
        'num_dynamic_obs': 0,
        'map_size': (200.0, 200.0),
        'collision_grace_steps': 30,
        'stuck_max_steps': 100,
        'use_map_pool': False,
        'local_grid_size': 7,
        'grid': current_pool[0]['grid'],
        'goal_bonus': 20000,
        'wp_base': 3.0,
        'wp_scale': 1.0,
        'speed_reward_coef': 3.0,
        'death_penalty': -80,
        'stuck_penalty': -20,
        'min_spawn_dist_factor': 10,
    }
    min_steps_needed = int(path_len / (base_config['max_speed'] * base_config['dt']))
    base_config['max_steps'] = max(base_config['max_steps'], int(min_steps_needed * 2.5))
    logger.info("max_steps=%d (path %.0f units, need ~%d steps)", base_config['max_steps'], path_len, min_steps_needed)

    num_envs = int(os.environ.get("EXP_NUM_ENVS", "4"))
    steps_per_env = int(os.environ.get("EXP_STEPS_PER_ENV", "4096"))
    num_episodes = int(os.environ.get("EXP_NUM_EPISODES", "40000"))
    eval_episodes = int(os.environ.get("EXP_EVAL_EPISODES", "5"))
    logger.info("Config: num_envs=%d steps_per_env=%d num_episodes=%d", num_envs, steps_per_env, num_episodes)

    runner = ParallelEnvRunner(base_config, num_envs=num_envs, initial_pool=current_pool)
    agent = MAPPO(
        runner, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2,
        ppo_epochs=20, num_mini_batch=8, value_loss_coef=0.5, entropy_coef=0.03,
        actor_hidden=256, critic_hidden=512,
    )

    try:
        model_dir = os.path.join(project_root, "data", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "exp3_actor_latest.pth")
        best_a = os.path.join(model_dir, "exp3_actor_best.pth")
        best_c = os.path.join(model_dir, "exp3_critic_best.pth")

        best_score = -float("inf")
        current_stage = 0

        logger.info("=" * 60)
        logger.info("EXP3 — Obstacle Density Curriculum")
        for s_idx, (s_ep, s_den, s_n, s_desc) in enumerate(STAGES):
            logger.info("  Stage %d (ep≥%d): density=%.2f n=%d %s", s_idx, s_ep, s_den, s_n, s_desc)
        logger.info("=" * 60)

        for ep in range(num_episodes):
            # Curriculum check
            stage = _stage_for_ep(ep)
            if stage != current_stage:
                _, den, n, desc = STAGES[stage]
                logger.info("=" * 60)
                logger.info("CURRICULUM: Stage %d — density=%.2f %s", stage, den, desc)
                logger.info("Generating new map pool ...")
                new_pool = _make_pool(map_size, den, n, seed=ep)
                logger.info("  %d maps ready, sending to workers ...", len(new_pool))
                runner.set_params({"map_pool": new_pool})
                current_pool = new_pool
                current_stage = stage
                logger.info("  Done.")
                logger.info("=" * 60)

            t0 = time.perf_counter()
            buffer = agent.collect_experience(steps_per_env)
            ct = time.perf_counter() - t0
            t0 = time.perf_counter()
            agent.update(buffer)
            ut = time.perf_counter() - t0

            if ep % 10 == 0:
                logger.info("Ep %d | coll %.1fs upd %.1fs | stage %d", ep, ct, ut, current_stage)

            if ep % 50 == 0:
                evm, evs, evsr = evaluate(agent.actor, runner, episodes=eval_episodes, logger=logger)
                logger.info("Ep %d | eval %.0f ± %.0f sr %.2f | stage %d", ep, evm, evs, evsr, current_stage)
                torch.save(agent.actor.state_dict(), model_path)
                sc = evm + 5000.0 * evsr
                if sc > best_score:
                    best_score = sc
                    torch.save(agent.actor.state_dict(), best_a)
                    torch.save(agent.critic.state_dict(), best_c)
                    logger.info("  => new best")
    finally:
        runner.close()
    logger.info("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
