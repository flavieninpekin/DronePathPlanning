import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time
import random
import logging
import multiprocessing as mp
from contextlib import nullcontext
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    try:
        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler
    except ImportError:
        autocast = None
        GradScaler = None

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from neural_controller.env.drone_env_v2 import MultiDroneEnvV2
from actor_critic import Actor, Critic, try_load_actor_checkpoint
from map_generator.MetropolisGenerator import generate as generate_metropolis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    log_dir = os.path.join(project_root, "data", "log")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"exp1_single_drone_{ts}.log")
    logger = logging.getLogger("exp1_single_drone")
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
    if not enabled or autocast is None:
        return nullcontext()
    return autocast('cuda', enabled=True)


def _env_worker_v8(env_config, conn, seed):
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


class ParallelEnvRunnerV8:
    def __init__(self, env_config, num_envs, single_map):
        self.num_envs = num_envs
        self.num_agents = env_config['num_drones']
        self.obs_dim = 0
        self.action_dim = 2
        self.parent_conns = []
        self.workers = []
        for i in range(num_envs):
            parent, child = mp.Pipe()
            p = mp.Process(target=_env_worker_v8, args=(env_config, child, 42 + i), daemon=True)
            p.start()
            self.parent_conns.append(parent)
            self.workers.append(p)
            child.close()
            parent.send(single_map)
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
            with _amp_autocast(True):
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


class MAPPO_V8:
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
        self.scaler = GradScaler('cuda') if self._use_amp else None
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
            with _amp_autocast(self._use_amp):
                v = self.critic(gs_f).squeeze(-1)
            vals = v.view(steps, -1)
            advs = torch.zeros_like(rew_buf.view(steps, -1))
            gae = torch.zeros(num_envs * self.num_agents, device=self.device)
            nv = torch.zeros(num_envs * self.num_agents, device=self.device)
            for t in reversed(range(steps)):
                nt = 1.0 - don_buf.view(steps, -1)[t]
                delta = rew_buf.view(steps, -1)[t] + self.gamma * nv * nt - vals[t]
                gae = delta + self.gamma * self.gae_lambda * nt * gae
                advs[t] = gae
                nv = vals[t]
            rets = advs + vals
            rets_f = rets.view(-1)
            advs_f = advs.view(-1)
            advs_f = (advs_f - advs_f.mean()) / (advs_f.std() + 1e-8)
        chunk = total // self.num_mini_batch
        for _ in range(self.ppo_epochs):
            idxs = torch.randperm(total, device=self.device)
            for start in range(0, total, chunk):
                bidx = idxs[start: start + chunk]
                ob, ac = obs_f[bidx], act_f[bidx]
                olp, ab, rb = old_lp_f[bidx], advs_f[bidx], rets_f[bidx]
                gb = gs_f[bidx]
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                with _amp_autocast(self._use_amp):
                    m, s = self.actor(ob)
                    d = torch.distributions.Normal(m, s)
                    nlp = d.log_prob(ac).sum(dim=-1)
                    ent = d.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(nlp - olp)
                    s1 = ratio * ab
                    s2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * ab
                    al = -torch.min(s1, s2).mean()
                    nv = self.critic(gb).squeeze(-1)
                    vl = torch.nn.functional.mse_loss(nv, rb.float())
                    loss = al + self.value_loss_coef * vl - self.entropy_coef * ent
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()


def evaluate(agent, runner, episodes=5, max_steps=4000, logger=None):
    returns, success_count = [], 0
    for ep in range(episodes):
        all_obs = runner.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            all_actions, _ = runner.get_actions_batched(all_obs, agent, deterministic=True)
            all_obs, rewards, dones = runner.step(all_actions)
            total_reward += sum(rewards[0].values())
            if dones[0]["__all__"]:
                break
        succ = runner.get_success(0)
        if succ:
            success_count += 1
        dbg = runner.get_debug(0)
        if logger and ep == 0:
            logger.info("[eval] step=%d wp=%s active=%s nwp=%d ok=%s",
                        dbg["step"], dbg["wp_indices"], dbg["active"], dbg["num_wp"], succ)
        returns.append(total_reward)
    arr = np.array(returns, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), success_count / max(1, episodes)


def main():
    logger, _ = setup_logger()
    logger.info("Device: %s", device)

    grid, wps = generate_metropolis((200, 200))
    wp_arr = np.array(wps)
    path_len = sum(np.linalg.norm(wp_arr[i+1] - wp_arr[i]) for i in range(len(wp_arr) - 1))
    logger.info("Metropolis: %s free=%.0f%% wp=%d path=%.0f steps~%.0f",
                str(grid.shape), 100*(grid==0).sum()/grid.size, len(wps), path_len, path_len/0.2)
    single_map = [{"grid": grid, "waypoints": wps}]

    base_config = {
        'num_drones': 1,
        'max_steps': 4000,
        'dt': 0.1,
        'max_speed': 2.0,
        'collision_radius': 0.25,
        'num_dynamic_obs': 0,
        'w_track': 1.0,
        'w_formation': 0.0,
        'w_collision': 0.0,
        'death_penalty': -80.0,
        'stuck_max_steps': 100,
        'stuck_progress_eps': 0.005,
        'stuck_speed_eps': 0.02,
        'stuck_penalty': -20.0,
        'collision_grace_steps': 30,
        'min_spawn_dist_factor': 10.0,
        'map_size': (200.0, 200.0),
        'use_jps_rrt': False,
        'local_grid_size': 7,
        'use_map_pool': False,
        'formation_sight_range': 6.0,
        'w_formation_a': 0,
        'w_formation_b': 0,
        'grid': grid,
        'goal_bonus': 20000.0,
        'wp_base': 3.0,
        'wp_scale': 1.0,
        'speed_reward_coef': 3.0,
    }

    num_envs = 4
    steps_per_env = 4096

    runner = ParallelEnvRunnerV8(base_config, num_envs=num_envs, single_map=single_map)
    agent = MAPPO_V8(
        runner, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2,
        ppo_epochs=20, num_mini_batch=8, value_loss_coef=0.5, entropy_coef=0.03,
        actor_hidden=256, critic_hidden=512,
    )

    try:
        writer = SummaryWriter("runs/exp1_single_drone")
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "exp1_actor_latest.pth")
        best_a = os.path.join(model_dir, "exp1_actor_best.pth")
        best_c = os.path.join(model_dir, "exp1_critic_best.pth")

        num_episodes = 10000
        eval_episodes = 5
        best_score = -float("inf")

        logger.info("=" * 60)
        logger.info("EXP1 — Single Drone (num_drones=1)")
        logger.info("  goal_bonus=%.0f wp_base=%.1f wp_scale=%.1f speed_coef=%.1f",
                    base_config['goal_bonus'], base_config['wp_base'],
                    base_config['wp_scale'], base_config['speed_reward_coef'])
        logger.info("  grace=%d death=%d stuck_max=%d",
                    base_config['collision_grace_steps'],
                    base_config['death_penalty'],
                    base_config['stuck_max_steps'])
        logger.info("  path=%.0f units ~%.0f steps", path_len, path_len/0.2)
        logger.info("=" * 60)

        for ep in range(num_episodes):
            t0 = time.perf_counter()
            buffer = agent.collect_experience(steps_per_env)
            ct = time.perf_counter() - t0
            t0 = time.perf_counter()
            agent.update(buffer)
            ut = time.perf_counter() - t0
            writer.add_scalar("train/collect_s", ct, ep)
            writer.add_scalar("train/update_s", ut, ep)

            if ep % 10 == 0:
                logger.info("Ep %d | coll %.1fs upd %.1fs", ep, ct, ut)

            if ep % 50 == 0:
                evm, evs, evsr = evaluate(agent.actor, runner, episodes=eval_episodes, logger=logger)
                writer.add_scalar("eval/mean", evm, ep)
                writer.add_scalar("eval/sr", evsr, ep)
                logger.info("Ep %d | eval %.0f +- %.0f sr %.2f", ep, evm, evs, evsr)
                torch.save(agent.actor.state_dict(), model_path)
                sc = evm + 5000.0 * evsr
                if sc > best_score:
                    best_score = sc
                    torch.save(agent.actor.state_dict(), best_a)
                    torch.save(agent.critic.state_dict(), best_c)
                    logger.info("  => new best")
        writer.close()
    finally:
        runner.close()
    logger.info("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
