import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from typing import List, Tuple, Dict

class MultiDroneEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.num_drones = config.get('num_drones', 2)
        self.max_steps = config.get('max_steps', 300)
        self.dt = config.get('dt', 0.1)
        self.max_speed = config.get('max_speed', 2.0)
        self.collision_radius = config.get('collision_radius', 0.25)
        self.goal_tolerance = config.get('goal_tolerance', 0.5)
        
        self.num_dynamic_obs = config.get('num_dynamic_obs', 0)  # 先设为0
        self.dynamic_obs_speed = config.get('dynamic_obs_speed', 1.0)
        
        self.w_track = config.get('w_track', 1.0)
        self.w_formation = config.get('w_formation', 0.3)
        self.w_collision = config.get('w_collision', 2.0)
        self.death_penalty = config.get('death_penalty', -35.0)
        self.collision_alert_radius_factor = config.get("collision_alert_radius_factor", 4.0)
        self.obstacle_alert_coef = config.get("obstacle_alert_coef", 1.5)
        self.stuck_max_steps = config.get("stuck_max_steps", 35)
        self.stuck_progress_eps = config.get("stuck_progress_eps", 0.015)
        self.stuck_speed_eps = config.get("stuck_speed_eps", 0.08)
        self.stuck_penalty = config.get("stuck_penalty", -45.0)
        
        self.map_size = config.get('map_size', (20.0, 20.0))
        
        # 生成路径
        self.waypoints = self._generate_smooth_path()
        
        # 每个动态障碍增加相对位置(2) + 相对速度(2)
        self.obs_dim = 4 + 2 + 2*(self.num_drones-1) + 4*self.num_dynamic_obs
        self.action_dim = 2
        
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
        
        self.agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.possible_agents = self.agents[:]
        
        #self.waypoints = []                     # 明确为 list
        self.waypoint_indices = np.array([], dtype=int)            # 或者 np.array([])
        self.prev_dist_to_target = []
        self.drone_positions = np.empty((0,2))
        self.drone_velocities = np.empty((0,2))
        self.drone_active = np.array([], dtype=bool)
        self.dynamic_obstacles = []  
        self.collision_grace_steps = config.get("collision_grace_steps", 8)
        self.min_spawn_dist_factor = config.get("min_spawn_dist_factor", 5.0)
        self.min_obs_start_dist_factor = config.get("min_obs_start_dist_factor", 8.0)
        self.stuck_counters = np.array([], dtype=np.int32)
              
    def _generate_smooth_path(self) -> List[Tuple[float, float]]:
        """生成一条从起点到终点的直线路径（保证非空）"""
        start = np.array([1.0, 1.0])
        goal = np.array([self.map_size[0] - 1.0, self.map_size[1] - 1.0])
    
        # 直接生成20个直线路径点
        path = []
        for t in np.linspace(0, 1, 20):
            x = start[0] * (1 - t) + goal[0] * t
            y = start[1] * (1 - t) + goal[1] * t
            path.append((x, y))
    
        # 确保路径非空（如果因为某种原因还是空，手动添加起点和终点）
        if len(path) == 0:
            path = [(1.0, 1.0), (self.map_size[0]-1.0, self.map_size[1]-1.0)]
    
        # 可选：打印路径长度以确认
        print(f"Generated path with {len(path)} waypoints")
        return path
    
    def _interpolate_path(self, path, step_dist):
        if len(path) < 2:
            return path
        new_path = [path[0]]
        for i in range(1, len(path)):
            p1 = np.array(path[i-1])
            p2 = np.array(path[i])
            dist = np.linalg.norm(p2-p1)
            if dist <= step_dist:
                new_path.append(path[i])
            else:
                num = int(np.ceil(dist / step_dist))
                for k in range(1, num):
                    t = k/num
                    interp = p1 + t*(p2-p1)
                    new_path.append(tuple(interp))
                new_path.append(path[i])
        return new_path
    
    def reset(self):
        start_point = self.waypoints[0]
        self.drone_positions = []
        # 避免初始碰撞：确保每架无人机之间距离至少 1.5 倍半径
        for i in range(self.num_drones):
            min_sep = self.min_spawn_dist_factor * self.collision_radius
            spawn_range = max(0.6, min_sep * 1.2)
            placed = False
            for _ in range(300):
                offset = np.random.uniform(-spawn_range, spawn_range, size=2)
                pos = np.array(start_point) + offset
                pos = np.clip(pos, 0, self.map_size[0])
                if i == 0:
                    placed = True
                    break
                # 检查与已有无人机的距离
                collision = False
                for j in range(i):
                    if np.linalg.norm(pos - self.drone_positions[j]) < min_sep:
                        collision = True
                        break
                if not collision:
                    placed = True
                    break
            if not placed:
                # 回退：按圆环分布强制给初始位置，避免死循环
                angle = 2 * np.pi * i / max(1, self.num_drones)
                pos = np.array(start_point) + min_sep * np.array([np.cos(angle), np.sin(angle)])
                pos = np.clip(pos, 0, self.map_size[0])
            self.drone_positions.append(pos)
        self.drone_positions = np.array(self.drone_positions, dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.drone_active = np.ones(self.num_drones, dtype=bool)
        self.stuck_counters = np.zeros(self.num_drones, dtype=np.int32)
        self.waypoint_indices = np.ones(self.num_drones, dtype=int)
        # 记录每个无人机到当前航点的距离（用于计算接近奖励）
        self.prev_dist_to_target = []
        for i in range(self.num_drones):
            wp_idx = self.waypoint_indices[i]
            if wp_idx < len(self.waypoints):
                dist = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
            else:
                dist = 0.0
            self.prev_dist_to_target.append(dist)
        
        self.dynamic_obstacles = []
        for _ in range(self.num_dynamic_obs):
            for _ in range(200):
                pos = np.random.uniform(0, self.map_size[0], size=2)
                near_start = np.linalg.norm(pos - np.array(start_point)) < self.min_obs_start_dist_factor * self.collision_radius
                near_drone = False
                for dpos in self.drone_positions:
                    if np.linalg.norm(pos - dpos) < self.min_obs_start_dist_factor * self.collision_radius:
                        near_drone = True
                        break
                if not near_start and not near_drone:
                    break
            angle = np.random.uniform(0, 2*np.pi)
            vel = self.dynamic_obs_speed * np.array([np.cos(angle), np.sin(angle)])
            self.dynamic_obstacles.append({'pos': pos, 'vel': vel})
        
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            if not self.drone_active[i]:
                obs_dict[agent] = np.zeros(self.obs_dim, dtype=np.float32)
                continue
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            obs = [pos[0], pos[1], vel[0], vel[1]]
            wp_idx = self.waypoint_indices[i]
            if wp_idx < len(self.waypoints):
                target = np.array(self.waypoints[wp_idx])
                rel_target = target - pos
                obs.extend(rel_target)
            else:
                obs.extend([0.0, 0.0])
            others = []
            for j in range(self.num_drones):
                if j == i:
                    continue
                rel = self.drone_positions[j] - pos
                others.extend(rel)
            max_others = (self.num_drones - 1) * 2
            if len(others) < max_others:
                others.extend([0.0] * (max_others - len(others)))
            obs.extend(others[:max_others])
            obs_rel_obs = []
            for ob in self.dynamic_obstacles:
                rel = ob['pos'] - pos
                rel_v = ob['vel'] - vel
                obs_rel_obs.extend(rel)
                obs_rel_obs.extend(rel_v)
            max_obs = self.num_dynamic_obs * 4
            if len(obs_rel_obs) < max_obs:
                obs_rel_obs.extend([0.0] * (max_obs - len(obs_rel_obs)))
            obs.extend(obs_rel_obs[:max_obs])
            obs_dict[agent] = np.array(obs, dtype=np.float32)
        return obs_dict
    
    def step(self, actions: Dict[str, np.ndarray]):
        action_matrix = np.array([actions[agent] for agent in self.agents], dtype=np.float32)
        action_matrix = np.clip(action_matrix, -self.max_speed, self.max_speed)
        new_positions = self.drone_positions + action_matrix * self.dt
        new_positions = np.clip(new_positions, 0, self.map_size[0])
        self.drone_velocities = action_matrix
        self.drone_positions = new_positions
        
        # 更新动态障碍物
        for obs in self.dynamic_obstacles:
            obs['pos'] += obs['vel'] * self.dt
            for dim in range(2):
                if obs['pos'][dim] < 0 or obs['pos'][dim] > self.map_size[0]:
                    obs['vel'][dim] *= -1
                    obs['pos'][dim] = np.clip(obs['pos'][dim], 0, self.map_size[0])
        
        # 记录旧航点索引和距离
        old_waypoint_indices = self.waypoint_indices.copy()
        old_dist_to_target = self.prev_dist_to_target.copy()
        
        # 更新航点索引
        for i in range(self.num_drones):
            if not self.drone_active[i]:
                continue
            wp_idx = self.waypoint_indices[i]
            if wp_idx < len(self.waypoints) - 1:
                dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
                if dist_to_wp < self.goal_tolerance:
                    self.waypoint_indices[i] = wp_idx + 1
            elif wp_idx == len(self.waypoints) - 1:
                # 最后一个航点到达后，将索引推进到 len(waypoints)，触发终点奖励与完成判定
                dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
                if dist_to_wp < self.goal_tolerance:
                    self.waypoint_indices[i] = len(self.waypoints)
            # 更新当前距离
            if self.waypoint_indices[i] < len(self.waypoints):
                self.prev_dist_to_target[i] = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[self.waypoint_indices[i]]))
            else:
                self.prev_dist_to_target[i] = 0.0
        
        prev_active = self.drone_active.copy()
        # 碰撞检测（前若干步启用保护，避免 reset 后立刻全灭）
        collision_occurred = np.zeros(self.num_drones, dtype=bool)
        if self.current_step >= self.collision_grace_steps:
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for j in range(i+1, self.num_drones):
                    if not self.drone_active[j]:
                        continue
                    if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        collision_occurred[j] = True
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for obs in self.dynamic_obstacles:
                    if np.linalg.norm(self.drone_positions[i] - obs['pos']) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        break
        self.drone_active = np.logical_and(self.drone_active, ~collision_occurred)
        newly_inactive_collision = np.logical_and(prev_active, collision_occurred)
        newly_inactive_stuck = np.zeros(self.num_drones, dtype=bool)
        
        # 计算奖励
        rewards = {}
        for i, agent in enumerate(self.agents):
            if not self.drone_active[i]:
                # 一次性死亡惩罚，避免失活后每步持续扣分掩盖其他智能体表现
                if newly_inactive_collision[i]:
                    rewards[agent] = self.death_penalty
                elif newly_inactive_stuck[i]:
                    rewards[agent] = self.stuck_penalty
                else:
                    rewards[agent] = 0.0
                continue
            
            reward = 0.0
            
            # 1. 路径跟踪奖励：距离减少量（势能奖励），但要限制范围
            if self.waypoint_indices[i] < len(self.waypoints):
                current_dist = self.prev_dist_to_target[i]
                prev_dist = old_dist_to_target[i]
                # 只有航点索引未变时，才计算距离变化
                if self.waypoint_indices[i] == old_waypoint_indices[i]:
                    delta_dist = prev_dist - current_dist
                    delta_dist = np.clip(delta_dist, -2.0, 2.0)   # 限制单步变化最大2米
                    reward += delta_dist * self.w_track * 2.0
                    speed = float(np.linalg.norm(self.drone_velocities[i]))
                    if delta_dist < self.stuck_progress_eps and speed < self.stuck_speed_eps:
                        self.stuck_counters[i] += 1
                    else:
                        self.stuck_counters[i] = 0
                # 如果到达新航点，直接给固定正奖励（你已经有了+5）
                if self.waypoint_indices[i] > old_waypoint_indices[i]:
                    reward += 5.0
                    self.stuck_counters[i] = 0
            
            # 2. 编队奖励：与其他无人机的平均距离与期望距离的偏差
            if self.num_drones > 1:
                total_dist = 0.0
                count = 0
                for j in range(self.num_drones):
                    if j != i and self.drone_active[j]:
                        d = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                        total_dist += d
                        count += 1
                if count > 0:
                    mean_dist = total_dist / count
                    # 期望距离为 1.0 到 1.5 之间
                    desired_dist = 1.2
                    reward -= abs(mean_dist - desired_dist) * self.w_formation
            
            # 3. 密集避障奖励：根据最小距离给予平滑惩罚
            min_dist = float('inf')
            min_obs_dist = float('inf')
            for j in range(self.num_drones):
                if j != i and self.drone_active[j]:
                    d = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                    min_dist = min(min_dist, d)
            for obs in self.dynamic_obstacles:
                d = np.linalg.norm(self.drone_positions[i] - obs['pos'])
                min_dist = min(min_dist, d)
                min_obs_dist = min(min_obs_dist, d)
            alert_radius = self.collision_alert_radius_factor * self.collision_radius
            if min_dist < alert_radius:
                penalty = -self.w_collision * (1.0 - min_dist / alert_radius)
                reward += penalty
            if min_obs_dist < alert_radius:
                # 动态障碍使用额外提前惩罚，鼓励提前避让而不是临撞规避
                obs_penalty = -self.w_collision * self.obstacle_alert_coef * (1.0 - min_obs_dist / alert_radius)
                reward += obs_penalty
            
            # 4. 到达终点奖励
            if self.waypoint_indices[i] >= len(self.waypoints):
                reward += 20.0
                self.drone_active[i] = False
                self.stuck_counters[i] = 0

            if self.stuck_counters[i] >= self.stuck_max_steps:
                self.drone_active[i] = False
                newly_inactive_stuck[i] = True
                reward += self.stuck_penalty
                self.stuck_counters[i] = 0
            
            rewards[agent] = reward
        
        self.current_step += 1
        done_all = (not np.any(self.drone_active)) or (self.current_step >= self.max_steps)
        dones = {agent: (not self.drone_active[i]) for i, agent in enumerate(self.agents)}
        dones["__all__"] = done_all
        
        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, dones, infos
    
    def render(self, mode='human'):
        pass