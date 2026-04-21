import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from map_generator import MapGenerator

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
        
        self.num_dynamic_obs = config.get('num_dynamic_obs', 0)
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
        
        # ========== 地图与路径配置 ==========
        self.use_jps_rrt = config.get('use_jps_rrt', False)
        self.obstacle_density = config.get('obstacle_density', 0.1)
        self.local_grid_size = config.get('local_grid_size', 5)
        self.regenerate_map = config.get('regenerate_map', False)
        self.map_generation_attempts = config.get('map_generation_attempts', 10)
        
        # 地图池配置
        self.use_map_pool = config.get('use_map_pool', False)
        self.map_pool_size = config.get('map_pool_size', 100)
        self.map_pool = []  # 每个元素: {'grid': np.ndarray, 'waypoints': List[Tuple[float, float]]}
        
        # 计算观测维度
        self.obs_dim = (4 + 2 + 2*(self.num_drones-1) + 4*self.num_dynamic_obs 
                        + self.local_grid_size * self.local_grid_size)
        self.action_dim = 2
        
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
        
        self.agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.possible_agents = self.agents[:]
        
        # 初始化地图和路径
        if self.use_map_pool:
            self._build_map_pool()
            # 从池中随机选一张作为初始地图
            self._sample_from_pool()
        else:
            self.grid = config.get('grid', None)
            self._initialize_single_map()
        
        # 其他状态变量
        self.waypoint_indices = np.array([], dtype=int)
        self.prev_dist_to_target = []
        self.drone_positions = np.empty((0,2))
        self.drone_velocities = np.empty((0,2))
        self.drone_active = np.array([], dtype=bool)
        self.dynamic_obstacles = []  
        self.collision_grace_steps = config.get("collision_grace_steps", 8)
        self.min_spawn_dist_factor = config.get("min_spawn_dist_factor", 5.0)
        self.min_obs_start_dist_factor = config.get("min_obs_start_dist_factor", 8.0)
        self.stuck_counters = np.array([], dtype=np.int32)

    # ========== 地图池构建 ==========
    def _build_map_pool(self):
        """预生成地图池，只保留成功生成 JPS-RRT 路径的地图"""
        print(f"Building map pool of size {self.map_pool_size} (valid JPS-RRT maps only)...")
        
        valid_count = 0
        total_attempts = 0
        max_total_attempts = self.map_pool_size * self.map_generation_attempts * 5
        
        while valid_count < self.map_pool_size and total_attempts < max_total_attempts:
            total_attempts += 1
            
            try:
                # 生成栅格地图
                grid = MapGenerator.generate_map_with_path(
                    size=tuple(int(s) for s in self.map_size) if isinstance(self.map_size, (list, tuple)) else (int(self.map_size),)*2,
                    obstacle_density=self.obstacle_density,
                    dim=2,
                    channel_expansion=2
                )
                
                # 尝试生成 JPS-RRT 路径（直接传 grid，不碰 self.grid）
                if self.use_jps_rrt:
                    waypoints = self._generate_jps_rrt_path(grid=grid)
                    
                    # 检查是否为有效路径（非回退直线）
                    # 直线路径恰好 20 个点，JPS-RRT 路径通常远大于 20
                    if len(waypoints) <= 20:
                        print(f"  Attempt {total_attempts}: JPS-RRT failed (fallback straight line, {len(waypoints)} pts), discarding.")
                        continue
                else:
                    waypoints = self._generate_smooth_path()
                
                # 成功：加入地图池
                self.map_pool.append({'grid': grid, 'waypoints': waypoints})
                valid_count += 1
                
                if valid_count % 10 == 0 or valid_count == self.map_pool_size:
                    print(f"  Progress: {valid_count}/{self.map_pool_size} valid maps (attempted {total_attempts})")
                    
            except Exception as e:
                print(f"  Attempt {total_attempts} error: {e}")
                continue
        
        if valid_count < self.map_pool_size:
            raise RuntimeError(
                f"Could only generate {valid_count}/{self.map_pool_size} valid maps "
                f"after {total_attempts} total attempts."
            )
        
        print(f"Map pool built successfully with {len(self.map_pool)} valid maps.")

    def _sample_from_pool(self):
        """从地图池中随机选取一张地图和路径"""
        selected = random.choice(self.map_pool)
        self.grid = selected['grid']
        self.waypoints = selected['waypoints']
    
    # ========== 单张地图初始化（非池模式） ==========
    def _initialize_single_map(self):
        """生成单张地图和路径（用于非池模式）"""
        if self.grid is None:
            from map_generator import MapGenerator
            self.grid = MapGenerator.generate_map_with_path(
                size=tuple(int(s) for s in self.map_size) if isinstance(self.map_size, (list, tuple)) else (int(self.map_size),)*2,
                obstacle_density=self.obstacle_density,
                dim=2,
                channel_expansion=2
            )
        else:
            self.grid = np.asarray(self.grid, dtype=np.uint8)
        
        if self.use_jps_rrt:
            self.waypoints = self._generate_jps_rrt_path(grid=self.grid)
        else:
            self.waypoints = self._generate_smooth_path()
    
    def _initialize_map(self):
        """生成新地图和路径，带重试机制"""
        for attempt in range(self.map_generation_attempts):
            try:
                # 生成栅格地图
                from map_generator import MapGenerator
                self.grid = MapGenerator.generate_map_with_path(
                    size=tuple(int(s) for s in self.map_size) if isinstance(self.map_size, (list, tuple)) else (int(self.map_size),)*2,
                    obstacle_density=self.obstacle_density,
                    dim=2,
                    channel_expansion=2  # 新增：通道膨胀宽度
                )
                # 生成全局路径
                if self.use_jps_rrt:
                    self.waypoints = self._generate_jps_rrt_path()
                else:
                    self.waypoints = self._generate_smooth_path()
                return  # 成功生成，退出
            except Exception as e:
                print(f"Map generation attempt {attempt+1} failed: {e}")
        raise RuntimeError("Failed to generate a valid map after multiple attempts.")
    
    # ========== 新增：JPS-RRT 路径生成 ==========
    def _generate_jps_rrt_path(self, grid: Optional[np.ndarray] = None, silent: bool = False) -> List[Tuple[float, float]]:
        """生成 JPS-RRT 路径，可接受外部传入的 grid，完全不依赖 self.grid"""
        if grid is None:
            grid = self.grid
        if grid is None:
            if not silent:
                print("Warning: No grid provided, falling back to straight line.")
            return self._generate_smooth_path()
        
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) == 0:
            if not silent:
                print("Warning: No free cells in grid, falling back to straight line.")
            return self._generate_smooth_path()
        
        start_coord = np.array([0, 0])
        distances = np.linalg.norm(free_cells - start_coord, axis=1)
        start = tuple(free_cells[np.argmin(distances)].tolist())
        
        goal_coord = np.array([self.map_size[0] - 1, self.map_size[1] - 1])
        distances = np.linalg.norm(free_cells - goal_coord, axis=1)
        goal = tuple(free_cells[np.argmin(distances)].tolist())
        
        if not silent:
            print(f"Adjusted start: {start}, goal: {goal}")
        
        try:
            from algo_combinations.jps_rrt import run_jps_rrt_pipeline
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from algo_combinations.jps_rrt import run_jps_rrt_pipeline
        
        path = run_jps_rrt_pipeline(
            grid=grid,
            size=grid.shape,
            density=float(grid.mean()),
            threshold=0.3,
            ratio=4,
            start=start,
            goal=goal,
            step_size=1.0,
            goal_tolerance=0.5,
            max_iter=5000,
            bias_prob=0.9
        )
        
        if path is None:
            if not silent:
                print("Warning: JPS-RRT failed to find a path. Falling back to straight line.")
            return self._generate_smooth_path()
        
        # 插值时传入 grid，不访问 self.grid
        interpolated = self._interpolate_path_safe(path, step_dist=0.5, grid=grid)
        
        if not silent:
            print(f"JPS-RRT path generated with {len(interpolated)} waypoints.")
        return interpolated
    
    def _is_in_obstacle(self, pos: np.ndarray, grid: Optional[np.ndarray] = None) -> bool:
        """检查连续坐标是否落在栅格障碍物内"""
        if grid is None:
            grid = self.grid
        if grid is None:
            return False  # 如果连 self.grid 也没有，默认安全（但这种情况不应该发生）
        x, y = pos
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]:
            return bool(grid[ix, iy] == 1)
        return True  # 超出边界视为障碍物
    # ========================================
    
    def _generate_smooth_path(self) -> List[Tuple[float, float]]:
        """生成一条从起点到终点的直线路径（保证非空）"""
        start = np.array([1.0, 1.0])
        goal = np.array([self.map_size[0] - 1.0, self.map_size[1] - 1.0])
        path = []
        for t in np.linspace(0, 1, 20):
            x = start[0] * (1 - t) + goal[0] * t
            y = start[1] * (1 - t) + goal[1] * t
            path.append((x, y))
        if len(path) == 0:
            path = [(1.0, 1.0), (self.map_size[0]-1.0, self.map_size[1]-1.0)]
        print(f"Generated straight path with {len(path)} waypoints")
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
    
    def _interpolate_path_safe(self, path, step_dist, grid=None):
        """带碰撞检测的路径插值"""
        if len(path) < 2:
            return path
        new_path = [path[0]]
        for i in range(1, len(path)):
            p1 = np.array(path[i-1])
            p2 = np.array(path[i])
            dist = np.linalg.norm(p2 - p1)
            num = int(np.ceil(dist / step_dist))
            for k in range(1, num + 1):
                t = k / num
                interp = p1 + t * (p2 - p1)
                if not self._is_in_obstacle(interp, grid=grid):
                    new_path.append(tuple(interp))
                else:
                    for _ in range(5):
                        offset = np.random.uniform(-0.5, 0.5, size=2)
                        cand = interp + offset
                        if not self._is_in_obstacle(cand, grid=grid):
                            new_path.append(tuple(cand))
                            break
        return new_path
    
    def reset(self):
        # 地图切换逻辑
        if self.use_map_pool:
            self._sample_from_pool()
        elif self.regenerate_map:
            self._initialize_single_map()
        # 否则保持原有地图不变
        
        start_point = self.waypoints[0]
        self.drone_positions = []
        for i in range(self.num_drones):
            min_sep = self.min_spawn_dist_factor * self.collision_radius
            spawn_range = max(0.6, min_sep * 1.2)
            placed = False
            for _ in range(300):
                offset = np.random.uniform(-spawn_range, spawn_range, size=2)
                pos = np.array(start_point) + offset
                pos = np.clip(pos, 0, self.map_size[0])
                if self._is_in_obstacle(pos):
                    continue
                if i == 0:
                    placed = True
                    break
                collision = False
                for j in range(i):
                    if np.linalg.norm(pos - self.drone_positions[j]) < min_sep:
                        collision = True
                        break
                if not collision:
                    placed = True
                    break
            if not placed:
                angle = 2 * np.pi * i / max(1, self.num_drones)
                pos = np.array(start_point) + min_sep * np.array([np.cos(angle), np.sin(angle)])
                pos = np.clip(pos, 0, self.map_size[0])
                for _ in range(10):
                    if not self._is_in_obstacle(pos):
                        break
                    pos = np.clip(pos + np.random.uniform(-0.5, 0.5, size=2), 0, self.map_size[0])
            self.drone_positions.append(pos)
        self.drone_positions = np.array(self.drone_positions, dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.drone_active = np.ones(self.num_drones, dtype=bool)
        self.stuck_counters = np.zeros(self.num_drones, dtype=np.int32)
        self.waypoint_indices = np.ones(self.num_drones, dtype=int)
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
                if self._is_in_obstacle(pos):
                    continue
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
            # ========== 新增：局部栅格感知 ==========
            local_grid = self._get_local_grid(self.drone_positions[i])
            obs.extend(local_grid)
        # ======================================
            obs_dict[agent] = np.array(obs, dtype=np.float32)
        return obs_dict
    
    def _get_local_grid(self, pos: np.ndarray) -> np.ndarray:
        """提取以无人机为中心的 local_grid_size x local_grid_size 局部栅格，展平返回。"""
        half = self.local_grid_size // 2
        x, y = int(np.floor(pos[0])), int(np.floor(pos[1]))
        local = np.zeros((self.local_grid_size, self.local_grid_size), dtype=np.float32)
    
        for i in range(self.local_grid_size):
            for j in range(self.local_grid_size):
                xi = x - half + i
                yj = y - half + j
                if 0 <= xi < self.grid.shape[0] and 0 <= yj < self.grid.shape[1]: # type: ignore
                    local[i, j] = float(self.grid[xi, yj]) # type: ignore
                else:
                    local[i, j] = 1.0  # 边界外视为障碍物
        return local.flatten()
    
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
                dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
                if dist_to_wp < self.goal_tolerance:
                    self.waypoint_indices[i] = len(self.waypoints)
            # 更新当前距离
            if self.waypoint_indices[i] < len(self.waypoints):
                self.prev_dist_to_target[i] = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[self.waypoint_indices[i]]))
            else:
                self.prev_dist_to_target[i] = 0.0
        
        prev_active = self.drone_active.copy()
        collision_occurred = np.zeros(self.num_drones, dtype=bool)
        
        if self.current_step >= self.collision_grace_steps:
            # 无人机间碰撞
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for j in range(i+1, self.num_drones):
                    if not self.drone_active[j]:
                        continue
                    if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        collision_occurred[j] = True
            # 与动态障碍物碰撞
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for obs in self.dynamic_obstacles:
                    if np.linalg.norm(self.drone_positions[i] - obs['pos']) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        break
            # ========== 新增：与静态障碍物碰撞检测 ==========
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                if self._is_in_obstacle(self.drone_positions[i]):
                    collision_occurred[i] = True
            # ============================================
        
        self.drone_active = np.logical_and(self.drone_active, ~collision_occurred)
        newly_inactive_collision = np.logical_and(prev_active, collision_occurred)
        newly_inactive_stuck = np.zeros(self.num_drones, dtype=bool)
        
        # 计算奖励
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
            
            # 1. 路径跟踪奖励
            if self.waypoint_indices[i] < len(self.waypoints):
                current_dist = self.prev_dist_to_target[i]
                prev_dist = old_dist_to_target[i]
                if self.waypoint_indices[i] == old_waypoint_indices[i]:
                    delta_dist = prev_dist - current_dist
                    delta_dist = np.clip(delta_dist, -2.0, 2.0)
                    reward += delta_dist * self.w_track * 2.0
                    speed = float(np.linalg.norm(self.drone_velocities[i]))
                    if delta_dist < self.stuck_progress_eps and speed < self.stuck_speed_eps:
                        self.stuck_counters[i] += 1
                    else:
                        self.stuck_counters[i] = 0
                if self.waypoint_indices[i] > old_waypoint_indices[i]:
                    reward += 5.0
                    self.stuck_counters[i] = 0
            
            # 2. 编队奖励
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
                    desired_dist = 1.2
                    reward -= abs(mean_dist - desired_dist) * self.w_formation
            
            # 3. 避障奖励
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