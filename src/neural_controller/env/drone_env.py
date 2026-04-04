import gym
from gym import spaces
import numpy as np
import math

class DroneEnv(gym.Env):
    """
    最小化无人机避障环境（简化版）
    - 二维连续空间，无人机视为质点
    - 静态圆形障碍物（动态障碍物后续添加）
    - 全局路径是一个点列（从 RRT 得到）
    - 观测：无人机位置 + 目标位置 + 最近障碍物距离
    """
    def __init__(self, config, global_path):
        super().__init__()
        self.config = config
        self.global_path = global_path          # 全局路径点 [(x0,y0), (x1,y1), ...]
        self.target = global_path[-1]           # 最终目标
        self.goal_threshold = config['rl']['env'].get('goal_threshold', 0.5)
        self.max_steps = config['rl']['env'].get('max_steps_per_episode', 500)
        self.dt = config['simulation']['time_step']
        self.max_speed = config['rl']['action']['max_speed']
        
        # 障碍物列表（静态圆形，可扩展为动态）
        self.obstacles = [
            {'pos': np.array([15.0, 20.0]), 'radius': 2.0},
            {'pos': np.array([30.0, 40.0]), 'radius': 2.5},
            {'pos': np.array([60.0, 30.0]), 'radius': 2.0},
        ]
        
        # 定义观测空间和动作空间
        # 观测： [drone_x, drone_y, target_x, target_y, min_obstacle_dist]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 100, 100, 20]),
            dtype=np.float32
        )
        # 动作：速度增量 (vx, vy) 范围 [-1,1] 映射到 [-max_speed, max_speed]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.drone_pos = np.array(self.global_path[0], dtype=np.float32)
        self.steps = 0
        self.done = False
        return self._get_obs()
    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}
        
        # 执行动作：速度控制
        vel = np.clip(action, -1.0, 1.0) * self.max_speed
        self.drone_pos += vel * self.dt
        self.steps += 1
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 判断终止
        done = self._is_done()
        if done:
            reward -= 50.0   # 碰撞惩罚
        elif self.steps >= self.max_steps:
            done = True
            reward -= 10.0   # 超时惩罚
        
        self.done = done
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        # 计算到最近障碍物的距离
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(self.drone_pos - obs['pos']) - obs['radius']
            min_dist = min(min_dist, dist)
        if min_dist == float('inf'):
            min_dist = 20.0
        
        obs = np.array([
            self.drone_pos[0], self.drone_pos[1],
            self.target[0], self.target[1],
            min_dist
        ], dtype=np.float32)
        return obs
    
    def _compute_reward(self):
        # 到达目标奖励
        dist_to_target = np.linalg.norm(self.drone_pos - self.target)
        if dist_to_target < self.goal_threshold:
            return 100.0
        
        # 碰撞检测（提前给负奖励，但不立刻结束，让agent学习避开）
        collision = self._collision()
        if collision:
            return -50.0
        
        # 距离奖励：越接近目标越好
        reward = -0.1 * dist_to_target
        
        # 步数惩罚（鼓励快速到达）
        reward -= 0.01
        
        return reward
    
    def _collision(self):
        for obs in self.obstacles:
            dist = np.linalg.norm(self.drone_pos - obs['pos'])
            if dist < obs['radius']:
                return True
        return False
    
    def _is_done(self):
        # 如果碰撞或到达目标，结束
        if self._collision():
            return True
        if np.linalg.norm(self.drone_pos - self.target) < self.goal_threshold:
            return True
        return False