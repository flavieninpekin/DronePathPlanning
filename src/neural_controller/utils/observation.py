import numpy as np
from typing import List, Tuple

def build_obs(drone_idx: int,
              drone_positions: np.ndarray,
              drone_velocities: np.ndarray,
              waypoints: List[Tuple[float, float]],
              waypoint_idx: int,
              dynamic_obstacles: List[dict],
              collision_radius: float,
              num_drones: int,
              num_dynamic_obs: int,
              map_size: Tuple[float, float]) -> np.ndarray:
    """
    构建单个无人机的观测向量
    """
    # 自身位置 (2) + 速度 (2)
    pos = drone_positions[drone_idx]
    vel = drone_velocities[drone_idx]
    obs = [pos[0], pos[1], vel[0], vel[1]]
    
    # 相对下一个航点 (2)
    if waypoint_idx < len(waypoints):
        target = np.array(waypoints[waypoint_idx])
        rel_target = target - pos
        obs.extend(rel_target)
    else:
        obs.extend([0.0, 0.0])
    
    # 其他无人机的相对位置 (最多 num_drones-1，不足补0)
    other_positions = []
    for j in range(num_drones):
        if j == drone_idx:
            continue
        other_pos = drone_positions[j]
        rel = other_pos - pos
        # 归一化到[-1,1]（可选）
        other_positions.extend(rel)
    # 填充到固定长度
    max_others = (num_drones - 1) * 2
    if len(other_positions) < max_others:
        other_positions.extend([0.0] * (max_others - len(other_positions)))
    obs.extend(other_positions[:max_others])
    
    # 动态障碍物的相对位置 (最多 num_dynamic_obs，不足补0)
    obs_rel_obs = []
    for ob in dynamic_obstacles:
        rel = ob['pos'] - pos
        obs_rel_obs.extend(rel)
    max_obs = num_dynamic_obs * 2
    if len(obs_rel_obs) < max_obs:
        obs_rel_obs.extend([0.0] * (max_obs - len(obs_rel_obs)))
    obs.extend(obs_rel_obs[:max_obs])
    
    return np.array(obs, dtype=np.float32)