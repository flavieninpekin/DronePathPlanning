import numpy as np

def check_circle_collision(pos1: np.ndarray, pos2: np.ndarray, radius: float) -> bool:
    """检测两个圆心距离是否小于2倍半径"""
    dist = np.linalg.norm(pos1 - pos2)
    return bool(dist < 2 * radius)

def check_circle_obstacle_collision(drone_pos: np.ndarray, obs_pos: np.ndarray, radius: float) -> bool:
    return check_circle_collision(drone_pos, obs_pos, radius)