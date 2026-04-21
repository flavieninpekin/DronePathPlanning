import numpy as np
import random
from collections import deque

def is_collision(grid, point, radius=0.0):
    """检查以 point 为中心、radius 为半径的圆是否与障碍物相交"""
    x, y = point[0], point[1]
    # 检查圆内所有整数坐标（简化：检查圆心所在格及周围8格）
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = int(np.floor(x + dx * radius))
            ny = int(np.floor(y + dy * radius))
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] == 1:
                    return True
    return False

def get_nearest(tree, point):
    dists = [np.linalg.norm(np.array(n) - point) for n in tree]
    return np.argmin(dists)

def steer(from_point, to_point, step_size):
    direction = np.array(to_point) - np.array(from_point)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_point
    direction = direction / length
    new_point = np.array(from_point) + direction * min(step_size, length)
    return new_point

def backtrace(parents, end_idx):
    path = []
    idx = end_idx
    while idx != -1:
        path.append(parents[idx][0])
        idx = parents[idx][1]
    return path[::-1]

def rrt_2d(grid, start, goal, step_size, max_iter=10000):
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    tree = [start]
    parents = [(start, -1)]  # (point, parent_idx); use -1 to indicate no parent
    for i in range(max_iter):
        rand_point = np.array([random.uniform(0, grid.shape[0]), random.uniform(0, grid.shape[1])])
        if random.random() < 0.1:
            rand_point = goal
        nearest_idx = get_nearest(tree, rand_point)
        nearest_idx = int(nearest_idx)
        new_point = steer(tree[nearest_idx], rand_point, step_size)
        if is_collision(grid, new_point, radius=0.0):
            continue
        tree.append(new_point)
        parents.append((new_point, int(nearest_idx)))
        if np.linalg.norm(new_point - goal) < step_size and not is_collision(grid, goal, radius=0.0):
            tree.append(goal)
            parents.append((goal, len(tree)-2))
            path = backtrace(parents, len(tree)-1)
            return np.array(path)
    return None

def rrt_3d(grid, start, goal, step_size, max_iter=10000):
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    tree = [start]
    parents = [(start, -1)]
    for i in range(max_iter):
        rand_point = np.array([random.uniform(0, grid.shape[0]), 
                               random.uniform(0, grid.shape[1]), 
                               random.uniform(0, grid.shape[2])])
        if random.random() < 0.1:
            rand_point = goal
        nearest_idx = get_nearest(tree, rand_point)
        new_point = steer(tree[nearest_idx], rand_point, step_size)
        if is_collision(grid, new_point, radius=0.0):
            continue
        tree.append(new_point)
        parents.append((new_point, int(nearest_idx)))
        if np.linalg.norm(new_point - goal) < step_size and not is_collision(grid, goal, radius=0.0):
            tree.append(goal)
            parents.append((goal, len(tree)-2))
            path = backtrace(parents, len(tree)-1)
            return np.array(path)
    return None

# 测试用例
if __name__ == "__main__":
    # 2D 测试
    grid2d = np.zeros((10, 10), dtype=int)
    grid2d[3:7, 5] = 1  # 障碍物
    start2d = (0, 0)
    goal2d = (9, 9)
    path2d = rrt_2d(grid2d, start2d, goal2d, step_size=1.5)
    print("2D Path:")
    if path2d is not None:
        print(np.round(path2d, 3))
    else:
        print("No path found.")

    # 3D 测试
    grid3d = np.zeros((8, 8, 8), dtype=int)
    grid3d[2:6, 4, 2:6] = 1  # 障碍物
    start3d = (0, 0, 0)
    goal3d = (7, 7, 7)
    path3d = rrt_3d(grid3d, start3d, goal3d, step_size=2.0)
    print("3D Path:")
    if path3d is not None:
        print(np.round(path3d, 3))
    else:
        print("No path found.")