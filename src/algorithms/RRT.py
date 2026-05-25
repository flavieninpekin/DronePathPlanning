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

# ==================== RRT* (RRT-Star) ====================
def rrt_star_2d(grid, start, goal, step_size, max_iter=10000, goal_sample_rate=0.1, rewire_radius=None, tolerance=1.0):
    """
    RRT* 路径规划算法（带渐进最优性保证的RRT改进版）。

    参数:
        grid: 二维 0/1 numpy 数组，0=自由，1=障碍物
        start, goal: 起点/终点坐标
        step_size: 扩展步长
        max_iter: 最大迭代次数
        goal_sample_rate: 以目标为采样点的概率
        rewire_radius: 重连半径，默认 step_size * 3
        tolerance: 到达目标的容差

    返回:
        成功时返回路径数组 (Nx2)，失败返回 None
    """
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    if rewire_radius is None:
        rewire_radius = step_size * 3

    tree = [start]
    parents = [(start, -1)]
    costs = [0.0]

    def near_vertices(point, radius):
        return [i for i, v in enumerate(tree) if np.linalg.norm(v - point) < radius]

    best_path = None
    best_cost = float('inf')

    for i in range(max_iter):
        if random.random() < goal_sample_rate:
            rand_point = goal
        else:
            rand_point = np.array([random.uniform(0, grid.shape[0]),
                                   random.uniform(0, grid.shape[1])])

        nearest_idx = int(get_nearest(tree, rand_point))
        new_point = steer(tree[nearest_idx], rand_point, step_size)
        if is_collision(grid, new_point):
            continue
        if not edge_collision_free(grid, tree[nearest_idx], new_point):
            continue

        near_idxs = near_vertices(new_point, rewire_radius)
        min_cost = costs[nearest_idx] + np.linalg.norm(new_point - tree[nearest_idx])
        best_parent = nearest_idx
        for ni in near_idxs:
            if ni == nearest_idx:
                continue
            if not edge_collision_free(grid, tree[ni], new_point):
                continue
            candidate = costs[ni] + np.linalg.norm(new_point - tree[ni])
            if candidate < min_cost:
                min_cost = candidate
                best_parent = ni

        new_idx = len(tree)
        tree.append(new_point)
        parents.append((new_point, best_parent))
        costs.append(costs[best_parent] + np.linalg.norm(new_point - tree[best_parent]))

        for ni in near_idxs:
            if ni == best_parent:
                continue
            if not edge_collision_free(grid, new_point, tree[ni]):
                continue
            new_cost = costs[new_idx] + np.linalg.norm(tree[ni] - new_point)
            if new_cost < costs[ni]:
                parents[ni] = (tree[ni], new_idx)
                costs[ni] = new_cost
                propagate_cost(tree, parents, costs, ni)

        if np.linalg.norm(new_point - goal) < tolerance and not is_collision(grid, goal):
            if edge_collision_free(grid, new_point, goal):
                goal_idx = len(tree)
                tree.append(goal)
                parents.append((goal, new_idx))
                costs.append(costs[new_idx] + np.linalg.norm(goal - new_point))
                path_cost = costs[goal_idx]
                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = backtrace(parents, goal_idx)

    return np.array(best_path) if best_path is not None else None


def edge_collision_free(grid, p1, p2, resolution=10):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    for i in range(resolution + 1):
        pt = p1 + (i / resolution) * (p2 - p1)
        if is_collision(grid, pt):
            return False
    return True


def propagate_cost(tree, parents, costs, start_idx):
    for i in range(start_idx + 1, len(tree)):
        if parents[i][1] == start_idx:
            costs[i] = costs[start_idx] + np.linalg.norm(tree[i] - tree[start_idx])


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