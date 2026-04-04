import numpy as np
import random
from collections import deque

def generate_map_with_path(size, obstacle_density, start=None, goal=None, dim=2,
                           step_factor=2.5, p=0.7, max_attempts=100):
    """
    先生成一条从起点到终点的蜿蜒路径（仅轴向移动），再随机填充障碍物。
    障碍物密度精确等于总格子数 * obstacle_density（除非路径占用了太多格子导致候选不足）。

    参数:
        size: 地图尺寸，如 (10,10) 或 (5,5,5)
        obstacle_density: 障碍物密度 (0~1)
        start, goal: 起点和终点坐标，默认分别为 (0,...,0) 和 (size[i]-1, ...)
        dim: 维度 (2 或 3)
        step_factor: 最大步数 = 曼哈顿距离 * step_factor，越大路径越蜿蜒
        p: 每一步选择朝向终点方向的概率（其余概率随机选其他方向）
        max_attempts: 随机游走失败时的最大重试次数
    """
    size = tuple(size)
    if start is None:
        start = tuple([0] * dim)
    if goal is None:
        goal = tuple([s - 1 for s in size])
    assert len(size) == dim
    assert len(start) == dim
    assert len(goal) == dim

    # 辅助函数：曼哈顿距离
    def manhattan(a, b):
        return sum(abs(a[i] - b[i]) for i in range(dim))

    # 获取轴向邻居（不检查边界和访问状态）
    def get_neighbors(pos):
        neighbors = []
        for d in range(dim):
            for delta in (-1, 1):
                n = list(pos)
                n[d] += delta
                if 0 <= n[d] < size[d]:
                    neighbors.append(tuple(n))
        return neighbors

    # 随机游走生成路径
    def random_walk_path():
        path = [start]
        current = start
        visited = set([start])
        max_steps = int(manhattan(start, goal) * step_factor)

        for _ in range(max_steps):
            if current == goal:
                return path

            neighbors = get_neighbors(current)
            # 避免立刻回头
            if len(path) >= 2:
                prev = path[-2]
                neighbors = [n for n in neighbors if n != prev]

            if not neighbors:
                break

            # 区分朝向终点和背离/平行方向
            towards = []
            others = []
            for n in neighbors:
                if manhattan(n, goal) < manhattan(current, goal):
                    towards.append(n)
                else:
                    others.append(n)

            if towards and random.random() < p:
                next_pos = random.choice(towards)
            else:
                # 从所有邻居中随机选
                next_pos = random.choice(neighbors)

            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos

        return None

    # 重试机制
    for attempt in range(max_attempts):
        path = random_walk_path()
        if path is not None and path[-1] == goal:
            break
    else:
        raise RuntimeError(f"经过 {max_attempts} 次尝试，无法生成从起点到终点的路径，请调整参数（如增大 step_factor）。")

    # 构建地图
    arr = np.zeros(size, dtype=np.uint8)
    path_set = set(path)  # 路径上的所有格子（去重）

    # 精确计算障碍物数量：总格子数 * 密度
    total_cells = np.prod(size)
    num_obstacles = int(total_cells * obstacle_density)

    # 候选格子：不在路径上，且不是起点和终点
    candidate_cells = [tuple(idx) for idx in np.ndindex(size)
                       if idx not in path_set and idx != start and idx != goal]

    # 如果候选格子不足，则只能把现有候选全设为障碍物，并发出警告
    if len(candidate_cells) < num_obstacles:
        num_obstacles = len(candidate_cells)
        actual_density = num_obstacles / total_cells
        print(f"警告：路径占用格子过多，障碍物密度只能达到 {actual_density:.3f}（目标 {obstacle_density}）")

    # 随机选择并设置障碍物
    if num_obstacles > 0:
        obstacle_indices = random.sample(candidate_cells, num_obstacles)
        for idx in obstacle_indices:
            arr[idx] = 1

    # 确保起点终点可通行
    arr[start] = 0
    arr[goal] = 0
    return arr


class MapGenerator:
    @staticmethod
    def generate_map_with_path(size, obstacle_density, start=None, goal=None, dim=2,
                               step_factor=2.5, p=0.7, max_attempts=100):
        return generate_map_with_path(size, obstacle_density, start=start, goal=goal, dim=dim,
                                      step_factor=step_factor, p=p, max_attempts=max_attempts)


# 测试用例
def test_generate_map():
    size = (10, 10)
    density = 0.2
    arr = generate_map_with_path(size, density)
    assert arr.shape == size
    assert arr[0, 0] == 0
    assert arr[-1, -1] == 0

    # 检查连通性（4方向BFS）
    def bfs(arr, start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (curr[0] + d[0], curr[1] + d[1])
                if 0 <= n[0] < arr.shape[0] and 0 <= n[1] < arr.shape[1]:
                    if arr[n] == 0 and n not in visited:
                        visited.add(n)
                        queue.append(n)
        return False

    assert bfs(arr, (0, 0), (size[0] - 1, size[1] - 1))
    print("Test passed.")


if __name__ == "__main__":
    test_generate_map()