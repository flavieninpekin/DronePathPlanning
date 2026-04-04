import numpy as np
from collections import deque

def generate_map(size, obstacle_density, start=None, goal=None, dim=2, max_tries=1000):
    """
    随机生成二维或三维0-1 np.array，1为障碍物，0为可通行。
    保证起点与终点间有可行路径，整体可到达。
    """
    size = tuple(size)
    if start is None:
        start = tuple([0]*dim)
    if goal is None:
        goal = tuple([s-1 for s in size])
    assert len(size) == dim
    assert len(start) == dim
    assert len(goal) == dim

    def neighbors(pos):
        for d in range(dim):
            for delta in [-1, 1]:
                n = list(pos)
                n[d] += delta
                if 0 <= n[d] < size[d]:
                    yield tuple(n)

    def is_connected(arr, s, g):
        visited = set()
        queue = deque([s])
        while queue:
            curr = queue.popleft()
            if curr == g:
                return True
            for n in neighbors(curr):
                if arr[n] == 0 and n not in visited:
                    visited.add(n)
                    queue.append(n)
        return False

    for _ in range(max_tries):
        arr = np.zeros(size, dtype=np.uint8)
        arr[start] = 0
        arr[goal] = 0
        # 随机生成障碍物
        total = np.prod(size)
        num_obstacles = int(total * obstacle_density)
        # 排除起点终点
        free_indices = [idx for idx in np.ndindex(size) if idx != start and idx != goal]
        obstacles = np.random.choice(len(free_indices), num_obstacles, replace=False)
        for i in obstacles:
            arr[free_indices[i]] = 1
        # 检查连通性
        if is_connected(arr, start, goal):
            return arr
    raise RuntimeError("无法生成满足条件的地图，请调整参数。")


class MapGenerator:
    @staticmethod
    def generate_map(size, obstacle_density, start=None, goal=None, dim=2, max_tries=1000):
        return generate_map(size, obstacle_density, start=start, goal=goal, dim=dim, max_tries=max_tries)


# 测试用例
def test_generate_map():
    size = (10, 10)
    density = 0.2
    arr = generate_map(size, density)
    assert arr.shape == size
    assert arr[0,0] == 0
    assert arr[-1,-1] == 0
    # 检查连通性
    def bfs(arr, start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            for d in [(-1,0),(1,0),(0,-1),(0,1)]:
                n = (curr[0]+d[0], curr[1]+d[1])
                if 0<=n[0]<arr.shape[0] and 0<=n[1]<arr.shape[1]:
                    if arr[n]==0 and n not in visited:
                        visited.add(n)
                        queue.append(n)
        return False
    assert bfs(arr, (0,0), (size[0]-1, size[1]-1))
    print("Test passed.")

if __name__ == "__main__":
    test_generate_map()