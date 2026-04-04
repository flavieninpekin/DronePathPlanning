import time
import numpy as np
import random
from collections import deque
import sys

# ========== 原版 generate_map ==========
def generate_map(size, obstacle_density, start=None, goal=None, dim=2, max_tries=100):
    # 将 max_tries 从 1000 降到 100，避免耗时过长
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
        total = np.prod(size)
        num_obstacles = int(total * obstacle_density)
        free_indices = [idx for idx in np.ndindex(size) if idx != start and idx != goal]
        if num_obstacles > len(free_indices):
            num_obstacles = len(free_indices)
        obstacles = np.random.choice(len(free_indices), num_obstacles, replace=False)
        for i in obstacles:
            arr[free_indices[i]] = 1
        if is_connected(arr, start, goal):
            return arr
    raise RuntimeError("无法生成连通地图")

# ========== 新版 generate_map_with_path ==========
import random as pyrandom
def generate_map_with_path(size, obstacle_density, start=None, goal=None, dim=2,
                           step_factor=2.5, p=0.7, max_attempts=100):
    size = tuple(size)
    if start is None:
        start = tuple([0] * dim)
    if goal is None:
        goal = tuple([s - 1 for s in size])
    assert len(size) == dim
    assert len(start) == dim
    assert len(goal) == dim

    def manhattan(a, b):
        return sum(abs(a[i] - b[i]) for i in range(dim))

    def get_neighbors(pos):
        neighbors = []
        for d in range(dim):
            for delta in (-1, 1):
                n = list(pos)
                n[d] += delta
                if 0 <= n[d] < size[d]:
                    neighbors.append(tuple(n))
        return neighbors

    def random_walk_path():
        path = [start]
        current = start
        visited = set([start])
        max_steps = int(manhattan(start, goal) * step_factor)
        for _ in range(max_steps):
            if current == goal:
                return path
            neighbors = get_neighbors(current)
            if len(path) >= 2:
                prev = path[-2]
                neighbors = [n for n in neighbors if n != prev]
            if not neighbors:
                break
            towards = []
            others = []
            for n in neighbors:
                if manhattan(n, goal) < manhattan(current, goal):
                    towards.append(n)
                else:
                    others.append(n)
            if towards and pyrandom.random() < p:
                next_pos = pyrandom.choice(towards)
            else:
                next_pos = pyrandom.choice(neighbors)
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos
        return None

    for attempt in range(max_attempts):
        path = random_walk_path()
        if path is not None and path[-1] == goal:
            break
    else:
        raise RuntimeError("无法生成路径")

    arr = np.zeros(size, dtype=np.uint8)
    path_set = set(path)
    total_cells = np.prod(size)
    num_obstacles = int(total_cells * obstacle_density)
    candidate_cells = [tuple(idx) for idx in np.ndindex(size)
                       if idx not in path_set and idx != start and idx != goal]
    if len(candidate_cells) < num_obstacles:
        num_obstacles = len(candidate_cells)
    if num_obstacles > 0:
        obstacle_indices = pyrandom.sample(candidate_cells, num_obstacles)
        for idx in obstacle_indices:
            arr[idx] = 1
    arr[start] = 0
    arr[goal] = 0
    return arr

# ========== 测试函数（带进度输出）==========
def run_test(method, name, dim, size_range, densities, num_tests_per_config=3):
    print(f"\n>>> 开始测试 {name} (dim={dim})")
    successes = 0
    total_time = 0.0
    total_tests = 0
    for density in densities:
        for i in range(num_tests_per_config):
            # 随机尺寸
            if dim == 2:
                size = (np.random.randint(size_range[0], size_range[1]+1),
                        np.random.randint(size_range[0], size_range[1]+1))
            else:
                size = (np.random.randint(size_range[0], size_range[1]+1),
                        np.random.randint(size_range[0], size_range[1]+1),
                        np.random.randint(size_range[0], size_range[1]+1))
            start = tuple([0]*dim)
            goal = tuple([s-1 for s in size])
            print(f"  尝试: dim={dim}, size={size}, density={density}, test={i+1}/{num_tests_per_config}", end='', flush=True)
            try:
                t0 = time.perf_counter()
                arr = method(size, density, start=start, goal=goal, dim=dim)
                elapsed = time.perf_counter() - t0
                successes += 1
                total_time += elapsed
                print(f" ✓ 成功, 耗时 {elapsed*1000:.1f} ms")
            except Exception as e:
                print(f" ✗ 失败: {e}")
            total_tests += 1
    success_rate = successes / total_tests if total_tests>0 else 0
    avg_time = total_time / successes if successes>0 else float('inf')
    print(f"{name} 完成: 成功率 {success_rate*100:.1f}%, 平均耗时 {avg_time*1000:.2f} ms")
    return success_rate, avg_time

def main():
    random.seed(42)
    np.random.seed(42)
    # 降低测试规模（尤其三维）
    num_tests_per_config = 3   # 每组测试3次
    densities_2d = [0.1, 0.3, 0.5]
    densities_3d = [0.1, 0.3, 0.5]
    size_range_2d = (100, 500)   # 2D 可以保持大尺寸，因为2D BFS较快
    size_range_3d = (20, 50)     # 3D 减小到 20~50，否则原版太慢

    print("="*60)
    print("地图生成算法对比测试")
    print("="*60)
    
    # 测试原版 2D
    run_test(generate_map, "原版(随机+连通)", 2, size_range_2d, densities_2d, num_tests_per_config)
    # 测试原版 3D
    run_test(generate_map, "原版(随机+连通)", 3, size_range_3d, densities_3d, num_tests_per_config)
    # 测试新版 2D
    run_test(generate_map_with_path, "新版(蜿蜒路径)", 2, size_range_2d, densities_2d, num_tests_per_config)
    # 测试新版 3D
    run_test(generate_map_with_path, "新版(蜿蜒路径)", 3, size_range_3d, densities_3d, num_tests_per_config)

if __name__ == "__main__":
    main()