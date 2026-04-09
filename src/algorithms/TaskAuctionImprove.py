import numpy as np
from collections import deque

# ========== 障碍物距离计算函数（与之前一致）==========
def compute_distance_map(grid, sources):
    """多源 BFS，返回每个格子到最近源点的路径距离（四连通）"""
    h, w = grid.shape
    dist = np.full((h, w), np.inf)
    q = deque()
    for sx, sy in sources:
        if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == 0:
            dist[sy, sx] = 0
            q.append((sx, sy))
    while q:
        x, y = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0 and dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist

def point_to_grid(point, grid):
    """连续坐标转网格整数坐标，并避开障碍物（螺旋搜索最近自由格）"""
    x, y = point
    ix, iy = int(round(x)), int(round(y))
    h, w = grid.shape
    ix = max(0, min(w-1, ix))
    iy = max(0, min(h-1, iy))
    if grid[iy, ix] == 0:
        return ix, iy
    for r in range(1, 10):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = ix+dx, iy+dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    return nx, ny
    raise ValueError(f"无法为点 ({x},{y}) 找到自由格子")

def compute_all_distances(task_points, drone_positions, grid):
    """预计算距离矩阵：无人机到任务、任务到任务"""
    num_tasks = len(task_points)
    num_drones = len(drone_positions)
    task_grid = [point_to_grid(p, grid) for p in task_points]
    drone_grid = [point_to_grid(p, grid) for p in drone_positions]

    # 无人机到任务距离
    dist_drone_to_task = np.zeros((num_drones, num_tasks))
    for d, dg in enumerate(drone_grid):
        dmap = compute_distance_map(grid, [dg])
        for t, (tx, ty) in enumerate(task_grid):
            dist_drone_to_task[d, t] = dmap[ty, tx]

    # 任务间距离
    dist_task_to_task = np.zeros((num_tasks, num_tasks))
    for i, tg in enumerate(task_grid):
        imap = compute_distance_map(grid, [tg])
        for j, (tx, ty) in enumerate(task_grid):
            dist_task_to_task[i, j] = imap[ty, tx]

    # 处理不可达（inf）情况
    if np.any(np.isinf(dist_drone_to_task)) or np.any(np.isinf(dist_task_to_task)):
        print("警告：存在不可达任务，将替换为大数")
        dist_drone_to_task = np.nan_to_num(dist_drone_to_task, nan=1e9, posinf=1e9)
        dist_task_to_task = np.nan_to_num(dist_task_to_task, nan=1e9, posinf=1e9)
    return dist_drone_to_task, dist_task_to_task

def tsp_approx_cost(task_indices, dist_task_to_task, dist_drone_to_task, drone_idx):
    """最近邻启发式估算从无人机起点出发遍历所有任务的路径长度"""
    if not task_indices:
        return 0.0
    unvisited = set(task_indices)
    # 第一个任务：选择离起点最近的任务
    first = min(unvisited, key=lambda t: dist_drone_to_task[drone_idx, t])
    total = dist_drone_to_task[drone_idx, first]
    current = first
    unvisited.remove(first)
    while unvisited:
        nxt = min(unvisited, key=lambda t: dist_task_to_task[current, t])
        total += dist_task_to_task[current, nxt]
        current = nxt
        unvisited.remove(nxt)
    return total

def total_cost_all(assignments, dist_task_to_task, dist_drone_to_task):
    """计算所有无人机的总路径代价"""
    total = 0.0
    for d, tasks in assignments.items():
        total += tsp_approx_cost(tasks, dist_task_to_task, dist_drone_to_task, d)
    return total

# ========== 改进的拍卖/分配算法 ==========
def auction_algorithm_improved(task_points, drone_positions, grid, max_iter=50):
    """
    改进算法：
    1. 贪心初始分配：每个任务分配给距离最近的无人机（保证负载大致均衡？不保证数量均衡，但保证每个任务分配）
    2. 局部搜索：尝试移动单个任务或交换两个任务，若降低总代价则接受
    返回：assignments, total_cost
    """
    num_tasks = len(task_points)
    num_drones = len(drone_positions)
    
    # 预计算距离矩阵
    dist_drone_to_task, dist_task_to_task = compute_all_distances(task_points, drone_positions, grid)
    
    # ---- 1. 贪心初始分配：每个任务独立分配给最近的无人机 ----
    assignments = {i: [] for i in range(num_drones)}
    for t in range(num_tasks):
        best_drone = np.argmin(dist_drone_to_task[:, t])
        best_drone = int(best_drone)
        assignments[best_drone].append(t)
    
    # 可选：如果某个无人机任务太多，可以后续通过局部搜索平衡，这里先保留
    
    # ---- 2. 局部搜索：移动任务 和 交换任务 ----
    current_cost = total_cost_all(assignments, dist_task_to_task, dist_drone_to_task)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        # 尝试所有无人机对 (i, j)
        for i in range(num_drones):
            for j in range(num_drones):
                if i == j:
                    continue
                # 2.1 尝试把 i 中的一个任务移动到 j
                for t_idx, task in enumerate(assignments[i]):
                    # 计算移动后的新代价
                    new_assign_i = assignments[i][:t_idx] + assignments[i][t_idx+1:]
                    new_assign_j = assignments[j] + [task]
                    new_assignments = assignments.copy()
                    new_assignments[i] = new_assign_i
                    new_assignments[j] = new_assign_j
                    new_cost = total_cost_all(new_assignments, dist_task_to_task, dist_drone_to_task)
                    if new_cost < current_cost - 1e-6:
                        assignments = new_assignments
                        current_cost = new_cost
                        improved = True
                        break  # 立即接受，重新开始扫描
                if improved:
                    break
                # 2.2 尝试交换 i 中的一个任务和 j 中的一个任务
                for t_i in range(len(assignments[i])):
                    for t_j in range(len(assignments[j])):
                        task_i = assignments[i][t_i]
                        task_j = assignments[j][t_j]
                        new_assign_i = assignments[i][:t_i] + [task_j] + assignments[i][t_i+1:]
                        new_assign_j = assignments[j][:t_j] + [task_i] + assignments[j][t_j+1:]
                        new_assignments = assignments.copy()
                        new_assignments[i] = new_assign_i
                        new_assignments[j] = new_assign_j
                        new_cost = total_cost_all(new_assignments, dist_task_to_task, dist_drone_to_task)
                        if new_cost < current_cost - 1e-6:
                            assignments = new_assignments
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        iter_count += 1
    
    # ---- 3. 最终整体代价（已经是最新 cost）----
    total_cost = current_cost
    return assignments, total_cost

# ========== 测试用例（需要地图生成模块）==========
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from map_generator.MapGenerator import generate_map_with_path
    from map_generator.TaskPointGeneration import generate_task_points
    
    # 生成地图
    size = (20, 20)
    density = 0.2
    grid = generate_map_with_path(size, density)
    
    # 生成任务点
    num_tasks = 15
    task_points = generate_task_points(grid, num_tasks)
    
    # 生成无人机起点（避开障碍物）
    num_drones = 4
    drone_positions = []
    h, w = grid.shape # type: ignore
    while len(drone_positions) < num_drones:
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        ix, iy = int(x), int(y)
        if grid[iy, ix] == 0:
            drone_positions.append([x, y])
    
    assignments, total_cost = auction_algorithm_improved(task_points, drone_positions, grid)
    
    print("改进算法分配结果：")
    for d, tasks in assignments.items():
        print(f"无人机{d}: 任务索引 {tasks} (数量 {len(tasks)})")
    print(f"整体预估路径代价: {total_cost:.2f}")