import numpy as np
from collections import deque
import itertools

def compute_distance_map(grid, sources):
    """
    多源 BFS，计算网格上每个点到最近源点的最短路径距离（四连通）。
    sources: list of (x, y) 整数坐标
    返回: 二维数组，dist[y, x] = 距离（不可达为 inf）
    """
    h, w = grid.shape
    dist = np.full((h, w), np.inf)
    q = deque()
    for sx, sy in sources:
        if 0 <= sx < w and 0 <= sy < h and grid[sy, sx] == 0:
            dist[sy, sx] = 0
            q.append((sx, sy))
    while q:
        x, y = q.popleft()
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0 and dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist

def point_to_grid(point, grid):
    """将连续坐标 (x,y) 转换为网格整数坐标，并确保在边界内且不是障碍物"""
    x, y = point
    ix, iy = int(round(x)), int(round(y))
    h, w = grid.shape
    ix = max(0, min(w-1, ix))
    iy = max(0, min(h-1, iy))
    # 如果该格子是障碍物，寻找最近的自由格子（简单回退：小范围搜索）
    if grid[iy, ix] == 1:
        # 螺旋搜索最近自由格子
        for r in range(1, 10):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = ix+dx, iy+dy
                    if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                        return nx, ny
        raise ValueError(f"无法为点 ({x},{y}) 找到附近自由格子")
    return ix, iy

def compute_path_distances(task_points, drone_positions, grid):
    """
    预计算所有任务点到无人机起点的路径距离，以及任务点之间的路径距离。
    返回: (dist_drone_to_task, dist_task_to_task)
    """
    num_tasks = len(task_points)
    num_drones = len(drone_positions)
    
    # 将所有任务点和无人机起点转换为网格整数坐标
    task_grid = [point_to_grid(p, grid) for p in task_points]
    drone_grid = [point_to_grid(p, grid) for p in drone_positions]
    
    # 计算每个任务点到所有格子的距离（多源BFS）
    drone_dist_maps = []
    for dg in drone_grid:
        dist_map = compute_distance_map(grid, [dg])
        drone_dist_maps.append(dist_map)
    
    task_dist_maps = []
    for tg in task_grid:
        dist_map = compute_distance_map(grid, [tg])
        task_dist_maps.append(dist_map)
    
    # 构建距离矩阵
    dist_drone_to_task = np.zeros((num_drones, num_tasks))
    for i, dmap in enumerate(drone_dist_maps):
        for j, (tx, ty) in enumerate(task_grid):
            dist_drone_to_task[i, j] = dmap[ty, tx]
    
    dist_task_to_task = np.zeros((num_tasks, num_tasks))
    for i, tmap in enumerate(task_dist_maps):
        for j, (tx, ty) in enumerate(task_grid):
            dist_task_to_task[i, j] = tmap[ty, tx]
    
    # 检查是否有不可达的任务（距离inf）
    if np.any(np.isinf(dist_drone_to_task)) or np.any(np.isinf(dist_task_to_task)):
        print("警告：存在不可达的任务点，可能由于障碍物完全隔离。")
        # 将inf替换为大数，但实际应避免这种情况
        dist_drone_to_task = np.nan_to_num(dist_drone_to_task, nan=1e9, posinf=1e9)
        dist_task_to_task = np.nan_to_num(dist_task_to_task, nan=1e9, posinf=1e9)
    
    return dist_drone_to_task, dist_task_to_task

def estimate_tsp_cost(task_indices, dist_task_to_task, dist_drone_to_task, drone_idx):
    """
    使用最近邻启发式估算从无人机起点出发，遍历指定任务列表的路径总长度。
    task_indices: 要访问的任务点索引列表
    dist_drone_to_task: (num_drones, num_tasks) 矩阵
    dist_task_to_task: (num_tasks, num_tasks) 矩阵
    drone_idx: 无人机索引
    返回: 总路径长度（浮点数）
    """
    if not task_indices:
        return 0.0
    unvisited = set(task_indices)
    current = None  # 当前所在的任务索引（None表示起点）
    total = 0.0
    # 从起点出发，选择最近的任务
    first_task = min(unvisited, key=lambda t: dist_drone_to_task[drone_idx, t])
    total += dist_drone_to_task[drone_idx, first_task]
    current = first_task
    unvisited.remove(first_task)
    
    while unvisited:
        next_task = min(unvisited, key=lambda t: dist_task_to_task[current, t])
        total += dist_task_to_task[current, next_task]
        current = next_task
        unvisited.remove(next_task)
    return total

def auction_algorithm(task_points, drone_positions, grid, epsilon=0.2, max_iter=1000, load_penalty=5.0):
    """
    障碍物感知的拍卖算法，用于将任务分配给多架无人机（每个无人机可执行多个任务）。
    参数:
        task_points: list of (x,y) 连续坐标
        drone_positions: list of (x,y) 连续坐标
        grid: 二维 0/1 numpy 数组，0=自由，1=障碍物
        epsilon: 价格上升步长
        max_iter: 最大迭代次数
    返回:
        assignments: 字典 {drone_idx: [task_idx, ...]}
        total_cost: 整体预估路径代价（基于最近邻TSP）
    """
    num_tasks = len(task_points)
    num_drones = len(drone_positions)
    
    # 1. 预计算路径距离矩阵
    dist_drone_to_task, dist_task_to_task = compute_path_distances(task_points, drone_positions, grid)
    
    # 2. 初始化
    task_prices = np.zeros(num_tasks)
    assignments = {i: [] for i in range(num_drones)}
    assigned_tasks = set()
    
    # 3. 多轮拍卖
    for _ in range(max_iter):
        if len(assigned_tasks) == num_tasks:
            break
        
        # 本轮每个无人机选择最有价值的未分配任务
        new_assignments = {}  # drone -> task
        for d in range(num_drones):
            best_task = None
            best_value = -np.inf
            for t in range(num_tasks):
                if t in assigned_tasks:
                    continue
                penalty = load_penalty * len(assignments[d])
                value = -dist_drone_to_task[d, t] - task_prices[t] - penalty
                if value > best_value:
                    best_value = value
                    best_task = t
            if best_task is not None:
                new_assignments[d] = best_task
        
        # 处理竞争：每个任务只分配给价值最高的无人机
        task_bidders = {}
        for d, t in new_assignments.items():
            task_bidders.setdefault(t, []).append(d)
        
        for t, bidders in task_bidders.items():
            if t in assigned_tasks:
                continue
            # 找出价值最高的竞拍者
            best_drone = None
            best_value = -np.inf
            for d in bidders:
                value = -dist_drone_to_task[d, t] - task_prices[t]
                if value > best_value:
                    best_value = value
                    best_drone = d
            if best_drone is not None:
                # 显式转换为 int 类型，避免 numpy.int64 作为字典键的问题
                best_drone = int(best_drone)
                assignments[best_drone].append(t)
                assigned_tasks.add(t)
                task_prices[t] += epsilon
        
        # 如果本轮没有新分配，且还有未分配任务，降低 epsilon 或退出
        if len(assigned_tasks) == len(task_bidders) == 0:
            break
    
    # 补充分配剩余任务（理论上不会，但安全起见）
    remaining = set(range(num_tasks)) - assigned_tasks
    for t in remaining:
        best_drone = int(np.argmin(dist_drone_to_task[:, t]))
        assignments[best_drone].append(t)
    
    # 4. 计算整体预估路径代价（每个无人机独立TSP，采用最近邻启发式）
    total_cost = 0.0
    for d in range(num_drones):
        tasks = assignments[d]
        if tasks:
            cost = estimate_tsp_cost(tasks, dist_task_to_task, dist_drone_to_task, d)
            total_cost += cost
    
    return assignments, total_cost

# 示例测试（需结合 MapGenerator 和 TaskPointGeneration）
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到路径，以便导入 map_generator 模块
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from map_generator.MapGenerator import generate_map_with_path
    from map_generator.TaskPointGeneration import generate_task_points
    
    # 生成地图
    size = (20, 20)
    density = 0.2
    grid = generate_map_with_path(size, density)
    
    # 生成任务点（避开障碍物）
    num_tasks = 15
    task_points = generate_task_points(grid, num_tasks)
    
    # 随机生成无人机起点（避开障碍物）
    num_drones = 4
    drone_positions = []
    h, w = grid.shape  # type: ignore # 修复：正确解包 grid.shape
    while len(drone_positions) < num_drones:
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        ix, iy = int(x), int(y)
        if grid[iy, ix] == 0:
            drone_positions.append([x, y])
    
    assignments, total_cost = auction_algorithm(task_points, drone_positions, grid)
    
    print("任务分配结果：")
    for d, tasks in assignments.items():
        print(f"无人机{d}: 任务索引 {tasks}")
    print(f"整体预估路径代价: {total_cost:.2f}")