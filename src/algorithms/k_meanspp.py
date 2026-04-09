import numpy as np
from collections import deque

# ========== 障碍物距离计算（复用）==========
def compute_distance_map(grid, sources):
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

def compute_task_distance_matrix(task_points, grid):
    num_tasks = len(task_points)
    task_grid = [point_to_grid(p, grid) for p in task_points]
    dist_mat = np.zeros((num_tasks, num_tasks))
    for i, tg in enumerate(task_grid):
        dmap = compute_distance_map(grid, [tg])
        for j, (tx, ty) in enumerate(task_grid):
            dist_mat[i, j] = dmap[ty, tx]
    if np.any(np.isinf(dist_mat)):
        print("警告：存在不可达的任务对，将替换为大数")
        dist_mat = np.nan_to_num(dist_mat, nan=1e9, posinf=1e9)
    return dist_mat

def tsp_approx_cost(task_indices, dist_task_to_task, dist_drone_to_task, drone_idx):
    if not task_indices:
        return 0.0
    unvisited = set(task_indices)
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


# ========== 障碍物感知的 K-Means++（实际是 K-Medoids++）==========
class ObstacleAwareKMeansPlusPlus:
    """
    使用路径距离矩阵，中心点必须是实际任务点（Medoid）。
    初始化采用 K-Means++ 策略：第一个中心随机选，后续按距离平方概率选。
    """
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, dist_matrix):
        """
        dist_matrix: (n_samples, n_samples) 对称距离矩阵（路径距离）
        返回: labels_, medoid_indices_
        """
        np.random.seed(self.random_state)
        n = dist_matrix.shape[0]
        # ---- K-Means++ 初始化 medoids ----
        medoids = []
        # 1. 随机选择第一个 medoid
        first = np.random.choice(n)
        medoids.append(first)
        # 2. 迭代选择剩余 medoids
        for _ in range(1, self.n_clusters):
            # 计算每个点到最近已选 medoid 的距离
            min_dist = np.full(n, np.inf)
            for m in medoids:
                d = dist_matrix[:, m]
                min_dist = np.minimum(min_dist, d)
            # 按距离平方概率选择下一个 medoid
            probs = min_dist ** 2
            probs /= probs.sum()
            next_medoid = np.random.choice(n, p=probs)
            medoids.append(next_medoid)
        self.medoid_indices_ = np.array(medoids)

        # ---- 迭代优化 ----
        for _ in range(self.max_iter):
            # 分配每个样本到最近的 medoid
            distances_to_medoids = dist_matrix[:, self.medoid_indices_]
            self.labels_ = np.argmin(distances_to_medoids, axis=1)

            # 更新每个簇的 medoid：选择簇内到其他点距离和最小的点
            new_medoids = []
            for k in range(self.n_clusters):
                cluster_points = np.where(self.labels_ == k)[0]
                if len(cluster_points) == 0:
                    # 空簇：重新随机选择未使用的点
                    used = set(self.medoid_indices_)
                    available = [i for i in range(n) if i not in used]
                    if available:
                        new_medoids.append(np.random.choice(available))
                    else:
                        new_medoids.append(np.random.choice(range(n)))
                    continue
                sub_dist = dist_matrix[np.ix_(cluster_points, cluster_points)]
                total_dist = np.sum(sub_dist, axis=1)
                best_idx = cluster_points[np.argmin(total_dist)]
                new_medoids.append(best_idx)
            new_medoids = np.array(new_medoids)

            if np.array_equal(new_medoids, self.medoid_indices_):
                break
            self.medoid_indices_ = new_medoids

        return self.labels_, self.medoid_indices_


def assign_tasks_with_kmeanspp(task_points, drone_positions, grid, max_iter=100):
    """
    使用 K-Means++（实际为 K-Medoids++）将任务聚成与无人机数量相等的簇，
    然后将每个簇分配给最近的无人机起点。
    返回:
        assignments: dict {drone_idx: [task_idx, ...]}
        total_cost: 整体预估路径代价
    """
    num_drones = len(drone_positions)
    num_tasks = len(task_points)
    if num_drones > num_tasks:
        raise ValueError("无人机数量不能大于任务数量")

    # 1. 计算任务间路径距离矩阵
    dist_task = compute_task_distance_matrix(task_points, grid)

    # 2. K-Means++ 聚类
    kmeans = ObstacleAwareKMeansPlusPlus(n_clusters=num_drones, max_iter=max_iter, random_state=42)
    labels, medoid_indices = kmeans.fit(dist_task)

    # 3. 计算每个无人机起点到每个簇中心（medoid）的路径距离
    drone_grid = [point_to_grid(p, grid) for p in drone_positions]
    medoid_grid = [point_to_grid(task_points[idx], grid) for idx in medoid_indices]

    dist_drone_to_medoid = np.zeros((num_drones, num_drones))
    for d, dg in enumerate(drone_grid):
        dmap = compute_distance_map(grid, [dg])
        for k, (mx, my) in enumerate(medoid_grid):
            dist_drone_to_medoid[d, k] = dmap[my, mx]

    # 4. 分配每个簇给最近的无人机（贪心匹配，避免冲突）
    pairs = []
    for k in range(num_drones):
        for d in range(num_drones):
            pairs.append((dist_drone_to_medoid[d, k], k, d))
    pairs.sort(key=lambda x: x[0])
    cluster_to_drone = {}
    used_drones = set()
    for _, cluster, drone in pairs:
        if cluster not in cluster_to_drone and drone not in used_drones:
            cluster_to_drone[cluster] = drone
            used_drones.add(drone)
    # 补充未分配（理论上不会）
    for k in range(num_drones):
        if k not in cluster_to_drone:
            for d in range(num_drones):
                if d not in used_drones:
                    cluster_to_drone[k] = d
                    used_drones.add(d)
                    break

    # 5. 构建每个无人机的任务列表
    assignments = {d: [] for d in range(num_drones)}
    for task_idx, cluster_id in enumerate(labels):
        drone_id = cluster_to_drone[cluster_id]
        assignments[drone_id].append(task_idx)

    # 6. 计算整体代价
    task_grid = [point_to_grid(p, grid) for p in task_points]
    dist_drone_to_task = np.zeros((num_drones, num_tasks))
    for d, dg in enumerate(drone_grid):
        dmap = compute_distance_map(grid, [dg])
        for t, (tx, ty) in enumerate(task_grid):
            dist_drone_to_task[d, t] = dmap[ty, tx]
    if np.any(np.isinf(dist_drone_to_task)):
        dist_drone_to_task = np.nan_to_num(dist_drone_to_task, nan=1e9, posinf=1e9)

    total_cost = 0.0
    for d in range(num_drones):
        tasks = assignments[d]
        if tasks:
            cost = tsp_approx_cost(tasks, dist_task, dist_drone_to_task, d)
            total_cost += cost

    return assignments, total_cost


# ========== 测试用例 ==========
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

    # 生成无人机起点
    num_drones = 4
    drone_positions = []
    h, w = grid.shape # type: ignore
    while len(drone_positions) < num_drones:
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        ix, iy = int(x), int(y)
        if grid[iy, ix] == 0:
            drone_positions.append([x, y])

    assignments, total_cost = assign_tasks_with_kmeanspp(task_points, drone_positions, grid)

    print("K-Means++ 分配结果：")
    for d, tasks in assignments.items():
        print(f"无人机{d}: 任务索引 {tasks} (数量 {len(tasks)})")
    print(f"整体预估路径代价: {total_cost:.2f}")