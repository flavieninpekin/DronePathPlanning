import numpy as np
import time
import csv
import os
import sys
import random
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_generator.MapGenerator import generate_map_with_path
from map_generator.TaskPointGeneration import generate_task_points

from algorithms.TaskAuction import auction_algorithm
from algorithms.TaskAuctionImprove import auction_algorithm_improved
from algorithms.k_means import assign_tasks_with_kmedoids
from algorithms.k_meanspp import assign_tasks_with_kmeanspp

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


# ========== 配置区 ==========
MAP_SIZE = (100, 100)
TASK_COUNTS = [15, 20, 30]
DRONE_COUNT = 5
EXP1_DENSITY = 0.2
DENSITIES = [0.1, 0.2, 0.3, 0.4]
EXP2_TASK_COUNT = 20
EXP2_DRONE_COUNT = 5
RUNS = 10

ALGORITHMS = {
    "BFS匹配":      lambda t, d, g: bfs_match_assign(d, t, g),
    "拍卖算法":     lambda t, d, g: auction_algorithm(t, d, g),
    "改进拍卖":     lambda t, d, g: auction_algorithm_improved(t, d, g),
    "K-Means":      lambda t, d, g: assign_tasks_with_kmedoids(t, d, g),
    "K-Means++":    lambda t, d, g: assign_tasks_with_kmeanspp(t, d, g),
}


def bfs_distance(start, goal, grid):
    """BFS最短路径，不可达返回inf"""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))

    if not (0 <= sx < cols and 0 <= sy < rows and grid[sy, sx] == 0):
        return float('inf')
    if not (0 <= gx < cols and 0 <= gy < rows and grid[gy, gx] == 0):
        return float('inf')

    queue = deque([(sx, sy, 0)])
    visited[sy, sx] = True

    while queue:
        x, y, d = queue.popleft()
        if (x, y) == (gx, gy):
            return d
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                queue.append((nx, ny, d+1))
    return float('inf')


def bfs_match_assign(drone_positions, task_positions, grid):
    """贪心BFS匹配"""
    num_drones = len(drone_positions)
    num_tasks = len(task_positions)
    dist_matrix = np.zeros((num_drones, num_tasks))

    for i in range(num_drones):
        for j in range(num_tasks):
            dist_matrix[i, j] = bfs_distance(drone_positions[i], task_positions[j], grid)

    assigned_tasks = set()
    assignment = {}
    total_cost = 0.0
    all_reachable = True

    for i in range(num_drones):
        unassigned = [j for j in range(num_tasks) if j not in assigned_tasks]
        if not unassigned:
            break
        best_j = min(unassigned, key=lambda j: dist_matrix[i, j])
        d = dist_matrix[i, best_j]
        if d == float('inf'):
            all_reachable = False
            continue
        assignment[i] = best_j
        assigned_tasks.add(best_j)
        total_cost += d

    success = all_reachable and (len(assignment) == min(num_drones, num_tasks))
    return assignment, total_cost, success


def get_reachable_cells(grid, start=(0, 0)):
    """BFS获取从start出发所有可达的空闲格子，返回 [(x,y),...]"""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([start])
    visited[start[1], start[0]] = True
    reachable = [start]
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                reachable.append((nx, ny))
                queue.append((nx, ny))
    return reachable


def generate_scenario(num_tasks, num_drones, density, max_attempts=100):
    """根据密度动态调整尝试次数"""
    for attempt in range(max_attempts):
        grid = generate_map_with_path(MAP_SIZE, density, dim=2, channel_expansion=0)
        reachable = get_reachable_cells(grid, (0, 0))
        
        if len(reachable) >= num_tasks + num_drones:
            selected = random.sample(reachable, num_tasks + num_drones)
            points = [[float(x), float(y)] for x, y in selected]
            return grid, points[:num_tasks], points[num_tasks:]
        
        if attempt % 20 == 19:
            print(f"      尝试{attempt+1}次仍未找到足够大的连通分量，继续...")
    
    raise RuntimeError(f"无法生成有效场景，{max_attempts}次尝试后连通分量仍不足")


def check_reachable(assignment, drone_positions, task_positions, grid):
    """检查分配结果中所有任务是否可达"""
    if not assignment:
        return True
    for drone_idx, task_idx in assignment.items():
        tasks = task_idx if isinstance(task_idx, list) else [task_idx]
        for t in tasks:
            if bfs_distance(drone_positions[drone_idx], task_positions[t], grid) == float('inf'):
                return False
    return True


def run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid):
    """统一运行算法并解析结果"""
    try:
        result = alg_func(task_pos, drone_pos, grid)
        if isinstance(result, tuple):
            assignment, total_cost = result[0], result[1] if len(result) > 1 else 0.0
        else:
            assignment, total_cost = result, 0.0

        success = check_reachable(assignment, drone_pos, task_pos, grid)
        return assignment, total_cost, success
    except Exception as e:
        print(f"  算法 {alg_name} 出错: {e}")
        return {}, 0.0, False


# ========== 实验1 ==========
def run_experiment_1():
    print("开始实验1：不同任务数量...")
    results = {name: [] for name in ALGORITHMS}

    for task_cnt in TASK_COUNTS:
        print(f"  处理任务数: {task_cnt}")
        costs = {name: [] for name in ALGORITHMS}

        for run_idx in range(RUNS):
            print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(task_cnt, DRONE_COUNT, EXP1_DENSITY)
                for alg_name, alg_func in ALGORITHMS.items():
                    _, total_cost, success = run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid)
                    costs[alg_name].append(total_cost if success else np.nan)
            except RuntimeError as e:
                print(f"    跳过: {e}")
                for alg_name in ALGORITHMS:
                    costs[alg_name].append(np.nan)

        for alg_name in ALGORITHMS:
            valid_costs = [c for c in costs[alg_name] if not np.isnan(c)]
            if valid_costs:
                avg_cost = np.mean(valid_costs)
            else:
                avg_cost = np.nan
            results[alg_name].append(avg_cost)
            print(f"    {alg_name}: 成功{len(valid_costs)}/{RUNS}次, 平均成本={avg_cost:.2f}")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp1_task_count_cost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法"] + [f"任务数{n}" for n in TASK_COUNTS])
        for alg_name in ALGORITHMS:
            writer.writerow([alg_name] + results[alg_name])

    x = np.arange(len(TASK_COUNTS))
    width = 0.15
    fig, ax = plt.subplots()
    for i, (alg_name, costs) in enumerate(results.items()):
        ax.bar(x + i*width, costs, width, label=alg_name)
    ax.set_xlabel('任务数量')
    ax.set_ylabel('平均总成本')
    ax.set_title('不同任务数量下的算法成本对比')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(n) for n in TASK_COUNTS])
    ax.legend()
    plt.tight_layout()
    plt.savefig("experiments/exp1_task_cost.png", dpi=150)
    plt.show()
    print("实验1完成。")


# ========== 实验2 ==========
def run_experiment_2():
    print("开始实验2：不同障碍密度...")
    results = {name: [] for name in ALGORITHMS}

    for density in DENSITIES:
        print(f"  处理密度: {density}")
        success_counts = {name: 0 for name in ALGORITHMS}

        for run_idx in range(RUNS):
            print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(EXP2_TASK_COUNT, EXP2_DRONE_COUNT, density)
                for alg_name, alg_func in ALGORITHMS.items():
                    _, _, success = run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid)
                    if success:
                        success_counts[alg_name] += 1
            except RuntimeError as e:
                print(f"    跳过: {e}")

        for alg_name in ALGORITHMS:
            results[alg_name].append(success_counts[alg_name])
            print(f"    {alg_name}: 成功{success_counts[alg_name]}/{RUNS}次")

    with open("experiments/exp2_density_success.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法"] + [f"密度{d}" for d in DENSITIES])
        for alg_name in ALGORITHMS:
            writer.writerow([alg_name] + results[alg_name])

    fig, ax = plt.subplots()
    for alg_name, successes in results.items():
        ax.plot(DENSITIES, successes, marker='o', label=alg_name)
    ax.set_xlabel('障碍密度')
    ax.set_ylabel('成功分配次数')
    ax.set_title('不同障碍密度下各算法分配成功次数')
    ax.legend()
    plt.savefig("experiments/exp2_density_success.png", dpi=150)
    plt.show()
    print("实验2完成。")


# ========== 实验3 ==========
def run_experiment_3():
    print("开始实验3：运行时间...")
    times = {}

    for alg_name, alg_func in ALGORITHMS.items():
        t_list = []
        for run_idx in range(RUNS):
            try:
                grid, task_pos, drone_pos = generate_scenario(20, DRONE_COUNT, 0.2)
                start = time.time()
                run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid)
                t_list.append(time.time() - start)
            except RuntimeError:
                continue
        if t_list:
            times[alg_name] = np.mean(t_list)
        else:
            times[alg_name] = 0.0
        print(f"  {alg_name}: 平均时间={times[alg_name]:.4f}s")

    with open("experiments/exp3_runtime.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法", "平均运行时间(s)"])
        for alg_name, t in times.items():
            writer.writerow([alg_name, t])

    fig, ax = plt.subplots()
    names = list(times.keys())
    values = list(times.values())
    ax.bar(names, values)
    ax.set_ylabel('平均运行时间 (s)')
    ax.set_title('各算法运行时间对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("experiments/exp3_runtime.png", dpi=150)
    plt.show()
    print("实验3完成。")


# ========== 实验4 ==========
def draw_allocation_map():
    print("绘制分配可视化...")
    try:
        grid, task_pos, drone_pos = generate_scenario(20, 5, 0.2)
    except RuntimeError as e:
        print(f"无法生成场景: {e}")
        return

    assignment, _, success = run_algorithm("拍卖算法", ALGORITHMS["拍卖算法"], task_pos, drone_pos, grid)
    if not success:
        print("警告：当前地图分配失败，无法可视化。")
        return

    plt.figure(figsize=(8,8))
    plt.imshow(grid.T, cmap='gray', origin='lower', interpolation='nearest')

    drones = np.array(drone_pos)
    tasks = np.array(task_pos)
    plt.scatter(drones[:,0], drones[:,1], c='blue', s=100, marker='^', label='UAV')
    plt.scatter(tasks[:,0], tasks[:,1], c='red', s=60, marker='o', label='Task')

    for drone_idx, task_indices in assignment.items():
        d_pos = drone_pos[drone_idx]
        if isinstance(task_indices, int):
            task_indices = [task_indices]
        for task_idx in task_indices:
            t_pos = task_pos[task_idx]
            plt.plot([d_pos[0], t_pos[0]], [d_pos[1], t_pos[1]], 'lime', linewidth=2)

    plt.legend()
    plt.title("Task Allocation Result")
    plt.tight_layout()
    plt.savefig("experiments/allocation_visualization.png", dpi=150)
    plt.show()
    print("可视化已保存。")


if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    draw_allocation_map()
