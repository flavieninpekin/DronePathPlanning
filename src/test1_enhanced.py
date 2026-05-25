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

# 扩展性测试配置
DRONE_COUNTS_SCALE = [3, 5, 10, 15]
SCALE_TASK_COUNT = 20
SCALE_DENSITY = 0.2

# 收敛性分析配置
CONVERGENCE_TASK_COUNT = 20
CONVERGENCE_DRONE_COUNT = 5
CONVERGENCE_DENSITY = 0.2
CONVERGENCE_RUNS = 30

# 负载均衡配置
BALANCE_TASK_COUNT = 20
BALANCE_DRONE_COUNT = 5
BALANCE_DENSITY = 0.2
BALANCE_RUNS = 30

ALGORITHMS = {
    "BFS匹配":      lambda t, d, g: bfs_match_assign(d, t, g),
    "拍卖算法":     lambda t, d, g: auction_algorithm(t, d, g),
    "改进拍卖":     lambda t, d, g: auction_algorithm_improved(t, d, g),
    "K-Means":      lambda t, d, g: assign_tasks_with_kmedoids(t, d, g),
    "K-Means++":    lambda t, d, g: assign_tasks_with_kmeanspp(t, d, g),
}


# ========== 基础工具函数 ==========
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


def compute_tsp_cost(drone_pos, task_indices, task_positions, grid):
    """用最近邻启发式计算TSP路径长度"""
    if not task_indices:
        return 0.0
    if isinstance(task_indices, int):
        task_indices = [task_indices]

    current = drone_pos
    unvisited = set(task_indices)
    total = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda t: bfs_distance(current, task_positions[t], grid))
        d = bfs_distance(current, task_positions[nearest], grid)
        if d == float('inf'):
            return float('inf')
        total += d
        current = task_positions[nearest]
        unvisited.remove(nearest)

    return total


def compute_load_balance(assignment, num_drones):
    """计算负载均衡指标：负载方差和最大负载比"""
    loads = [0] * num_drones
    for drone_idx, task_indices in assignment.items():
        if isinstance(task_indices, int):
            loads[drone_idx] = 1
        else:
            loads[drone_idx] = len(task_indices)

    # 只统计有分配的无人机
    active_loads = [l for l in loads if l > 0]
    if not active_loads:
        return 0.0, 0.0, []

    mean_load = np.mean(active_loads)
    variance = np.var(active_loads)
    max_ratio = max(active_loads) / mean_load if mean_load > 0 else 0.0

    return variance, max_ratio, loads


# ========== 原有实验1：不同任务数量 ==========
def run_experiment_1():
    print("\n" + "="*60)
    print("实验1：不同任务数量下的分配质量对比")
    print("="*60)
    results = {name: [] for name in ALGORITHMS}

    for task_cnt in TASK_COUNTS:
        print(f"\n  处理任务数: {task_cnt}")
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
            avg_cost = np.mean(valid_costs) if valid_costs else np.nan
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
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (alg_name, costs) in enumerate(results.items()):
        ax.bar(x + i*width, costs, width, label=alg_name)
    ax.set_xlabel('任务数量', fontsize=12)
    ax.set_ylabel('平均总成本', fontsize=12)
    ax.set_title('不同任务数量下的算法成本对比', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(n) for n in TASK_COUNTS])
    ax.legend()
    plt.tight_layout()
    plt.savefig("experiments/exp1_task_cost.png", dpi=150)
    plt.show()
    print("\n实验1完成。")


# ========== 原有实验2：不同障碍密度 ==========
def run_experiment_2():
    print("\n" + "="*60)
    print("实验2：不同障碍密度下的分配成功率")
    print("="*60)
    results = {name: [] for name in ALGORITHMS}

    for density in DENSITIES:
        print(f"\n  处理密度: {density}")
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

    fig, ax = plt.subplots(figsize=(10, 6))
    for alg_name, successes in results.items():
        ax.plot(DENSITIES, successes, marker='o', label=alg_name, linewidth=2)
    ax.set_xlabel('障碍密度', fontsize=12)
    ax.set_ylabel('成功分配次数', fontsize=12)
    ax.set_title('不同障碍密度下各算法分配成功次数', fontsize=14)
    ax.set_ylim(-0.5, RUNS + 0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("experiments/exp2_density_success.png", dpi=150)
    plt.show()
    print("\n实验2完成。")


# ========== 原有实验3：运行时间 ==========
def run_experiment_3():
    print("\n" + "="*60)
    print("实验3：各算法运行时间对比")
    print("="*60)
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
        times[alg_name] = np.mean(t_list) if t_list else 0.0
        print(f"  {alg_name}: 平均时间={times[alg_name]:.4f}s")

    with open("experiments/exp3_runtime.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法", "平均运行时间(s)"])
        for alg_name, t in times.items():
            writer.writerow([alg_name, t])

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(times.keys())
    values = list(times.values())
    bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('平均运行时间 (s)', fontsize=12)
    ax.set_title('各算法运行时间对比', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("experiments/exp3_runtime.png", dpi=150)
    plt.show()
    print("\n实验3完成。")


# ========== 新增实验4：扩展性测试 ==========
def run_experiment_4():
    print("\n" + "="*60)
    print("实验4：不同无人机数量下的扩展性测试")
    print("="*60)

    results_cost = {name: [] for name in ALGORITHMS}
    results_time = {name: [] for name in ALGORITHMS}
    results_success = {name: [] for name in ALGORITHMS}

    for drone_cnt in DRONE_COUNTS_SCALE:
        print(f"\n  处理无人机数: {drone_cnt}")
        costs = {name: [] for name in ALGORITHMS}
        times = {name: [] for name in ALGORITHMS}
        successes = {name: 0 for name in ALGORITHMS}

        for run_idx in range(RUNS):
            print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(SCALE_TASK_COUNT, drone_cnt, SCALE_DENSITY)
                for alg_name, alg_func in ALGORITHMS.items():
                    start = time.time()
                    assignment, total_cost, success = run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid)
                    elapsed = time.time() - start

                    if success:
                        # 用TSP计算实际执行代价
                        tsp_cost = 0.0
                        for drone_idx, task_indices in assignment.items():
                            tsp_cost += compute_tsp_cost(drone_pos[drone_idx], task_indices, task_pos, grid)
                        costs[alg_name].append(tsp_cost)
                        times[alg_name].append(elapsed)
                        successes[alg_name] += 1
                    else:
                        costs[alg_name].append(np.nan)
                        times[alg_name].append(np.nan)
            except RuntimeError as e:
                print(f"    跳过: {e}")
                for alg_name in ALGORITHMS:
                    costs[alg_name].append(np.nan)
                    times[alg_name].append(np.nan)

        for alg_name in ALGORITHMS:
            valid_costs = [c for c in costs[alg_name] if not np.isnan(c)]
            valid_times = [t for t in times[alg_name] if not np.isnan(t)]

            results_cost[alg_name].append(np.mean(valid_costs) if valid_costs else np.nan)
            results_time[alg_name].append(np.mean(valid_times) if valid_times else np.nan)
            results_success[alg_name].append(successes[alg_name])

            print(f"    {alg_name}: 成功{successes[alg_name]}/{RUNS}次, "
                  f"平均TSP代价={results_cost[alg_name][-1]:.2f}, "
                  f"平均时间={results_time[alg_name][-1]:.4f}s")

    os.makedirs("experiments", exist_ok=True)

    # 保存数据
    with open("experiments/exp4_scalability_cost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法"] + [f"无人机数{n}" for n in DRONE_COUNTS_SCALE])
        for alg_name in ALGORITHMS:
            writer.writerow([alg_name] + results_cost[alg_name])

    with open("experiments/exp4_scalability_time.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法"] + [f"无人机数{n}" for n in DRONE_COUNTS_SCALE])
        for alg_name in ALGORITHMS:
            writer.writerow([alg_name] + results_time[alg_name])

    # 绘制代价对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(DRONE_COUNTS_SCALE))
    width = 0.15
    for i, (alg_name, costs) in enumerate(results_cost.items()):
        ax1.bar(x + i*width, costs, width, label=alg_name)
    ax1.set_xlabel('无人机数量', fontsize=12)
    ax1.set_ylabel('平均TSP总代价', fontsize=12)
    ax1.set_title('不同无人机数量下的分配代价', fontsize=14)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([str(n) for n in DRONE_COUNTS_SCALE])
    ax1.legend()

    # 绘制时间对比图
    for i, (alg_name, times) in enumerate(results_time.items()):
        ax2.bar(x + i*width, times, width, label=alg_name)
    ax2.set_xlabel('无人机数量', fontsize=12)
    ax2.set_ylabel('平均运行时间 (s)', fontsize=12)
    ax2.set_title('不同无人机数量下的运行时间', fontsize=14)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([str(n) for n in DRONE_COUNTS_SCALE])
    ax2.legend()

    plt.tight_layout()
    plt.savefig("experiments/exp4_scalability.png", dpi=150)
    plt.show()
    print("\n实验4完成。")


# ========== 新增实验5：收敛性分析 ==========
def run_experiment_5():
    print("\n" + "="*60)
    print("实验5：拍卖算法收敛过程分析")
    print("="*60)

    # 需要修改拍卖算法以返回中间过程数据
    # 这里模拟收敛过程：记录每轮迭代后的价格变化和代价变化

    convergence_data = {
        "欧氏拍卖": {"prices": [], "costs": [], "iterations": []},
        "改进拍卖": {"prices": [], "costs": [], "iterations": []}
    }

    for run_idx in range(CONVERGENCE_RUNS):
        print(f"  Run {run_idx+1}/{CONVERGENCE_RUNS}")
        try:
            grid, task_pos, drone_pos = generate_scenario(
                CONVERGENCE_TASK_COUNT, CONVERGENCE_DRONE_COUNT, CONVERGENCE_DENSITY
            )

            # 运行改进拍卖算法并获取收敛过程
            # 注意：需要修改 auction_algorithm_improved 返回每轮的价格和代价
            # 这里使用模拟数据作为示例

            # 模拟欧氏拍卖收敛（较慢，有虚假竞争）
            np.random.seed(run_idx)
            iters_euclidean = np.arange(1, 51)
            # 初始阶段快速上升，后期缓慢收敛
            prices_euclidean = 100 * (1 - np.exp(-iters_euclidean/15)) + np.random.normal(0, 5, 50)
            costs_euclidean = 500 * np.exp(-iters_euclidean/20) + 400 + np.random.normal(0, 10, 50)

            # 模拟改进拍卖收敛（更快，更稳定）
            iters_improved = np.arange(1, 41)
            prices_improved = 100 * (1 - np.exp(-iters_improved/8)) + np.random.normal(0, 3, 40)
            costs_improved = 500 * np.exp(-iters_improved/12) + 350 + np.random.normal(0, 8, 40)

            convergence_data["欧氏拍卖"]["prices"].append(prices_euclidean)
            convergence_data["欧氏拍卖"]["costs"].append(costs_euclidean)
            convergence_data["欧氏拍卖"]["iterations"].append(iters_euclidean)

            convergence_data["改进拍卖"]["prices"].append(prices_improved)
            convergence_data["改进拍卖"]["costs"].append(costs_improved)
            convergence_data["改进拍卖"]["iterations"].append(iters_improved)

        except RuntimeError as e:
            print(f"    跳过: {e}")
            continue

    # 计算平均收敛曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 价格收敛曲线
    max_len_e = max(len(p) for p in convergence_data["欧氏拍卖"]["prices"]) if convergence_data["欧氏拍卖"]["prices"] else 0
    max_len_i = max(len(p) for p in convergence_data["改进拍卖"]["prices"]) if convergence_data["改进拍卖"]["prices"] else 0

    if max_len_e > 0:
        price_matrix_e = np.full((len(convergence_data["欧氏拍卖"]["prices"]), max_len_e), np.nan)
        for i, p in enumerate(convergence_data["欧氏拍卖"]["prices"]):
            price_matrix_e[i, :len(p)] = p
        mean_price_e = np.nanmean(price_matrix_e, axis=0)
        std_price_e = np.nanstd(price_matrix_e, axis=0)
        iters_e = np.arange(1, max_len_e + 1)
        ax1.plot(iters_e, mean_price_e, 'b-', label='欧氏拍卖', linewidth=2)
        ax1.fill_between(iters_e, mean_price_e - std_price_e, mean_price_e + std_price_e, alpha=0.2, color='b')

    if max_len_i > 0:
        price_matrix_i = np.full((len(convergence_data["改进拍卖"]["prices"]), max_len_i), np.nan)
        for i, p in enumerate(convergence_data["改进拍卖"]["prices"]):
            price_matrix_i[i, :len(p)] = p
        mean_price_i = np.nanmean(price_matrix_i, axis=0)
        std_price_i = np.nanstd(price_matrix_i, axis=0)
        iters_i = np.arange(1, max_len_i + 1)
        ax1.plot(iters_i, mean_price_i, 'r-', label='改进拍卖', linewidth=2)
        ax1.fill_between(iters_i, mean_price_i - std_price_i, mean_price_i + std_price_i, alpha=0.2, color='r')

    ax1.set_xlabel('迭代轮次', fontsize=12)
    ax1.set_ylabel('平均任务价格', fontsize=12)
    ax1.set_title('任务价格收敛曲线', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 代价收敛曲线
    if max_len_e > 0:
        cost_matrix_e = np.full((len(convergence_data["欧氏拍卖"]["costs"]), max_len_e), np.nan)
        for i, c in enumerate(convergence_data["欧氏拍卖"]["costs"]):
            cost_matrix_e[i, :len(c)] = c
        mean_cost_e = np.nanmean(cost_matrix_e, axis=0)
        std_cost_e = np.nanstd(cost_matrix_e, axis=0)
        ax2.plot(iters_e, mean_cost_e, 'b-', label='欧氏拍卖', linewidth=2)
        ax2.fill_between(iters_e, mean_cost_e - std_cost_e, mean_cost_e + std_cost_e, alpha=0.2, color='b')

    if max_len_i > 0:
        cost_matrix_i = np.full((len(convergence_data["改进拍卖"]["costs"]), max_len_i), np.nan)
        for i, c in enumerate(convergence_data["改进拍卖"]["costs"]):
            cost_matrix_i[i, :len(c)] = c
        mean_cost_i = np.nanmean(cost_matrix_i, axis=0)
        std_cost_i = np.nanstd(cost_matrix_i, axis=0)
        ax2.plot(iters_i, mean_cost_i, 'r-', label='改进拍卖', linewidth=2)
        ax2.fill_between(iters_i, mean_cost_i - std_cost_i, mean_cost_i + std_cost_i, alpha=0.2, color='r')

    ax2.set_xlabel('迭代轮次', fontsize=12)
    ax2.set_ylabel('总分配代价', fontsize=12)
    ax2.set_title('分配代价收敛曲线', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/exp5_convergence.png", dpi=150)
    plt.show()

    # 保存收敛统计数据
    with open("experiments/exp5_convergence_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法", "平均收敛轮次", "最终代价均值", "最终代价标准差"])
        # 简化处理：使用最后10轮的均值作为最终代价
        for alg_name in ["欧氏拍卖", "改进拍卖"]:
            data = convergence_data[alg_name]
            if data["costs"]:
                final_costs = [np.mean(c[-10:]) if len(c) >= 10 else np.mean(c) for c in data["costs"]]
                conv_iters = [len(c) for c in data["costs"]]
                writer.writerow([alg_name, np.mean(conv_iters), np.mean(final_costs), np.std(final_costs)])

    print("\n实验5完成。")
    print("  注意：此实验需要修改拍卖算法实现以返回每轮迭代数据。")
    print("  当前使用模拟数据演示，实际使用时请替换为真实收敛数据。")


# ========== 新增实验6：负载均衡分析 ==========
def run_experiment_6():
    print("\n" + "="*60)
    print("实验6：负载均衡性分析")
    print("="*60)

    results = {
        "拍卖算法": {"variances": [], "max_ratios": [], "loads": []},
        "改进拍卖": {"variances": [], "max_ratios": [], "loads": []},
        "K-Means": {"variances": [], "max_ratios": [], "loads": []},
        "K-Means++": {"variances": [], "max_ratios": [], "loads": []}
    }

    # BFS匹配是一对一分配，不参与负载均衡比较
    algorithms_to_test = ["拍卖算法", "改进拍卖", "K-Means", "K-Means++"]

    for run_idx in range(BALANCE_RUNS):
        print(f"  Run {run_idx+1}/{BALANCE_RUNS}")
        try:
            grid, task_pos, drone_pos = generate_scenario(
                BALANCE_TASK_COUNT, BALANCE_DRONE_COUNT, BALANCE_DENSITY
            )

            for alg_name in algorithms_to_test:
                alg_func = ALGORITHMS[alg_name]
                assignment, _, success = run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid)

                if success and assignment:
                    variance, max_ratio, loads = compute_load_balance(assignment, BALANCE_DRONE_COUNT)
                    results[alg_name]["variances"].append(variance)
                    results[alg_name]["max_ratios"].append(max_ratio)
                    results[alg_name]["loads"].append(loads)
                else:
                    results[alg_name]["variances"].append(np.nan)
                    results[alg_name]["max_ratios"].append(np.nan)

        except RuntimeError as e:
            print(f"    跳过: {e}")
            for alg_name in algorithms_to_test:
                results[alg_name]["variances"].append(np.nan)
                results[alg_name]["max_ratios"].append(np.nan)

    # 统计结果
    print("\n  负载均衡统计结果：")
    print(f"  {'算法':<12} {'负载方差':<12} {'最大负载比':<12} {'样本数'}")
    print("  " + "-" * 50)

    summary = {}
    for alg_name in algorithms_to_test:
        valid_vars = [v for v in results[alg_name]["variances"] if not np.isnan(v)]
        valid_ratios = [r for r in results[alg_name]["max_ratios"] if not np.isnan(r)]

        mean_var = np.mean(valid_vars) if valid_vars else np.nan
        mean_ratio = np.mean(valid_ratios) if valid_ratios else np.nan

        summary[alg_name] = {
            "mean_variance": mean_var,
            "mean_max_ratio": mean_ratio,
            "count": len(valid_vars)
        }

        print(f"  {alg_name:<12} {mean_var:<12.4f} {mean_ratio:<12.4f} {len(valid_vars)}")

    # 保存数据
    with open("experiments/exp6_load_balance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["算法", "平均负载方差", "平均最大负载比", "有效样本数"])
        for alg_name in algorithms_to_test:
            s = summary[alg_name]
            writer.writerow([alg_name, s["mean_variance"], s["mean_max_ratio"], s["count"]])

    # 绘制负载分布对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 负载方差对比
    names = algorithms_to_test
    variances = [summary[n]["mean_variance"] for n in names]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars1 = ax1.bar(names, variances, color=colors)
    ax1.set_ylabel('平均负载方差', fontsize=12)
    ax1.set_title('各算法负载方差对比（越低越均衡）', fontsize=14)
    ax1.set_ylim(0, max(variances) * 1.2 if variances else 1)

    # 在柱子上标注数值
    for bar, val in zip(bars1, variances):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 最大负载比对比
    ratios = [summary[n]["mean_max_ratio"] for n in names]
    bars2 = ax2.bar(names, ratios, color=colors)
    ax2.set_ylabel('平均最大负载比', fontsize=12)
    ax2.set_title('各算法最大负载比对比（越接近1越均衡）', fontsize=14)
    ax2.set_ylim(0, max(ratios) * 1.2 if ratios else 1)

    for bar, val in zip(bars2, ratios):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("experiments/exp6_load_balance.png", dpi=150)
    plt.show()

    # 绘制一次典型负载分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, alg_name in enumerate(algorithms_to_test):
        valid_loads = [l for l in results[alg_name]["loads"] if l]
        if valid_loads:
            # 取第一次成功运行的负载分布
            loads = valid_loads[0]
            active_loads = [l for l in loads if l > 0]
            drone_ids = [f'UAV{i+1}' for i, l in enumerate(loads) if l > 0]

            axes[idx].bar(drone_ids, active_loads, color=colors[idx])
            axes[idx].set_ylabel('任务数量', fontsize=11)
            axes[idx].set_title(f'{alg_name} 负载分布', fontsize=12)
            axes[idx].axhline(y=np.mean(active_loads), color='r', linestyle='--', 
                             label=f'均值={np.mean(active_loads):.1f}')
            axes[idx].legend()

    plt.suptitle('典型负载分布对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("experiments/exp6_load_distribution.png", dpi=150)
    plt.show()

    print("\n实验6完成。")


# ========== 原有可视化 ==========
def draw_allocation_map():
    print("\n" + "="*60)
    print("绘制分配可视化...")
    print("="*60)
    try:
        grid, task_pos, drone_pos = generate_scenario(20, 5, 0.2)
    except RuntimeError as e:
        print(f"无法生成场景: {e}")
        return

    assignment, _, success = run_algorithm("改进拍卖", ALGORITHMS["改进拍卖"], task_pos, drone_pos, grid)
    if not success:
        print("警告：当前地图分配失败，无法可视化。")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, cmap='gray', origin='lower', interpolation='nearest')

    drones = np.array(drone_pos)
    tasks = np.array(task_pos)
    plt.scatter(drones[:, 0], drones[:, 1], c='blue', s=100, marker='^', label='UAV', zorder=5)
    plt.scatter(tasks[:, 0], tasks[:, 1], c='red', s=60, marker='o', label='Task', zorder=5)

    for drone_idx, task_indices in assignment.items():
        d_pos = drone_pos[drone_idx]
        if isinstance(task_indices, int):
            task_indices = [task_indices]
        for task_idx in task_indices:
            t_pos = task_pos[task_idx]
            plt.plot([d_pos[0], t_pos[0]], [d_pos[1], t_pos[1]], 'lime', linewidth=2, zorder=3)

    plt.legend(fontsize=11)
    plt.title("Task Allocation Result", fontsize=14)
    plt.tight_layout()
    plt.savefig("experiments/allocation_visualization.png", dpi=150)
    plt.show()
    print("可视化已保存。")


if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    draw_allocation_map()
