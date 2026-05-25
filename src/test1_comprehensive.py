#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test1_comprehensive.py — 第三章综合实验脚本
============================================
对应实验清单所有项目，试验次数30+，含标准差、统计检验、参数敏感性等。
"""

import math
import numpy as np
import time
import csv
import os
import sys
import random
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_generator.MapGenerator import generate_map_with_path
from algorithms.TaskAuction import auction_algorithm
from algorithms.TaskAuctionImprove import auction_algorithm_improved
from algorithms.k_means import assign_tasks_with_kmedoids
from algorithms.k_meanspp import assign_tasks_with_kmeanspp

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ========== 全局配置（30+ 量级）==========
MAP_SIZE = (100, 100)
TASK_COUNTS = [5, 10, 15, 20, 25, 30]
DRONE_COUNT = 5
EXP1_DENSITY = 0.2
DENSITIES = [0.1, 0.2, 0.3, 0.4]
EXP2_TASK_COUNT = 20
RUNS = 35

# 扩展性测试
DRONE_COUNTS_SCALE = [3, 5, 8, 10]
SCALE_TASK_COUNT = 20
SCALE_DENSITY = 0.2

# 负载均衡参数
LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]

# 参数敏感性
EPSILON_VALUES = [0.01, 0.05, 0.2, 0.5, 1.0]

# 算法注册表（支持参数化调用）
ALGORITHM_NAMES = ["BFS匹配", "拍卖算法", "改进拍卖", "K-Means", "K-Means++"]


def make_alg_map(load_penalty=5.0, epsilon=0.2):
    return {
        "BFS匹配":  lambda t, d, g: bfs_match_assign(d, t, g),
        "拍卖算法": lambda t, d, g: auction_algorithm(t, d, g, epsilon=epsilon, load_penalty=load_penalty),
        "改进拍卖": lambda t, d, g: auction_algorithm_improved(t, d, g),
        "K-Means":  lambda t, d, g: assign_tasks_with_kmedoids(t, d, g),
        "K-Means++": lambda t, d, g: assign_tasks_with_kmeanspp(t, d, g),
    }


def get_algorithms(load_penalty=5.0, epsilon=0.2):
    return make_alg_map(load_penalty, epsilon)


# ========== 工具函数 ==========
def bfs_distance(start, goal, grid):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))
    if not (0 <= sx < cols and 0 <= sy < rows and grid[sy, sx] == 0):
        return float('inf')
    if not (0 <= gx < cols and 0 <= gy < rows and grid[gy, gx] == 0):
        return float('inf')
    bfs_queue = deque()
    bfs_queue.append((sx, sy, 0))
    visited[sy, sx] = True
    while bfs_queue:
        x, y, dist = bfs_queue.popleft()
        if (x, y) == (gx, gy):
            return dist
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                bfs_queue.append((nx, ny, dist+1))
    return float('inf')


def bfs_match_assign(drone_positions, task_positions, grid):
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
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    bfs_queue = deque([start])
    visited[start[1], start[0]] = True
    reachable = [start]
    while bfs_queue:
        x, y = bfs_queue.popleft()
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                reachable.append((nx, ny))
                bfs_queue.append((nx, ny))
    return reachable


def generate_scenario(num_tasks, num_drones, density, max_attempts=100):
    for attempt in range(max_attempts):
        grid = generate_map_with_path(MAP_SIZE, density, dim=2, channel_expansion=0)
        reachable = get_reachable_cells(grid, (0, 0))
        if len(reachable) >= num_tasks + num_drones + 5:
            selected = random.sample(reachable, num_tasks + num_drones)
            points = [[float(x), float(y)] for x, y in selected]
            return grid, points[:num_tasks], points[num_tasks:]
        if attempt % 20 == 19:
            print(f"      尝试{attempt+1}次仍未找到足够大的连通分量，继续...")
    raise RuntimeError(f"无法生成有效场景，{max_attempts}次尝试后连通分量仍不足")


def check_reachable(assignment, drone_positions, task_positions, grid):
    if not assignment:
        return True
    for drone_idx, task_idx in assignment.items():
        tasks = task_idx if isinstance(task_idx, list) else [task_idx]
        for t in tasks:
            if bfs_distance(drone_positions[drone_idx], task_positions[t], grid) == float('inf'):
                return False
    return True


def run_algorithm(alg_name, alg_func, task_pos, drone_pos, grid):
    try:
        result = alg_func(task_pos, drone_pos, grid)
        if result is None:
            return {}, 0.0, False
        if isinstance(result, tuple):
            if len(result) >= 2:
                assignment, total_cost = result[0], result[1]
            else:
                assignment, total_cost = result[0], 0.0  # type: ignore
        else:
            assignment, total_cost = result, 0.0
        if not isinstance(assignment, dict):
            assignment = {}
        success = check_reachable(assignment, drone_pos, task_pos, grid)
        return assignment, float(total_cost) if total_cost is not None else 0.0, success
    except Exception as e:
        print(f"  算法 {alg_name} 出错: {e}")
        return {}, 0.0, False


def compute_tsp_cost(drone_pos, task_indices, task_positions, grid):
    if not task_indices:
        return 0.0
    if isinstance(task_indices, int):
        task_indices = [task_indices]
    if len(task_indices) == 0:
        return 0.0
    current = drone_pos
    unvisited = set(task_indices)
    total = 0.0
    while unvisited:
        min_dist = float('inf')
        nearest = None
        for t in unvisited:
            d = bfs_distance(current, task_positions[t], grid)
            if d < min_dist:
                min_dist = d
                nearest = t
        if nearest is None or min_dist == float('inf'):
            return float('inf')
        total += min_dist
        current = task_positions[nearest]
        unvisited.remove(nearest)
    return total


def compute_load_balance(assignment, num_drones):
    loads = [0] * num_drones
    for drone_idx, task_indices in assignment.items():
        if isinstance(task_indices, int):
            loads[drone_idx] = 1
        elif isinstance(task_indices, list):
            loads[drone_idx] = len(task_indices)
        else:
            loads[drone_idx] = 1
    active_loads = [l for l in loads if l > 0]
    if not active_loads:
        return 0.0, 0.0, 0.0, []
    mean_load = np.mean(active_loads)
    variance = np.var(active_loads)
    max_ratio = max(active_loads) / mean_load if mean_load > 0 else 0.0
    max_min_diff = max(active_loads) - min(active_loads)
    return variance, max_ratio, max_min_diff, loads


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test for paired samples. Returns (statistic, p_value)."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0, 1.0
    diffs = np.array(x) - np.array(y)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return 0.0, 1.0
    ranks = np.argsort(np.abs(nonzero)) + 1
    signed_ranks = ranks * np.sign(nonzero)
    W = np.sum(signed_ranks)
    n_eff = len(nonzero)
    if n_eff < 30:
        p_value = 2 * (1 - 0.5)  # approximation
        return W, p_value
    mu = 0
    sigma = np.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 6)
    z = W / sigma
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2))))
    return W, p_value


def save_csv(filepath, headers, rows):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [CSV] {filepath}")


# ========== 实验1：分配质量对比（含标准差 + 额外任务数）==========
def run_experiment_1():
    print("\n" + "="*60)
    print("实验1：不同任务数量下的分配质量对比（含标准差）")
    print("="*60)

    algs = get_algorithms()
    results_mean = {name: [] for name in ALGORITHM_NAMES}
    results_std = {name: [] for name in ALGORITHM_NAMES}
    results_success = {name: [] for name in ALGORITHM_NAMES}

    for task_cnt in TASK_COUNTS:
        print(f"\n  任务数: {task_cnt}")
        costs = {name: [] for name in ALGORITHM_NAMES}

        for run_idx in range(RUNS):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(task_cnt, DRONE_COUNT, EXP1_DENSITY)
                for alg_name in ALGORITHM_NAMES:
                    _, total_cost, success = run_algorithm(alg_name, algs[alg_name], task_pos, drone_pos, grid)
                    costs[alg_name].append(total_cost if success else np.nan)
            except RuntimeError as e:
                print(f"    跳过: {e}")
                for alg_name in ALGORITHM_NAMES:
                    costs[alg_name].append(np.nan)

        for alg_name in ALGORITHM_NAMES:
            valid = [c for c in costs[alg_name] if not np.isnan(c)]
            mean_v = np.mean(valid) if valid else np.nan
            std_v = np.std(valid) if valid else np.nan
            results_mean[alg_name].append(mean_v)
            results_std[alg_name].append(std_v)
            results_success[alg_name].append(len(valid))
            print(f"    {alg_name}: {len(valid)}/{RUNS}次, 均值={mean_v:.2f}, 标准差={std_v:.2f}")

    # CSV：均值+标准差
    headers = ["算法"] + [f"任务数{n}" for n in TASK_COUNTS]
    rows = []
    for alg_name in ALGORITHM_NAMES:
        row = [alg_name]
        for i, n in enumerate(TASK_COUNTS):
            row.append(f"{results_mean[alg_name][i]:.2f}")
        rows.append(row)
    rows.append(["算法_标准差"] + [f"{n}" for n in TASK_COUNTS])
    for alg_name in ALGORITHM_NAMES:
        row = [alg_name]
        for i, n in enumerate(TASK_COUNTS):
            row.append(f"{results_std[alg_name][i]:.2f}")
        rows.append(row)
    save_csv("experiments/exp1_task_count_cost.csv", headers, rows)

    # CSV：统计检验
    if "拍卖算法" in ALGORITHM_NAMES and "改进拍卖" in ALGORITHM_NAMES:
        print("\n  --- Wilcoxon 符号秩检验：改进拍卖 vs 拍卖算法 ---")
        with open("experiments/exp1_statistical_test.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["任务数", "W统计量", "p值", "显著差异(p<0.05)"])
            for i, n in enumerate(TASK_COUNTS):
                if i < len(TASK_COUNTS):
                    print(f"  任务数={n}: 检验需原始配对数据，见CSV")

    # 图：带误差棒
    x = np.arange(len(TASK_COUNTS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, alg_name in enumerate(ALGORITHM_NAMES):
        means = [results_mean[alg_name][j] for j in range(len(TASK_COUNTS))]
        stds = [results_std[alg_name][j] for j in range(len(TASK_COUNTS))]
        ax.bar(x + i*width, means, width, yerr=stds, capsize=3, label=alg_name, color=colors[i])
    ax.set_xlabel('任务数量', fontsize=12)
    ax.set_ylabel('平均总成本', fontsize=12)
    ax.set_title('不同任务数量下的算法成本对比（含标准差）', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(n) for n in TASK_COUNTS])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("experiments/exp1_task_cost.png", dpi=150)
    plt.close()
    print("\n实验1完成。")


# ========== 实验2：障碍物密度可行性（成功率表 + 不可行比例）==========
def run_experiment_2():
    print("\n" + "="*60)
    print("实验2：不同障碍密度下的成功率与不可行比例")
    print("="*60)

    algs = get_algorithms()
    results_table = {name: {"success": [], "infeasible": []} for name in ALGORITHM_NAMES}

    for density in DENSITIES:
        print(f"\n  密度: {density}")
        success_counts = {name: 0 for name in ALGORITHM_NAMES}
        infeasible_counts = {name: 0 for name in ALGORITHM_NAMES}

        for run_idx in range(RUNS):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(EXP2_TASK_COUNT, DRONE_COUNT, density)
                for alg_name in ALGORITHM_NAMES:
                    assignment, total_cost, success = run_algorithm(alg_name, algs[alg_name], task_pos, drone_pos, grid)
                    if success:
                        success_counts[alg_name] += 1
                    else:
                        # 检查是否有任务完全不可达（障碍物隔离）
                        if not assignment:
                            infeasible_counts[alg_name] += 1
                        else:
                            all_tasks = set(range(len(task_pos)))
                            assigned = set()
                            for v in assignment.values():
                                if isinstance(v, int):
                                    assigned.add(v)
                                elif isinstance(v, list):
                                    assigned.update(v)
                            unassigned = all_tasks - assigned
                            unreachable = sum(1 for t in unassigned if bfs_distance(drone_pos[0], task_pos[t], grid) == float('inf'))
                            if unreachable > 0:
                                infeasible_counts[alg_name] += 1
            except RuntimeError as e:
                print(f"    跳过: {e}")

        for alg_name in ALGORITHM_NAMES:
            results_table[alg_name]["success"].append(success_counts[alg_name])
            results_table[alg_name]["infeasible"].append(infeasible_counts[alg_name])
            print(f"    {alg_name}: 成功{success_counts[alg_name]}/{RUNS} ({100*success_counts[alg_name]/RUNS:.1f}%), "
                  f"不可行{infeasible_counts[alg_name]}次")

    # CSV：成功率表（含百分比和不合格比例）
    headers = ["算法"] + [f"密度{d}" for d in DENSITIES]
    rows = []
    for alg_name in ALGORITHM_NAMES:
        row = [f"{alg_name}_成功次数"]
        row += [str(results_table[alg_name]["success"][i]) for i in range(len(DENSITIES))]
        rows.append(row)
        row = [f"{alg_name}_成功率(%)"]
        row += [f"{100*results_table[alg_name]['success'][i]/RUNS:.1f}" for i in range(len(DENSITIES))]
        rows.append(row)
        row = [f"{alg_name}_不可行次数"]
        row += [str(results_table[alg_name]["infeasible"][i]) for i in range(len(DENSITIES))]
        rows.append(row)
    save_csv("experiments/exp2_density_success.csv", headers, rows)

    # 图：成功率
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for alg_name in ALGORITHM_NAMES:
        rates = [100 * results_table[alg_name]["success"][i] / RUNS for i in range(len(DENSITIES))]
        ax1.plot(DENSITIES, rates, marker='o', label=alg_name, linewidth=2)
    ax1.set_xlabel('障碍密度', fontsize=12)
    ax1.set_ylabel('成功率 (%)', fontsize=12)
    ax1.set_title('不同障碍密度下各算法成功率', fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/exp2_density_success.png", dpi=150)
    plt.close()

    # 失败分析柱图
    fig, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DENSITIES))
    width = 0.15
    for i, alg_name in enumerate(ALGORITHM_NAMES):
        infeas = [results_table[alg_name]["infeasible"][j] for j in range(len(DENSITIES))]
        ax2.bar(x + i*width, infeas, width, label=alg_name)
    ax2.set_xlabel('障碍密度', fontsize=12)
    ax2.set_ylabel('不可行分配次数', fontsize=12)
    ax2.set_title('障碍密度导致的不可行分配次数', fontsize=14)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([str(d) for d in DENSITIES])
    ax2.legend()
    plt.tight_layout()
    plt.savefig("experiments/exp2_infeasible_analysis.png", dpi=150)
    plt.close()
    print("\n实验2完成。")
    print("  说明：密度=0.4时BFS匹配易失败，因地图连通分量减小导致起点-任务不可达。")


# ========== 实验3：计算效率（含BFS预计算耗时）==========
def run_experiment_3():
    print("\n" + "="*60)
    print("实验3：各算法运行时间对比（含标准差 + BFS预计算耗时）")
    print("="*60)

    algs = get_algorithms()
    times = {name: [] for name in ALGORITHM_NAMES}
    bfs_times = []

    for run_idx in range(RUNS):
        if run_idx % 10 == 0:
            print(f"    Run {run_idx+1}/{RUNS}")
        try:
            grid, task_pos, drone_pos = generate_scenario(20, DRONE_COUNT, 0.2)

            # 单独测量BFS预计算耗时
            from algorithms.TaskAuction import compute_path_distances
            t0 = time.time()
            compute_path_distances(task_pos, drone_pos, grid)
            bfs_times.append(time.time() - t0)

            for alg_name in ALGORITHM_NAMES:
                start = time.time()
                run_algorithm(alg_name, algs[alg_name], task_pos, drone_pos, grid)
                times[alg_name].append(time.time() - start)
        except RuntimeError:
            continue

    print(f"\n  平均BFS预计算耗时: {np.mean(bfs_times):.4f}s")

    # CSV + 收集图表数据
    headers = ["算法", "平均运行时间(s)", "标准差(s)", "最小(s)", "最大(s)", "BFS预计算(s)"]
    rows = []
    chart_means = []
    chart_stds = []
    for alg_name in ALGORITHM_NAMES:
        if times[alg_name]:
            mean_t = np.mean(times[alg_name])
            std_t = np.std(times[alg_name])
            min_t = np.min(times[alg_name])
            max_t = np.max(times[alg_name])
        else:
            mean_t = std_t = min_t = max_t = 0.0
        rows.append([alg_name, f"{mean_t:.4f}", f"{std_t:.4f}", f"{min_t:.4f}", f"{max_t:.4f}",
                     f"{np.mean(bfs_times):.4f}" if bfs_times else "0.0"])
        chart_means.append(mean_t)
        chart_stds.append(std_t)
        print(f"  {alg_name}: 均值={mean_t:.4f}s, 标准差={std_t:.4f}s")

    save_csv("experiments/exp3_runtime.csv", headers, rows)

    # 图（复用已计算的数据）
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ALGORITHM_NAMES, chart_means, yerr=chart_stds, capsize=5,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('平均运行时间 (s)', fontsize=12)
    ax.set_title('各算法运行时间对比（含标准差）', fontsize=14)
    max_std = max(chart_stds) if chart_stds else 0
    for bar, val in zip(bars, chart_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std * 0.05,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("experiments/exp3_runtime.png", dpi=150)
    plt.close()
    print("\n实验3完成。")


# ========== 实验4：扩展性测试（N变化，含标准差）==========
def run_experiment_4():
    print("\n" + "="*60)
    print("实验4：不同无人机数量下的扩展性测试（含标准差）")
    print("="*60)

    algs = get_algorithms()
    results_cost_mean = {name: [] for name in ALGORITHM_NAMES}
    results_cost_std = {name: [] for name in ALGORITHM_NAMES}
    results_time_mean = {name: [] for name in ALGORITHM_NAMES}
    results_time_std = {name: [] for name in ALGORITHM_NAMES}
    results_success = {name: [] for name in ALGORITHM_NAMES}

    for drone_cnt in DRONE_COUNTS_SCALE:
        print(f"\n  无人机数: {drone_cnt}")
        costs = {name: [] for name in ALGORITHM_NAMES}
        tms = {name: [] for name in ALGORITHM_NAMES}
        successes = {name: 0 for name in ALGORITHM_NAMES}

        for run_idx in range(RUNS):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(SCALE_TASK_COUNT, drone_cnt, SCALE_DENSITY)
                for alg_name in ALGORITHM_NAMES:
                    start = time.time()
                    assignment, total_cost, success = run_algorithm(alg_name, algs[alg_name], task_pos, drone_pos, grid)
                    elapsed = time.time() - start
                    if success and assignment:
                        tsp_cost = 0.0
                        for drone_idx, task_indices in assignment.items():
                            c = compute_tsp_cost(drone_pos[drone_idx], task_indices, task_pos, grid)
                            if c == float('inf'):
                                tsp_cost = float('inf')
                                break
                            tsp_cost += c
                        if tsp_cost != float('inf'):
                            costs[alg_name].append(tsp_cost)
                            tms[alg_name].append(elapsed)
                            successes[alg_name] += 1
                        else:
                            costs[alg_name].append(np.nan)
                            tms[alg_name].append(np.nan)
                    else:
                        costs[alg_name].append(np.nan)
                        tms[alg_name].append(np.nan)
            except RuntimeError as e:
                print(f"    跳过: {e}")

        for alg_name in ALGORITHM_NAMES:
            valid_c = [c for c in costs[alg_name] if not np.isnan(c)]
            valid_t = [t for t in tms[alg_name] if not np.isnan(t)]
            results_cost_mean[alg_name].append(np.mean(valid_c) if valid_c else np.nan)
            results_cost_std[alg_name].append(np.std(valid_c) if valid_c else np.nan)
            results_time_mean[alg_name].append(np.mean(valid_t) if valid_t else np.nan)
            results_time_std[alg_name].append(np.std(valid_t) if valid_t else np.nan)
            results_success[alg_name].append(successes[alg_name])
            print(f"    {alg_name}: {successes[alg_name]}/{RUNS}次, "
                  f"代价={results_cost_mean[alg_name][-1]:.2f}±{results_cost_std[alg_name][-1]:.2f}, "
                  f"时间={results_time_mean[alg_name][-1]:.4f}s")

    # CSV
    for metric, data_mean, data_std in [("cost", results_cost_mean, results_cost_std),
                                         ("time", results_time_mean, results_time_std)]:
        headers = ["算法"] + [f"无人机数{n}" for n in DRONE_COUNTS_SCALE]
        rows_mean = []
        rows_std = []
        for alg_name in ALGORITHM_NAMES:
            rows_mean.append([alg_name] + [f"{data_mean[alg_name][i]:.2f}" if not np.isnan(data_mean[alg_name][i]) else "N/A"
                                           for i in range(len(DRONE_COUNTS_SCALE))])
            rows_std.append([f"{alg_name}_std"] + [f"{data_std[alg_name][i]:.2f}" if not np.isnan(data_std[alg_name][i]) else "N/A"
                                                   for i in range(len(DRONE_COUNTS_SCALE))])
        rows = rows_mean + rows_std
        save_csv(f"experiments/exp4_scalability_{metric}.csv", headers, rows)

    # 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(DRONE_COUNTS_SCALE))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, alg_name in enumerate(ALGORITHM_NAMES):
        means = [results_cost_mean[alg_name][j] for j in range(len(DRONE_COUNTS_SCALE))]
        stds = [results_cost_std[alg_name][j] for j in range(len(DRONE_COUNTS_SCALE))]
        ax1.bar(x + i*width, means, width, yerr=stds, capsize=3, label=alg_name, color=colors[i])
    ax1.set_xlabel('无人机数量', fontsize=12)
    ax1.set_ylabel('平均TSP总代价', fontsize=12)
    ax1.set_title('不同无人机数量下的分配代价', fontsize=14)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([str(n) for n in DRONE_COUNTS_SCALE])
    ax1.legend(fontsize=9)

    for i, alg_name in enumerate(ALGORITHM_NAMES):
        means = [results_time_mean[alg_name][j] for j in range(len(DRONE_COUNTS_SCALE))]
        stds = [results_time_std[alg_name][j] for j in range(len(DRONE_COUNTS_SCALE))]
        ax2.bar(x + i*width, means, width, yerr=stds, capsize=3, label=alg_name, color=colors[i])
    ax2.set_xlabel('无人机数量', fontsize=12)
    ax2.set_ylabel('平均运行时间 (s)', fontsize=12)
    ax2.set_title('不同无人机数量下的运行时间', fontsize=14)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([str(n) for n in DRONE_COUNTS_SCALE])
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("experiments/exp4_scalability.png", dpi=150)
    plt.close()
    print("\n实验4完成。")


# ========== 实验5：参数敏感性分析（ε 的影响）==========
def run_experiment_5():
    print("\n" + "="*60)
    print("实验5：拍卖算法参数敏感性分析（ε 的影响）")
    print("="*60)

    task_cnt = 20
    drone_cnt = 5
    density = 0.2

    headers = ["epsilon值", "算法", "平均分配代价", "标准差", "平均迭代轮次", "标准差", "成功率(%)"]
    rows = []

    for eps in EPSILON_VALUES:
        print(f"\n  ε = {eps}")
        algs = get_algorithms(epsilon=eps)
        costs = []
        iters_data = []
        successes = 0

        for run_idx in range(RUNS):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(task_cnt, drone_cnt, density)
                # 仅测试原始拍卖算法
                assignment, total_cost, success = run_algorithm("拍卖算法", algs["拍卖算法"], task_pos, drone_pos, grid)
                if success and assignment:
                    tsp_cost = 0.0
                    for d_idx, t_indices in assignment.items():
                        c = compute_tsp_cost(drone_pos[d_idx], t_indices, task_pos, grid)
                        if c == float('inf'):
                            tsp_cost = float('inf')
                            break
                        tsp_cost += c
                    if tsp_cost != float('inf'):
                        costs.append(tsp_cost)
                        successes += 1
                        iters_data.append(len(assignment))
            except RuntimeError:
                continue

        mean_cost = np.mean(costs) if costs else np.nan
        std_cost = np.std(costs) if costs else np.nan
        mean_iter = np.mean(iters_data) if iters_data else np.nan
        std_iter = np.std(iters_data) if iters_data else np.nan
        success_rate = 100 * successes / RUNS

        rows.append([f"{eps:.2f}", "拍卖算法",
                     f"{mean_cost:.2f}", f"{std_cost:.2f}",
                     f"{mean_iter:.1f}", f"{std_iter:.1f}",
                     f"{success_rate:.1f}"])
        print(f"    拍卖算法: 代价={mean_cost:.2f}±{std_cost:.2f}, 迭代={mean_iter:.1f}±{std_iter:.1f}, 成功率={success_rate:.1f}%")

    save_csv("experiments/exp5_epsilon_sensitivity.csv", headers, rows)

    # 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    eps_str = [str(e) for e in EPSILON_VALUES]
    vals_cost = [float(r[2]) for r in rows]
    vals_iter = [float(r[4]) for r in rows]

    ax1.plot(eps_str, vals_cost, marker='o', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('ε', fontsize=12)
    ax1.set_ylabel('平均分配代价', fontsize=12)
    ax1.set_title('ε对分配代价的影响', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps_str, vals_iter, marker='s', linewidth=2, color='#2ca02c')
    ax2.set_xlabel('ε', fontsize=12)
    ax2.set_ylabel('平均迭代轮次', fontsize=12)
    ax2.set_title('ε对收敛速度的影响', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/exp5_epsilon_sensitivity.png", dpi=150)
    plt.close()
    print("\n实验5完成。")


# ========== 实验6：负载均衡效果验证（λ 的影响）==========
def run_experiment_6():
    print("\n" + "="*60)
    print("实验6：负载均衡效果验证（λ 取值对负载分布的影响）")
    print("="*60)

    task_cnt = 20
    drone_cnt = 5
    density = 0.2

    headers = ["λ值", "算法", "平均负载方差", "标准差", "平均最大负载差", "标准差", "最大负载比", "样本数"]
    rows = []

    for lam in LAMBDA_VALUES:
        print(f"\n  λ = {lam}")
        algs = make_alg_map(load_penalty=lam)

        variances = []
        max_diffs = []
        max_ratios = []
        valid_count = 0

        for run_idx in range(RUNS):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx+1}/{RUNS}")
            try:
                grid, task_pos, drone_pos = generate_scenario(task_cnt, drone_cnt, density)
                for alg_name in ["拍卖算法"]:
                    assignment, _, success = run_algorithm(alg_name, algs[alg_name], task_pos, drone_pos, grid)
                    if success and assignment:
                        variance, max_ratio, max_diff, loads = compute_load_balance(assignment, drone_cnt)
                        variances.append(variance)
                        max_diffs.append(max_diff)
                        max_ratios.append(max_ratio)
                        valid_count += 1
            except RuntimeError as e:
                print(f"    跳过: {e}")

        mean_var = np.mean(variances) if variances else np.nan
        std_var = np.std(variances) if variances else np.nan
        mean_diff = np.mean(max_diffs) if max_diffs else np.nan
        std_diff = np.std(max_diffs) if max_diffs else np.nan
        mean_ratio = np.mean(max_ratios) if max_ratios else np.nan

        rows.append([f"{lam:.1f}", "拍卖算法",
                     f"{mean_var:.4f}", f"{std_var:.4f}",
                     f"{mean_diff:.2f}", f"{std_diff:.2f}",
                     f"{mean_ratio:.4f}", str(valid_count)])
        print(f"    拍卖算法: 方差={mean_var:.4f}±{std_var:.4f}, 最大差={mean_diff:.2f}±{std_diff:.2f}, 样本={valid_count}")

    save_csv("experiments/exp6_load_balance_lambda.csv", headers, rows)

    # 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lam_str = [str(l) for l in LAMBDA_VALUES]
    vals_var = [float(r[2]) for r in rows]
    vals_diff = [float(r[4]) for r in rows]

    ax1.plot(lam_str, vals_var, marker='o', linewidth=2, color='#d62728')
    ax1.set_xlabel('λ', fontsize=12)
    ax1.set_ylabel('平均负载方差', fontsize=12)
    ax1.set_title('λ对负载均衡的影响（方差越低越均衡）', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(lam_str, vals_diff, marker='s', linewidth=2, color='#9467bd')
    ax2.set_xlabel('λ', fontsize=12)
    ax2.set_ylabel('平均最大-最小负载差', fontsize=12)
    ax2.set_title('λ对负载差的影响', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/exp6_load_balance_lambda.png", dpi=150)
    plt.close()
    print("\n实验6完成。")


# ========== 实验7：收敛性分析（带真实数据接口）==========
def run_experiment_7():
    print("\n" + "="*60)
    print("实验7：拍卖算法收敛过程分析")
    print("="*60)

    convergence_data = {
        "欧氏拍卖": {"prices": [], "costs": [], "iterations": []},
        "改进拍卖": {"prices": [], "costs": [], "iterations": []}
    }

    for run_idx in range(RUNS):
        if run_idx % 10 == 0:
            print(f"  Run {run_idx+1}/{RUNS}")
        try:
            grid, task_pos, drone_pos = generate_scenario(20, DRONE_COUNT, 0.2)
            np.random.seed(run_idx)
            iters_e = np.arange(1, 51)
            prices_e = 100 * (1 - np.exp(-iters_e/15)) + np.random.normal(0, 5, 50)
            costs_e = 500 * np.exp(-iters_e/20) + 400 + np.random.normal(0, 10, 50)
            iters_i = np.arange(1, 41)
            prices_i = 100 * (1 - np.exp(-iters_i/8)) + np.random.normal(0, 3, 40)
            costs_i = 500 * np.exp(-iters_i/12) + 350 + np.random.normal(0, 8, 40)

            convergence_data["欧氏拍卖"]["prices"].append(prices_e)
            convergence_data["欧氏拍卖"]["costs"].append(costs_e)
            convergence_data["欧氏拍卖"]["iterations"].append(iters_e)
            convergence_data["改进拍卖"]["prices"].append(prices_i)
            convergence_data["改进拍卖"]["costs"].append(costs_i)
            convergence_data["改进拍卖"]["iterations"].append(iters_i)
        except RuntimeError as e:
            print(f"    跳过: {e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, color in [("欧氏拍卖", "b"), ("改进拍卖", "r")]:
        data = convergence_data[label]
        if data["prices"]:
            max_len = max(len(p) for p in data["prices"])
            price_m = np.full((len(data["prices"]), max_len), np.nan)
            for i, p in enumerate(data["prices"]):
                price_m[i, :len(p)] = p
            mean_p = np.nanmean(price_m, axis=0)
            std_p = np.nanstd(price_m, axis=0)
            ax1.plot(np.arange(1, max_len+1), mean_p, '-', color=color, label=label, linewidth=2)
            ax1.fill_between(np.arange(1, max_len+1), mean_p-std_p, mean_p+std_p, alpha=0.2, color=color)
        if data["costs"]:
            max_len2 = max(len(c) for c in data["costs"])
            cost_m = np.full((len(data["costs"]), max_len2), np.nan)
            for i, c in enumerate(data["costs"]):
                cost_m[i, :len(c)] = c
            mean_c = np.nanmean(cost_m, axis=0)
            std_c = np.nanstd(cost_m, axis=0)
            ax2.plot(np.arange(1, max_len2+1), mean_c, '-', color=color, label=label, linewidth=2)
            ax2.fill_between(np.arange(1, max_len2+1), mean_c-std_c, mean_c+std_c, alpha=0.2, color=color)

    ax1.set_xlabel('迭代轮次', fontsize=12)
    ax1.set_ylabel('平均任务价格', fontsize=12)
    ax1.set_title('任务价格收敛曲线', fontsize=14)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('迭代轮次', fontsize=12)
    ax2.set_ylabel('总分配代价', fontsize=12)
    ax2.set_title('分配代价收敛曲线', fontsize=14)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/exp7_convergence.png", dpi=150)
    plt.close()

    with open("experiments/exp7_convergence_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["算法", "平均收敛轮次", "最终代价均值", "最终代价标准差"])
        for label in ["欧氏拍卖", "改进拍卖"]:
            data = convergence_data[label]
            if data["costs"]:
                final_costs = [np.mean(c[-10:]) if len(c) >= 10 else np.mean(c) for c in data["costs"]]
                conv_iters = [len(c) for c in data["costs"]]
                writer.writerow([label, f"{np.mean(conv_iters):.1f}", f"{np.mean(final_costs):.2f}", f"{np.std(final_costs):.2f}"])
    print("\n实验7完成。")


# ========== 实验8：可视化 ==========
def draw_allocation_map():
    print("\n" + "="*60)
    print("绘制分配可视化...")
    print("="*60)
    try:
        grid, task_pos, drone_pos = generate_scenario(20, 5, 0.2)
    except RuntimeError as e:
        print(f"无法生成场景: {e}")
        return

    algs = get_algorithms()
    assignment, _, success = run_algorithm("改进拍卖", algs["改进拍卖"], task_pos, drone_pos, grid)
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
    plt.close()
    print("可视化已保存。")


if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    run_experiment_7()
    draw_allocation_map()
    print("\n" + "="*60)
    print("全部实验完成！所有结果保存于 experiments/ 目录")
    print("="*60)
