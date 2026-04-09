import numpy as np
import time
import sys
import os
from collections import deque
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_generator.MapGenerator import generate_map_with_path
from map_generator.TaskPointGeneration import generate_task_points

# 导入四个算法（根据实际文件名调整）
from algorithms.TaskAuction import auction_algorithm
from algorithms.TaskAuctionImprove import auction_algorithm_improved
from algorithms.k_means import assign_tasks_with_kmedoids
from algorithms.k_meanspp import assign_tasks_with_kmeanspp


# ========== 辅助函数：连通性检查 ==========
def is_connected(grid, start, goal):
    """BFS 检查起点到终点是否连通（四连通）"""
    h, w = grid.shape
    visited = np.zeros((h, w), bool)
    q = deque([start])
    visited[start[1], start[0]] = True
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0 and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((nx, ny))
    return False

def point_to_grid_safe(point, grid, max_radius=20):
    """安全版：将连续坐标转为自由格子坐标，如果周围 max_radius 内无自由格子则返回 None"""
    x, y = point
    ix, iy = int(round(x)), int(round(y))
    h, w = grid.shape
    ix = max(0, min(w-1, ix))
    iy = max(0, min(h-1, iy))
    if grid[iy, ix] == 0:
        return ix, iy
    for r in range(1, max_radius+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = ix+dx, iy+dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    return nx, ny
    return None

def generate_valid_scenario(map_size, obstacle_density, num_tasks, num_drones, max_attempts=20):
    """生成一个所有任务点和无人机起点都位于自由区域且地图连通的场景"""
    for attempt in range(max_attempts):
        # 生成地图（确保起点终点连通）
        grid = generate_map_with_path(map_size, obstacle_density, dim=2,
                                      step_factor=3.0, p=0.7, max_attempts=100)
        start = (0, 0)
        goal = (map_size[0]-1, map_size[1]-1)
        if not is_connected(grid, start, goal):
            continue

        # 生成任务点
        try:
            task_points = generate_task_points(grid, num_tasks)
        except RuntimeError:
            continue

        # 生成无人机起点（避开障碍物）
        h, w = grid.shape # type: ignore
        drone_positions = []
        for _ in range(num_drones * 3):  # 多生成一些候选
            x = np.random.uniform(0, w)
            y = np.random.uniform(0, h)
            result = point_to_grid_safe([x, y], grid)
            if result is not None:
                gx, gy = result
                drone_positions.append([x, y])
            if len(drone_positions) >= num_drones:
                break
        if len(drone_positions) < num_drones:
            continue

        # 检查所有任务点和无人机起点都能映射到自由格子
        ok = True
        for tp in task_points:
            if point_to_grid_safe(tp, grid) is None:
                ok = False
                break
        for dp in drone_positions:
            if point_to_grid_safe(dp, grid) is None:
                ok = False
                break
        if ok:
            return grid, task_points, drone_positions

    raise RuntimeError("无法生成有效场景，请降低障碍物密度或增加地图尺寸")


# ========== 负载均衡指标 ==========
def compute_load_balance(assignments):
    task_counts = [len(assignments[d]) for d in range(len(assignments))]
    return np.std(task_counts)


# ========== 单次测试 ==========
def run_single_test(algorithm_func, task_points, drone_positions, grid, **kwargs):
    start_time = time.perf_counter()
    assignments, total_cost = algorithm_func(task_points, drone_positions, grid, **kwargs)
    elapsed = time.perf_counter() - start_time
    load_balance = compute_load_balance(assignments)
    return total_cost, load_balance, elapsed


# ========== 主对比函数 ==========
def compare_algorithms(num_tests=10, map_size=(30,30), obstacle_density=0.15,
                       num_tasks=20, num_drones=5, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    results = {
        'Auction': {'cost': [], 'balance': [], 'time': []},
        'AuctionImproved': {'cost': [], 'balance': [], 'time': []},
        'KMedoids': {'cost': [], 'balance': [], 'time': []},
        'KMeansPP': {'cost': [], 'balance': [], 'time': []}
    }

    for test_idx in range(num_tests):
        print(f"测试 {test_idx+1}/{num_tests} ...")
        try:
            grid, task_points, drone_positions = generate_valid_scenario(
                map_size, obstacle_density, num_tasks, num_drones)
        except RuntimeError as e:
            print(f"跳过本次测试: {e}")
            continue

        # 运行四个算法
        for name, func in [('Auction', auction_algorithm),
                           ('AuctionImproved', auction_algorithm_improved),
                           ('KMedoids', assign_tasks_with_kmedoids),
                           ('KMeansPP', assign_tasks_with_kmeanspp)]:
            try:
                cost, bal, t = run_single_test(func, task_points, drone_positions, grid)
                results[name]['cost'].append(cost)
                results[name]['balance'].append(bal)
                results[name]['time'].append(t)
            except Exception as e:
                print(f"  算法 {name} 出错: {e}")

    # 计算平均值和标准差
    summary = {}
    for name, metrics in results.items():
        if len(metrics['cost']) == 0:
            summary[name] = {'avg_cost': 0, 'std_cost': 0, 'avg_balance': 0, 'std_balance': 0, 'avg_time': 0, 'std_time': 0}
            continue
        summary[name] = {
            'avg_cost': np.mean(metrics['cost']),
            'std_cost': np.std(metrics['cost']),
            'avg_balance': np.mean(metrics['balance']),
            'std_balance': np.std(metrics['balance']),
            'avg_time': np.mean(metrics['time']),
            'std_time': np.std(metrics['time']),
        }

    # 打印表格
    print("\n" + "="*80)
    print(f"算法性能对比 (基于 {num_tests} 次有效测试)")
    print("="*80)
    print(f"{'Algorithm':<20} {'Total Cost':<20} {'Load Balance (std)':<20} {'Time (s)':<15}")
    print("-"*80)
    for name in ['Auction', 'AuctionImproved', 'KMedoids', 'KMeansPP']:
        s = summary[name]
        if s['avg_cost'] == 0:
            continue
        print(f"{name:<20} {s['avg_cost']:8.2f} ± {s['std_cost']:6.2f}   "
              f"{s['avg_balance']:8.2f} ± {s['std_balance']:6.2f}   "
              f"{s['avg_time']:8.3f} ± {s['std_time']:6.3f}")
    print("="*80)

    # 绘图（英文标签，避免字体问题）
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        algorithms = list(summary.keys())
        costs = [summary[a]['avg_cost'] for a in algorithms if summary[a]['avg_cost'] > 0]
        cost_err = [summary[a]['std_cost'] for a in algorithms if summary[a]['avg_cost'] > 0]
        balances = [summary[a]['avg_balance'] for a in algorithms if summary[a]['avg_cost'] > 0]
        balance_err = [summary[a]['std_balance'] for a in algorithms if summary[a]['avg_cost'] > 0]
        times = [summary[a]['avg_time'] for a in algorithms if summary[a]['avg_cost'] > 0]
        time_err = [summary[a]['std_time'] for a in algorithms if summary[a]['avg_cost'] > 0]
        alg_names = [a for a in algorithms if summary[a]['avg_cost'] > 0]

        axes[0].bar(alg_names, costs, yerr=cost_err, capsize=5, color='skyblue')
        axes[0].set_title('Total Path Cost')
        axes[0].set_ylabel('Cost')
        axes[1].bar(alg_names, balances, yerr=balance_err, capsize=5, color='lightgreen')
        axes[1].set_title('Load Balance (Std of Task Counts)')
        axes[1].set_ylabel('Std Dev')
        axes[2].bar(alg_names, times, yerr=time_err, capsize=5, color='salmon')
        axes[2].set_title('Execution Time')
        axes[2].set_ylabel('Seconds')
        plt.tight_layout()
        plt.savefig('algorithm_comparison_en.png')
        plt.show()
        print("图表已保存为 algorithm_comparison_en.png")
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")

    return summary


if __name__ == "__main__":
    compare_algorithms(
        num_tests=10,
        map_size=(30,30),
        obstacle_density=0.15,   # 降低密度，提高连通性
        num_tasks=20,
        num_drones=5,
        random_seed=42
    )