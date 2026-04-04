import sys
import os
import time
import csv
import numpy as np
import random
from pathlib import Path
import multiprocessing as mp
from functools import partial

# 添加模块路径（根据你的实际结构调整）
current_dir = Path(__file__).parent
algorithms_dir = current_dir / 'algorithms'
mapgen_dir = current_dir / 'map_generator'
sys.path.insert(0, str(algorithms_dir))
sys.path.insert(0, str(mapgen_dir))

from algorithms.Astar import astar
from algorithms.JPS import jps_2d
from map_generator.MapGenerator import generate_map_with_path

# ================== 配置参数 ==================
# 只测试 2D
DIM = 2
SIZES = [(100, 100), (200, 200), (500, 500)]
DENSITIES = [0.2, 0.3, 0.6]          # 去掉 0.45
NUM_TRIALS = 25                      # 每种配置重复次数

# 算法列表（不包含 RRT）
ALGORITHMS = [
    ('Astar', astar),
    ('JPS', jps_2d)              # JPS 2D 函数
]

# 每个任务（一个地图）的超时时间（秒），包括地图生成 + 两个算法
TASK_TIMEOUT = 240                   # 3分钟，足够 500x500 地图

# 并行进程数（充分利用你的 32 线程，但避免内存争抢，设为 24 即可）
NUM_WORKERS = min(24, mp.cpu_count())

# 输出目录
MAP_DIR = Path("./data/map")
LOG_DIR = Path("./data/log")
MAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_LOG = LOG_DIR / "benchmark_2d_fast.csv"
SUMMARY_LOG = LOG_DIR / "benchmark_2d_summary.csv"

# ================== 辅助函数 ==================
def get_start_goal(size):
    """2D 起点和终点"""
    return (0, 0), (size[0]-1, size[1]-1)

def run_single_algorithm(alg_name, func, grid, start, goal):
    """运行一个算法，返回 (success, time_ms)"""
    try:
        t0 = time.perf_counter()
        path = func(grid, start, goal)
        elapsed = time.perf_counter() - t0
        success = (path is not None and len(path) > 0)
        return success, elapsed * 1000.0
    except Exception as e:
        print(f"    {alg_name} 出错: {e}")
        return False, 0.0

def worker_task(size, density, trial):
    """单个任务的执行函数（生成地图 + 运行所有算法）"""
    # 为每个任务设置独立随机种子，保证可重复
    seed = hash((size, density, trial)) & 0xffffffff
    random.seed(seed)
    np.random.seed(seed % 2**32)

    start, goal = get_start_goal(size)

    # 1. 生成地图
    try:
        grid = generate_map_with_path(
            size=size,
            obstacle_density=density,
            start=start,
            goal=goal,
            dim=2,
            step_factor=2.5,
            p=0.7,
            max_attempts=50
        )
    except Exception as e:
        print(f"地图生成失败 {size} dens{density} trial{trial}: {e}")
        return None   # 表示该任务失败

    # 2. 保存地图（可选，文件名包含完整参数）
    map_filename = f"2D_{size[0]}x{size[1]}_dens{density}_trial{trial}.npy"
    map_path = MAP_DIR / map_filename
    np.save(map_path, grid)

    # 3. 运行所有算法
    results = []
    for alg_name, alg_func in ALGORITHMS:
        success, time_ms = run_single_algorithm(alg_name, alg_func, grid, start, goal)
        results.append([size, density, trial, alg_name, success, f"{time_ms:.3f}"])
    return results

# ================== 断点续跑 ==================
def get_completed_tasks(log_file):
    """从日志文件中提取已经成功完成的地图（以 size, density, trial 为标识）"""
    completed = set()
    if not log_file.exists():
        return completed
    with open(log_file, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return completed
        for row in reader:
            if len(row) < 4:
                continue
            # 日志格式: size, density, trial, algorithm, success, time_ms
            size_str, density_str, trial_str, alg = row[0], row[1], row[2], row[3]
            # 只要地图已经有一个算法记录（不管是哪个算法），就认为该地图已处理
            # 注意：同一个地图会被多个算法写入多行，我们只记录一次
            completed.add((size_str, density_str, trial_str))
    return completed

# ================== 主函数 ==================
def main():
    print(f"使用 {NUM_WORKERS} 个进程并行测试")
    print(f"2D 尺寸: {SIZES}")
    print(f"障碍物密度: {DENSITIES}")
    print(f"每个配置重复 {NUM_TRIALS} 次")
    print(f"算法: {[a[0] for a in ALGORITHMS]}")
    print("=" * 60)

    # 生成所有任务（size, density, trial）
    all_tasks = []
    for size in SIZES:
        for density in DENSITIES:
            for trial in range(1, NUM_TRIALS + 1):
                all_tasks.append((size, density, trial))

    # 断点续跑：过滤已完成的任务
    completed = get_completed_tasks(DETAIL_LOG)
    tasks = []
    for size, density, trial in all_tasks:
        key = (str(size), str(density), str(trial))
        if key in completed:
            print(f"跳过已完成: size={size} density={density} trial={trial}")
            continue
        tasks.append((size, density, trial))

    print(f"总任务数: {len(all_tasks)}, 剩余未完成: {len(tasks)}")
    if not tasks:
        print("所有任务已完成，退出。")
        return

    # 准备日志文件（追加模式）
    file_exists = DETAIL_LOG.exists()
    f_detail = open(DETAIL_LOG, 'a', newline='')
    writer = csv.writer(f_detail)
    if not file_exists:
        writer.writerow(['size', 'density', 'trial', 'algorithm', 'success', 'time_ms'])

    # 使用进程池执行，带超时控制
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # 提交所有任务，获得 AsyncResult 对象
        async_results = []
        for task in tasks:
            ar = pool.apply_async(worker_task, task)
            async_results.append((task, ar))

        # 逐个获取结果（按提交顺序，但为了进度显示，可以用索引）
        completed_count = 0
        for idx, (task, ar) in enumerate(async_results):
            size, density, trial = task
            try:
                results = ar.get(timeout=TASK_TIMEOUT)
                if results is not None:
                    for row in results:
                        writer.writerow(row)
                    completed_count += 1
                    print(f"进度: {completed_count}/{len(tasks)} 完成 (当前: {size} 密度{density} trial{trial})")
                else:
                    # 地图生成失败
                    writer.writerow([size, density, trial, 'MAP_FAILED', False, '0'])
                    print(f"地图生成失败: {size} 密度{density} trial{trial}")
            except mp.TimeoutError:
                print(f"任务超时 ({TASK_TIMEOUT}s): {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, 'TIMEOUT', False, '0'])
            except Exception as e:
                print(f"任务异常: {e} - {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, 'ERROR', False, '0'])
            f_detail.flush()

    f_detail.close()

    # 生成汇总统计
    print("\n===== 生成汇总统计 =====")
    with open(DETAIL_LOG, 'r') as f_in, open(SUMMARY_LOG, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        header = next(reader)
        writer_sum = csv.writer(f_out)
        writer_sum.writerow(['size', 'density', 'algorithm', 'success_rate', 'avg_time_ms', 'num_success', 'total_trials'])

        data = {}
        for row in reader:
            size_str, density_str, trial, alg, success_str, time_str = row
            # 过滤掉失败标记的行（MAP_FAILED, TIMEOUT, ERROR）
            if alg in ['MAP_FAILED', 'TIMEOUT', 'ERROR']:
                continue
            size = size_str
            density = float(density_str)
            success = (success_str == 'True')
            time_ms = float(time_str) if success else None
            key = (size, density, alg)
            if key not in data:
                data[key] = {'times': [], 'success_count': 0, 'total': 0}
            data[key]['total'] += 1
            if success:
                data[key]['success_count'] += 1
                data[key]['times'].append(time_ms)

        for (size, density, alg), stats in sorted(data.items()):
            success_rate = stats['success_count'] / stats['total'] if stats['total'] > 0 else 0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0.0
            writer_sum.writerow([size, density, alg, f"{success_rate*100:.1f}%", f"{avg_time:.3f}", stats['success_count'], stats['total']])
            print(f"{size} 密度{density} {alg}: 成功率{success_rate*100:.1f}% 平均{avg_time:.3f}ms (成功{stats['success_count']}/{stats['total']})")

    print(f"\n详细日志: {DETAIL_LOG}")
    print(f"汇总日志: {SUMMARY_LOG}")

if __name__ == "__main__":
    # Windows 多进程需要
    mp.freeze_support()
    main()