import sys
import os
import time
import csv
import numpy as np
import random
from pathlib import Path
import multiprocessing as mp

# 添加模块路径
current_dir = Path(__file__).resolve().parent
algorithms_dir = current_dir / 'algorithms'
mapgen_dir = current_dir / 'map_generator'
sys.path.insert(0, str(algorithms_dir))
sys.path.insert(0, str(mapgen_dir))

from algorithms.Astar import astar
from algorithms.JPS import jps_3d
from map_generator.MapGenerator import generate_map_with_path

# ================== 配置参数 ==================
DIM = 3
SIZES = [(100, 100, 100), (200, 200, 200)]
DENSITIES = [0.2, 0.3, 0.6]
NUM_TRIALS = 25
ALGORITHMS = [
    ('Astar', astar),
    ('JPS', jps_3d)
]
TASK_TIMEOUT = 300
NUM_WORKERS = min(24, mp.cpu_count())

# 输出目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
MAP_DIR = project_root / "data" / "map"
LOG_DIR = project_root / "data" / "log"
MAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_LOG = LOG_DIR / "benchmark_3d_fast.csv"
SUMMARY_LOG = LOG_DIR / "benchmark_3d_summary.csv"

# ================== 辅助函数 ==================
def get_start_goal(size):
    return (0, 0, 0), (size[0]-1, size[1]-1, size[2]-1)

def run_single_algorithm(alg_name, func, grid, start, goal):
    """运行一个算法，返回 (success, time_ms, path_length)"""
    try:
        t0 = time.perf_counter()
        path = func(grid, start, goal)
        elapsed = time.perf_counter() - t0
        if path is None or len(path) == 0:
            return False, 0.0, 0
        success = True
        path_length = len(path)
        return success, elapsed * 1000.0, path_length
    except Exception as e:
        print(f"    {alg_name} 出错: {e}")
        return False, 0.0, 0

def worker_task(size, density, trial):
    seed = hash((size, density, trial)) & 0xffffffff
    random.seed(seed)
    np.random.seed(seed % 2**32)

    start, goal = get_start_goal(size)

    try:
        grid = generate_map_with_path(
            size=size,
            obstacle_density=density,
            start=start,
            goal=goal,
            dim=3,
            step_factor=2.5,
            p=0.7,
            max_attempts=50
        )
    except Exception as e:
        print(f"地图生成失败 {size} dens{density} trial{trial}: {e}")
        return None

    # 保存地图
    map_filename = f"3D_{size[0]}x{size[1]}x{size[2]}_dens{density}_trial{trial}.npy"
    map_path = MAP_DIR / map_filename
    np.save(map_path, grid)

    results = []
    for alg_name, alg_func in ALGORITHMS:
        success, time_ms, length = run_single_algorithm(alg_name, alg_func, grid, start, goal)
        results.append([size, density, trial, alg_name, success, f"{time_ms:.3f}", length])
    return results

# ================== 断点续跑 ==================
def get_completed_tasks(log_file):
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
            size_str, density_str, trial_str, alg = row[0], row[1], row[2], row[3]
            completed.add((size_str, density_str, trial_str))
    return completed

# ================== 主函数 ==================
def main():
    print(f"使用 {NUM_WORKERS} 个进程并行测试 3D")
    print(f"3D 尺寸: {SIZES}")
    print(f"障碍物密度: {DENSITIES}")
    print(f"每个配置重复 {NUM_TRIALS} 次")
    print(f"算法: {[a[0] for a in ALGORITHMS]}")
    print("=" * 60)

    all_tasks = []
    for size in SIZES:
        for density in DENSITIES:
            for trial in range(1, NUM_TRIALS + 1):
                all_tasks.append((size, density, trial))

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

    file_exists = DETAIL_LOG.exists()
    f_detail = open(DETAIL_LOG, 'a', newline='')
    writer = csv.writer(f_detail)
    if not file_exists:
        writer.writerow(['size', 'density', 'trial', 'algorithm', 'success', 'time_ms', 'path_length'])

    with mp.Pool(processes=NUM_WORKERS) as pool:
        async_results = []
        for task in tasks:
            ar = pool.apply_async(worker_task, task)
            async_results.append((task, ar))

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
                    writer.writerow([size, density, trial, 'MAP_FAILED', False, '0', '0'])
                    print(f"地图生成失败: {size} 密度{density} trial{trial}")
            except mp.TimeoutError:
                print(f"任务超时 ({TASK_TIMEOUT}s): {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, 'TIMEOUT', False, '0', '0'])
            except Exception as e:
                print(f"任务异常: {e} - {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, 'ERROR', False, '0', '0'])
            f_detail.flush()

    f_detail.close()

    # 生成汇总统计（增加平均路径长度）
    print("\n===== 生成汇总统计 =====")
    with open(DETAIL_LOG, 'r') as f_in, open(SUMMARY_LOG, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        header = next(reader)
        writer_sum = csv.writer(f_out)
        writer_sum.writerow(['size', 'density', 'algorithm', 'success_rate', 'avg_time_ms', 'avg_path_length', 'num_success', 'total_trials'])

        data = {}
        for row in reader:
            size_str, density_str, trial, alg, success_str, time_str, length_str = row
            if alg in ['MAP_FAILED', 'TIMEOUT', 'ERROR']:
                continue
            size = size_str
            density = float(density_str)
            success = (success_str == 'True')
            time_ms = float(time_str) if success else None
            length = int(length_str) if success else None
            key = (size, density, alg)
            if key not in data:
                data[key] = {'times': [], 'lengths': [], 'success_count': 0, 'total': 0}
            data[key]['total'] += 1
            if success:
                data[key]['success_count'] += 1
                data[key]['times'].append(time_ms)
                data[key]['lengths'].append(length)

        for (size, density, alg), stats in sorted(data.items()):
            success_rate = stats['success_count'] / stats['total'] if stats['total'] > 0 else 0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0.0
            avg_length = sum(stats['lengths']) / len(stats['lengths']) if stats['lengths'] else 0.0
            writer_sum.writerow([size, density, alg, f"{success_rate*100:.1f}%", f"{avg_time:.3f}", f"{avg_length:.1f}", stats['success_count'], stats['total']])
            print(f"{size} 密度{density} {alg}: 成功率{success_rate*100:.1f}% 平均{avg_time:.3f}ms 平均路径长度{avg_length:.1f} (成功{stats['success_count']}/{stats['total']})")

    print(f"\n详细日志: {DETAIL_LOG}")
    print(f"汇总日志: {SUMMARY_LOG}")

if __name__ == "__main__":
    mp.freeze_support()
    main()