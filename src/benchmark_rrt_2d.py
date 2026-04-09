import sys
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

from algorithms.RRT import rrt_2d
from map_generator.MapGenerator import generate_map_with_path

# ================== 配置 ==================
SIZES = [(100, 100), (200, 200), (500, 500)]
DENSITIES = [0.2, 0.3, 0.6]
NUM_TRIALS = 25
RRT_STEP_SIZE = 1.5          # 步长
RRT_MAX_ITER = 50000         # 最大迭代次数
TASK_TIMEOUT = 300           # 每个任务超时（秒），RRT可能较慢
NUM_WORKERS = min(16, mp.cpu_count())  # RRT 计算密集，适当减少并行数

# 输出目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
MAP_DIR = project_root / "data" / "map"
LOG_DIR = project_root / "data" / "log"
MAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_LOG = LOG_DIR / "benchmark_rrt_2d.csv"
SUMMARY_LOG = LOG_DIR / "benchmark_rrt_2d_summary.csv"

def get_start_goal(size):
    return (0, 0), (size[0]-1, size[1]-1)

def run_rrt(grid, start, goal):
    """运行 RRT，返回 (success, time_ms, path_length)"""
    try:
        t0 = time.perf_counter()
        path = rrt_2d(grid, start, goal, step_size=RRT_STEP_SIZE, max_iter=RRT_MAX_ITER)
        elapsed = time.perf_counter() - t0
        if path is None or len(path) == 0:
            return False, 0.0, 0
        # path 是 numpy 数组，计算路径长度（欧氏距离累计）
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        return True, elapsed * 1000.0, length
    except Exception as e:
        print(f"    RRT 出错: {e}")
        return False, 0.0, 0

def worker_task(size, density, trial):
    seed = hash((size, density, trial)) & 0xffffffff
    random.seed(seed)
    np.random.seed(seed % 2**32)

    start, goal = get_start_goal(size)

    # 生成地图
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
        return None

    # 保存地图（可选）
    map_filename = f"RRT2D_{size[0]}x{size[1]}_dens{density}_trial{trial}.npy"
    np.save(MAP_DIR / map_filename, grid)

    success, time_ms, length = run_rrt(grid, start, goal)
    return [size, density, trial, success, f"{time_ms:.3f}", f"{length:.3f}"]

def get_completed_tasks(log_file):
    completed = set()
    if not log_file.exists():
        return completed
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return completed
        for row in reader:
            if len(row) < 4:
                continue
            size_str, density_str, trial_str = row[0], row[1], row[2]
            completed.add((size_str, density_str, trial_str))
    return completed

def main():
    print(f"RRT 2D 测试，并行进程数: {NUM_WORKERS}")
    print(f"尺寸: {SIZES}")
    print(f"密度: {DENSITIES}")
    print(f"每组重复: {NUM_TRIALS} 次")
    print("=" * 60)

    # 生成所有任务
    all_tasks = []
    for size in SIZES:
        for density in DENSITIES:
            for trial in range(1, NUM_TRIALS + 1):
                all_tasks.append((size, density, trial))

    # 断点续跑
    completed = get_completed_tasks(DETAIL_LOG)
    tasks = []
    for size, density, trial in all_tasks:
        key = (str(size), str(density), str(trial))
        if key in completed:
            print(f"跳过已完成: {size} 密度{density} trial{trial}")
            continue
        tasks.append((size, density, trial))

    print(f"总任务数: {len(all_tasks)}, 剩余: {len(tasks)}")
    if not tasks:
        print("所有任务已完成")
        return

    # 准备日志文件
    file_exists = DETAIL_LOG.exists()
    f_detail = open(DETAIL_LOG, 'a', newline='')
    writer = csv.writer(f_detail)
    if not file_exists:
        writer.writerow(['size', 'density', 'trial', 'success', 'time_ms', 'path_length'])

    # 并行执行
    with mp.Pool(processes=NUM_WORKERS) as pool:
        async_results = [pool.apply_async(worker_task, task) for task in tasks]
        completed_cnt = 0
        for idx, (task, ar) in enumerate(zip(tasks, async_results)):
            size, density, trial = task
            try:
                res = ar.get(timeout=TASK_TIMEOUT)
                if res:
                    writer.writerow(res)
                    completed_cnt += 1
                    print(f"进度: {completed_cnt}/{len(tasks)} 完成 ({size} 密度{density} trial{trial})")
                else:
                    writer.writerow([size, density, trial, False, '0', '0'])
                    print(f"任务失败: {size} 密度{density} trial{trial}")
            except mp.TimeoutError:
                print(f"任务超时: {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, False, '0', '0'])
            except Exception as e:
                print(f"任务异常: {e} - {size} 密度{density} trial{trial}")
                writer.writerow([size, density, trial, False, '0', '0'])
            f_detail.flush()
    f_detail.close()

    # 汇总统计
    print("\n===== RRT 2D 汇总 =====")
    with open(DETAIL_LOG, 'r') as f_in, open(SUMMARY_LOG, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        header = next(reader)
        writer_sum = csv.writer(f_out)
        writer_sum.writerow(['size', 'density', 'success_rate', 'avg_time_ms', 'avg_path_length', 'num_success', 'total'])

        data = {}
        for row in reader:
            size_str, density_str, trial, success_str, time_str, length_str = row
            success = (success_str == 'True')
            time_ms = float(time_str) if success else None
            length = float(length_str) if success else None
            key = (size_str, density_str)
            if key not in data:
                data[key] = {'times': [], 'lengths': [], 'success': 0, 'total': 0}
            data[key]['total'] += 1
            if success:
                data[key]['success'] += 1
                data[key]['times'].append(time_ms)
                data[key]['lengths'].append(length)

        for (size_str, density_str), stats in sorted(data.items()):
            success_rate = stats['success'] / stats['total']
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            avg_len = sum(stats['lengths']) / len(stats['lengths']) if stats['lengths'] else 0
            writer_sum.writerow([size_str, density_str, f"{success_rate*100:.1f}%", f"{avg_time:.3f}", f"{avg_len:.3f}", stats['success'], stats['total']])
            print(f"{size_str} 密度{density_str}: 成功率{success_rate*100:.1f}% 平均{avg_time:.3f}ms 平均长度{avg_len:.3f} ({stats['success']}/{stats['total']})")

    print(f"详细日志: {DETAIL_LOG}")
    print(f"汇总日志: {SUMMARY_LOG}")

if __name__ == "__main__":
    mp.freeze_support()
    main()