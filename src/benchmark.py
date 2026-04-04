import sys
import os
import time
import csv
import numpy as np
import random
from pathlib import Path

# 添加模块搜索路径（根据你的实际目录结构调整）
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'mapgenerator'))

# 导入算法和地图生成器
from algorithms.Astar import astar
from algorithms.JPS import jps_2d, jps_3d
from algorithms.RRT import rrt_2d, rrt_3d
from map_generator.MapGenerator import generate_map_with_path

# ================== 配置参数 ==================
DIMENSIONS = {
    '2D': {
        'sizes': [(100,100), (200,200), (500,500)],
        'densities': [0.2, 0.3, 0.45, 0.6],
    },
    '3D': {
        'sizes': [(100,100,100), (200,200,200)],
        'densities': [0.2, 0.3, 0.45, 0.6],
    }
}
NUM_TRIALS = 25                     # 每种配置重复测试次数
RRT_STEP_SIZE = 1.5                # RRT 步长（可根据地图尺寸调整）
RRT_MAX_ITER = 50000               # RRT 最大迭代次数

# 输出目录
MAP_DIR = Path("./data/map")
LOG_DIR = Path("./data/log")
MAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件（详细记录）
DETAIL_LOG = LOG_DIR / "benchmark_details.csv"
# 汇总文件（每个配置的平均值和成功率）
SUMMARY_LOG = LOG_DIR / "benchmark_summary.csv"

# ================== 辅助函数 ==================
def get_start_goal(size, dim):
    """返回起点和终点坐标"""
    if dim == 2:
        return (0, 0), (size[0]-1, size[1]-1)
    else:
        return (0, 0, 0), (size[0]-1, size[1]-1, size[2]-1)

def run_algorithm(alg_name, func, grid, start, goal, dim, **kwargs):
    """运行一次算法，返回 (success, elapsed_ms)"""
    try:
        t0 = time.perf_counter()
        if alg_name == 'RRT':
            if dim == 2:
                path = func(grid, start, goal, step_size=kwargs.get('step_size', 1.5),
                            max_iter=kwargs.get('max_iter', 50000))
            else:
                path = func(grid, start, goal, step_size=kwargs.get('step_size', 1.5),
                            max_iter=kwargs.get('max_iter', 50000))
        else:
            # Astar 和 JPS 统一接口
            path = func(grid, start, goal)
        elapsed = time.perf_counter() - t0
        success = (path is not None and len(path) > 0)
        return success, elapsed * 1000.0
    except Exception as e:
        print(f"    {alg_name} 出错: {e}")
        return False, 0.0

# ================== 主测试循环 ==================
def main():
    # 打开详细日志文件
    with open(DETAIL_LOG, 'w', newline='') as f_detail:
        writer = csv.writer(f_detail)
        writer.writerow(['dim', 'size', 'density', 'trial', 'algorithm', 'success', 'time_ms'])

        # 遍历所有配置
        for dim_label, config in DIMENSIONS.items():
            dim = 2 if dim_label == '2D' else 3
            for size in config['sizes']:
                for density in config['densities']:
                    print(f"\n===== 测试 {dim_label} 尺寸 {size} 密度 {density} =====")
                    start, goal = get_start_goal(size, dim)

                    for trial in range(1, NUM_TRIALS + 1):
                        print(f"  第 {trial} 次试运行...", end='', flush=True)

                        # 生成地图（每次独立生成）
                        try:
                            grid = generate_map_with_path(
                                size=size,
                                obstacle_density=density,
                                start=start,
                                goal=goal,
                                dim=dim,
                                step_factor=2.5,      # 蜿蜒程度
                                p=0.7,
                                max_attempts=50
                            )
                        except RuntimeError as e:
                            print(f" 地图生成失败: {e}")
                            continue

                        # 保存地图文件（可选，为了节省空间只保存第一张？这里每张都保存）
                        map_filename = f"{dim_label}_{size[0]}x{size[1]}" + (f"x{size[2]}" if dim==3 else "") + f"_dens{density}_trial{trial}.npy"
                        map_path = MAP_DIR / map_filename
                        np.save(map_path, grid)

                        # 测试三种算法
                        # Astar
                        success, t = run_algorithm('Astar', astar, grid, start, goal, dim)
                        writer.writerow([dim_label, size, density, trial, 'Astar', success, f"{t:.3f}"])
                        # JPS
                        if dim == 2:
                            success, t = run_algorithm('JPS', jps_2d, grid, start, goal, dim)
                        else:
                            success, t = run_algorithm('JPS', jps_3d, grid, start, goal, dim)
                        writer.writerow([dim_label, size, density, trial, 'JPS', success, f"{t:.3f}"])
                        # RRT
                        if dim == 2:
                            success, t = run_algorithm('RRT', rrt_2d, grid, start, goal, dim,
                                                       step_size=RRT_STEP_SIZE, max_iter=RRT_MAX_ITER)
                        else:
                            success, t = run_algorithm('RRT', rrt_3d, grid, start, goal, dim,
                                                       step_size=RRT_STEP_SIZE, max_iter=RRT_MAX_ITER)
                        writer.writerow([dim_label, size, density, trial, 'RRT', success, f"{t:.3f}"])

                        print(" 完成")
                        f_detail.flush()   # 实时写入

    # 生成汇总统计
    print("\n===== 生成汇总统计 =====")
    with open(DETAIL_LOG, 'r') as f_detail, open(SUMMARY_LOG, 'w', newline='') as f_sum:
        reader = csv.reader(f_detail)
        header = next(reader)  # 跳过表头
        writer_sum = csv.writer(f_sum)
        writer_sum.writerow(['dim', 'size', 'density', 'algorithm', 'success_rate', 'avg_time_ms', 'num_success'])

        # 读取所有数据
        data = {}
        for row in reader:
            dim, size_str, density, trial, alg, success_str, time_str = row
            # 将 size 字符串转换为元组（用于排序）
            if dim == '2D':
                size = tuple(map(int, size_str.strip('()').split(',')))
            else:
                size = tuple(map(int, size_str.strip('()').split(',')))
            density = float(density)
            success = success_str == 'True'
            time_ms = float(time_str) if success else None

            key = (dim, size, density, alg)
            if key not in data:
                data[key] = {'times': [], 'success_count': 0, 'total': 0}
            data[key]['total'] += 1
            if success:
                data[key]['success_count'] += 1
                data[key]['times'].append(time_ms)

        # 输出汇总
        for (dim, size, density, alg), stats in sorted(data.items()):
            success_rate = stats['success_count'] / stats['total']
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0.0
            writer_sum.writerow([dim, size, density, alg, f"{success_rate*100:.1f}%", f"{avg_time:.3f}", stats['success_count']])
            print(f"{dim} {size} 密度{density} {alg}: 成功率{success_rate*100:.1f}% 平均时间{avg_time:.3f}ms (成功{stats['success_count']}/{stats['total']})")

    print(f"\n详细日志保存在: {DETAIL_LOG}")
    print(f"汇总日志保存在: {SUMMARY_LOG}")

if __name__ == "__main__":
    main()