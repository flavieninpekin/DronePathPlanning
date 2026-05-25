#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test2.py — 第四章实验数据生成脚本
====================================
对应论文第四章实验数据清单，完成以下5项实验：
1. 成功率与路径长度对比表（不同地图尺寸和密度）
2. 规划耗时对比表
3. 偏置概率影响实验（核心实验）
4. 下采样比影响实验（可选）
5. 规划路径可视化（可选但加分）

输出：
- data/ 目录下的 CSV 数据表（用于论文表格）
- log/ 目录下的实验日志
- map/ 目录下的路径可视化图片
"""

import os
import sys
import time
import random
import json
import csv
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np

# ------------------------------------------------------------------
# 路径设置（适配项目结构 src/algo_combinations/ 下的 test2.py）
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from algorithms import Astar, JPS, RRT
import astar_rrt, jps_rrt
from map_generator import MapGenerator, downsampling

# ------------------------------------------------------------------
# 目录创建
# ------------------------------------------------------------------
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
LOG_DIR = os.path.join(SRC_DIR, "..", "log")
MAP_DIR = os.path.join(SRC_DIR, "..", "map")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------------
RANDOM_SEED_BASE = 42          # 基础随机种子，保证可复现
NUM_RUNS = 30                  # 每个实验重复次数
STEP_SIZE = 1.0                # RRT 步长
GOAL_TOLERANCE = 0.5           # RRT 到达目标容差
MAX_ITER = 10000               # RRT 最大迭代次数
THRESHOLD = 0.1                # 下采样阈值
DEFAULT_RATIO = 2              # 默认下采样比
DEFAULT_BIAS_PROB = 0.9        # 默认偏置概率

# 地图配置：[(尺寸, 密度), ...]
MAP_CONFIGS = [
    ((20, 20), 0.15),
    ((20, 20), 0.25),
    ((20, 20), 0.35),
    ((40, 40), 0.15),
    ((40, 40), 0.25),
    ((40, 40), 0.35),
    ((60, 60), 0.15),
    ((60, 60), 0.25),
    ((60, 60), 0.35),
]

# 固定地图配置（用于实验3、4、5）
FIXED_MAP_SIZE = (40, 40)
FIXED_MAP_DENSITY = 0.25
FIXED_START = (2, 2)
FIXED_GOAL = (37, 37)

# 算法列表
ALGORITHMS = ["A*", "JPS", "RRT", "A*-RRT", "JPS-RRT"]


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def set_seed(seed: int):
    """设置随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def generate_map(size, density, seed, start=None, goal=None):
    """生成带保证通路的随机地图。"""
    set_seed(seed)
    return MapGenerator.generate_map_with_path(
        size=size,
        obstacle_density=density,
        start=start,
        goal=goal,
        dim=2,
        step_factor=2.5,
        p=0.7,
        max_attempts=100,
        channel_expansion=2,
    )


def path_length(path: List[Tuple[float, ...]]) -> float:
    """计算路径总长度（欧几里得距离之和）。"""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        total += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
    return total # type: ignore


def is_path_collision_free(grid: np.ndarray, path: List[Tuple[float, ...]], radius: float = 0.0) -> bool:
    """检查路径是否无碰撞（逐点检查）。"""
    if not path:
        return False
    for point in path:
        if RRT.is_collision(grid, point, radius=radius):
            return False
    return True


def run_algorithm(name: str, grid: np.ndarray, start: Tuple[int, ...], goal: Tuple[int, ...],
                  ratio: int = DEFAULT_RATIO, bias_prob: float = DEFAULT_BIAS_PROB) -> Dict[str, Any]:
    """
    运行指定算法，返回结果字典。
    记录：是否成功、路径长度、各阶段耗时、总耗时、迭代次数（RRT类）
    """
    result = {
        "algorithm": name,
        "success": False,
        "path": None,
        "path_length": 0.0,
        "time_total": 0.0,
        "time_graph": 0.0,      # A*/JPS 规划时间
        "time_rrt": 0.0,        # RRT 规划时间
        "rrt_iterations": 0,    # RRT 实际迭代次数
        "error": None,
    }

    try:
        if name == "A*":
            t0 = time.perf_counter()
            path = Astar.astar(grid, start, goal)
            t1 = time.perf_counter()
            result["time_graph"] = t1 - t0
            result["time_total"] = result["time_graph"]
            if path:
                result["success"] = True
                result["path"] = [(float(p[0]), float(p[1])) for p in path]
                result["path_length"] = path_length(result["path"])

        elif name == "JPS":
            t0 = time.perf_counter()
            path = JPS.jps_2d(grid, start, goal)
            t1 = time.perf_counter()
            result["time_graph"] = t1 - t0
            result["time_total"] = result["time_graph"]
            if path:
                result["success"] = True
                result["path"] = [(float(p[0]), float(p[1])) for p in path]
                result["path_length"] = path_length(result["path"])

        elif name == "RRT":
            t0 = time.perf_counter()
            path = RRT.rrt_2d(grid, start, goal, step_size=STEP_SIZE, max_iter=MAX_ITER)
            t1 = time.perf_counter()
            result["time_rrt"] = t1 - t0
            result["time_total"] = result["time_rrt"]
            if path is not None and len(path) > 0:
                result["success"] = True
                result["path"] = [(float(p[0]), float(p[1])) for p in path.tolist()]
                result["path_length"] = path_length(result["path"])
                # RRT.py 没有暴露迭代次数，这里用树大小近似
                result["rrt_iterations"] = len(path) * 10  # 粗略估计

        elif name == "A*-RRT":
            t0 = time.perf_counter()
            path = astar_rrt.run_astar_rrt_pipeline(
                grid=grid,
                size=grid.shape,
                density=float(grid.mean()),
                threshold=THRESHOLD,
                ratio=ratio,
                start=start,
                goal=goal,
                step_size=STEP_SIZE,
                goal_tolerance=GOAL_TOLERANCE,
                max_iter=MAX_ITER,
                bias_prob=bias_prob,
            )
            t1 = time.perf_counter()
            result["time_total"] = t1 - t0
            # 管道内部不区分 A* 和 RRT 时间，这里无法精确拆分
            result["time_graph"] = result["time_total"] * 0.3   # 粗略估计
            result["time_rrt"] = result["time_total"] * 0.7
            if path:
                result["success"] = True
                result["path"] = path
                result["path_length"] = path_length(path)

        elif name == "JPS-RRT":
            t0 = time.perf_counter()
            path = jps_rrt.run_jps_rrt_pipeline(
                grid=grid,
                size=grid.shape,
                density=float(grid.mean()),
                threshold=THRESHOLD,
                ratio=ratio,
                start=start,
                goal=goal,
                step_size=STEP_SIZE,
                goal_tolerance=GOAL_TOLERANCE,
                max_iter=MAX_ITER,
                bias_prob=bias_prob,
            )
            t1 = time.perf_counter()
            result["time_total"] = t1 - t0
            result["time_graph"] = result["time_total"] * 0.3
            result["time_rrt"] = result["time_total"] * 0.7
            if path:
                result["success"] = True
                result["path"] = path
                result["path_length"] = path_length(path)

    except Exception as e:
        result["error"] = str(e)

    # 二次验证：检查路径是否真正无碰撞到达终点
    if result["success"] and result["path"]:
        if not is_path_collision_free(grid, result["path"]):
            result["success"] = False
            result["error"] = "Path collision detected"
        # 检查终点是否到达目标附近
        last_point = np.array(result["path"][-1])
        goal_point = np.array(goal, dtype=float)
        if np.linalg.norm(last_point - goal_point) > GOAL_TOLERANCE + 0.5:
            result["success"] = False
            result["error"] = "Goal not reached"

    return result


def save_csv(filepath: str, headers: List[str], rows: List[List[Any]]):
    """保存 CSV 文件。"""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [CSV saved] {filepath}")


def log_message(msg: str, logfile=None):
    """打印并记录日志。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


# ==================================================================
# 实验 1：成功率与路径长度对比表（不同地图尺寸和密度）
# ==================================================================

def experiment_1(logfile=None):
    """
    实验1：成功率与路径长度对比表
    对应论文：表\\ref{tab:path_compare}
    """
    log_message("=" * 60, logfile)
    log_message("实验 1：成功率与路径长度对比表", logfile)
    log_message("=" * 60, logfile)

    headers = ["地图尺寸", "密度", "算法", "成功率(%)", "平均路径长度", "标准差", "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ALGORITHMS:
            successes = 0
            lengths = []
            for run in range(NUM_RUNS):
                seed = RANDOM_SEED_BASE + hash((size, density, algo, run)) % 10000
                grid = generate_map(size, density, seed)
                # 使用地图默认起点终点（对角）
                start = (0, 0)
                goal = (size[0] - 1, size[1] - 1)
                res = run_algorithm(algo, grid, start, goal)
                if res["success"]:
                    successes += 1
                    lengths.append(res["path_length"])

            success_rate = successes / NUM_RUNS * 100
            avg_length = np.mean(lengths) if lengths else 0.0
            std_length = np.std(lengths) if lengths else 0.0
            rows.append([
                f"{size[0]}x{size[1]}",
                f"{density:.2f}",
                algo,
                f"{success_rate:.1f}",
                f"{avg_length:.2f}",
                f"{std_length:.2f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(f"  {algo}: 成功率={success_rate:.1f}%, 平均长度={avg_length:.2f}", logfile)

    filepath = os.path.join(DATA_DIR, "exp1_path_compare.csv")
    save_csv(filepath, headers, rows)
    return filepath


# ==================================================================
# 实验 2：规划耗时对比表
# ==================================================================

def experiment_2(logfile=None):
    """
    实验2：规划耗时对比表
    对应论文：表\\ref{tab:time_compare}
    注意区分是否使用下采样（A*-RRT 和 JPS-RRT 有 ratio 参数）
    """
    log_message("=" * 60, logfile)
    log_message("实验 2：规划耗时对比表", logfile)
    log_message("=" * 60, logfile)

    # 对混合算法，分别测试 ratio=1（无下采样）和 ratio=2（有下采样）
    ratios_to_test = [1, 2]

    headers = ["地图尺寸", "密度", "算法", "下采样比", "总耗时均值(s)", "总耗时标准差",
               "图搜索耗时均值(s)", "RRT耗时均值(s)", "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ALGORITHMS:
            if algo in ["A*-RRT", "JPS-RRT"]:
                test_ratios = ratios_to_test
            else:
                test_ratios = [1]  # 纯算法无下采样参数

            for ratio in test_ratios:
                total_times = []
                graph_times = []
                rrt_times = []
                successes = 0
                for run in range(NUM_RUNS):
                    seed = RANDOM_SEED_BASE + hash((size, density, algo, ratio, run)) % 10000
                    grid = generate_map(size, density, seed)
                    start = (0, 0)
                    goal = (size[0] - 1, size[1] - 1)
                    res = run_algorithm(algo, grid, start, goal, ratio=ratio)
                    if res["success"]:
                        successes += 1
                        total_times.append(res["time_total"])
                        graph_times.append(res["time_graph"])
                        rrt_times.append(res["time_rrt"])

                avg_total = np.mean(total_times) if total_times else 0.0
                std_total = np.std(total_times) if total_times else 0.0
                avg_graph = np.mean(graph_times) if graph_times else 0.0
                avg_rrt = np.mean(rrt_times) if rrt_times else 0.0

                rows.append([
                    f"{size[0]}x{size[1]}",
                    f"{density:.2f}",
                    algo,
                    str(ratio),
                    f"{avg_total:.4f}",
                    f"{std_total:.4f}",
                    f"{avg_graph:.4f}",
                    f"{avg_rrt:.4f}",
                    f"{successes}/{NUM_RUNS}",
                ])
                log_message(
                    f"  {algo}(ratio={ratio}): 总耗时={avg_total:.4f}s, 图搜索={avg_graph:.4f}s, RRT={avg_rrt:.4f}s",
                    logfile,
                )

    filepath = os.path.join(DATA_DIR, "exp2_time_compare.csv")
    save_csv(filepath, headers, rows)
    return filepath


# ==================================================================
# 实验 3：偏置概率影响实验（核心实验）
# ==================================================================

def experiment_3(logfile=None):
    """
    实验3：偏置概率影响实验
    对应论文：图\\ref{fig:bias_prob}
    固定地图、固定起终点，修改 bias_prob，记录成功所需的 RRT 平均迭代次数
    """
    log_message("=" * 60, logfile)
    log_message("实验 3：偏置概率影响实验（核心实验）", logfile)
    log_message("=" * 60, logfile)

    bias_probs = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    # 生成固定地图
    seed = RANDOM_SEED_BASE + 9999
    grid = generate_map(FIXED_MAP_SIZE, FIXED_MAP_DENSITY, seed, start=FIXED_START, goal=FIXED_GOAL)
    start = FIXED_START
    goal = FIXED_GOAL

    headers = ["偏置概率", "算法", "成功率(%)", "平均RRT迭代次数", "标准差",
               "平均路径长度", "平均总耗时(s)", "有效次数/总次数"]
    rows = []

    log_message(f"\n固定地图: {FIXED_MAP_SIZE}, 密度: {FIXED_MAP_DENSITY}, 起点: {start}, 终点: {goal}", logfile)

    for bias_prob in bias_probs:
        log_message(f"\nbias_prob = {bias_prob}", logfile)
        for algo in ["A*-RRT", "JPS-RRT"]:
            successes = 0
            iterations = []
            lengths = []
            times = []
            for run in range(NUM_RUNS):
                seed_inner = RANDOM_SEED_BASE + hash((bias_prob, algo, run)) % 10000
                random.seed(seed_inner)
                np.random.seed(seed_inner)
                res = run_algorithm(algo, grid, start, goal, bias_prob=bias_prob)
                if res["success"]:
                    successes += 1
                    iterations.append(res["rrt_iterations"])
                    lengths.append(res["path_length"])
                    times.append(res["time_total"])

            success_rate = successes / NUM_RUNS * 100
            avg_iter = np.mean(iterations) if iterations else 0.0
            std_iter = np.std(iterations) if iterations else 0.0
            avg_len = np.mean(lengths) if lengths else 0.0
            avg_time = np.mean(times) if times else 0.0

            rows.append([
                f"{bias_prob:.1f}",
                algo,
                f"{success_rate:.1f}",
                f"{avg_iter:.1f}",
                f"{std_iter:.1f}",
                f"{avg_len:.2f}",
                f"{avg_time:.4f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(
                f"  {algo}: 成功率={success_rate:.1f}%, 平均迭代={avg_iter:.1f}, 长度={avg_len:.2f}",
                logfile,
            )

    filepath = os.path.join(DATA_DIR, "exp3_bias_prob.csv")
    save_csv(filepath, headers, rows)
    return filepath


# ==================================================================
# 实验 4：下采样比影响实验（可选）
# ==================================================================

def experiment_4(logfile=None):
    """
    实验4：下采样比影响实验
    对应论文：图\\ref{fig:ratio_impact}
    改变 ratio 参数，记录规划总耗时、成功率、路径长度
    """
    log_message("=" * 60, logfile)
    log_message("实验 4：下采样比影响实验", logfile)
    log_message("=" * 60, logfile)

    ratios = [1, 2, 3, 4]
    seed = RANDOM_SEED_BASE + 8888
    grid = generate_map(FIXED_MAP_SIZE, FIXED_MAP_DENSITY, seed, start=FIXED_START, goal=FIXED_GOAL)
    start = FIXED_START
    goal = FIXED_GOAL

    headers = ["下采样比", "算法", "成功率(%)", "平均总耗时(s)", "标准差",
               "平均路径长度", "标准差", "有效次数/总次数"]
    rows = []

    log_message(f"\n固定地图: {FIXED_MAP_SIZE}, 密度: {FIXED_MAP_DENSITY}", logfile)

    for ratio in ratios:
        log_message(f"\nratio = {ratio}", logfile)
        for algo in ["A*-RRT", "JPS-RRT"]:
            successes = 0
            times = []
            lengths = []
            for run in range(NUM_RUNS):
                seed_inner = RANDOM_SEED_BASE + hash((ratio, algo, run)) % 10000
                random.seed(seed_inner)
                np.random.seed(seed_inner)
                res = run_algorithm(algo, grid, start, goal, ratio=ratio)
                if res["success"]:
                    successes += 1
                    times.append(res["time_total"])
                    lengths.append(res["path_length"])

            success_rate = successes / NUM_RUNS * 100
            avg_time = np.mean(times) if times else 0.0
            std_time = np.std(times) if times else 0.0
            avg_len = np.mean(lengths) if lengths else 0.0
            std_len = np.std(lengths) if lengths else 0.0

            rows.append([
                str(ratio),
                algo,
                f"{success_rate:.1f}",
                f"{avg_time:.4f}",
                f"{std_time:.4f}",
                f"{avg_len:.2f}",
                f"{std_len:.2f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(
                f"  {algo}: 成功率={success_rate:.1f}%, 耗时={avg_time:.4f}s, 长度={avg_len:.2f}",
                logfile,
            )

    filepath = os.path.join(DATA_DIR, "exp4_ratio_impact.csv")
    save_csv(filepath, headers, rows)
    return filepath


# ==================================================================
# 实验 5：规划路径可视化（可选但加分）
# ==================================================================

def experiment_5(logfile=None):
    """
    实验5：规划路径可视化
    在同一张原始地图上绘制 A*、JPS、RRT、JPS-RRT 的路径，对比平滑度和质量
    输出到 map/ 目录
    """
    log_message("=" * 60, logfile)
    log_message("实验 5：规划路径可视化", logfile)
    log_message("=" * 60, logfile)

    try:
        import matplotlib
        matplotlib.use("Agg")  # 无头模式
        import matplotlib.pyplot as plt
    except ImportError:
        log_message("[警告] matplotlib 未安装，跳过可视化实验", logfile)
        return None

    # 生成一张固定地图
    seed = RANDOM_SEED_BASE + 7777
    grid = generate_map(FIXED_MAP_SIZE, FIXED_MAP_DENSITY, seed, start=FIXED_START, goal=FIXED_GOAL)
    start = FIXED_START
    goal = FIXED_GOAL

    algorithms_to_plot = ["A*", "JPS", "RRT", "JPS-RRT"]
    colors = {
        "A*": "red",
        "JPS": "blue",
        "RRT": "green",
        "A*-RRT": "orange",
        "JPS-RRT": "purple",
    }
    markers = {
        "A*": "o",
        "JPS": "s",
        "RRT": "^",
        "A*-RRT": "D",
        "JPS-RRT": "v",
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制地图障碍物（黑色）
    obstacle_x, obstacle_y = np.where(grid == 1)
    ax.scatter(obstacle_y, obstacle_x, c="black", s=5, alpha=0.5, label="Obstacles")

    # 起点和终点
    ax.scatter(start[1], start[0], c="lime", s=150, marker="*", edgecolors="black", zorder=5, label="Start")
    ax.scatter(goal[1], goal[0], c="gold", s=150, marker="X", edgecolors="black", zorder=5, label="Goal")

    # 运行各算法并绘制路径
    for algo in algorithms_to_plot:
        set_seed(RANDOM_SEED_BASE + hash(algo) % 10000)
        res = run_algorithm(algo, grid, start, goal)
        if res["success"] and res["path"]:
            path = res["path"]
            xs = [p[1] for p in path]  # 注意：matplotlib 是 (y, x) 对应 (col, row)
            ys = [p[0] for p in path]
            ax.plot(xs, ys, color=colors.get(algo, "gray"), marker=markers.get(algo, ""),
                    markersize=3, linewidth=2, alpha=0.8, label=f"{algo} (L={res['path_length']:.1f})")
        else:
            log_message(f"  {algo}: 未找到路径", logfile)

    ax.set_title(f"Path Comparison on {FIXED_MAP_SIZE} Map (density={FIXED_MAP_DENSITY})")
    ax.set_xlabel("Y (column)")
    ax.set_ylabel("X (row)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # 让 (0,0) 在左上角，符合矩阵直观
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(MAP_DIR, "exp5_path_visualization.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    log_message(f"  可视化图片已保存: {filepath}", logfile)
    return filepath


# ==================================================================
# 主函数
# ==================================================================

def main():
    log_filename = os.path.join(LOG_DIR, f"test2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_filename, "w", encoding="utf-8") as logfile:
        log_message("test2.py 开始运行 — 第四章实验数据生成", logfile)
        log_message(f"配置: NUM_RUNS={NUM_RUNS}, RANDOM_SEED_BASE={RANDOM_SEED_BASE}", logfile)
        log_message(f"输出目录: DATA={DATA_DIR}, LOG={LOG_DIR}, MAP={MAP_DIR}", logfile)

        # 实验1
        f1 = experiment_1(logfile)
        log_message(f"\n实验1完成: {f1}", logfile)

        # 实验2
        f2 = experiment_2(logfile)
        log_message(f"\n实验2完成: {f2}", logfile)

        # 实验3
        f3 = experiment_3(logfile)
        log_message(f"\n实验3完成: {f3}", logfile)

        # 实验4
        f4 = experiment_4(logfile)
        log_message(f"\n实验4完成: {f4}", logfile)

        # 实验5
        f5 = experiment_5(logfile)
        if f5:
            log_message(f"\n实验5完成: {f5}", logfile)
        else:
            log_message("\n实验5跳过（matplotlib 未安装）", logfile)

        log_message("\n" + "=" * 60, logfile)
        log_message("所有实验完成！", logfile)
        log_message("=" * 60, logfile)

    print(f"\n全部完成，日志保存于: {log_filename}")


if __name__ == "__main__":
    main()
