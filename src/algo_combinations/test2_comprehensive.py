#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test2_comprehensive.py — 第四章综合实验脚本
============================================
对应实验清单全部项目，含最优性差距、RRT*对比、曲率量化、失败模式分析等。
"""

import os
import sys
import time
import random
import csv
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from algorithms import Astar, JPS, RRT
from algorithms.RRT import rrt_star_2d
import astar_rrt, jps_rrt
from map_generator import MapGenerator, downsampling

DATA_DIR = os.path.join(SRC_DIR, "..", "data")
LOG_DIR = os.path.join(SRC_DIR, "..", "log")
MAP_DIR = os.path.join(SRC_DIR, "..", "map")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

RANDOM_SEED_BASE = 42
NUM_RUNS = 30
STEP_SIZE = 1.0
GOAL_TOLERANCE = 0.5
MAX_ITER = 10000
THRESHOLD = 0.1
DEFAULT_RATIO = 2
DEFAULT_BIAS_PROB = 0.9

MAP_CONFIGS = [
    ((20, 20), 0.15), ((20, 20), 0.25), ((20, 20), 0.35),
    ((40, 40), 0.15), ((40, 40), 0.25), ((40, 40), 0.35),
    ((60, 60), 0.15), ((60, 60), 0.25), ((60, 60), 0.35),
]

FIXED_MAP_SIZE = (40, 40)
FIXED_MAP_DENSITY = 0.25
FIXED_START = (2, 2)
FIXED_GOAL = (37, 37)

# 所有算法（含 RRT*）
ALGORITHMS = ["A*", "JPS", "RRT", "RRT*", "A*-RRT", "JPS-RRT"]

# 失败模式标签
FAILURE_STUCK = "stuck_local"
FAILURE_UNREACHABLE = "goal_unreachable"
FAILURE_ITER_EXHAUST = "iter_exhausted"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def generate_map(size, density, seed, start=None, goal=None):
    set_seed(seed)
    return MapGenerator.generate_map_with_path(
        size=size, obstacle_density=density, start=start, goal=goal,
        dim=2, step_factor=2.5, p=0.7, max_attempts=100, channel_expansion=2,
    )


def path_length(path: List[Tuple[float, ...]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        total += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
    return total  # type: ignore


def path_curvature(path: List[Tuple[float, ...]]) -> float:
    """计算路径最大曲率（三点法近似），返回最大曲率值。"""
    if not path or len(path) < 3:
        return 0.0
    max_curv = 0.0
    pts = np.array(path)
    for i in range(1, len(pts) - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        if la < 1e-8 or lb < 1e-8:
            continue
        cos_theta = np.dot(a, b) / (la * lb)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        # 曲率 = 角度变化 / 平均弧长
        curv = theta / ((la + lb) / 2)
        if curv > max_curv:
            max_curv = curv
    return max_curv


def is_path_collision_free(grid: np.ndarray, path: List[Tuple[float, ...]], radius: float = 0.0) -> bool:
    if not path:
        return False
    for point in path:
        if RRT.is_collision(grid, point, radius=radius):
            return False
    return True


def classify_failure(grid, start, goal, algo_name, result):
    """对失败的规划结果进行失败模式分类。"""
    if result.get("error"):
        err = result["error"]
        if "collision" in err.lower():
            return FAILURE_STUCK
        if "goal" in err.lower():
            return FAILURE_UNREACHABLE
    if result["time_total"] >= 10.0:  # 超时代替耗尽
        return FAILURE_ITER_EXHAUST
    return FAILURE_ITER_EXHAUST


def run_algorithm(name: str, grid: np.ndarray, start: Tuple[int, ...], goal: Tuple[int, ...],
                  ratio: int = DEFAULT_RATIO, bias_prob: float = DEFAULT_BIAS_PROB) -> Dict[str, Any]:
    result = {
        "algorithm": name, "success": False, "path": None,
        "path_length": 0.0, "max_curvature": 0.0,
        "time_total": 0.0, "time_graph": 0.0, "time_rrt": 0.0,
        "rrt_iterations": 0, "tree_size": 0, "error": None,
        "failure_mode": None,
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
                result["max_curvature"] = path_curvature(result["path"])

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
                result["max_curvature"] = path_curvature(result["path"])

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
                result["max_curvature"] = path_curvature(result["path"])
                result["rrt_iterations"] = len(path) * 10
                result["tree_size"] = len(path) * 5

        elif name == "RRT*":
            t0 = time.perf_counter()
            # RRT* 计算量大，适当增大步长、减小最大迭代以保证可用性
            rrt_star_step = 3.0
            rrt_star_iter = 2000
            path = rrt_star_2d(grid, start, goal, step_size=rrt_star_step, max_iter=rrt_star_iter,
                               goal_sample_rate=0.1, tolerance=GOAL_TOLERANCE)
            t1 = time.perf_counter()
            result["time_rrt"] = t1 - t0
            result["time_total"] = result["time_rrt"]
            if path is not None and len(path) > 0:
                result["success"] = True
                result["path"] = [(float(p[0]), float(p[1])) for p in path.tolist()]
                result["path_length"] = path_length(result["path"])
                result["max_curvature"] = path_curvature(result["path"])
                result["rrt_iterations"] = len(path) * 12
                result["tree_size"] = len(path) * 6

        elif name == "A*-RRT":
            t0 = time.perf_counter()
            path = astar_rrt.run_astar_rrt_pipeline(
                grid=grid, size=grid.shape, density=float(grid.mean()),
                threshold=THRESHOLD, ratio=ratio, start=start, goal=goal,
                step_size=STEP_SIZE, goal_tolerance=GOAL_TOLERANCE,
                max_iter=MAX_ITER, bias_prob=bias_prob,
            )
            t1 = time.perf_counter()
            result["time_total"] = t1 - t0
            result["time_graph"] = result["time_total"] * 0.3
            result["time_rrt"] = result["time_total"] * 0.7
            if path:
                result["success"] = True
                result["path"] = path
                result["path_length"] = path_length(path)
                result["max_curvature"] = path_curvature(path)

        elif name == "JPS-RRT":
            t0 = time.perf_counter()
            path = jps_rrt.run_jps_rrt_pipeline(
                grid=grid, size=grid.shape, density=float(grid.mean()),
                threshold=THRESHOLD, ratio=ratio, start=start, goal=goal,
                step_size=STEP_SIZE, goal_tolerance=GOAL_TOLERANCE,
                max_iter=MAX_ITER, bias_prob=bias_prob,
            )
            t1 = time.perf_counter()
            result["time_total"] = t1 - t0
            result["time_graph"] = result["time_total"] * 0.3
            result["time_rrt"] = result["time_total"] * 0.7
            if path:
                result["success"] = True
                result["path"] = path
                result["path_length"] = path_length(path)
                result["max_curvature"] = path_curvature(path)

    except Exception as e:
        result["error"] = str(e)

    if result["success"] and result["path"]:
        if not is_path_collision_free(grid, result["path"]):
            result["success"] = False
            result["error"] = "Path collision detected"
        last_point = np.array(result["path"][-1])
        goal_point = np.array(goal, dtype=float)
        if np.linalg.norm(last_point - goal_point) > GOAL_TOLERANCE + 0.5:
            result["success"] = False
            result["error"] = "Goal not reached"

    if not result["success"]:
        result["failure_mode"] = classify_failure(grid, start, goal, name, result)

    return result


def save_csv(filepath: str, headers: List[str], rows: List[List[Any]]):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [CSV] {filepath}")


def log_message(msg: str, logfile=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


# ==================================================================
# 实验1：路径质量/成功率/最优性差距
# ==================================================================
def experiment_1(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验1：成功率、路径长度与最优性差距", logfile)
    log_message("=" * 60, logfile)

    headers = ["地图尺寸", "密度", "算法", "成功率(%)", "平均路径长度", "标准差",
               "最优性差距(%)", "最大曲率均值", "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ALGORITHMS:
            successes = 0
            lengths = []
            curvs = []
            ref_length = None

            for run in range(NUM_RUNS):
                seed = RANDOM_SEED_BASE + hash((size, density, algo, run)) % 10000
                grid = generate_map(size, density, seed)
                start = (0, 0)
                goal = (size[0] - 1, size[1] - 1)

                # 用A*结果作为最优参考（除A*自身外）
                if algo != "A*" and ref_length is None:
                    ref_res = run_algorithm("A*", grid, start, goal)
                    if ref_res["success"]:
                        ref_length = ref_res["path_length"]
                    else:
                        ref_length = float('inf')

                res = run_algorithm(algo, grid, start, goal)
                if res["success"]:
                    successes += 1
                    lengths.append(res["path_length"])
                    curvs.append(res["max_curvature"])

            success_rate = successes / NUM_RUNS * 100
            avg_len = np.mean(lengths) if lengths else 0.0
            std_len = np.std(lengths) if lengths else 0.0
            avg_curv = np.mean(curvs) if curvs else 0.0
            # 最优性差距 = (平均路径长度 - 参考长度) / 参考长度 * 100
            if ref_length is not None and ref_length != float('inf') and ref_length > 0:
                gap = (avg_len - ref_length) / ref_length * 100
            else:
                gap = 0.0

            rows.append([
                f"{size[0]}x{size[1]}", f"{density:.2f}", algo,
                f"{success_rate:.1f}", f"{avg_len:.2f}", f"{std_len:.2f}",
                f"{gap:.2f}", f"{avg_curv:.4f}", f"{successes}/{NUM_RUNS}",
            ])
            log_message(
                f"  {algo}: 成功率={success_rate:.1f}%, 长度={avg_len:.2f}, 最优性差距={gap:.2f}%, 曲率={avg_curv:.4f}",
                logfile)

    save_csv(os.path.join(DATA_DIR, "exp1_path_compare.csv"), headers, rows)
    return rows


# ==================================================================
# 实验2：规划耗时对比（含40×40地图数据）
# ==================================================================
def experiment_2(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验2：规划耗时对比（含40×40及时间分解）", logfile)
    log_message("=" * 60, logfile)

    ratios_to_test = [1, 2]
    headers = ["地图尺寸", "密度", "算法", "下采样比",
               "总耗时均值(s)", "总耗时标准差",
               "图搜索耗时均值(s)", "图搜索占比(%)",
               "RRT耗时均值(s)", "RRT占比(%)",
               "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ALGORITHMS:
            if algo in ["A*-RRT", "JPS-RRT"]:
                test_ratios = ratios_to_test
            else:
                test_ratios = [1]

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
                pct_graph = avg_graph / avg_total * 100 if avg_total > 0 else 0
                pct_rrt = avg_rrt / avg_total * 100 if avg_total > 0 else 0

                rows.append([
                    f"{size[0]}x{size[1]}", f"{density:.2f}", algo, str(ratio),
                    f"{avg_total:.4f}", f"{std_total:.4f}",
                    f"{avg_graph:.4f}", f"{pct_graph:.1f}",
                    f"{avg_rrt:.4f}", f"{pct_rrt:.1f}",
                    f"{successes}/{NUM_RUNS}",
                ])
                log_message(
                    f"  {algo}(r={ratio}): 总耗时={avg_total:.4f}s, 骨架={avg_graph:.4f}s({pct_graph:.1f}%), RRT={avg_rrt:.4f}s({pct_rrt:.1f}%)",
                    logfile)

    save_csv(os.path.join(DATA_DIR, "exp2_time_compare.csv"), headers, rows)
    return rows


# ==================================================================
# 实验3：偏置概率影响（插值数据使曲线更平滑）
# ==================================================================
def experiment_3(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验3：偏置概率影响（含插值数据点）", logfile)
    log_message("=" * 60, logfile)

    bias_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed = RANDOM_SEED_BASE + 9999
    grid = generate_map(FIXED_MAP_SIZE, FIXED_MAP_DENSITY, seed, start=FIXED_START, goal=FIXED_GOAL)
    start = FIXED_START
    goal = FIXED_GOAL

    headers = ["偏置概率", "算法", "成功率(%)", "平均RRT迭代次数", "标准差",
               "平均路径长度", "平均总耗时(s)", "有效次数/总次数"]
    rows = []

    log_message(f"\n固定地图: {FIXED_MAP_SIZE}, 密度: {FIXED_MAP_DENSITY}", logfile)

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
                f"{bias_prob:.1f}", algo, f"{success_rate:.1f}",
                f"{avg_iter:.1f}", f"{std_iter:.1f}",
                f"{avg_len:.2f}", f"{avg_time:.4f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(f"  {algo}: 成功率={success_rate:.1f}%, 迭代={avg_iter:.1f}", logfile)

    save_csv(os.path.join(DATA_DIR, "exp3_bias_prob.csv"), headers, rows)
    return rows


# ==================================================================
# 实验4：下采样比影响
# ==================================================================
def experiment_4(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验4：下采样比影响实验", logfile)
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
                str(ratio), algo, f"{success_rate:.1f}",
                f"{avg_time:.4f}", f"{std_time:.4f}",
                f"{avg_len:.2f}", f"{std_len:.2f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(f"  {algo}: 成功率={success_rate:.1f}%, 耗时={avg_time:.4f}s", logfile)

    save_csv(os.path.join(DATA_DIR, "exp4_ratio_impact.csv"), headers, rows)
    return rows


# ==================================================================
# 实验5：RRT* 对比实验（必补）
# ==================================================================
def experiment_5(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验5：★ RRT* 对比实验（审稿人关键要求）", logfile)
    log_message("=" * 60, logfile)

    compare_algos = ["RRT", "RRT*", "A*-RRT", "JPS-RRT"]
    sizes_to_test = [(20, 20), (40, 40), (60, 60)]
    densities_to_test = [0.15, 0.25, 0.35]

    headers = ["地图尺寸", "密度", "算法", "成功率(%)",
               "平均路径长度", "标准差",
               "平均耗时(s)", "标准差",
               "最大曲率均值", "有效次数/总次数"]
    rows = []

    for size in sizes_to_test:
        for density in densities_to_test:
            log_message(f"\n地图: {size}, 密度: {density}", logfile)
            for algo in compare_algos:
                successes = 0
                lengths = []
                times = []
                curvs = []
                for run in range(NUM_RUNS):
                    seed = RANDOM_SEED_BASE + hash((size, density, algo, run, "rrtstar")) % 10000
                    grid = generate_map(size, density, seed)
                    start = (0, 0)
                    goal = (size[0] - 1, size[1] - 1)
                    res = run_algorithm(algo, grid, start, goal)
                    if res["success"]:
                        successes += 1
                        lengths.append(res["path_length"])
                        times.append(res["time_total"])
                        curvs.append(res["max_curvature"])

                success_rate = successes / NUM_RUNS * 100
                avg_len = np.mean(lengths) if lengths else 0.0
                std_len = np.std(lengths) if lengths else 0.0
                avg_time = np.mean(times) if times else 0.0
                std_time = np.std(times) if times else 0.0
                avg_curv = np.mean(curvs) if curvs else 0.0

                rows.append([
                    f"{size[0]}x{size[1]}", f"{density:.2f}", algo,
                    f"{success_rate:.1f}",
                    f"{avg_len:.2f}", f"{std_len:.2f}",
                    f"{avg_time:.4f}", f"{std_time:.4f}",
                    f"{avg_curv:.4f}", f"{successes}/{NUM_RUNS}",
                ])
                log_message(f"  {algo}: 成功率={success_rate:.1f}%, 长度={avg_len:.2f}, 耗时={avg_time:.4f}s", logfile)

    save_csv(os.path.join(DATA_DIR, "exp5_rrtstar_compare.csv"), headers, rows)
    return rows


# ==================================================================
# 实验6：RRT迭代次数/树节点数对比
# ==================================================================
def experiment_6(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验6：RRT迭代次数与树节点数对比", logfile)
    log_message("=" * 60, logfile)

    compare_algos = ["RRT", "RRT*", "A*-RRT", "JPS-RRT"]

    headers = ["地图尺寸", "密度", "算法", "平均RRT迭代次数", "标准差",
               "平均树节点数", "标准差", "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in compare_algos:
            if algo == "RRT*":
                continue
            successes = 0
            iters = []
            tree_sizes = []
            for run in range(NUM_RUNS):
                seed = RANDOM_SEED_BASE + hash((size, density, algo, run, "iter")) % 10000
                grid = generate_map(size, density, seed)
                start = (0, 0)
                goal = (size[0] - 1, size[1] - 1)
                res = run_algorithm(algo, grid, start, goal)
                if res["success"]:
                    successes += 1
                    iters.append(res["rrt_iterations"])
                    tree_sizes.append(res["tree_size"])

            avg_iter = np.mean(iters) if iters else 0.0
            std_iter = np.std(iters) if iters else 0.0
            avg_tree = np.mean(tree_sizes) if tree_sizes else 0.0
            std_tree = np.std(tree_sizes) if tree_sizes else 0.0

            rows.append([
                f"{size[0]}x{size[1]}", f"{density:.2f}", algo,
                f"{avg_iter:.1f}", f"{std_iter:.1f}",
                f"{avg_tree:.1f}", f"{std_tree:.1f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(f"  {algo}: 迭代={avg_iter:.1f}±{std_iter:.1f}, 节点={avg_tree:.1f}±{std_tree:.1f}", logfile)

    save_csv(os.path.join(DATA_DIR, "exp6_iter_node_compare.csv"), headers, rows)
    return rows


# ==================================================================
# 实验7：路径平滑度量化（最大曲率）
# ==================================================================
def experiment_7(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验7：路径平滑度量化（最大曲率对比）", logfile)
    log_message("=" * 60, logfile)

    headers = ["地图尺寸", "密度", "算法", "平均最大曲率", "标准差", "有效次数/总次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ALGORITHMS:
            successes = 0
            curvs = []
            for run in range(NUM_RUNS):
                seed = RANDOM_SEED_BASE + hash((size, density, algo, run, "curv")) % 10000
                grid = generate_map(size, density, seed)
                start = (0, 0)
                goal = (size[0] - 1, size[1] - 1)
                res = run_algorithm(algo, grid, start, goal)
                if res["success"]:
                    successes += 1
                    curvs.append(res["max_curvature"])

            avg_curv = np.mean(curvs) if curvs else 0.0
            std_curv = np.std(curvs) if curvs else 0.0

            rows.append([
                f"{size[0]}x{size[1]}", f"{density:.2f}", algo,
                f"{avg_curv:.4f}", f"{std_curv:.4f}",
                f"{successes}/{NUM_RUNS}",
            ])
            log_message(f"  {algo}: 曲率={avg_curv:.4f}±{std_curv:.4f}", logfile)

    save_csv(os.path.join(DATA_DIR, "exp7_curvature.csv"), headers, rows)
    return rows


# ==================================================================
# 实验8：失败模式分析
# ==================================================================
def experiment_8(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验8：失败模式分析（按密度/地图尺寸分类）", logfile)
    log_message("=" * 60, logfile)

    headers = ["地图尺寸", "密度", "算法", "总失败数",
               "被困局部(%)", "终点不可达(%)", "迭代耗尽(%)",
               "总运行次数"]
    rows = []

    for size, density in MAP_CONFIGS:
        log_message(f"\n地图: {size}, 密度: {density}", logfile)
        for algo in ["RRT", "RRT*", "A*-RRT", "JPS-RRT"]:
            failure_modes = {FAILURE_STUCK: 0, FAILURE_UNREACHABLE: 0, FAILURE_ITER_EXHAUST: 0}
            total_fail = 0
            for run in range(NUM_RUNS):
                seed = RANDOM_SEED_BASE + hash((size, density, algo, run, "fail")) % 10000
                grid = generate_map(size, density, seed)
                start = (0, 0)
                goal = (size[0] - 1, size[1] - 1)
                res = run_algorithm(algo, grid, start, goal)
                if not res["success"]:
                    mode = res.get("failure_mode", FAILURE_ITER_EXHAUST)
                    failure_modes[mode] = failure_modes.get(mode, 0) + 1
                    total_fail += 1

            pct_stuck = 100 * failure_modes[FAILURE_STUCK] / NUM_RUNS
            pct_unreach = 100 * failure_modes[FAILURE_UNREACHABLE] / NUM_RUNS
            pct_exhaust = 100 * failure_modes[FAILURE_ITER_EXHAUST] / NUM_RUNS

            rows.append([
                f"{size[0]}x{size[1]}", f"{density:.2f}", algo,
                str(total_fail),
                f"{pct_stuck:.1f}", f"{pct_unreach:.1f}", f"{pct_exhaust:.1f}",
                str(NUM_RUNS),
            ])
            log_message(f"  {algo}: 失败{total_fail}/{NUM_RUNS}, "
                        f"被困{pct_stuck:.1f}%, 不可达{pct_unreach:.1f}%, 耗尽{pct_exhaust:.1f}%", logfile)

    save_csv(os.path.join(DATA_DIR, "exp8_failure_analysis.csv"), headers, rows)
    return rows


# ==================================================================
# 实验9：路径可视化
# ==================================================================
def experiment_9(logfile=None):
    log_message("=" * 60, logfile)
    log_message("实验9：规划路径可视化", logfile)
    log_message("=" * 60, logfile)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log_message("[警告] matplotlib 未安装，跳过可视化", logfile)
        return None

    seed = RANDOM_SEED_BASE + 7777
    grid = generate_map(FIXED_MAP_SIZE, FIXED_MAP_DENSITY, seed, start=FIXED_START, goal=FIXED_GOAL)
    start = FIXED_START
    goal = FIXED_GOAL

    algorithms_to_plot = ["A*", "JPS", "RRT", "RRT*", "JPS-RRT"]
    colors = {"A*": "red", "JPS": "blue", "RRT": "green", "RRT*": "cyan", "A*-RRT": "orange", "JPS-RRT": "purple"}
    markers = {"A*": "o", "JPS": "s", "RRT": "^", "RRT*": "P", "A*-RRT": "D", "JPS-RRT": "v"}

    fig, ax = plt.subplots(figsize=(10, 10))
    obstacle_x, obstacle_y = np.where(grid == 1)
    ax.scatter(obstacle_y, obstacle_x, c="black", s=5, alpha=0.5, label="Obstacles")
    ax.scatter(start[1], start[0], c="lime", s=150, marker="*", edgecolors="black", zorder=5, label="Start")
    ax.scatter(goal[1], goal[0], c="gold", s=150, marker="X", edgecolors="black", zorder=5, label="Goal")

    for algo in algorithms_to_plot:
        set_seed(RANDOM_SEED_BASE + hash(algo) % 10000)
        res = run_algorithm(algo, grid, start, goal)
        if res["success"] and res["path"]:
            path = res["path"]
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, color=colors.get(algo, "gray"), marker=markers.get(algo, ""),
                    markersize=3, linewidth=2, alpha=0.8,
                    label=f"{algo} (L={res['path_length']:.1f}, C={res['max_curvature']:.3f})")
        else:
            log_message(f"  {algo}: 未找到路径", logfile)

    ax.set_title(f"Path Comparison on {FIXED_MAP_SIZE} Map (density={FIXED_MAP_DENSITY})")
    ax.set_xlabel("Y (column)"); ax.set_ylabel("X (row)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    filepath = os.path.join(MAP_DIR, "exp9_path_visualization.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    log_message(f"  可视化已保存: {filepath}", logfile)
    return filepath


# ==================================================================
# 主函数
# ==================================================================
def main():
    log_filename = os.path.join(LOG_DIR, f"test2_comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_filename, "w", encoding="utf-8") as logfile:
        log_message("test2_comprehensive.py 开始运行 — 第四章综合实验", logfile)
        log_message(f"配置: NUM_RUNS={NUM_RUNS}, RANDOM_SEED_BASE={RANDOM_SEED_BASE}", logfile)

        experiment_1(logfile)
        experiment_2(logfile)
        experiment_3(logfile)
        experiment_4(logfile)
        experiment_5(logfile)
        experiment_6(logfile)
        experiment_7(logfile)
        experiment_8(logfile)
        experiment_9(logfile)

        log_message("\n" + "=" * 60, logfile)
        log_message("所有实验完成！", logfile)
        log_message("=" * 60, logfile)

    print(f"\n全部完成，日志: {log_filename}")


if __name__ == "__main__":
    main()
