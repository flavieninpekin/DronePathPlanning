import math
import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ..algorithms import JPS
    from ..algorithms.RRT import get_nearest, is_collision, steer
    from ..map_generator import downsampling
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from algorithms import JPS
    from algorithms.RRT import get_nearest, is_collision, steer
    from map_generator import downsampling


def _crop_grid_for_ratio(grid: np.ndarray, ratio: int) -> np.ndarray:
    if ratio <= 1:
        return grid

    if grid.ndim == 2:
        h, w = grid.shape
        h_crop = (h // ratio) * ratio
        w_crop = (w // ratio) * ratio
        return grid[:h_crop, :w_crop]

    if grid.ndim == 3:
        d, h, w = grid.shape
        d_crop = (d // ratio) * ratio
        h_crop = (h // ratio) * ratio
        w_crop = (w // ratio) * ratio
        return grid[:d_crop, :h_crop, :w_crop]

    raise ValueError("grid must be 2D or 3D")


def _validate_size(grid: np.ndarray, size: Optional[Sequence[int]]):
    if size is None:
        return
    if tuple(size) != tuple(grid.shape):
        raise ValueError(f"Provided size {tuple(size)} does not match grid shape {grid.shape}")


def _find_default_endpoints(grid: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise ValueError("Grid contains no free cells")
    sums = np.sum(free, axis=1)
    start_idx = int(np.argmin(sums))
    goal_idx = int(np.argmax(sums))
    start = tuple(int(x) for x in free[start_idx])
    goal = tuple(int(x) for x in free[goal_idx])
    return start, goal


def _scale_point_to_downsample(point: Sequence[float], ratio: int, shape: Sequence[int]) -> Tuple[int, ...]:
    scaled = []
    for i, coord in enumerate(point):
        idx = int(math.floor(coord / ratio))
        idx = max(0, min(idx, shape[i] - 1))
        scaled.append(idx)
    return tuple(scaled)


def _scale_path_to_original(path: List[Tuple[int, ...]], ratio: int) -> List[Tuple[float, ...]]:
    scaled = []
    for node in path:
        scaled.append(tuple(float(v * ratio + ratio / 2.0) for v in node))
    return scaled


def _sample_uniform(grid: np.ndarray) -> np.ndarray:
    dims = grid.ndim
    if dims == 2:
        return np.array([random.random() * grid.shape[0], random.random() * grid.shape[1]])
    return np.array([random.random() * grid.shape[0], random.random() * grid.shape[1], random.random() * grid.shape[2]])


def _sample_near_jps_point(jps_points: List[Tuple[float, ...]], grid: np.ndarray, ratio: int) -> np.ndarray:
    if not jps_points:
        return _sample_uniform(grid)
    max_attempts = 50
    dims = grid.ndim
    for _ in range(max_attempts):
        base = np.array(random.choice(jps_points))
        offset = np.random.uniform(-ratio, ratio, size=dims)
        candidate = base + offset
        if any(candidate < 0) or any(candidate >= np.array(grid.shape)):
            continue
        if not is_collision(grid, candidate):
            return candidate
    return _sample_uniform(grid)


def _sample_rrt_point(grid: np.ndarray, jps_points: List[Tuple[float, ...]], ratio: int, bias_prob: float) -> np.ndarray:
    if random.random() < bias_prob and jps_points:
        return _sample_near_jps_point(jps_points, grid, ratio)
    return _sample_uniform(grid)


def _ensure_valid_endpoint(grid: np.ndarray, point: Sequence[int]) -> Tuple[int, ...]:
    if all(0 <= coord < grid.shape[i] for i, coord in enumerate(point)) and grid[tuple(point)] == 0:
        return tuple(int(coord) for coord in point)
    return _find_default_endpoints(grid)[0]


def _rrt_biased(
    grid: np.ndarray,
    start: Sequence[float],
    goal: Sequence[float],
    jps_points: List[Tuple[float, ...]],
    ratio: int,
    bias_prob: float,
    step_size: float,
    goal_tolerance: float,
    max_iter: int,
) -> Optional[List[Tuple[float, ...]]]:
    start_point = np.array(start, dtype=float)
    goal_point = np.array(goal, dtype=float)
    tree = [start_point]
    parents = [(start_point, -1)]

    for _ in range(max_iter):
        rand_point = _sample_rrt_point(grid, jps_points, ratio, bias_prob)
        nearest_idx = get_nearest(tree, rand_point)
        nearest_idx = int(nearest_idx)
        new_point = steer(tree[nearest_idx], rand_point, step_size)
        if is_collision(grid, new_point, radius=0.0):
            continue
        tree.append(new_point)
        parents.append((new_point, nearest_idx))
        if np.linalg.norm(new_point - goal_point) < goal_tolerance and not is_collision(grid, goal_point, radius=0.0):
            tree.append(goal_point)
            parents.append((goal_point, len(tree) - 2))
            path = _backtrace(parents, len(tree) - 1)
            return path
    return None


def _backtrace(parents: List[Tuple[np.ndarray, int]], end_index: int) -> List[Tuple[float, ...]]:
    path: List[Tuple[float, ...]] = []
    idx = end_index
    while idx != -1:
        path.append(tuple(parents[idx][0].tolist()))
        idx = parents[idx][1]
    return path[::-1]


def _derive_downsampled_endpoint(point: Sequence[int], ratio: int, shape: Sequence[int]) -> Tuple[int, ...]:
    return _scale_point_to_downsample(point, ratio, shape)


def _run_jps(downsampled: np.ndarray, start: Tuple[int, ...], goal: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    if downsampled.ndim == 2:
        return JPS.jps_2d(downsampled, start, goal)
    if downsampled.ndim == 3:
        return JPS.jps_3d(downsampled, start, goal)
    raise ValueError("JPS currently supports only 2D or 3D grids")


def run_jps_rrt_pipeline(
    grid: Union[np.ndarray, List],
    size: Optional[Sequence[int]],
    density: Optional[float],
    threshold: float,
    ratio: int,
    start: Optional[Sequence[int]] = None,
    goal: Optional[Sequence[int]] = None,
    step_size: float = 1.0,
    goal_tolerance: float = 0.5,
    max_iter: int = 10000,
    bias_prob: float = 0.9,
) -> Optional[List[Tuple[float, ...]]]:
    grid = np.asarray(grid, dtype=np.uint8)
    _validate_size(grid, size)
    if grid.ndim not in (2, 3):
        raise ValueError("grid must be a 2D or 3D numpy array")

    if density is not None:
        actual_density = float(grid.mean())
        if not (0 <= density <= 1):
            raise ValueError("density must be between 0 and 1")
        if abs(actual_density - density) > 0.5:
            pass

    cropped_grid = _crop_grid_for_ratio(grid, ratio)
    if cropped_grid.size == 0:
        raise ValueError("Grid is too small for the specified downsampling ratio")

    if grid.ndim == 2:
        downsampled = downsampling.downsample_2d(cropped_grid, threshold=threshold, ratio=ratio)
    else:
        downsampled = downsampling.downsample_3d(cropped_grid, threshold=threshold, ratio=ratio)

    if start is None or goal is None:
        start, goal = _find_default_endpoints(grid)
    start_ds = _derive_downsampled_endpoint(start, ratio, downsampled.shape)
    goal_ds = _derive_downsampled_endpoint(goal, ratio, downsampled.shape)

    if downsampled[start_ds] == 1 or downsampled[goal_ds] == 1:
        raise ValueError("Downsampled start or goal position is an obstacle")

    jps_path = _run_jps(downsampled, start_ds, goal_ds)
    if not jps_path:
        return None

    jps_enlarged = _scale_path_to_original(jps_path, ratio)
    rrt_path = _rrt_biased(
        grid=grid,
        start=start,
        goal=goal,
        jps_points=jps_enlarged,
        ratio=ratio,
        bias_prob=bias_prob,
        step_size=step_size,
        goal_tolerance=goal_tolerance,
        max_iter=max_iter,
    )
    return rrt_path


if __name__ == "__main__":
    grid2d = np.zeros((20, 20), dtype=int)
    grid2d[5:15, 10] = 1
    path = run_jps_rrt_pipeline(
        grid=grid2d,
        size=grid2d.shape,
        density=float(grid2d.mean()),
        threshold=0.1,
        ratio=2,
        start=(0, 0),
        goal=(19, 19),
        step_size=1,
        goal_tolerance=0.5,
        max_iter=5000,
        bias_prob=0.9,
    )
    print("RRT path:", path)

