import numpy as np
import os
import sys
from typing import List, Tuple

_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _find_free_cell_near(grid: np.ndarray, target: Tuple[int, int]) -> Tuple[int, int]:
    h, w = grid.shape
    for r in range(max(0, target[0] - 1), min(h, target[0] + 2)):
        for c in range(max(0, target[1] - 1), min(w, target[1] + 2)):
            if grid[r, c] == 0:
                return (r, c)
    best, best_d = None, float("inf")
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                d = (r - target[0]) ** 2 + (c - target[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = (r, c)
    if best is None:
        raise ValueError("No free cell in grid")
    return best


def _interpolate_path(path: List[Tuple[int, int]], step=1.0) -> List[Tuple[float, float]]:
    if len(path) < 2:
        return [(float(p[1]), float(p[0])) for p in path]
    result = [(float(path[0][1]), float(path[0][0]))]
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        dr, dc = r1 - r0, c1 - c0
        dist = (dr * dr + dc * dc) ** 0.5
        steps = max(1, int(dist / step))
        for k in range(1, steps + 1):
            t = k / steps
            result.append((c0 + dc * t, r0 + dr * t))
    return result


def _road_start_end(grid: np.ndarray, block_size: int = 20, road_width: int = 5) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return guaranteed-free (start, goal) cells on the road network."""
    period = block_size + road_width
    num_blocks_y = grid.shape[0] // period
    num_blocks_x = grid.shape[1] // period
    first_road = block_size
    last_road_y = (num_blocks_y - 1) * period + block_size
    if last_road_y >= grid.shape[0]:
        last_road_y = grid.shape[0] - 1
    last_road_x = (num_blocks_x - 1) * period + block_size
    if last_road_x >= grid.shape[1]:
        last_road_x = grid.shape[1] - 1
    return (first_road, first_road), (last_road_y, last_road_x)


def _find_jps_rrt_path(grid: np.ndarray) -> List[Tuple[float, float]]:
    # BFS pathfinding (avoids Astar.py compatibility issues)
    from collections import deque
    h, w = grid.shape
    start, goal = _road_start_end(grid)

    visited = np.zeros_like(grid, dtype=bool)
    parent = {}
    q = deque()
    q.append(start)
    visited[start] = True
    found = False
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            found = True
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    if not found:
        raise RuntimeError(f"BFS failed: start={start}, goal={goal}")

    path_int = [goal]
    cur = goal
    while cur in parent:
        cur = parent[cur]
        path_int.append(cur)
    path_int.reverse()
    deduped = [path_int[0]]
    for p in path_int[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    # Snap any integer points on buildings to nearest free cell
    snapped = [deduped[0]]
    for (r, c) in deduped[1:]:
        if grid[r, c] == 1:
            for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                    r, c = nr, nc
                    break
        if (r, c) != snapped[-1]:
            snapped.append((r, c))
    raw = _interpolate_path(snapped, step=0.5)
    safe = []
    for (x, y) in raw:
        ix, iy = int(x), int(y)
        if grid[iy, ix] == 1:
            continue
        if not safe or (x, y) != safe[-1]:
            safe.append((x, y))
    if len(safe) < 2:
        safe = [(float(c), float(r)) for (r, c) in snapped]
    return safe


def generate_city(
    size: Tuple[int, int] = (200, 200),
    block_size: int = 20,
    road_width: int = 5,
    seed: int = None,
    use_jps_rrt: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Generate a regular city grid map with building blocks separated by roads.

    Parameters:
        size: (height, width) of the full grid.
        block_size: size of each square building block.
        road_width: width of roads between blocks.
        seed: random seed (affects plaza placement).
        use_jps_rrt: if True, use JPS-RRT to find path; else fallback to L-route.

    Returns:
        grid: uint8 array, 0=road/free, 1=building/obstacle.
        waypoints: list of (x, y) continuous coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    period = block_size + road_width
    num_blocks_y = (h + road_width) // period
    num_blocks_x = (w + road_width) // period

    grid = np.zeros((h, w), dtype=np.uint8)

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * period
            x_start = bx * period
            y_end = min(y_start + block_size, h)
            x_end = min(x_start + block_size, w)
            if y_end > y_start and x_end > x_start:
                grid[y_start:y_end, x_start:x_end] = 1

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * period
            x_start = bx * period
            if y_start + block_size <= h and x_start + block_size <= w:
                if np.random.random() < 0.25:
                    py = y_start + np.random.randint(1, block_size - 4)
                    px = x_start + np.random.randint(1, block_size - 4)
                    grid[py:py + 4, px:px + 4] = 0

    if use_jps_rrt:
        waypoints = _find_jps_rrt_path(grid)
    else:
        # Fallback: simple L-shaped path along road centerlines
        road_centers_y = [i * period + block_size + road_width // 2 for i in range(num_blocks_y) if i * period + block_size + road_width // 2 < h]
        road_centers_x = [i * period + block_size + road_width // 2 for i in range(num_blocks_x) if i * period + block_size + road_width // 2 < w]
        first_y = road_centers_y[0]
        last_x = road_centers_x[-1]
        last_y = road_centers_y[-1]
        path_cells = [(first_y, 2)]
        for c in range(3, last_x + 1):
            if 0 <= first_y < h and 0 <= c < w and grid[first_y, c] == 0:
                path_cells.append((first_y, c))
        for r in range(first_y + 1, last_y + 1):
            if 0 <= r < h and 0 <= last_x < w and grid[r, last_x] == 0:
                path_cells.append((r, last_x))
        seen = set()
        unique_path = []
        for p in path_cells:
            if p not in seen:
                seen.add(p)
                unique_path.append(p)
        waypoints = _interpolate_path(unique_path, step=0.5)

    return grid, waypoints


def build_city_map_pool(
    size: Tuple[int, int] = (200, 200),
    block_size: int = 20,
    road_width: int = 5,
    pool_size: int = 50,
    base_seed: int = 0,
    use_jps_rrt: bool = True,
) -> List[dict]:
    pool = []
    for i in range(pool_size):
        grid, waypoints = generate_city(size, block_size, road_width, seed=base_seed + i, use_jps_rrt=use_jps_rrt)
        pool.append({"grid": grid, "waypoints": waypoints})
    return pool
