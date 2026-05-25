import numpy as np
import random
from collections import deque
from typing import List, Tuple


def _bfs_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """BFS find path from start to goal on free cells (grid == 0)."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    parent = {}
    q = deque()
    q.append(start)
    visited[start] = True
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            cur = goal
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return []


def _interpolate_path(path: List[Tuple[int, int]], step=1.0) -> List[Tuple[float, float]]:
    """Interpolate integer path to continuous (x, y) waypoints.
    Input path is (row, col), output is (x=col, y=row)."""
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


def _find_free_cell_near(grid: np.ndarray, target: Tuple[int, int]) -> Tuple[int, int]:
    """Find nearest free cell to target position."""
    h, w = grid.shape
    best = None
    best_d = float("inf")
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


# ========================================================================
# Maze
# ========================================================================
def generate_maze(size: Tuple[int, int], seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Perfect maze using recursive backtracker. size is the full grid dimensions."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    h, w = size
    if h < 5 or w < 5:
        raise ValueError("Maze too small, minimum 5x5")
    cells_h = (h - 1) // 2
    cells_w = (w - 1) // 2
    grid_h = cells_h * 2 + 1
    grid_w = cells_w * 2 + 1
    grid = np.ones((grid_h, grid_w), dtype=np.uint8)
    visited = np.zeros((cells_h, cells_w), dtype=bool)

    order = np.zeros((cells_h, cells_w, 4, 2), dtype=np.int32)
    all_dirs = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int32)
    for r in range(cells_h):
        for c in range(cells_w):
            perm = np.random.permutation(4)
            order[r, c] = all_dirs[perm]

    stack = [(0, 0, 0)]
    visited[0, 0] = True
    grid[1, 1] = 0
    while stack:
        r, c, idx = stack[-1]
        if idx >= 4:
            stack.pop()
            continue
        dr, dc = order[r, c, idx]
        stack[-1] = (r, c, idx + 1)
        nr, nc = r + dr, c + dc
        if 0 <= nr < cells_h and 0 <= nc < cells_w and not visited[nr, nc]:
            visited[nr, nc] = True
            grid[2 * r + 1 + dr, 2 * c + 1 + dc] = 0
            grid[2 * nr + 1, 2 * nc + 1] = 0
            stack.append((nr, nc, 0))

    start = (1, 1)
    goal = (grid_h - 2, grid_w - 2)
    path_int = _bfs_path(grid, start, goal)
    waypoints = _interpolate_path(path_int, step=0.5)
    if (grid_h, grid_w) != (h, w):
        full = np.ones((h, w), dtype=np.uint8)
        full[:grid_h, :grid_w] = grid
        grid = full
    return grid, waypoints


# ========================================================================
# Corridor
# ========================================================================
def generate_corridor(size: Tuple[int, int], seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Winding corridor with obstacles on both sides."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    h, w = size
    grid = np.ones((h, w), dtype=np.uint8)
    path_cells = []
    r, c = 0, 0
    path_cells.append((r, c))
    grid[r, c] = 0
    while r < h - 1 or c < w - 1:
        if r >= h - 1:
            c += 1
        elif c >= w - 1:
            r += 1
        else:
            if random.random() < 0.55:
                r += 1
            else:
                c += 1
        r = min(r, h - 1)
        c = min(c, w - 1)
        grid[r, c] = 0
        path_cells.append((r, c))
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                dist_to_path = min(abs(r - pr) + abs(c - pc) for pr, pc in path_cells)
                if dist_to_path > 1 and random.random() < 0.7:
                    grid[r, c] = 0
    grid = 1 - grid  # invert: 0=free, 1=obstacle
    waypoints = _interpolate_path(path_cells, step=0.5)
    return grid, waypoints


# ========================================================================
# Checkerboard
# ========================================================================
def generate_checkerboard(size: Tuple[int, int], block: int = 3) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Checkerboard obstacle pattern. block controls obstacle block size."""
    h, w = size
    grid = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if ((r // block) + (c // block)) % 2 == 0:
                grid[r, c] = 1
    start = _find_free_cell_near(grid, (1, 1))
    goal = _find_free_cell_near(grid, (h - 2, w - 2))
    path_int = _bfs_path(grid, start, goal)
    if not path_int:
        raise RuntimeError("No path in checkerboard")
    waypoints = _interpolate_path(path_int, step=0.5)
    return grid, waypoints


# ========================================================================
# Rooms
# ========================================================================
def generate_rooms(size: Tuple[int, int], seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Partition map into rooms with doorways."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    h, w = size
    grid = np.ones((h, w), dtype=np.uint8)
    room_h = max(3, h // 3)
    room_w = max(3, w // 3)
    rooms = []
    for r0 in range(0, h, room_h):
        for c0 in range(0, w, room_w):
            r1 = min(r0 + room_h, h)
            c1 = min(c0 + room_w, w)
            if r1 - r0 > 1 and c1 - c0 > 1:
                rooms.append((r0, r1, c0, c1))
                grid[r0:r1, c0:c1] = 0
    for r0, r1, c0, c1 in rooms:
        walls_added = 0
        for r in (r0, r1 - 1):
            if r < h:
                for c in range(c0, c1):
                    if random.random() < 0.85:
                        grid[r, c] = 1
                        walls_added += 1
        for c in (c0, c1 - 1):
            if c < w:
                for r in range(r0, r1):
                    if random.random() < 0.85:
                        grid[r, c] = 1
                        walls_added += 1
    start = _find_free_cell_near(grid, (1, 1))
    goal = _find_free_cell_near(grid, (h - 2, w - 2))
    path_int = _bfs_path(grid, start, goal)
    if not path_int:
        grid[max(1, h // 3):min(h - 1, 2 * h // 3), max(1, w // 3)] = 0
        path_int = _bfs_path(grid, start, goal)
        if not path_int:
            for door_r in range(h):
                if grid[door_r, w // 2] == 1:
                    grid[door_r, w // 2] = 0
                    break
            path_int = _bfs_path(grid, start, goal)
            if not path_int:
                raise RuntimeError("No path in rooms")
    waypoints = _interpolate_path(path_int, step=0.5)
    return grid, waypoints


# ========================================================================
# Ring
# ========================================================================
def generate_ring(size: Tuple[int, int], seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Concentric obstacle rings with corridors."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    h, w = size
    grid = np.zeros((h, w), dtype=np.uint8)
    cx, cy = h // 2, w // 2
    max_r = min(cx, cy)
    for r in range(2, max_r, 4):
        rr = r + random.randint(0, 1)
        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                dist = (dx * dx + dy * dy) ** 0.5
                if rr - 0.5 <= dist <= rr + 0.5:
                    grid[y, x] = 1
        escape = random.randint(0, 3)
        for _ in range(3):
            angle = random.uniform(0, 2 * np.pi)
            ex = int(cx + rr * np.cos(angle))
            ey = int(cy + rr * np.sin(angle))
            if 0 <= ex < w and 0 <= ey < h:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ey2, ex2 = ey + dy, ex + dx
                        if 0 <= ey2 < h and 0 <= ex2 < w:
                            grid[ey2, ex2] = 0
    start = _find_free_cell_near(grid, (1, 1))
    goal = _find_free_cell_near(grid, (h - 2, w - 2))
    path_int = _bfs_path(grid, start, goal)
    if not path_int:
        raise RuntimeError("No path in ring")
    waypoints = _interpolate_path(path_int, step=0.5)
    return grid, waypoints


# ========================================================================
# Factory
# ========================================================================
MAP_TYPES = {
    "maze": generate_maze,
    "corridor": generate_corridor,
    "checkerboard": generate_checkerboard,
    "rooms": generate_rooms,
    "ring": generate_ring,
}


def generate(map_type: str, size: Tuple[int, int], seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    if map_type not in MAP_TYPES:
        raise ValueError(f"Unknown map type: {map_type}. Available: {list(MAP_TYPES.keys())}")
    return MAP_TYPES[map_type](size, seed)


def build_rule_map_pool(
    map_type: str,
    size: Tuple[int, int],
    pool_size: int = 50,
    base_seed: int = 0,
) -> List[dict]:
    pool = []
    for i in range(pool_size):
        grid, waypoints = generate(map_type, size, seed=base_seed + i)
        pool.append({"grid": grid, "waypoints": waypoints})
    return pool


def build_mixed_pool(
    size: Tuple[int, int],
    pool_size: int = 50,
    base_seed: int = 0,
) -> List[dict]:
    types = list(MAP_TYPES.keys())
    pool = []
    for i in range(pool_size):
        mt = types[i % len(types)]
        grid, waypoints = generate(mt, size, seed=base_seed + i)
        pool.append({"grid": grid, "waypoints": waypoints, "map_type": mt})
    return pool
