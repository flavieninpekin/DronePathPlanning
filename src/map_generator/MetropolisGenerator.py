"""Metropolis: a rigid city grid with roads at exact specified ranges.

Road (free) when x OR y is in: 0-4, 30-34, 60-64, 90-94, 116-120, 146-150, 172-176, 195-199
All other cells = building (obstacle).
"""
import numpy as np
from collections import deque
from typing import List, Tuple

ROAD_RANGES = [
    (0, 4), (30, 34), (60, 64), (90, 94),
    (116, 120), (146, 150), (172, 176), (195, 199),
]


def _is_road(x: int, y: int) -> bool:
    for lo, hi in ROAD_RANGES:
        if lo <= x <= hi:
            return True
        if lo <= y <= hi:
            return True
    return False


def _bfs_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    parent = {}
    q = deque()
    q.append(start)
    visited[start] = True
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = [(r, c)]
            cur = (r, c)
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return []


def generate(size: Tuple[int, int] = (200, 200)) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    h, w = size
    grid = np.ones((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if _is_road(x, y):
                grid[y, x] = 0

    start = (2, 2)
    goal = (197, 197)
    path_int = _bfs_path(grid, start, goal)
    if not path_int:
        raise RuntimeError("BFS failed on metropolis grid")

    deduped = [path_int[0]]
    for p in path_int[1:]:
        if p != deduped[-1]:
            deduped.append(p)

    waypoints = []
    waypoints.append((float(deduped[0][1]), float(deduped[0][0])))
    for i in range(1, len(deduped)):
        r0, c0 = deduped[i - 1]
        r1, c1 = deduped[i]
        dr, dc = r1 - r0, c1 - c0
        dist = (dr * dr + dc * dc) ** 0.5
        steps = max(1, int(dist / 0.5))
        for k in range(1, steps + 1):
            t = k / steps
            waypoints.append((c0 + dc * t, r0 + dr * t))

    return grid, waypoints


def build_pool(size=(200, 200), pool_size=60) -> List[dict]:
    grid, waypoints = generate(size)
    return [{"grid": grid.copy(), "waypoints": waypoints} for _ in range(pool_size)]
