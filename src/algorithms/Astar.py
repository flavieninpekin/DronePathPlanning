import math
import numpy as np
import heapq
from typing import Sequence, List, Tuple

def _reconstruct_path(came_from: dict, current: tuple, start: tuple) -> List[tuple]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _validate_positions(grid: np.ndarray, start: tuple, goal: tuple):
    if start == goal:
        return
    if grid.ndim not in (2, 3):
        raise ValueError("grid must be a 2D or 3D numpy array")
    if any(s < 0 or s >= grid.shape[i] for i, s in enumerate(start)):
        raise ValueError("start position is outside grid bounds")
    if any(g < 0 or g >= grid.shape[i] for i, g in enumerate(goal)):
        raise ValueError("goal position is outside grid bounds")
    if grid[start] == 1:
        raise ValueError("start position is an obstacle")
    if grid[goal] == 1:
        raise ValueError("goal position is an obstacle")


def astar_2D(grid: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> List[tuple]:
    start = tuple(start)
    goal = tuple(goal)
    _validate_positions(grid, start, goal)
    rows, cols = grid.shape

    def heuristic(a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct_path(came_from, current, start)
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:
                    continue
                move_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return []


def astar_3D(grid: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> List[tuple]:
    start = tuple(start)
    goal = tuple(goal)
    _validate_positions(grid, start, goal)
    depth, rows, cols = grid.shape

    def heuristic(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    directions = [
        (dz, dx, dy)
        for dz in (-1, 0, 1)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dz == 0 and dx == 0 and dy == 0)
    ]

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct_path(came_from, current, start)
        if current in visited:
            continue
        visited.add(current)

        for dz, dx, dy in directions:
            neighbor = (current[0] + dz, current[1] + dx, current[2] + dy)
            if 0 <= neighbor[0] < depth and 0 <= neighbor[1] < rows and 0 <= neighbor[2] < cols:
                if grid[neighbor] == 1:
                    continue
                move_cost = math.sqrt(dz * dz + dx * dx + dy * dy)
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return []


def astar(grid: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> List[tuple]:
    grid = np.asarray(grid)
    if grid.ndim == 2:
        return astar_2D(grid, start, goal)
    if grid.ndim == 3:
        return astar_3D(grid, start, goal)
    raise ValueError("astar currently supports only 2D or 3D grids")


if __name__ == "__main__":
    grid2d = np.array([
        [0,0,0,0,0],
        [1,1,0,1,0],
        [0,0,0,1,0],
        [0,1,1,0,0],
        [0,0,0,0,0]
    ])
    start2d = (0,0)
    goal2d = (4,4)
    print("2D path:", astar(grid2d, start2d, goal2d))

    grid3d = np.zeros((3, 5, 5), dtype=int)
    grid3d[1, 2, 2] = 1
    start3d = (0, 0, 0)
    goal3d = (2, 4, 4)
    print("3D path:", astar(grid3d, start3d, goal3d))

