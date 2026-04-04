import numpy as np
from heapq import heappush, heappop

def jps_2d(grid: np.ndarray, start: tuple, end: tuple):
    """
    Jump Point Search (JPS) for 2D grid.
    grid: 2D np.array, 0 for free, 1 for obstacle
    start, end: (x, y) tuples
    Returns: list of (x, y) path points from start to end (inclusive), or [] if no path
    """
    def neighbors(pos):
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if grid[nx, ny] == 0:
                        yield (nx, ny)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))
    closed = set()
    while open_set:
        _, cost, current, path = heappop(open_set)
        if current == end:
            return path
        if current in closed:
            continue
        closed.add(current)
        for n in neighbors(current):
            if n not in closed:
                heappush(open_set, (cost + 1 + heuristic(n, end), cost + 1, n, path + [n]))
    return []

def jps_3d(grid: np.ndarray, start: tuple, end: tuple):
    """
    JPS for 3D grid.
    grid: 3D np.array, 0 for free, 1 for obstacle
    start, end: (x, y, z) tuples
    Returns: list of (x, y, z) path points from start to end (inclusive), or [] if no path
    """
    def neighbors(pos):
        x, y, z = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < grid.shape[0] and
                        0 <= ny < grid.shape[1] and
                        0 <= nz < grid.shape[2]):
                        if grid[nx, ny, nz] == 0:
                            yield (nx, ny, nz)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    open_set = []
    heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))
    closed = set()
    while open_set:
        _, cost, current, path = heappop(open_set)
        if current == end:
            return path
        if current in closed:
            continue
        closed.add(current)
        for n in neighbors(current):
            if n not in closed:
                heappush(open_set, (cost + 1 + heuristic(n, end), cost + 1, n, path + [n]))
    return []

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
    print("2D path:", jps_2d(grid2d, start2d, goal2d))

    grid3d = np.zeros((3, 5, 5), dtype=int)
    grid3d[1, 2, 2] = 1
    start3d = (0, 0, 0)
    goal3d = (2, 4, 4)
    print("3D path:", jps_3d(grid3d, start3d, goal3d))

