import numpy as np
from typing import Sequence, Iterable, Tuple


def point_in_bounds(grid: np.ndarray, point: Sequence[int]) -> bool:
    return all(0 <= int(coord) < dim for coord, dim in zip(point, grid.shape))


def is_collision(grid: np.ndarray, point: Sequence[float]) -> bool:
    coords = tuple(int(np.floor(coord)) for coord in point)
    if not point_in_bounds(grid, coords):
        return True
    return bool(grid[coords] == 1)


def path_is_clear(grid: np.ndarray, path: Iterable[Sequence[float]]) -> bool:
    return all(not is_collision(grid, point) for point in path)
