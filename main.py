import numpy as np

from src.algorithms.Astar import astar
from src.map_generator.MapGenerator import generate_map_with_path
from src.algo_combinations.astar_rrt import run_astar_rrt_pipeline


def main():
    print('=== DronePathPlanning sample run ===')
    grid, path = generate_map_with_path((20, 20), obstacle_density=0.12)
    print(f'Generated grid shape: {grid.shape}')
    path = astar(grid, (0, 0), (19, 19))
    print('A* path length:', len(path) if path else 0)

    rrt_path = run_astar_rrt_pipeline(
        grid=grid,
        size=grid.shape,
        density=float(grid.mean()),
        threshold=0.2,
        ratio=2,
        start=(0, 0),
        goal=(19, 19),
        step_size=1.0,
        goal_tolerance=0.5,
        max_iter=2000,
        bias_prob=0.8,
    )
    print('A*+RRT pipeline path:', rrt_path is not None)


if __name__ == '__main__':
    main()
