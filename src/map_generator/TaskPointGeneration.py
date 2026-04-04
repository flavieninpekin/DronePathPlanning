import numpy as np

def generate_task_points(grid: np.ndarray, num_points: int):
    """
    在二维0-1 np.array内随机生成指定数量的任务点，避开障碍物（1）。
    返回点群的连续坐标（二维数组，shape=(num_points, 2)）。
    """
    h, w = grid.shape
    points = []
    attempts = 0
    max_attempts = num_points * 100  # 防止死循环

    while len(points) < num_points and attempts < max_attempts:
        # 随机生成连续坐标
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        ix, iy = int(x), int(y)
        # 检查是否为障碍物
        if grid[iy, ix] == 0:
            points.append([x, y])
        attempts += 1

    if len(points) < num_points:
        raise RuntimeError("无法在给定地图上生成足够的任务点，请检查障碍物比例或任务点数量。")

    return np.array(points)

# 示例用法
if __name__ == "__main__":
    grid = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    points = generate_task_points(grid, 5)
    print(points)