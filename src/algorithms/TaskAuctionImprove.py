import numpy as np

def auction_algorithm_improved(task_points, drone_positions):
    num_drones = len(drone_positions)
    num_tasks = len(task_points)
    # 1. 贪心初始分配
    assignments = {i: [] for i in range(num_drones)}
    unassigned = set(range(num_tasks))
    while unassigned:
        # 找出当前成本最小的无人机-任务对
        best_cost = np.inf
        best_pair = None
        for i in range(num_drones):
            for j in unassigned:
                cost = np.linalg.norm(np.array(drone_positions[i]) - np.array(task_points[j]))
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)
        if best_pair is None:
            # No valid pair found, break to avoid error
            break
        i, j = best_pair
        assignments[i].append(j)
        unassigned.remove(j)

    # 2. 局部交换优化（2-opt 风格）
    improved = True
    while improved:
        improved = False
        # 尝试所有无人机对 (i1, i2)
        for i1 in range(num_drones):
            for i2 in range(i1+1, num_drones):
                # 尝试交换 i1 和 i2 的某个任务
                for idx1, t1 in enumerate(assignments[i1]):
                    for idx2, t2 in enumerate(assignments[i2]):
                        # 计算当前总成本（简化：只考虑涉及的两个任务）
                        cost_before = (
                            np.linalg.norm(np.array(drone_positions[i1]) - np.array(task_points[t1])) +
                            np.linalg.norm(np.array(drone_positions[i2]) - np.array(task_points[t2]))
                        )
                        cost_after = (
                            np.linalg.norm(np.array(drone_positions[i1]) - np.array(task_points[t2])) +
                            np.linalg.norm(np.array(drone_positions[i2]) - np.array(task_points[t1]))
                        )
                        if cost_after < cost_before - 1e-6:
                            # 交换
                            assignments[i1][idx1], assignments[i2][idx2] = t2, t1
                            improved = True
    return assignments

# 测试用例
if __name__ == "__main__":
    np.random.seed(842)
    num_tasks = 20
    num_drones = 5
    coord_range = 50

    task_points = np.random.randint(0, coord_range + 1, size=(num_tasks, 2)).tolist()
    drone_positions = np.random.randint(0, coord_range + 1, size=(num_drones, 2)).tolist()

    assignments = auction_algorithm_improved(task_points, drone_positions)

    print("任务点坐标：")
    for i, p in enumerate(task_points):
        print(f"任务{i}: {p}")
    print("\n无人机初始位置：")
    for i, p in enumerate(drone_positions):
        print(f"无人机{i}: {p}")
    print("\n任务分配结果：")
    for drone, tasks in assignments.items():
        print(f"无人机{drone}分配到任务: {tasks}")