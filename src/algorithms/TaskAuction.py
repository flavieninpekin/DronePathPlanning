import numpy as np

def auction_algorithm(task_points, drone_positions):
    num_tasks = len(task_points)
    num_drones = len(drone_positions)
    assignments = {i: [] for i in range(num_drones)}
    unassigned_tasks = set(range(num_tasks))
    task_prices = np.zeros(num_tasks)
    epsilon = 1e-3

    while unassigned_tasks:
        for drone_idx, drone_pos in enumerate(drone_positions):
            # Find the best task for this drone
            best_task = None
            best_value = -np.inf
            for task_idx in unassigned_tasks:
                task_pos = task_points[task_idx]
                cost = np.linalg.norm(np.array(drone_pos) - np.array(task_pos))
                value = -cost - task_prices[task_idx]
                if value > best_value:
                    best_value = value
                    best_task = task_idx
            if best_task is not None:
                # Assign the task to this drone
                assignments[drone_idx].append(best_task)
                unassigned_tasks.remove(best_task)
                task_prices[best_task] += epsilon
    return assignments

# 测试用例
if __name__ == "__main__":
    np.random.seed(842)
    num_tasks = 20
    num_drones = 5
    coord_range = 50

    task_points = np.random.randint(0, coord_range + 1, size=(num_tasks, 2)).tolist()
    drone_positions = np.random.randint(0, coord_range + 1, size=(num_drones, 2)).tolist()

    assignments = auction_algorithm(task_points, drone_positions)

    print("任务点坐标：")
    for i, p in enumerate(task_points):
        print(f"任务{i}: {p}")
    print("\n无人机初始位置：")
    for i, p in enumerate(drone_positions):
        print(f"无人机{i}: {p}")
    print("\n任务分配结果：")
    for drone, tasks in assignments.items():
        print(f"无人机{drone}分配到任务: {tasks}")