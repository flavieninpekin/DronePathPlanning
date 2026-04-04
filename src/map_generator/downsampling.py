import numpy as np

def downsample_2d(arr: np.ndarray, threshold: float, ratio: int = 10) -> np.ndarray:
    h, w = arr.shape
    new_h, new_w = h // ratio, w // ratio
    downsampled = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            block = arr[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
            mean_val = block.mean()
            downsampled[i, j] = 1 if mean_val > threshold else 0
    return downsampled

def downsample_3d(arr: np.ndarray, threshold: float, ratio: int = 10) -> np.ndarray:
    d, h, w = arr.shape
    new_d, new_h, new_w = d // ratio, h // ratio, w // ratio
    downsampled = np.zeros((new_d, new_h, new_w), dtype=np.uint8)
    for k in range(new_d):
        for i in range(new_h):
            for j in range(new_w):
                block = arr[k*ratio:(k+1)*ratio, i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
                mean_val = block.mean()
                downsampled[k, i, j] = 1 if mean_val > threshold else 0
    return downsampled

# 测试用例
if __name__ == "__main__":
    # 2D 测试
    arr2d = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    print("2D Downsampled:")
    print(downsample_2d(arr2d, threshold=0.5, ratio=5))

    # 3D 测试
    arr3d = np.zeros((10, 10, 10), dtype=np.uint8)
    arr3d[:5, :5, :5] = 1
    print("3D Downsampled:")
    print(downsample_3d(arr3d, threshold=0.5, ratio=5))