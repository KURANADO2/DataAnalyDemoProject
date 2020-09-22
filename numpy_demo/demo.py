import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)
print(arr[1][2])
# 取第 2 列
print(arr[:, 1])
# 取第 2 行
print(arr[1, :])
# 转换为 1 行多列的 single sample multiple feature
print(arr.reshape(1, -1))
# 转换为 1 列多行的 multiple sample single feature
print(arr.reshape(-1, 1))
arr = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 9, 10, 0, 15, 4, 0,
                0, 0, 3, 16, 12, 14, 2, 0,
                0, 0, 4, 16, 16, 2, 0, 0,
                0, 3, 16, 8, 10, 13, 2, 0,
                0, 1, 15, 1, 3, 16, 8, 0,
                0, 0, 11, 16, 15, 11, 1, 0])
print(arr.reshape(1, -1))
