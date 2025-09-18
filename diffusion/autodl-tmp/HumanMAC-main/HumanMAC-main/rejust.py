import numpy as np

# 加载原始数据
data = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy')  # shape: (1000, 21, 3)

# 方法 1：重复一次（复制为2000帧）
data_expanded = np.tile(data, (2, 1, 1))  # shape: (2000, 21, 3)

# 方法 2（可选）：也可以使用 np.concatenate
# data_expanded = np.concatenate([data, data], axis=0)

# 保存新文件
np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy', data_expanded)
