import numpy as np
import matplotlib.pyplot as plt

# === 加载轨迹 ===
trajectory = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy')
trajectory = trajectory[0]

# === 计算每个原子的平均位置 ===
mean_coords = trajectory.mean(axis=0)  # shape: (N, 3)

# === 计算 RMSF（每个原子）===
# 对每个原子计算 (r_t - r_mean)^2 再平均
squared_displacements = (trajectory - mean_coords[np.newaxis, :, :]) ** 2  # shape: (T, N, 3)
rmsf = np.sqrt(squared_displacements.mean(axis=0).sum(axis=1))  # shape: (N,)
rmsf = rmsf.mean()

print(rmsf)
