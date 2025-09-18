import numpy as np

def compute_smoothness_rho(trajectory):
    """
    输入:
        trajectory: ndarray of shape (T, N, 3)
            T: 时间步数
            N: 原子数
            3: 每个原子的 (x, y, z) 坐标
    输出:
        ρ: 系统平滑度指标
    """
    T, N, _ = trajectory.shape

    # 初始化 Δ_ijt 累加器
    total_delta = 0.0

    # 遍历时间步 t = 0 到 T-2 （因为用到 t 和 t+1）
    for t in range(T - 1):
        coords_t = trajectory[t]     # shape: (N, 3)
        coords_t1 = trajectory[t+1]  # shape: (N, 3)

        # 计算 t 和 t+1 时间步的距离矩阵（N x N）
        dist_t = np.linalg.norm(coords_t[:, None, :] - coords_t[None, :, :], axis=-1)
        dist_t1 = np.linalg.norm(coords_t1[:, None, :] - coords_t1[None, :, :], axis=-1)

        # 计算 Δ_ijt = |a_{ij}^{t+1} - a_{ij}^t|，取上三角部分（避免重复和对角线）
        delta_ijt = np.abs(dist_t1 - dist_t)
        delta_sum = np.sum(np.triu(delta_ijt, k=1))  # 上三角（不含对角线）
        total_delta += delta_sum

    # 归一化因子：（T - 1） × 所有原子对数 = (T - 1) × N(N - 1)/2
    normalization = (T - 1) * (N * (N - 1) / 2)

    rho = total_delta / normalization
    return rho
trajectory = np.load("/root/rmd17/extracted_data/aspirin_coord.npy")  # shape: (T, N, 3)
trajectory = trajectory[:1000]
rho = compute_smoothness_rho(trajectory)
print("系统的平滑度指标 rho =", rho)
