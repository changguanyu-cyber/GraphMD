import numpy as np
import matplotlib.pyplot as plt


def calculate_rmsd_per_frame(traj_coords, ref_coords):
    """
    计算轨迹中每一帧的 RMSD 值，并返回所有帧的 RMSD 结果。

    参数：
    traj_coords -- ndarray, 轨迹坐标 (125, 20, 3)
    ref_coords  -- ndarray, 参考结构坐标 (20, 3)

    返回：
    rmsd_values -- ndarray, 每一帧的 RMSD 值 (125,)
    avg_rmsd    -- float, RMSD 的均值
    """
    num_frames, num_atoms, _ = traj_coords.shape
    rmsd_values = np.zeros(num_frames)

    for i in range(num_frames):
        diff = traj_coords[i] - ref_coords
        squared_diff = np.sum(diff ** 2, axis=1)
        rmsd_values[i] = np.sqrt(np.sum(squared_diff) / num_atoms)

    avg_rmsd = np.mean(rmsd_values)
    return rmsd_values, avg_rmsd


# 加载轨迹数据，形状为 (64, 125, 21, 3)
traj_coords = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy')

# 去除第一个原子，调整形状为 (64, 125, 20, 3)
traj_coords = traj_coords[:, :, 1:, :]

# 仅取前10条轨迹
num_traj_to_plot = 10
traj_coords = traj_coords[:num_traj_to_plot]

# 初始化保存所有轨迹的平均 RMSD
all_avg_rmsd = []

# 计算前10条轨迹的 RMSD
for i in range(traj_coords.shape[0]):
    single_traj = traj_coords[i]  # 当前轨迹 (125, 20, 3)
    ref_coords = single_traj[0]  # 使用第1帧作为参考 (20, 3)
    single_traj = single_traj[1:]  # 去除第一帧 (124, 20, 3)

    _, avg_rmsd = calculate_rmsd_per_frame(single_traj, ref_coords)
    all_avg_rmsd.append(avg_rmsd)

# 打印前10条轨迹的平均 RMSD
for i, avg in enumerate(all_avg_rmsd):
    print(f"轨迹 {i + 1} 的平均 RMSD: {avg:.3f} Å")

# 绘制平均 RMSD 的柱状图
plt.figure(figsize=(8, 5))
plt.bar(range(1, num_traj_to_plot + 1), all_avg_rmsd, color='b', alpha=0.7, label='Average RMSD')
plt.axhline(1.59, color='r', linestyle='--', label='Reference RMSD = 1.59 Å')
plt.xlabel("Trajectory Index")
plt.ylabel("Average RMSD (Å)")
plt.title("Average RMSD for First 10 Trajectories")
plt.legend()
plt.grid(True)

# 保存图像到文件
plt.savefig("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/RMSD.png", dpi=300, bbox_inches='tight')

plt.show()
