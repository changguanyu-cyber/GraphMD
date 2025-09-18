import numpy as np
import os

# 假设 data 是形状为 (125, 21, 3) 的 NumPy 数组
npy_file_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/uni_traj.npy'
data = np.load(npy_file_path)
# 输出文件夹路径
#data=data[1]
print(data.shape)
output_dir = "/root/autodl-tmp/HumanMAC-main/HumanMAC-main/bene_matrix"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹

# 遍历每一帧并保存对应的距离矩阵
for i, frame in enumerate(data):
    # 计算 21 个原子之间的两两距离矩阵
    distances = np.linalg.norm(frame[:, np.newaxis, :] - frame[np.newaxis, :, :], axis=-1)

    # 文件名
    output_file = os.path.join(output_dir, f"frame_{i + 1}.txt")

    # 保存距离矩阵到单独的文件
    np.savetxt(output_file, distances, fmt="%.6f")
    print(f"Saved Frame {i + 1} to {output_file}")

print(f"All distance matrices saved to {output_dir}/")
