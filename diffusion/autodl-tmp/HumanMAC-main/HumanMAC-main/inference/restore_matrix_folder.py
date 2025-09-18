import numpy as np
import os

# 读取 NumPy 数据
npy_file_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy'
data = np.load(npy_file_path)

# 确保数据格式正确
print(f"Loaded data shape: {data.shape}")  # 形状应为 (64, 125, 21, 3)

# 根目录路径
root_output_dir = "/root/autodl-tmp/HumanMAC-main/HumanMAC-main/nap_matrix"
os.makedirs(root_output_dir, exist_ok=True)  # 确保主目录存在

# 遍历 64 个不同的序列
for seq_idx, sequence in enumerate(data):
    sequence_folder = os.path.join(root_output_dir, f"sequence_{seq_idx+1:03d}")  # 例如 sequence_001
    os.makedirs(sequence_folder, exist_ok=True)  # 创建子文件夹

    print(f"Processing sequence {seq_idx+1}/64, saving to {sequence_folder}")

    # 遍历 125 帧
    for frame_idx, frame in enumerate(sequence):
        # 计算欧几里得距离矩阵
        distances = np.linalg.norm(frame[:, np.newaxis, :] - frame[np.newaxis, :, :], axis=-1)

        # 生成文件路径
        output_file = os.path.join(sequence_folder, f"frame_{frame_idx+1}.txt")

        # 如果文件已存在，跳过计算
        if os.path.exists(output_file):
            print(f"Frame {frame_idx+1} already exists in sequence {seq_idx+1}, skipping...")
            continue

        # 保存距离矩阵到 .txt
        np.savetxt(output_file, distances, fmt="%.6f")
        print(f"Saved Frame {frame_idx+1} (shape: {distances.shape}) to {output_file}")

print(f"All distance matrices saved to {root_output_dir}/")
