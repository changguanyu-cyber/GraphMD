import numpy as np

def read_multiframe_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    n_lines_per_frame = n_atoms + 2
    n_frames = len(lines) // n_lines_per_frame

    all_coords = []

    for i in range(n_frames):
        start = i * n_lines_per_frame + 2
        end = start + n_atoms
        frame_coords = []
        for line in lines[start:end]:
            parts = line.split()
            xyz = list(map(float, parts[1:4]))  # 忽略元素符号
            frame_coords.append(xyz)
        all_coords.append(frame_coords)

    coords_array = np.array(all_coords)  # shape: (T, N, 3)
    return coords_array

# 使用函数读取并保存
coords = read_multiframe_xyz("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/traj.xyz")
np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy", coords)
