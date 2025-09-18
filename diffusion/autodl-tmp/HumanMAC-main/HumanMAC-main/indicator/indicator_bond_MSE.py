import numpy as np

# === Step 1: 加载标准键长 ===
data = np.load("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/aspirin_bond_lengths.npz", allow_pickle=True)
bond_lengths_ref = data['bonds'].item()  # {(i, j): standard_length}


# === Step 2: 加载轨迹 xyz ===
def read_xyz_trajectory(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0])
    nframes = len(lines) // (natoms + 2)

    traj = []
    for i in range(nframes):
        start = i * (natoms + 2) + 2
        coords = []
        for j in range(natoms):
            parts = lines[start + j].split()
            coords.append([float(x) for x in parts[1:4]])
        traj.append(np.array(coords))
    return np.array(traj)  # shape: (n_frames, n_atoms, 3)


# === Step 3: 计算全轨迹的平均MSE ===
def compute_average_mse(traj, bond_lengths_ref):
    """
    返回所有帧上所有键长的MSE（单一标量）
    """
    total_mse = 0.0
    n_frames = traj.shape[0]
    n_bonds = len(bond_lengths_ref)

    for frame in traj:
        for (i, j), ref_len in bond_lengths_ref.items():
            dist = np.linalg.norm(frame[i] - frame[j])

            total_mse += abs(dist - ref_len)

    avg_mse = total_mse / (n_frames * n_bonds)
    return avg_mse


# === Step 4: 执行计算 ===
traj = read_xyz_trajectory("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/uff.xyz")
average_mse = compute_average_mse(traj, bond_lengths_ref)

print(f"全轨迹平均键长MSE: {average_mse:.6f}")

