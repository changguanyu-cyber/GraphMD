import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list, natural_cutoffs

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def extract_standard_angles_and_indices(xyz_path, save_path="/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/standard_angles.npz"):
    mol = read(xyz_path)
    cutoffs = natural_cutoffs(mol)
    i_list, j_list = neighbor_list('ij', mol, cutoffs)

    angle_indices = []
    angle_values = []

    N = len(mol)
    for j in range(N):  # j 是中心原子
        neighbors = i_list[j_list == j]
        if len(neighbors) < 2:
            continue
        for m in range(len(neighbors)):
            for n in range(m + 1, len(neighbors)):
                i, k = neighbors[m], neighbors[n]
                a, b, c = mol.positions[i], mol.positions[j], mol.positions[k]
                angle = compute_angle(a, b, c)
                angle_indices.append([i, j, k])
                angle_values.append(angle)

    angle_indices = np.array(angle_indices)
    angle_values = np.array(angle_values)

    np.savez(save_path, indices=angle_indices, angles=angle_values)
    print(f"保存了 {len(angle_values)} 个角度值到 {save_path}")



# 使用示例
extract_standard_angles_and_indices('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/uff.xyz')
