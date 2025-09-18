import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list, natural_cutoffs
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))
def compute_angles_from_indices(mol, angle_indices):
    angles = []
    for i, j, k in angle_indices:
        a, b, c = mol.positions[i], mol.positions[j], mol.positions[k]
        angle = compute_angle(a, b, c)
        angles.append(angle)
    return np.array(angles)
from ase.io import read
import numpy as np

# 加载标准角度信息
data = np.load("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/standard_angles.npz")
ref_indices = data["indices"]
ref_angles = data["angles"]

# 加载轨迹
frames = read("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/uff.xyz", index=":")

all_maes = []
for mol in frames:
    pred_angles = compute_angles_from_indices(mol, ref_indices)
    mae = np.mean(np.abs(pred_angles - ref_angles))
    all_maes.append(mae)

final_mae = np.mean(all_maes)
print(f"Angle MAE over trajectory: {final_mae:.3f} degrees")


