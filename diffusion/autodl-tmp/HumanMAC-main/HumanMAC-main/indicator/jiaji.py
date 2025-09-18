import numpy as np

def compute_dihedral(p0, p1, p2, p3):
    """四个点计算二面角（单位：度）"""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

# === 输入数据 ===
trajectory = np.load("/root/rmd17/extracted_data/ethanol_coord.npy")
trajectory = trajectory[:10000]# shape: (1000, 9, 3)
data = np.load("/root/rmd17/npz_data/rmd17_ethanol.npz")
charges = data['nuclear_charges']

# === 找到对应原子下标 ===
# 示例：H–C–C–O = idx0, idx1, idx2, idx3
print(charges)
H_indices = np.where(charges == 1)[0]  # 所有氢
C_indices = np.where(charges == 6)[0]  # 两个碳
O_index = np.where(charges == 8)[0][0] # 一个氧

# 选一个甲基氢和两个碳
H_index = H_indices[4]
C1_index = C_indices[0]
C2_index = C_indices[1]
print(H_index)
print(C1_index)
print(C2_index)
print(O_index)
# === 计算每一帧的扭转角 ===
angles = []
for frame in trajectory:
    p0 = frame[H_index]
    p1 = frame[C1_index]
    p2 = frame[C2_index]
    p3 = frame[O_index]
    angle = compute_dihedral(p0, p1, p2, p3)
    angles.append(angle)

angles = np.array(angles)
np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/methyl_dihedral_angles.npy", angles)
print("✅ 已保存到 methyl_dihedral_angles.npy，角度单位为度。")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 torsions 是一个 shape = (N,) 的数组，单位为角度
torsions = np.load("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/methyl_dihedral_angles.npy")

# KDE plot
sns.kdeplot(torsions, fill=True, bw_adjust=0.5, clip=[-180, 180])
plt.xticks(np.arange(-180, 181, 60))
plt.xlabel("Methyl torsion angle (°)")
plt.ylabel("Density")
plt.title("Ethanol Torsional Angle Distribution")
plt.tight_layout()
plt.savefig("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/ethanol_dihedral_distribution.png", dpi=300)
plt.show()

