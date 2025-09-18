import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def compute_dihedral(p0, p1, p2, p3):
    """计算二面角（torsion angle），单位为度"""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))

# 设定 torsion 原子索引（需根据结构确认）
carboxyl_torsion_indices = [5, 10, 9, 13]  # C–C–O–H
ester_torsion_indices = [6, 12, 11, 8]    # C–O–C–O

# 加载轨迹
trajectory = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy')
print(trajectory.shape)
torsion1 = []
torsion2 = []

for i, frame in enumerate(trajectory):

        a1 = frame[carboxyl_torsion_indices]  # shape (4, 3)
        a2 = frame[ester_torsion_indices]

        angle1 = compute_dihedral(*a1)
        angle2 = compute_dihedral(*a2)

        torsion1.append(angle1)
        torsion2.append(angle2)

torsion1 = np.array(torsion1)
torsion2 = np.array(torsion2)

# KDE 分布
plt.figure(figsize=(6, 5))
sns.kdeplot(
    x=torsion1,
    y=torsion2,
    fill=True,
    cmap="YlGnBu",
    levels=100,       # 更多的 contour 层级使颜色更平滑
    bw_adjust=0.5,
    thresh=1e-4
)

plt.xlabel("Carboxyl Torsion Angle (°)")
plt.ylabel("Ester Torsion Angle (°)")
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.title("2D KDE of Ester and Carboxyl Torsion Angles")
plt.tight_layout()

# 保存或显示图像
plt.savefig("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/asipirin_kde_plot.png", dpi=300)
plt.show()



