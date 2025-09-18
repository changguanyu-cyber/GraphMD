import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟角度数据
np.random.seed(42)
angle1 = np.random.vonmises(mu=0, kappa=5, size=1000) * 180 / np.pi
angle2 = angle1 + np.random.normal(0, 30, size=1000)

# 映射到 [-180, 180]
angle1 = (angle1 + 180) % 360 - 180
angle2 = (angle2 + 180) % 360 - 180

# 绘图
plt.figure(figsize=(6, 5))
sns.kdeplot(
    x=angle1, y=angle2,
    fill=True,
    cmap="YlGnBu",
    levels=10,
    bw_adjust=0.5,
    thresh=1e-4,
)

plt.xlabel("Aldehyde angle (°)")
plt.ylabel("Aldehyde angle (°)")
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.title("2D KDE of Aldehyde Angles")

# 保存图像
plt.tight_layout()
plt.savefig("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/kde_plot.png", dpi=300)  # 可改为 "kde_plot.pdf" 等
plt.show()
