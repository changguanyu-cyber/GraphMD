import numpy as np

# 加载 .npz 文件
data = np.load("/root/autodl-tmp/rmd17/npz_data/rmd17_aspirin.npz")  # 替换为你的文件名
print(data.files)
# 提取 nuclear_charge 和坐标
charges = data["nuclear_charges"]            # shape: (N,)
coords = data["coords"]                     # shape: (T, N, 3)

# 提取第一帧坐标
first_frame = coords[0]                     # shape: (N, 3)

# 原子序号映射到元素符号（可根据需要扩展）
periodic_table = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
}

# 将每个原子的元素符号和坐标格式化为 .xyz 行
xyz_lines = []
for z, pos in zip(charges, first_frame):
    element = periodic_table.get(int(z), "X")  # 若找不到元素符号用 "X" 占位
    line = f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
    xyz_lines.append(line)

# 写入 .xyz 文件
with open("/root/aspirin.xyz", "w") as f:
    f.write(f"{len(charges)}\n")
    f.write("Generated from npz\n")
    f.write("\n".join(xyz_lines))
