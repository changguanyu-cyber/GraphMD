import numpy as np

# 原子共价半径参考（单位：Å），只包含常见元素
cov_radii = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    # 需要可以继续补充
}


def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    coords = np.array(coords)
    return atoms, coords


def calculate_bond_lengths(atoms, coords, scale_factor=1.2):
    """
    scale_factor 用于容忍度，常用1.2-1.3倍共价半径和作为判断阈值
    返回键长字典：键为(atom_idx1, atom_idx2)，值为距离
    """
    n = len(atoms)
    bonds = {}
    for i in range(n):
        for j in range(i + 1, n):
            elem1 = atoms[i]
            elem2 = atoms[j]

            # 忽略未知元素
            if elem1 not in cov_radii or elem2 not in cov_radii:
                continue

            dist = np.linalg.norm(coords[i] - coords[j])
            cutoff = scale_factor * (cov_radii[elem1] + cov_radii[elem2])
            if dist <= cutoff:
                bonds[(i, j)] = dist
    return bonds


# 例子：读取xyz文件并计算键长
xyz_file = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/uff.xyz'  # 你的xyz文件路径
atoms, coords = read_xyz(xyz_file)
bond_lengths = calculate_bond_lengths(atoms, coords)

# 打印所有键长
for (i, j), length in bond_lengths.items():
    print(f"键 {atoms[i]}({i}) - {atoms[j]}({j}): {length:.3f} Å")

# 保存键长，方便后续使用（保存为numpy npz文件）
import numpy as np

np.savez('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/aspirin_bond_lengths.npz', bonds=bond_lengths)

print("键长已保存到 aspirin_bond_lengths.npz")
