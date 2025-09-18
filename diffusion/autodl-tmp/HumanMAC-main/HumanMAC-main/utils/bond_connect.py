import numpy as np

# 示例输入
data = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy')
data = data[:1000]
data2 = data[0]
atomic_number_to_symbol = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
            15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
            # 如有更多元素，请扩展
        }

        # 假设你有如下原子序号：
molecule = np.load('/root/rmd17/npz_data/rmd17_aspirin.npz')
print(molecule.files)
nuclear_charges = molecule['nuclear_charges']

coordinates = data2

# 共价半径表（单位：Å）
covalent_radii = {
    1: 0.31,  # H
    6: 0.76,  # C
    7: 0.71,
    8: 0.66,
    # 可扩展更多元素
}

# 计算连接
def infer_bond_connectivity(nuclear_charges, coordinates, tolerance=0.45):
    num_atoms = len(nuclear_charges)
    bond_list = [[] for _ in range(num_atoms)]

    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            ri = covalent_radii.get(nuclear_charges[i], 0.7)
            rj = covalent_radii.get(nuclear_charges[j], 0.7)
            cutoff = ri + rj + tolerance
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            if dist <= cutoff:
                bond_list[i].append(j)
                bond_list[j].append(i)

    return bond_list

bond_connectivity_list = infer_bond_connectivity(nuclear_charges, coordinates)
print(bond_connectivity_list)