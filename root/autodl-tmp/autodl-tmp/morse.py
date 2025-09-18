from ase import Atoms
import numpy as np
from schnetpack.data import ASEAtomsData
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList
import os


data = np.load('/root/rmd17_aspirin.npz')
print(data.files)
numbers = data["nuclear_charges"]
print(numbers)
import numpy as np
from ase.data import covalent_radii

def get_bond_and_nonbond_pairs(positions, nuclear_charges, threshold=1.2):
    """
    根据距离和共价半径判断分子中哪些原子之间存在/不存在化学键。

    参数:
        positions: numpy.ndarray, 形状为 (N, 3)，表示每个原子的坐标
        nuclear_charges: list[int] or np.ndarray, 长度为 N，表示每个原子的核电荷
        threshold: float, 距离因子，一般取 1.1 - 1.3

    返回:
        bonded_pairs: list[tuple[int, int]] - 认为有化学键的原子对
        non_bonded_pairs: list[tuple[int, int]] - 认为无化学键的原子对
    """
    num_atoms = len(nuclear_charges)
    bonded_pairs = []
    non_bonded_pairs = []

    radii = np.array([covalent_radii[Z] for Z in nuclear_charges])

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            cutoff = threshold * (radii[i] + radii[j])
            if dist < cutoff:
                bonded_pairs.append((i, j))
            else:
                non_bonded_pairs.append((i, j))

    return bonded_pairs, non_bonded_pairs
# 示例用法（以 aspirin 为例）
data_ = data["coords"]
positions = data_[1]
nuclear_charges = numbers              # 每个原子的原子序数（C:6, H:1, O:8）

bonds, no_bonds = get_bond_and_nonbond_pairs(positions, nuclear_charges)

import numpy as np

# Morse 势函数（化学键）
def morse_potential(r, D_e=4.0, a=1.0, r_e=1.5):
    return D_e * (1 - np.exp(-a * (r - r_e)))**2

# Lennard-Jones 势函数（非化学键）
def lennard_jones_potential(r, epsilon=0.1, sigma=3.5):
    if r == 0:
        return 0  # 避免除以零
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# 总势能计算函数
def compute_total_pairwise_energy(elements, positions, bonded_pairs, nonbonded_pairs,
                                  morse_params={}, lj_params={}):
    total_energy = 0.0

    # 处理存在化学键的原子对（Morse 势）
    for (i, j) in bonded_pairs:
        r = np.linalg.norm(positions[i] - positions[j])
        # 可根据原子对选择参数
        key = tuple(sorted((elements[i], elements[j])))
        D_e, a, r_e = morse_params.get(key, (4.0, 1.0, 1.5))
        morse_E = morse_potential(r, D_e, a, r_e)
        total_energy += morse_E
    print(total_energy)

    # 处理非化学键的原子对（Lennard-Jones 势）
    for (i, j) in nonbonded_pairs:
        r = np.linalg.norm(positions[i] - positions[j])
        key = tuple(sorted((elements[i], elements[j])))
        epsilon, sigma = lj_params.get(key, (0.05, 2.5))
        Lennard_Jones_E = lennard_jones_potential(r, epsilon, sigma)
        total_energy += Lennard_Jones_E

    return total_energy
morse_params = {
    ('C', 'C'): (4.5, 1.0, 1.54),
    ('C', 'H'): (4.3, 1.1, 1.09),
    ('C', 'O'): (5.0, 1.2, 1.43),
}
lj_params = {
    ('C', 'C'): (0.1, 3.4),
    ('C', 'H'): (0.053, 3.0),
    ('O', 'H'): (0.079, 2.7),
}

# 计算总势能
elements = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'O', 'O', 'O', 'C', 'C', 'O',
            'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
total_energy = compute_total_pairwise_energy(elements, positions, bonds, no_bonds,
                                             morse_params, lj_params)

print(f"分子总势能: {total_energy:.4f} eV")

