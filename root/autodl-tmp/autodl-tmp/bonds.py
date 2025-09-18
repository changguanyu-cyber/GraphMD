import numpy as np
from ase.data import covalent_radii
def get_bond_and_nonbond_pairs(positions, nuclear_charges, threshold=1.2):
    """
    每个分子处理：返回有键和无键的原子对索引列表
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
data = np.load('/root/rmd17_aspirin.npz')
positions = data["coords"][1]  # 第 2 帧坐标
nuclear_charges = data["nuclear_charges"]
bonded_pairs, nonbonded_pairs = get_bond_and_nonbond_pairs(positions, nuclear_charges)
import torch

torch.save({
    "bonded_pairs": bonded_pairs,
    "nonbonded_pairs": nonbonded_pairs
}, "/root/autodl-tmp/pair_indices.pt")
