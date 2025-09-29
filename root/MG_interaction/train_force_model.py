import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
import schnetpack


from schnetpack.utils.compatibility import load_model
device = "cuda"

# load model
model_path = "/root/toluene/best_inference_model"
best_model = load_model(model_path, device=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)

data = np.load('/root/autodl-tmp/rmd17/npz_data/rmd17_uracil.npz')
print(data.files)
pos1 = np.load('/root/autodl-tmp/interpolated_trajectory1.npy')
pos1 = pos1[:1000]
pos2 = np.load('/root/autodl-tmp/interpolated_trajectory2.npy')
pos2 = pos2[:1000]
pos3 = np.load('/root/autodl-tmp/interpolated_trajectory3.npy')
pos3 = pos3[:1000]
pos4 = np.load('/root/autodl-tmp/interpolated_trajectory4.npy')
pos4 = pos4[:1000]
numbers = data["nuclear_charges"]
energies = []

# 4. 逐帧预测
for i in range(pos1.shape[0]):
    coords = pos1[i]  # (12, 3)
    atoms = Atoms(numbers=numbers, positions=coords)
    inputs = converter(atoms)

    out = best_model(inputs)
    energy = out['energy'].item()  # 单帧能量标量
    energies.append(energy)

energies = np.array(energies)
print("Predicted energies shape:", energies.shape)

# 5. 保存能量序列
np.save("/root/autodl-tmp/predicted_energies1.npy", energies)
energies = []

# 4. 逐帧预测
for i in range(pos2.shape[0]):
    coords = pos2[i]  # (12, 3)
    atoms = Atoms(numbers=numbers, positions=coords)
    inputs = converter(atoms)

    out = best_model(inputs)
    energy = out['energy'].item()  # 单帧能量标量
    energies.append(energy)

energies = np.array(energies)
print("Predicted energies shape:", energies.shape)

# 5. 保存能量序列
np.save("/root/autodl-tmp/predicted_energies2.npy", energies)
energies = []

# 4. 逐帧预测
for i in range(pos3.shape[0]):
    coords = pos3[i]  # (12, 3)
    atoms = Atoms(numbers=numbers, positions=coords)
    inputs = converter(atoms)

    out = best_model(inputs)
    energy = out['energy'].item()  # 单帧能量标量
    energies.append(energy)

energies = np.array(energies)
print("Predicted energies shape:", energies.shape)

# 5. 保存能量序列
np.save("/root/autodl-tmp/predicted_energies3.npy", energies)
energies = []

# 4. 逐帧预测
for i in range(pos4.shape[0]):
    coords = pos4[i]  # (12, 3)
    atoms = Atoms(numbers=numbers, positions=coords)
    inputs = converter(atoms)

    out = best_model(inputs)
    energy = out['energy'].item()  # 单帧能量标量
    energies.append(energy)

energies = np.array(energies)
print("Predicted energies shape:", energies.shape)

# 5. 保存能量序列
np.save("/root/autodl-tmp/predicted_energies4.npy", energies)
print("Energies saved to predicted_energies.npy")
