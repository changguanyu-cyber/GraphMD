from ase import Atoms
import numpy as np
from schnetpack.data import ASEAtomsData
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList
import os


data = np.load('/root/autodl-tmp/rmd17/npz_data/rmd17_paracetamol.npz')
print(data.files)
numbers = data["nuclear_charges"]
atoms_list = []
property_list = []
for positions, energies, forces in zip(data["coords"], data["energies"], data["forces"]):
    ats = Atoms(positions=positions, numbers=numbers)
    properties = {'energy': energies, 'forces': forces}
    property_list.append(properties)
    atoms_list.append(ats)

print('Properties:', property_list[0])
for prop_dict in property_list:
    for key in prop_dict:
        # 确保每个属性值都是 NumPy 数组
        if isinstance(prop_dict[key], (float, int)):
            prop_dict[key] = np.array([prop_dict[key]], dtype=np.float32)
        elif isinstance(prop_dict[key], list):
            prop_dict[key] = np.array(prop_dict[key], dtype=np.float32)
db_path = './new_dataset.db'
if os.path.exists(db_path):
    os.remove(db_path)

# 创建新的数据库
new_dataset = ASEAtomsData.create(
    db_path,
    distance_unit='Ang',
    property_unit_dict={
        'energy': 'kcal/mol',
        'forces': 'kcal/mol/Ang'
    }
)

# 添加系统到数据库
new_dataset.add_systems(property_list, atoms_list)
print('Number of reference calculations:', len(new_dataset))
print('Available properties:')
# 在创建数据集后检查长度是否匹配
print(f"数据库条目数: {len(new_dataset)}")
print(f"尝试访问的索引: 0")  # 你尝试访问的是第0个元素

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)
