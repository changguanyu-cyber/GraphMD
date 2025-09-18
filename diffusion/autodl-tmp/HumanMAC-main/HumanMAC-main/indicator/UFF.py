import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
mol = Chem.MolFromSmiles(aspirin_smiles)
mol = Chem.AddHs(mol)

conf_id = AllChem.EmbedMolecule(mol, maxAttempts=10)

if conf_id == -1:
    print("构象生成失败！")
else:
    result = AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
    if result == 0:
        print("UFF优化成功！")

        conf = mol.GetConformer(conf_id)
        coords = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

        coords = np.array(coords)  # 转成numpy数组，shape=(原子数, 3)

        # 保存到npy文件
        coords = coords.reshape(1,21,3)
        print(coords.shape)
        np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/aspirin_uff_conf.npy', coords)
        print("分子构象坐标已保存到 aspirin_uff_conf.npy")
    else:
        print("UFF优化未收敛或失败。")
