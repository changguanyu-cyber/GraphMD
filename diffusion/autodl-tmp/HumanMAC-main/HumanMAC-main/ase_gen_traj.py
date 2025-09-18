import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, MMFFGetMoleculeForceField, MMFFGetMoleculeProperties


def generate_naphthalene_trajectory(num_frames=100000, random_disp=0.02):
    """
    使用 MMFF94 力场生成萘 (C10H8) 分子的运动轨迹。

    参数：
        num_frames (int): 生成的帧数，即萘分子在不同时间步的构象。
        random_disp (float): 施加在原子坐标上的随机扰动幅度（单位：Å）。

    返回：
        numpy.ndarray: 形状为 (num_frames, 18, 3) 的 NumPy 数组，表示所有帧的坐标。
    """
    # 1. 生成萘分子（C10H8）
    naphthalene = Chem.MolFromSmiles("CCO")  # 萘的 SMILES
    naphthalene = Chem.AddHs(naphthalene)  # 添加氢原子

    # 2. 生成 3D 坐标
    AllChem.EmbedMolecule(naphthalene, AllChem.ETKDG())  # 生成 3D 坐标

    # 3. 获取 MMFF94 力场
    MMFFOptimizeMolecule(naphthalene)  # 先全局优化一次
    mmff_props = MMFFGetMoleculeProperties(naphthalene)  # 获取 MMFF 参数
    forcefield = MMFFGetMoleculeForceField(naphthalene, mmff_props)  # 传入参数

    # 获取分子构象
    conf = naphthalene.GetConformer()

    # 4. 记录轨迹
    trajectory = []

    for _ in range(num_frames):
        # 随机扰动原子坐标（模拟热运动）
        for i in range(naphthalene.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(i))  # 获取原子坐标
            pos += np.random.uniform(-random_disp, random_disp, size=3)  # 添加随机扰动
            conf.SetAtomPosition(i, pos)  # 重新设置原子坐标

        # MMFF94 结构优化
        MMFFOptimizeMolecule(naphthalene)

        # 记录当前帧坐标
        frame_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(naphthalene.GetNumAtoms())])
        trajectory.append(frame_coords)

    return np.array(trajectory)  # 形状：(num_frames, 18, 3)


def save_npy(filename, data):
    """
    将轨迹数据保存为 .npy 文件。

    参数：
        filename (str): 输出的 .npy 文件名。
        data (numpy.ndarray): 形状为 (num_frames, 18, 3) 的坐标数据。
    """
    np.save(filename, data)
    print(f"✅ 轨迹数据已保存至 {filename}")


# 生成萘的轨迹并保存
naphthalene_trajectory = generate_naphthalene_trajectory(num_frames=100000)
save_npy("/root/rmd17/extracted_data/ethanol_ase.npy", naphthalene_trajectory)

print("✅ 轨迹数据已保存！")
