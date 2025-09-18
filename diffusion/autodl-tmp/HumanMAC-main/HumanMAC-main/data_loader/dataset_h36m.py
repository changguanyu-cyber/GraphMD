"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_h36m.py
"""

import numpy as np
import os
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton

class DatasetH36M(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = '/root/rmd17/extract_data_npz/ethanol_coord.npz'
        self.subjects_split = {'train': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                               'test': [10]}
        self.data_energy_file = '/root/rmd17/extract_data_npz/malonaldehyde_energy.npz'
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(
    parents=[-1, 0, 0, 0, 0, 1, 1, 1, 2],
    joints_left=[3, 5, 7],  # 左侧氢原子
    joints_right=[4, 6, 8]  # 右侧氢原子和 O-H
)





        #self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(9) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
 #       self.skeleton._parents[11] = 8
  #      self.skeleton._parents[14] = 8
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['coords'].item()
        data_e = np.load(self.data_energy_file, allow_pickle=True)['energies'].item()
        self.S1_skeleton = data_o['S1']['coords'][:1, self.kept_joints].copy()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        data_en = dict(filter(lambda x: x[0] in self.subjects, data_e.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f
        self.data_energies = data_en


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'all'}
    dataset = DatasetH36M('train', actions=actions)
    generator = dataset.sampling_generator()
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)
