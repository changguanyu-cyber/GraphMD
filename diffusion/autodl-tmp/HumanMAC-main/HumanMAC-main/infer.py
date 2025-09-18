import copy
import time
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import optim, nn
from torch_geometric.nn import GINEConv
from torch.utils.data import TensorDataset, DataLoader
from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from utils.script import sample_preprocessing
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_scatter import scatter
from abc import ABC, abstractmethod
from torch.nn import Softplus
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim=3*9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.float()
        return self.mlp(x).reshape(-1, 9, 3)
def compute_hooke_loss_batch_traj(pos_pred, bond_list, rest_lengths, k=1.0):
    """
    pos_pred: Tensor of shape (T, N, 3)  # T: number of frames, N: number of atoms, 3: 3D coordinates
    bond_list: list of (i, j) bond index pairs
    rest_lengths: Tensor of shape (len(bond_list),)  # Rest lengths of the bonds
    k: spring constant for the Hooke's law
    """


    bond_list = torch.tensor(bond_list, dtype=torch.long)
    rest_lengths = rest_lengths.squeeze()


    # List to store losses for each frame
    hooke_losses = []

    # Iterate over each frame (T)
    for t in range(pos_pred.shape[0]):
        pos = pos_pred[t]  # (N, 3) for the current frame

        # Ensure pos is a torch tensor, if it is not already
        if isinstance(pos, np.ndarray):
            pos = torch.tensor(pos, dtype=torch.float32)

        # Compute the bond distances
        bond_dists = torch.norm(pos[bond_list[:, 0]] - pos[bond_list[:, 1]], dim=1)

        # Compute the displacement from the rest length
        rest_lengths = rest_lengths.to(bond_dists.device)
        displacement = bond_dists - rest_lengths

        # Compute the Hooke's energy (spring potential energy)
        hooke_energy = 0.5 * k * (displacement ** 2)

        # Append the Hooke energy for each bond in this frame
        hooke_losses.append(hooke_energy)

    # Convert the losses into a tensor with shape (T, num_bonds)
    hooke_losses_tensor = torch.stack(hooke_losses, dim=0)  # shape: (T, num_bonds)

    # Reshape the tensor to (T * num_bonds, 1)
    return hooke_losses_tensor.reshape(-1, 1)
def compute_bond_lengths(trajectory, bond_list):
    """
    根据轨迹和键列表计算键长。

    Parameters:
        trajectory (torch.Tensor): 形状为 (9, 3)，每个原子在三维空间中的坐标。
        bond_list (list): 键连接的原子对索引，形状为 [(i, j), ...]。

    Returns:
        torch.Tensor: 形状为 (键数, 1) 的键长张量。
    """
    bond_lengths = []
    trajectory = trajectory.clone().detach().to(torch.float32).to(device)


    for bond in bond_list:
        i, j = bond  # 获取键连接的两个原子
        pos_i = trajectory[i].clone().detach()
        pos_j = trajectory[j].clone().detach()  # 确保pos_i是torch.Tensor类型

        bond_length = torch.norm(pos_i - pos_j)  # 计算两个原子之间的距离
        bond_lengths.append(bond_length.unsqueeze(0))  # 添加一个维度，保持形状 (1,)

    # 将所有键长拼接成一个 (键数, 1) 的 tensor
    return torch.cat(bond_lengths, dim=0)
def compute_rest_lengths(pos_batch, bond_list):
    """
    计算平衡态的键长（rest lengths），用于 Hooke 势能等。

    Parameters:
        pos_batch (torch.Tensor): shape (B, N, 3)，轨迹中的一批分子坐标
        bond_list (list or Tensor): 键连接，例如 [(0,1), (1,2), ...]

    Returns:
        torch.Tensor: shape (num_bonds, 1)，第一帧中每条键的长度
    """
    # 如果 bond_list 是列表，先转换为 tensor
    if isinstance(bond_list, list):
        bond_list = torch.tensor(bond_list, dtype=torch.long)

    # 确保 pos_batch 是 torch.Tensor 类型
    if isinstance(pos_batch, np.ndarray):
        pos_batch = torch.tensor(pos_batch, dtype=torch.float32)

    # 取第一帧的原子位置 (N, 3)


    # 提取键连接的原子位置
    atom_i = pos_batch[bond_list[:, 0]]  # shape: (num_bonds, 3)
    atom_j = pos_batch[bond_list[:, 1]]  # shape: (num_bonds, 3)

    # 计算键长 (num_bonds, 1)
    bond_lengths = torch.norm(atom_i - atom_j, dim=1, keepdim=True)  # shape: (num_bonds, 1)
    return bond_lengths.detach()
def compute_angle(a, b, c):
    """计算角度（单位：度），a-b-c"""
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def compute_bond_angles_single_frame(frame):
    """
    计算一帧中所有键角，返回 shape = (num_angles, 1)，单位为度（°）

    参数:
        frame: np.ndarray 或 torch.Tensor，shape=(9, 3)
    返回:
        torch.Tensor: shape=(num_angles, 1)
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()


    angle_triplets = [
        (3, 0, 4),  # H-C-H
        (3, 0, 1),  # H-C-C
        (4, 0, 1),  # H-C-C
        (0, 1, 2),  # C-C-O
        (1, 2, 5),  # C-O-H
        (1, 2, 6),  # C-O-H
        (5, 2, 6),  # H-O-H
        (7, 1, 8),  # H-C-H
    ]

    angles = []
    for i, j, k in angle_triplets:
        theta = compute_angle(frame[i], frame[j], frame[k])
        angles.append([theta])  # 使其为 (num_angles, 1)

    return torch.tensor(angles, dtype=torch.float32)
class SSPlus(Softplus):
    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold) - np.log(2.)


def gaussian(r_i, r_j, gamma: float, u_max: float, step: float):
    if u_max < 0.1:
        raise ValueError('u_max should not be smaller than 0.1')

    d = torch.norm(r_i - r_j, p=2, dim=1, keepdim=True)

    u_k = torch.arange(0, u_max, step, device=r_i.device).unsqueeze(0)

    out = torch.exp(-gamma * torch.square(d-u_k))
    return out


class CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step):
        super().__init__()
        self.gamma = gamma
        self.u_max = u_max
        self.step = step

        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_g = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )

    def forward(self, x, edge_index, z, position):
        v = self.propagate(edge_index, x=x, position=position)
        return v

    def message(self, x_i, x_j,  position_i, position_j, index):
        # g
        g = gaussian(position_i, position_j,
                     gamma=self.gamma, u_max=self.u_max, step=self.step)
        g = self.mlp_g(g)

        # out
        out = x_j * g
        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out


class Z_CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step):
        super().__init__()
        self.gamma = gamma
        self.u_max = u_max
        self.step = step

        self.embedding_z = nn.Embedding(100, n_filters, padding_idx=0)
        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_g = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )

    def forward(self, x, edge_index, z, position):
        v = self.propagate(edge_index, x=x, z=z, position=position)
        return v

    def message(self, x_i, x_j, z_i, z_j, position_i, position_j, index):
        # g
        g = gaussian(position_i, position_j,
                     gamma=self.gamma, u_max=self.u_max, step=self.step)
        g = self.mlp_g(g)

        # z
        z_j = self.embedding_z(z_j.reshape(-1))
        z_i = self.embedding_z(z_i.reshape(-1))
        z = self.mlp_z(z_j * z_i)

        w = g * z

        # out
        out = x_j * w
        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out

class Interaction(nn.Module):
    def __init__(self,
                 conv_module,
                 n_filters: int,
                 u_max: float,
                 gamma: float = 10.0,
                 step: float = 0.1
                 ):
        super().__init__()

        self.lin_1 = nn.Linear(n_filters, n_filters)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )

        # initialize a cfconv block
        self.cfconv = conv_module(n_filters=n_filters, gamma=gamma, u_max=u_max, step=step)

    def forward(self, x, edge_index, z, position):
        # x
        m = self.lin_1(x)
        v = self.cfconv(m, edge_index, z, position)
        v = self.mlp(v)
        x = x + v
        return x


class ModelBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, n_filters=64, n_interactions=2):
        super().__init__()
        #
        self.embedding_z = nn.Embedding(100, n_filters, padding_idx=0)  # atomic numbers are all smaller than 99
        self.embedding_solv = nn.Embedding(4, 64)

        # Interaction Module
        self.n_interactions = n_interactions
        self.convs = nn.ModuleList()

    @abstractmethod
    def forward(self, data):
        pass

    def loss(self, pred, label):
        pred, label = pred.reshape(-1), label.reshape(-1)
        return F.mse_loss(pred, label)


class SchNetAvg(ModelBase):
    def __init__(self, n_filters=64, n_interactions=2, u_max=5.0, output_dim=1):
        super().__init__(n_filters, n_interactions)

        # Interaction module
        for _ in range(self.n_interactions):
            self.convs.append(
                Interaction(
                    conv_module=CFConv,
                    n_filters=n_filters,
                    u_max=u_max
                )
            )

        # NNs
        self.post_mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, 64)
        )
        self.post_mlp2 = nn.Sequential(
            nn.Linear(64 + 32, 128),
            SSPlus(),
            nn.Linear(128, 32),
            SSPlus(),
            nn.Linear(32, output_dim)
        )
        self.mlp_solv = nn.Sequential(
            nn.Linear(64, 64),
            SSPlus(),
            nn.Linear(64, 32)
        )

    def forward(self, data):
        x, edge_index, position, z, batch = data.x, data.edge_index, data.pos, data.Z, data.batch
        solvent = data.solvent
        nuc_index = data.nuc_index - 1  # minus 1
        # first embedding
        x = self.embedding_z(z.reshape(-1))

        # solvent
        solvent = self.embedding_solv(solvent)
        solvent = self.mlp_solv(solvent)

        # interaction block
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=position)

        # post mlp
        x = self.post_mlp(x)

        #
        x = scatter(x, batch, dim=-2, reduce='mean')
        out = torch.cat((x, solvent), dim=1)
        out = self.post_mlp2(out)

        return out


class SchNetNuc(SchNetAvg):
    def __init__(self, n_filters=64, n_interactions=2, u_max=5.0, output_dim=1):
        super().__init__(
            n_filters=n_filters,
            n_interactions=n_interactions,
            u_max=u_max,
            output_dim=output_dim
        )

    def forward(self, data):
        x, edge_index, position, z, batch = data.x, data.edge_index, data.pos, data.Z, data.batch
        solvent = data.solvent
        nuc_index = data.nuc_index - 1  # minus 1
        # first embedding
        x = self.embedding_z(z.reshape(-1))

        # solvent
        solvent = self.embedding_solv(solvent)
        solvent = self.mlp_solv(solvent)

        # interaction block
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=position)

        # post mlp
        x = self.post_mlp(x)

        #
        x = x[nuc_index]
        out = torch.cat((x, solvent), dim=1)
        out = self.post_mlp2(out)

        return out
class ZSchNet(ModelBase):
    def __init__(self, n_filters=64, n_interactions=2, u_max=5.0, output_dim=1):
        super().__init__(n_filters, n_interactions)

        # Interaction module
        for _ in range(self.n_interactions):
            self.convs.append(
                Interaction(
                    conv_module=Z_CFConv,
                    n_filters=n_filters,
                    u_max=u_max
                )
            )

        # NNs
        self.post_mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, 64)
        )
        self.post_mlp2 = nn.Sequential(
            nn.Linear(64 + 128, 128),
            SSPlus(),
            nn.Linear(128, 32),
            SSPlus(),
            nn.Linear(32, output_dim)
        )
        self.mlp_solv = nn.Sequential(
            nn.Linear(52, 64),
            nn.Linear(64, 64),
            SSPlus(),
            nn.Linear(64, 128)
        )

    def forward(self, data):
        x, edge_index, pos0, z, batch = data.x, data.edge_index, data.pos, data.z, data.batch
        angle = data.angle
        keylen = data.keylen
        hook = data.hook
        morse = data.morse
        pos = pos0.reshape(-1,9,3)
        B, N, _ = pos.shape

        # 计算所有原子对之间的差向量，形状为 [64, 9, 9, 3]
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)

        # 计算 L2 距离，得到 [64, 9, 9]
        dist_matrix = torch.norm(diff, dim=-1)

        # 获取上三角索引（不含对角线）
        triu_indices = torch.triu_indices(N, N, offset=1)

        # 提取上三角部分，得到形状 [64, 36]
        dist_upper = dist_matrix[:, triu_indices[0], triu_indices[1]]

        # 添加一个维度，变为 [64, 36, 1]
        dist = dist_upper.unsqueeze(-1)

        morse = morse.reshape(-1, 8, 1)
        keylen = keylen.reshape(-1, 8, 1)
        hook = hook.reshape(-1, 8, 1)
        angle = angle.reshape(-1, 8, 1)

        edge_pos_attr = torch.cat((morse, keylen, hook), dim=2)
        solvent = torch.cat((angle, dist, morse), dim=1)
        solvent = solvent.reshape(-1,52)
        nuc_index = data.nuc_index - 1  # minus 1
        # first embedding
        x = self.embedding_z(z.reshape(-1))

        # solvent
        solvent = self.mlp_solv(solvent)

        # interaction block
        z = z.reshape(-1,1)

        edge_pos_attr = edge_pos_attr.reshape(-1,3)
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=pos0)

        # post mlp
        x = self.post_mlp(x)

        #
        x = x[nuc_index]

        out = torch.cat((x, solvent), dim=1)
        out = self.post_mlp2(out)
        return out
class ZSchNet_CDFT(ModelBase):
    def __init__(self, n_filters=64, n_interactions=2, u_max=5.0, output_dim=1):
        super().__init__(n_filters, n_interactions)

        # Interaction module
        for _ in range(self.n_interactions):
            self.convs.append(
                Interaction(
                    conv_module=Z_CFConv,
                    n_filters=n_filters,
                    u_max=u_max
                )
            )

        # NNs
        self.post_mlp = nn.Sequential(
            nn.Linear(n_filters + n_filters + 32, 256),
            SSPlus(),
            nn.Linear(256, 32),
            SSPlus(),
            nn.Linear(32, output_dim)
        )
        self.mlp_u0 = nn.Sequential(
            nn.Linear(10, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )
        self.mlp_u1 = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )
        self.mlp_u2 = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )
        self.mlp_u3 = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )
        self.mlp_solv = nn.Sequential(
            nn.Linear(64, 64),
            SSPlus(),
            nn.Linear(64, 32)
        )

    def forward(self, data):
        x, edge_index, position, z, batch = data.x, data.edge_index, data.pos, data.Z, data.batch
        solvent = data.solvent
        nuc_index = data.nuc_index - 1  # minus 1
        cdft = data.cdft
        # first embedding
        x = self.embedding_z(z.reshape(-1))

        # solvent
        solvent = self.embedding_solv(solvent)
        solvent = self.mlp_solv(solvent)

        # interaction block
        u = cdft
        u = self.mlp_u0(u)
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=position)
            m = self.mlp_u1(scatter(x, batch, dim=0, reduce='mean')) + self.mlp_u2(u)
            m = self.mlp_u3(m)
            u = u + m

        #
        x = x[nuc_index]
        out = torch.cat((x, u, solvent), dim=1)
        out = self.post_mlp(out)

        return out

def build_graph_from_frame(frame, bond_list):
    """
    frame: np.ndarray of shape (num_atoms, 3)
    morse_energy: np.ndarray of shape (num_atoms, 1)
    bond_list: list of tuples indicating bonded atom indices
    """
    atom_type_to_id = {'H': 0, 'C': 1, 'O': 2}
    atoms = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    x = torch.tensor([[atom_type_to_id[a]] for a in atoms], dtype=torch.float)

    num_atoms = frame.shape[0]
    frame_tensor = frame.clone().detach().to(torch.float32)


    # --- 1. Compute distance matrix
    diff = frame_tensor.unsqueeze(1) - frame_tensor.unsqueeze(0)
    dist_matrix = torch.norm(diff, dim=-1)  # shape: (num_atoms, num_atoms)


    # --- 2. Build edge_index and edge_attr
    edge_index = []
    edge_attr = []

    for i, j in bond_list:
        edge_index.append([i, j])
        edge_index.append([j, i])  # bidirectional
        edge_attr.append([dist_matrix[i, j].item()])
        edge_attr.append([dist_matrix[j, i].item()])  # same value, for symmetry

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # shape: (2*num_bonds, 1)

    # --- 3. Node features: Morse energy  # shape: (num_atoms, 1) or (num_atoms, d)

    return Data(x=x, edge_index=edge_index, edge_attr=dist_matrix)


class GNNMotionPredictor(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim=1, hidden_dim=64):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim = hidden_dim
        )
        self.out = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        out = self.out(x)
        return out.view(-1, 9, 3)


def infer_bond_type(i, j, bond_list):
    # 确保 (i, j) 和 (j, i) 被认为是相同的键
    bond_pair = (i, j) if (i, j) in bond_list else (j, i)

    # bond_types 和 bond_list 索引一一对应
    bond_types = ["C-H", "C-H", "C-H", "C-H", "C-C", "C-O", "C-O", "O-H"]

    # 如果找到了原子对 (i, j) 或 (j, i)，返回对应的键类型
    if bond_pair in bond_list:
        bond_index = bond_list.index(bond_pair)  # 找到原子对的索引
        return bond_types[bond_index]  # 返回对应的键类型

    # 如果找不到键类型，返回 None
    return None
def compute_morse_potential_batch(positions, bond_list, bond_params):
    """
    positions: [B, N, 3] or [N, 3] tensor
    bond_list: list of (i, j)
    bond_params: dict with bond type parameters
    return: [B, num_bonds, 1] tensor (Morse potentials for each bond)
    """
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.float32)

    if positions.dim() == 2:
        positions = positions.unsqueeze(0)  # [1, N, 3]

    B, N, _ = positions.shape
    energies = []

    for (i, j) in bond_list:
        if i >= N or j >= N:
            raise IndexError(f"Index out of bounds: i={i}, j={j}, but N={N}")

        dists = torch.norm(positions[:, i] - positions[:, j], dim=-1)  # [B]

        bond_type = infer_bond_type(i, j, bond_list)
        if bond_type is None:
            continue
        params = bond_params[bond_type]

        D_e = torch.tensor(params["D_e"], device=positions.device)
        r_e = torch.tensor(params["r_e"], device=positions.device)
        a = torch.tensor(params["a"], device=positions.device)

        V = D_e * (1 - torch.exp(-a * (dists - r_e))) ** 2  # [B]
        energies.append(V.unsqueeze(-1))  # [B, 1]

    if not energies:
        return torch.zeros((B, 0, 1), device=positions.device)

    # [B, num_bonds, 1]
    return torch.stack(energies, dim=1)
def recursive_inference(model_path, init_frame, bond_list, steps, device):
    """
    从初始帧开始，递归预测后续 steps 帧。

    Parameters:
        model_path (str): 模型权重路径
        init_frame (np.ndarray): 初始帧坐标，shape = (9, 3)
        bond_list (list): 分子结构键连接
        steps (int): 要预测的帧数（递归推理）
        device (torch.device): 运行设备
    Returns:
        np.ndarray: shape = (steps + 1, 9, 3)
    """
    # 1. 加载模型
    bond_params = {
        "C-H": {"D_e": 4.3, "r_e": 1.09, "a": 1.942},
        "C-C": {"D_e": 3.6, "r_e": 1.54, "a": 1.055},
        "C-O": {"D_e": 3.7, "r_e": 1.43, "a": 1.102},
        "O-H": {"D_e": 4.8, "r_e": 0.96, "a": 1.581},
    }
    model = ZSchNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    current_pos = torch.tensor(init_frame, dtype=torch.float32).to(device)  # (1, 9, 3)
    predicted_traj = [init_frame]
    rest_lengths = compute_rest_lengths(init_frame, bond_list)# 存储初始帧

    mole_weight_kg = 46.07e-3 / 6.022e23  # 转化为 kg

    # 玻尔兹曼常数 (J/K)
    k_B = 1.380649e-23

    # 模拟的温度 (K)
    T = 300.0

    # 计算速度的标准差 (m/s)
    std_velocity = np.sqrt(k_B * T / mole_weight_kg)

    # 假设我们有 N 个原子，3 维空间
    # 计算一个形状为 (N, 3) 的速度张量
    # 假设乙醇有 9 个原子
    N_atoms = 9

    # 初始化速度，使用正态分布生成速度
    velocity = torch.randn(N_atoms, 3).to(device) * std_velocity

    # 质量设置（原子质量单位 amu）
    masses = torch.tensor([12.01, 12.01, 16.00, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008], device=device).view(9, 1)

    # 时间步长（单位：秒）
    dt = 1e-7  # 10 fs

    for step in range(steps):
        frame = current_pos

        # === 构建图结构 ===
        graph = build_graph_from_frame(frame, bond_list)
        graph.pos = frame.clone().detach().requires_grad_(True).to(device)
        graph.angle = compute_bond_angles_single_frame(frame)
        graph.z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1]).to(device)
        graph.x = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1]).to(device)
        graph.morse = compute_morse_potential_batch(frame, bond_list, bond_params).detach()
        graph.keylen = compute_bond_lengths(frame, bond_list)
        graph.hook = compute_hooke_loss_batch_traj(frame.reshape(-1, 9, 3), bond_list, rest_lengths, k=1.0)
        graph.dist = graph.edge_attr
        graph.nuc_index = torch.tensor([2 + 1], dtype=torch.long).to(device)
        graph = graph.to(device)

        # === 模型预测能量 ===
        energy = model(graph)
        print(energy)

        # === 当前加速度 ===
        force = -torch.autograd.grad(energy, graph.pos, create_graph=True)[0]
        acceleration = force / masses

        # === 位置更新 ===
        next_pos = current_pos + velocity * dt + 0.5 * acceleration * dt ** 2

        # === 构建下一帧图结构以计算新力 ===
        next_graph = build_graph_from_frame(next_pos, bond_list)
        next_graph.pos = next_pos.clone().detach().requires_grad_(True).to(device)
        next_graph.angle = compute_bond_angles_single_frame(next_pos)
        next_graph.z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1]).to(device)
        next_graph.x = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1]).to(device)
        next_graph.morse = compute_morse_potential_batch(next_pos, bond_list, bond_params).detach()
        next_graph.keylen = compute_bond_lengths(next_pos, bond_list)
        next_graph.hook = compute_hooke_loss_batch_traj(next_pos.reshape(-1, 9, 3), bond_list, rest_lengths, k=1.0)
        next_graph.dist = next_graph.edge_attr
        next_graph.nuc_index = torch.tensor([2 + 1], dtype=torch.long).to(device)
        next_graph = next_graph.to(device)

        # === 计算新加速度 ===
        next_energy = model(next_graph)
        next_force = -torch.autograd.grad(next_energy, next_graph.pos, create_graph=True)[0]
        next_acceleration = next_force / masses

        # === 速度更新 ===
        velocity = velocity + 0.5 * (acceleration + next_acceleration) * dt

        # === 记录轨迹 ===
        predicted_traj.append(next_pos.squeeze(0).detach().cpu().numpy())

        # 准备下一步
        current_pos = next_pos

    return np.stack(predicted_traj)
model_path = "/root/autodl-tmp/HumanMAC-main/HumanMAC-main/results/h36m_86/models/ckpt_300.pt"
# 加载轨迹和能量数据
traj = np.load("/root/rmd17/extracted_data/ethanol_coord.npy")
init_frame = traj[0]
bond_list = [
    (0, 1), (1, 2), (1, 3), (1, 4),
    (2, 5), (2, 6), (2, 7), (7, 8)
]
# shape = (9, 3)
bond_params = {
    "C-H": {"D_e": 4.3, "r_e": 1.09, "a": 1.942},
    "C-C": {"D_e": 3.6, "r_e": 1.54, "a": 1.055},
    "C-O": {"D_e": 3.7, "r_e": 1.43, "a": 1.102},
    "O-H": {"D_e": 4.8, "r_e": 0.96, "a": 1.581},
}

predicted_traj = recursive_inference(model_path, init_frame, bond_list, steps=1, device=device)

np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy", predicted_traj)

