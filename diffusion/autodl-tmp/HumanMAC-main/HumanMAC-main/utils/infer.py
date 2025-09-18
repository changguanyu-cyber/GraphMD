import torch
import torch.nn as nn
from torch_scatter import scatter
from abc import ABC, abstractmethod
from torch.nn import Softplus
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
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
    def __init__(self, n_filters=64, n_interactions=2, u_max=5.0, output_dim=3*9):
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
            nn.Linear(44, 64),
            nn.Linear(64, 64),
            SSPlus(),
            nn.Linear(64, 128)
        )

    def forward(self, data):
        x, edge_index, position, z = data.x, data.edge_index, data.pos, data.z
        morse = data.morse
        dist = data.dist
        dist = dist.reshape(-1,9,9)
        triu_indices = torch.triu_indices(9, 9, offset=1)  # shape (2, num_elems)

        # 提取上三角元素（不包括对角线），结果是 [B, N*(N-1)/2]
        dist = dist[:, triu_indices[0], triu_indices[1]]
        dist = dist.reshape(-1, 36,1)
        solvent = torch.cat((morse, dist), dim=1)
        solvent = solvent.reshape(-1,44)
        nuc_index = data.nuc_index - 1  # minus 1
        # first embedding
        x = self.embedding_z(z.reshape(-1))

        # solvent
        solvent = self.mlp_solv(solvent)

        # interaction block
        z = z.reshape(-1,1)
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=position)

        # post mlp
        x = self.post_mlp(x)

        #
        x = x[nuc_index]

        out = torch.cat((x, solvent), dim=1)
        out = self.post_mlp2(out)

        return out.reshape(-1,9,3)


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
class MotionPredictor(nn.Module):
    def __init__(self):
        super(MotionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(27 + 81 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 27),  # 输出 (9, 3)
        )

    def forward(self, x_t, d_t, e_t):
        # Flatten inputs
        x_feat = x_t.view(x_t.shape[0], -1)   # [B, 27]
        d_feat = d_t.view(d_t.shape[0], -1)   # [B, 81]
        input_feat = torch.cat([x_feat, d_feat, e_t], dim=-1)  # [B, 109]
        delta_x_pred = self.net(input_feat).view(-1, 9, 3)     # [B, 9, 3]
        return delta_x_pred

def generate_trajectory(model, x0, d0, e0, steps=125):
    """
    x0: 初始坐标，(9, 3)
    d0: 初始距离矩阵，(9, 9)
    e0: 初始能量，(1,)
    返回：预测的轨迹序列，(125, 9, 3)
    """
    model.eval()
    traj = [x0.copy()]
    x = torch.tensor(x0[None], dtype=torch.float32).to(device)  # (1, 9, 3)
    d = torch.tensor(d0[None], dtype=torch.float32).to(device)  # (1, 9, 9)
    e = torch.tensor(e0[None], dtype=torch.float32).to(device)  # (1, 1)

    for _ in range(steps - 1):
        with torch.no_grad():
            delta = model(x, d, e)
        x_next = x + delta
        traj.append(x_next.squeeze(0).cpu().numpy())

        # 更新下一轮输入
        x = x_next
        d = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
        e = e  # 若能量可变可更新

    return np.stack(traj, axis=0)  # (125, 9, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZSchNet().to(device)
ckpt = torch.load('./results/h36m_86/models/ckpt_300.pt')
model.load_state_dict(ckpt)
traj = np.load("/root/rmd17/extracted_data/ethanol_coord.npy")
num_frames = traj.shape[0]

# 构造前一帧和当前帧
x_t = traj[:-1]  # (99999, 9, 3)
x_tp1 = traj[1:]  # (99999, 9, 3)

# 计算距离矩阵
diff = x_t[:, :, np.newaxis, :] - x_t[:, np.newaxis, :, :]  # (99999, 9, 9, 3)
d_t = np.linalg.norm(diff, axis=-1)  # (99999, 9, 9)

# 坐标差值作为目标
delta_x = x_tp1 - x_t  # (99999, 9, 3)

# 能量，如果你有的话
# 假设 energy.shape == (100000,)
# e_t = energy[:-1].reshape(-1, 1)
# 如果没有，可以初始化为0：
e_t = np.load("/root/rmd17/extracted_data/ethanol_energy.npy")
e_t = e_t[:-1]

# 转为 torch 张量
x_t = torch.tensor(x_t, dtype=torch.float32)
d_t = torch.tensor(d_t, dtype=torch.float32)
e_t = torch.tensor(e_t, dtype=torch.float32)
e_t = e_t.unsqueeze(-1)
x_t = x_t[0]
d_t = d_t[0]
e_t = e_t[0]
traj = generate_trajectory(model, x_t, d_t, e_t)
np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy", traj)
# 或其他方式加载
