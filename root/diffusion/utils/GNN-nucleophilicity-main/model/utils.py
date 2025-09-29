import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

    d = torch.linalg.vector_norm(r_i - r_j, ord=2, dim=1, keepdim=True)
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
