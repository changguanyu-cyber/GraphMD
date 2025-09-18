import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .utils import SSPlus, CFConv, Z_CFConv
from abc import ABC, abstractmethod


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
        x, edge_index, position, z = data.x, data.edge_index, data.pos, data.z
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
