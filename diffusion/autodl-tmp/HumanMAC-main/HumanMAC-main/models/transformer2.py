import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
from feature import get_features
import math

from utils import *
def variance_norm_regularization(K, α = 0.01):
    # K: (B, 1, T, D)
    K = K.squeeze(1)              # (B, T, D)
    norms = torch.norm(K, dim=-1)        # (B, T)
    mean_norm = norms.mean(dim=-1, keepdim=True)  # (B, 1)
    var = α * ((norms - mean_norm) ** 2).mean()
    var.backward()# scalar
    grad = K.grad
    return grad
def logsumexp_norm_loss_with_grad(K):
    """
    输入:
        K: Tensor, shape (B, 1, N, D), requires_grad=True
    输出:
        E: scalar, 损失值
        grad: Tensor, shape (B, N, D), 对 K 的梯度 (squeeze 掉第2维)
    """
    assert K.dim() == 4 and K.shape[1] == 1, "K must be shape (B, 1, N, D)"
    B, _, N, D = K.shape

    # 计算每个向量的平方范数 ||K||^2
    norms_sq = torch.sum(K ** 2, dim=-1)              # (B, 1, N)
    exps = torch.exp(0.5 * norms_sq)                  # (B, 1, N)
    S = torch.sum(exps, dim=2, keepdim=True)          # (B, 1, 1)
    E = torch.log(S).sum()                            # scalar loss

    # 反向传播
    E.backward()

    # 提取梯度
    grad = K.grad                                     # (B, 1, N, D)
    grad = grad.squeeze(1)                            # (B, N, D)

    return grad
def info_nce_loss(Q, K, temperature=0.07):
    Q = Q.squeeze(2)  # (B, T, D)
    K = K.squeeze(1)  # (B, N, D)
    B, T, D = Q.shape
    assert T == K.shape[1]

    # Normalize
    Q = F.normalize(Q, dim=-1)
    K = F.normalize(K, dim=-1)

    # Enable gradient tracking
    Q.requires_grad_(True)
    K.requires_grad_(True)

    # Compute logits
    logits = torch.matmul(Q, K.transpose(-1, -2)) / temperature  # (B, T, T)
    labels = torch.arange(T).to(Q.device).unsqueeze(0).expand(B, T)

    # Compute InfoNCE loss
    loss = F.cross_entropy(logits.reshape(B*T, T), labels.reshape(B*T))
    loss.backward()
    grad = K.grad
    return grad
def pad_2d_tensor(tensor, target_length, padding_value=0):
    """
    对二维 Tensor 的第二维进行 0 填充或截断，使其长度为 target_length。

    参数：
    - tensor: 输入 Tensor，形状为 (N, ?)
    - target_length: 希望第二维达到的长度
    - padding_value: 填充值（默认是 0）

    返回：
    - 新的 Tensor，形状为 (N, target_length)
    """
    padded_rows = []
    for row in tensor:
        length = row.shape[0]
        if length < target_length:
            pad = torch.full((target_length - length,), padding_value, dtype=row.dtype, device=row.device)
            new_row = torch.cat([row, pad])
        else:
            new_row = row[:target_length]
        padded_rows.append(new_row)

    return torch.stack(padded_rows)
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, features):
        """
        x: B, T, D
        """
        nonbonds = features['nonbonds']
        bonds = features['bonds']
        angles = features['angles']
        dihedrals = features['dihedrals']
        # 加和 bond + angle + diheral
        mu = features['bonds'].sum() + features['angles'].sum() + features['dihedrals'].sum()
        target_length = 20
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D(64,1,20,512)
        key = self.key(self.norm(x)).unsqueeze(1)
        bonds = pad_2d_tensor(bonds, target_length, padding_value=0)
        angles = pad_2d_tensor(angles, target_length, padding_value=0)
        nonbonds = pad_2d_tensor(nonbonds, target_length, padding_value=0)
        dihedrals = pad_2d_tensor(dihedrals, target_length, padding_value=0)
        total = bonds + angles + nonbonds + dihedrals
        total = total.reshape(total.shape[1], total.shape[0])
        total = pad_2d_tensor(total, D, padding_value=0)
        total = total.unsqueeze(0).unsqueeze(0)
        key = key + total
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb, features):
        x = self.sa_block(x, emb, features)
        x = self.ffn(x, emb)
        return x


class MotionTransformer2(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, features, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        if mod is not None:
            # mod_p=mod.reshape(B, -1)
            # print(f"mod shape before reshape: {mod_p.shape}")
            # print(f"cond_embed weight shape: {self.cond_embed.weight.shape}")

            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb, features)
            elif i >= (self.num_layers // 2):
                h = module(h, emb, features)
                h += prelist[-1]
                prelist.pop()
            i += 1

        output = self.out(h).view(B, T, -1).contiguous()
        return output

        # return output