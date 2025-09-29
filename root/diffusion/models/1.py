import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
def variance_norm_regularization(K, α = 0.01):
    # K: (B, 1, T, D)
    K.requires_grad_(True)
    K = K.squeeze(1)              # (B, T, D)
    norms = torch.norm(K, dim=-1)        # (B, T)
    mean_norm = norms.mean(dim=-1, keepdim=True)  # (B, 1)
    var = α * ((norms - mean_norm) ** 2).mean()

    return var, K
def logsumexp_norm_loss_with_grad(K):
    """
    输入:
        K: Tensor, shape (B, 1, N, D), requires_grad=True
    输出:
        E: scalar, 损失值
        grad: Tensor, shape (B, N, D), 对 K 的梯度 (squeeze 掉第2维)
    """
    K.requires_grad_(True)
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
    return loss, Q, K
Q_input = torch.randn(8, 32, 1, 64)  # B=8, T=32, D=64
K_input = torch.randn(8, 1, 32, 64)
var, K = variance_norm_regularization(K_input)
var.backward()

# 得到梯度
print(K)  # (B, T, D)

