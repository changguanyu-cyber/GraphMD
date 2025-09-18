import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mu = nn.Linear(256, latent_dim)  # 均值
        self.fc3_logvar = nn.Linear(256, latent_dim)  # 对数方差

    def forward(self, x):
        #print(x.shape)
        x = x.view(x.shape[0], -1)  # 让 PyTorch 自动计算合适的第二维
  # 扁平化输入数据
       # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar


# 定义解码器网络
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        x_reconstructed = torch.sigmoid(self.fc3(z))  # 使用sigmoid确保输出在[0, 1]范围内
        return x_reconstructed


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


# 计算VAE的损失函数
def vae_loss(reconstructed_x, x, mu, logvar):
    # 重构误差
    BCE = nn.MSELoss(reduction='sum')(reconstructed_x, x.view(x.shape[0], -1)  # 让 PyTorch 自动计算合适的第二维
)

    # KL散度
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KL


# 数据集加载
# 假设分子轨迹数据集保存在一个变量 `data`，形状为 (100000, 21, 3)
# 需要对数据进行预处理，例如标准化

data = np.load('/root/rmd17/npz_data/S1_coords_selected_frames.npy')
data = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data)

# 数据集拆分
train_val_size = 90000  # 用于训练和验证的数据集大小
test_size = 10000  # 用于推理（测试）集的大小

train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

# 再次拆分训练集和验证集
train_size = 80000  # 用于训练的数据集大小
val_size = 10000  # 用于验证的数据集大小

train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置VAE模型
latent_dim = 64  # 潜在空间维度
input_dim = 21 * 3  # 每个轨迹的特征维度
vae = VAE(input_dim, latent_dim)

# 优化器
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# 训练VAE模型
epochs = 300
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch in train_loader:
        data_batch = batch[0]
        data_batch = data_batch.float()

        optimizer.zero_grad()

        # 前向传播
        reconstructed_batch, mu, logvar = vae(data_batch)

        # 计算损失
        loss = vae_loss(reconstructed_batch, data_batch, mu, logvar)

        # 反向传播
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 在验证集上评估
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data_batch = batch[0]
            data_batch = data_batch.float()

            reconstructed_batch, mu, logvar = vae(data_batch)
            loss = vae_loss(reconstructed_batch, data_batch, mu, logvar)
            val_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

# 推理阶段（生成新轨迹）
vae.eval()
with torch.no_grad():
    # 从标准正态分布中采样潜在变量
    z = torch.randn(125, latent_dim)  # 生成10个新轨迹
    generated_trajectories = vae.decoder(z)

    # 将生成的轨迹重新形状为 (21, 3) 以表示21个原子的3D坐标
    generated_trajectories = generated_trajectories.view(125, 21, 3)
    print(generated_trajectories)
    generated_trajectories = generated_trajectories.numpy()

    # 保存为 .npy 文件
    np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/VAE_infer.npy", generated_trajectories)

# 测试集推理
vae.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        data_batch = batch[0]
        data_batch = data_batch.float()

        reconstructed_batch, mu, logvar = vae(data_batch)
        loss = vae_loss(reconstructed_batch, data_batch, mu, logvar)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader)}")
