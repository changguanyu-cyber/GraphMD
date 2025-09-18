import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator和Discriminator类的定义保持不变...
#[前面的Generator和Discriminator类代码保持不变]
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # 输入: (batch_size, latent_dim)
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 125 * 21 * 3),
            nn.Tanh()
        )

    def forward(self, z):
        # 生成分子轨迹
        trajectories = self.main(z)
        # 重塑为所需形状
        return trajectories.view(-1, 125, 21, 3)


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 输入: (batch_size, 125, 21, 3) -> (batch_size, 125*21*3)
            nn.Linear(125*21 * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.main(x_flat)

# 数据预处理和划分函数
def prepare_data(data, train_ratio=0.8, batch_size=64):
    """
    预处理数据并划分为训练集、验证集和测试集
    train_ratio: 训练集在训练+验证数据中的比例
    """
    # 获取90000条用于训练和验证
    train_val_data, test_data = train_test_split(data, test_size=10000, random_state=42)

    # 将90000条数据划分为训练集和验证集
    train_data, val_data = train_test_split(train_val_data,
                                            test_size=1 - train_ratio,
                                            random_state=42)

    # 数据归一化
    # 使用训练集的统计数据进行归一化
    train_min = train_data.min()
    train_max = train_data.max()

    train_normalized = (train_data - train_min) / (train_max - train_min) * 2 - 1
    val_normalized = (val_data - train_min) / (train_max - train_min) * 2 - 1
    test_normalized = (test_data - train_min) / (train_max - train_min) * 2 - 1
    train_normalized = train_normalized.reshape(-1, 125, 21, 3)
    val_normalized = val_normalized.reshape(-1, 125, 21, 3)
    test_normalized = test_normalized.reshape(-1, 125, 21, 3)

    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(train_normalized))
    val_dataset = TensorDataset(torch.FloatTensor(val_normalized))
    test_dataset = TensorDataset(torch.FloatTensor(test_normalized))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), (train_min, train_max)


# 验证函数
def validate(generator, discriminator, dataloader, criterion, latent_dim, device):
    generator.eval()
    discriminator.eval()
    total_g_loss = 0
    total_d_loss = 0
    num_batches = 0

    with torch.no_grad():
        for real_trajectories in dataloader:
            batch_size = real_trajectories[0].size(0)

            # 真实数据的损失
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(real_trajectories[0].to(device))
            d_loss_real = criterion(real_output, real_labels)

            # 生成数据的损失
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_trajectories = generator(z)
            fake_output = discriminator(fake_trajectories)

            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # 生成器损失
            g_loss = criterion(fake_output, real_labels)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1

    return total_g_loss / num_batches, total_d_loss / num_batches


# 修改后的训练函数
def train_gan(generator, discriminator, train_loader, val_loader, num_epochs, latent_dim, device):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 训练记录
    train_g_losses = []
    train_d_losses = []
    val_g_losses = []
    val_d_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        # 训练阶段
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0

        for i, real_trajectories in enumerate(train_loader):
            batch_size = real_trajectories[0].size(0)



            # 训练判别器
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(real_trajectories[0].to(device))
            d_loss_real = criterion(real_output, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_trajectories = generator(z)
            fake_output = discriminator(fake_trajectories.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            output = discriminator(fake_trajectories)
            g_loss = criterion(output, real_labels)

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')

        # 计算训练集平均损失
        train_g_loss = epoch_g_loss / num_batches
        train_d_loss = epoch_d_loss / num_batches
        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)

        # 验证阶段
        val_g_loss, val_d_loss = validate(generator, discriminator, val_loader,
                                          criterion, latent_dim, device)
        val_g_losses.append(val_g_loss)
        val_d_losses.append(val_d_loss)

        # 保存最佳模型
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
            }, '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/best_model_GAN.pth')

        print(f'Epoch [{epoch}/{num_epochs}] '
              f'Train - G_loss: {train_g_loss:.4f} D_loss: {train_d_loss:.4f} '
              f'Val - G_loss: {val_g_loss:.4f} D_loss: {val_d_loss:.4f}')

    return train_g_losses, train_d_losses, val_g_losses, val_d_losses


# 主训练流程
def main():
    # 超参数
    latent_dim = 100
    batch_size = 64
    num_epochs = 1

    # 加载数据（替换为实际数据）
    data = np.load('/root/rmd17/npz_data/S1_coords_selected_frames.npy')

    # 准备数据
    (train_loader, val_loader, test_loader), data_stats = prepare_data(data,
                                                                       train_ratio=0.8,
                                                                       batch_size=batch_size)

    # 初始化模型
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 训练模型
    train_g_losses, train_d_losses, val_g_losses, val_d_losses = train_gan(
        generator, discriminator, train_loader, val_loader, num_epochs, latent_dim, device
    )

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_g_losses, label='Train Generator Loss')
    plt.plot(train_d_losses, label='Train Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_g_losses, label='Val Generator Loss')
    plt.plot(val_d_losses, label='Val Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss')
    plt.show()

    return generator, discriminator, data_stats, test_loader


# 推理函数
def inference(generator, test_loader, latent_dim, data_stats, num_samples=10):
    generator.eval()
    train_min, train_max = data_stats
    generator.load_state_dict(torch.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/best_model_GAN.pth'))
    #discriminator.load_state_dict(torch.load('discriminator.pth'))

    # 生成新轨迹
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_trajectories = generator(z)

        # 反归一化
        generated_trajectories = (generated_trajectories + 1) / 2 * (train_max - train_min) + train_min
        generated_trajectories = generated_trajectories.cpu().numpy()

    # 计算测试集的统计特征
    test_trajectories = []
    for batch in test_loader:
        test_trajectories.append(batch[0].numpy())
    test_trajectories = np.concatenate(test_trajectories, axis=0)

    # 反归一化测试集
    test_trajectories = (test_trajectories + 1) / 2 * (train_max - train_min) + train_min

    # 计算和比较统计特征
    print("\n统计特征比较：")
    print("测试集形状:", test_trajectories.shape)
    print("生成轨迹形状:", generated_trajectories.shape)

    test_mean = np.mean(test_trajectories, axis=0)
    test_std = np.std(test_trajectories, axis=0)
    gen_mean = np.mean(generated_trajectories, axis=0)
    gen_std = np.std(generated_trajectories, axis=0)

    mean_mse = np.mean((test_mean - gen_mean) ** 2)
    std_mse = np.mean((test_std - gen_std) ** 2)

    print(f"平均值MSE: {mean_mse:.6f}")
    print(f"标准差MSE: {std_mse:.6f}")

    return generated_trajectories, test_trajectories


if __name__ == "__main__":
    # 训练模型
    generator, discriminator, data_stats, test_loader = main()

    # 进行推理
    generated_trajectories, test_trajectories = inference(
        generator, test_loader, latent_dim=100, data_stats=data_stats
    )
    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/GAN_infer.npy', generated_trajectories)