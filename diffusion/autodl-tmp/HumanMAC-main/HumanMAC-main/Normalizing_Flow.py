import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义耦合层
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.mask = mask

        # 缩放和平移网络
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2+1),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Linear(input_dim // 2+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2+1)
        )

    def forward(self, x, reverse=False):
        x_masked = x * self.mask
        x_pass = x * (1 - self.mask)
        x_masked_half = x_masked.reshape(x.shape[0], -1)[:, :x_masked.reshape(x.shape[0], -1).shape[1] // 2+1]


        if not reverse:
            x_transform = x_pass.reshape(x.shape[0], -1)
            split_idx = x_transform.shape[1] // 2  # 获取中间索引
            x_left, x_right = x_transform[:, :split_idx], x_transform[:, split_idx:]
            s = self.scale_net(x_masked_half)
            t = self.translation_net(x_masked_half)
            # 切分


            # 确保 s 的形状匹配 x_right
            s = s[:, :x_right.shape[1]]  # 截断 s 以匹配 x_right
            t = t[:, :x_right.shape[1]]  # 截断 t 以匹配 x_right

            x_right_transformed = x_right * torch.exp(s) + t  # 计算新值
            x_transform = torch.cat((x_left, x_right_transformed), dim=1)
            x_transform = x_transform.reshape(x.shape)

            log_det = torch.sum(s, dim=1)
        else:
            s = self.scale_net(x_masked_half)
            t = self.translation_net(x_masked_half)

            x_transform = x_pass.reshape(x.shape[0], -1)  # 将数据展平为 2D

            split_idx = x_transform.shape[1] // 2  # 计算分界点
            x_right = x_transform[:, split_idx:]  # 取后半部分

            # **确保 s 和 t 的形状与 x_right 匹配**
            s = s[:, :x_right.shape[1]]
            t = t[:, :x_right.shape[1]]

            x_transform[:, split_idx:] = (x_right - t) * torch.exp(-s)  # 计算逆变换
            x_transform = x_transform.reshape(x.shape)  # 重新 reshape 回原来的维度

            log_det = -torch.sum(s, dim=1)  # 计算对数行列式


        return x_transform + x_masked, log_det


# 定义RealNVP模型
class RealNVP(nn.Module):
    def __init__(self, input_shape, hidden_dim, num_layers):
        super().__init__()
        self.input_shape = input_shape
        self.flatten_dim = np.prod(input_shape)

        # 创建交替的掩码
        masks = []
        for i in range(num_layers):
            if self.flatten_dim % 2 == 0:
                if i % 2 == 0:
                    masks.append(torch.cat([torch.ones(self.flatten_dim // 2),
                                            torch.zeros(self.flatten_dim // 2)]))
                else:
                    masks.append(torch.cat([torch.zeros(self.flatten_dim // 2),
                                            torch.ones(self.flatten_dim // 2)]))
            else:
                # 如果 flatten_dim 是奇数, 需要保证掩码的长度是 flatten_dim
                if i % 2 == 0:
                    masks.append(torch.cat([torch.ones(self.flatten_dim // 2 + 1),
                                            torch.zeros(self.flatten_dim // 2)]))
                else:
                    masks.append(torch.cat([torch.zeros(self.flatten_dim // 2 + 1),
                                            torch.ones(self.flatten_dim // 2)]))

        # 创建耦合层
        #masks.append(0)
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(self.flatten_dim, hidden_dim, mask)
            for mask in masks
        ])

    def forward(self, x, reverse=False):
        log_det_sum = 0
        if not reverse:
            x = x.reshape(x.shape[0], -1)
            for layer in self.coupling_layers:
                x, log_det = layer(x, reverse=False)
                log_det_sum += log_det
        else:
            x = x.reshape(x.shape[0], -1)
            for layer in reversed(self.coupling_layers):
                x, log_det = layer(x, reverse=True)
                log_det_sum += log_det

        return x.reshape(x.shape[0], *self.input_shape), log_det_sum


# 数据预处理函数
def prepare_data(data, train_ratio=0.8, batch_size=32):
    # 划分数据集
    train_val_data, test_data = train_test_split(data, test_size=10000, random_state=42)
    train_data, val_data = train_test_split(train_val_data,
                                            test_size=1 - train_ratio,
                                            random_state=42)

    # 标准化
    mean = train_data.mean()
    std = train_data.std()

    train_normalized = (train_data - mean) / std
    val_normalized = (val_data - mean) / std
    test_normalized = (test_data - mean) / std
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

    return (train_loader, val_loader, test_loader), (mean, std)


# 训练函数
torch.autograd.set_detect_anomaly(True)
def train_flow(model, train_loader, val_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            # 前向传播
            z, log_det = model(x)

            # 计算损失（负对数似然）
            prior_ll = -0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - 0.5 * np.log(2 * np.pi) * np.prod(x.shape[1:])
            loss = -(prior_ll + log_det).mean()

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                z, log_det = model(x)
                prior_ll = -0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - 0.5 * np.log(2 * np.pi) * np.prod(x.shape[1:])
                val_loss -= (prior_ll + log_det).mean()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/best_flow_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


# 生成新轨迹
def generate_trajectories(model, num_samples, device, stats):
    model.eval()
    mean, std = stats

    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_samples, 125, 21, 3).to(device)

        # 通过模型生成轨迹
        x, _ = model(z, reverse=True)

        # 反标准化
        x = x.cpu().numpy() * std + mean

    return x


def main():
    # 超参数
    batch_size = 64
    num_epochs = 300
    hidden_dim = 256
    num_coupling_layers = 8

    # 加载数据（替换为实际数据）
    data = np.load('/root/rmd17/npz_data/S1_coords_selected_frames.npy')  # 示例数据

    # 准备数据
    (train_loader, val_loader, test_loader), stats = prepare_data(
        data, train_ratio=0.8, batch_size=batch_size
    )

    # 初始化模型
    model = RealNVP(
        input_shape=(125, 21, 3),
        hidden_dim=hidden_dim,
        num_layers=num_coupling_layers
    ).to(device)

    # 训练模型
    train_losses, val_losses = train_flow(
        model, train_loader, val_loader, num_epochs, device
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # 生成新轨迹
    generated_trajectories = generate_trajectories(
        model, num_samples=1, device=device, stats=stats
    )

    # 评估生成的轨迹
    test_trajectories = next(iter(test_loader))[0].numpy()

    print("\n统计特征比较：")
    print("测试集形状:", test_trajectories.shape)
    print("生成轨迹形状:", generated_trajectories.shape)

    # 计算统计特征
    test_mean = np.mean(test_trajectories, axis=0)
    test_std = np.std(test_trajectories, axis=0)
    gen_mean = np.mean(generated_trajectories, axis=0)
    gen_std = np.std(generated_trajectories, axis=0)

    mean_mse = np.mean((test_mean - gen_mean) ** 2)
    std_mse = np.mean((test_std - gen_std) ** 2)

    print(f"平均值MSE: {mean_mse:.6f}")
    print(f"标准差MSE: {std_mse:.6f}")

    return model, generated_trajectories, test_trajectories


if __name__ == "__main__":
    model, generated_trajectories, test_trajectories = main()
    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/Normalizing_Flow_result.npy', generated_trajectories)