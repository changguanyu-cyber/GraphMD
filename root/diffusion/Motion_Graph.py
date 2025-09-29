import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np


# Graph Neural Network部分
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        print(f"x shape: {x.shape}, edge_index max: {edge_index.max().item()}, edge_index shape: {edge_index.shape}")
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        print(tmp.shape)
        return self.mlp(tmp)


# Transformer Encoder层
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


# Motion Graph模型
class MotionGraph(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_frames=125):
        super(MotionGraph, self).__init__()

        # GNN部分
        self.edge_conv1 = EdgeConv(input_dim, hidden_dim)
        self.edge_conv2 = EdgeConv(hidden_dim, hidden_dim)

        # Transformer部分
        self.transformer = TransformerEncoder(hidden_dim, nhead=8)

        # 输出层
        self.output_frames = output_frames
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, edge_index):
        batch_size = x.size(0)

        # GNN处理空间信息
        h = self.edge_conv1(x, edge_index)
        h = self.edge_conv2(h, edge_index)

        # Transformer处理时序信息
        h = h.view(batch_size, -1, h.size(-1))  # [batch_size, num_atoms, hidden_dim]
        h = self.transformer(h)

        # 解码生成轨迹
        outputs = []
        current = h[:, -1:, :]  # 使用最后一帧作为初始状态

        for _ in range(self.output_frames):
            current = self.decoder(current)
            outputs.append(current)
            current = self.edge_conv1(current.view(-1, 3), edge_index).view(batch_size, 1, -1)

        return torch.cat(outputs, dim=1)


# 数据预处理
def prepare_data(trajectory_data, window_size=10):
    """
    处理输入数据，创建滑动窗口序列
    trajectory_data: shape (100000, 21, 3)
    """
    windows = []
    targets = []

    for i in range(len(trajectory_data) - window_size):
        windows.append(trajectory_data[i:i + window_size])
        targets.append(trajectory_data[i + window_size])

    return np.array(windows), np.array(targets)


# 构建边索引（完全图）
def build_edge_index(num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t()


# 训练函数
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

        output = model(x, edge_index)
        loss = F.mse_loss(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 验证函数
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            x, edge_index, y = batch.x, batch.edge_index, batch.y
            x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

            output = model(x, edge_index)
            loss = F.mse_loss(output, y)

            total_loss += loss.item()

    return total_loss / len(val_loader)


# 主函数
def main():
    # 假设我们有以下数据
    # trajectory_data shape: (100000, 21, 3)
    trajectory_data = np.load('/root/rmd17/npz_data/S1_coords_selected_frames.npy')  # 这里用随机数据代替

    # 数据预处理
    window_size = 10
    X, y = prepare_data(trajectory_data, window_size)

    # 划分训练集和验证集
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # 构建边索引
    num_nodes = 21
    edge_index = build_edge_index(21)
    assert edge_index.max().item() < num_nodes, f"edge_index 超界，最大索引: {edge_index.max().item()}, num_nodes: {num_nodes}"

    # 创建数据加载器
    train_dataset = [Data(x=torch.FloatTensor(x),
                          edge_index=edge_index,
                          y=torch.FloatTensor(y)) for x, y in zip(X_train, y_train)]
    val_dataset = [Data(x=torch.FloatTensor(x),
                        edge_index=edge_index,
                        y=torch.FloatTensor(y)) for x, y in zip(X_val, y_val)]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotionGraph().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 1
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/Motion_Graph_best_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

    # 推理
    model.load_state_dict(torch.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/Motion_Graph_best_model.pth'))
    model.eval()

    # 生成新的轨迹
    with torch.no_grad():
        # 使用最后一个窗口作为初始输入
        initial_window = torch.FloatTensor(X_val[-1]).unsqueeze(0).to(device)
        edge_index = edge_index.to(device)

        # 生成125帧的新轨迹
        generated_trajectory = model(initial_window, edge_index)
        generated_trajectory = generated_trajectory.cpu().numpy()

        print("Generated trajectory shape:", generated_trajectory.shape)
        # 生成的轨迹形状应该是 (1, 125, 21, 3)
    np.save("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/Motion_Graph_infer.npy", generated_trajectory)


if __name__ == "__main__":
    main()