import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def calculate_occupancy(trajectory, n_clusters=10):
    """
    计算轨迹的状态占有率

    Parameters:
    -----------
    trajectory : numpy.ndarray
        形状为(125, 21, 3)的轨迹数据
    n_clusters : int
        聚类的状态数量

    Returns:
    --------
    occupancy : numpy.ndarray
        各状态的占有率
    """
    # 将轨迹重塑为2D数组，每一行代表一帧的所有原子坐标
    n_frames, n_atoms, n_dims = trajectory.shape
    reshaped_traj = trajectory.reshape(n_frames, -1)

    # 使用K-means进行状态聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    states = kmeans.fit_predict(reshaped_traj)

    # 计算各状态的占有率
    occupancy = np.zeros(n_clusters)
    for i in range(n_clusters):
        occupancy[i] = np.sum(states == i) / len(states)

    return occupancy


def plot_occupancy_comparison(md_traj, our_traj, n_clusters=10, save_path=None):
    """
    比较两个轨迹的状态占有率并绘制散点图

    Parameters:
    -----------
    md_traj : numpy.ndarray
        MD模拟轨迹，形状为(125, 21, 3)
    our_traj : numpy.ndarray
        Our模型轨迹，形状为(125, 21, 3)
    n_clusters : int
        聚类的状态数量
    save_path : str, optional
        图片保存路径
    """
    # 计算两个轨迹的占有率
    md_occupancy = calculate_occupancy(md_traj, n_clusters)
    our_occupancy = calculate_occupancy(our_traj, n_clusters)

    # 设置图形样式
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制散点图
    ax.scatter(md_occupancy, our_occupancy,
               alpha=0.6, c='blue', label='States')

    # 添加对角线
    min_val = min(md_occupancy.min(), our_occupancy.min())
    max_val = max(md_occupancy.max(), our_occupancy.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.5, label='Perfect match')

    # 设置轴标签和标题
    ax.set_xlabel('MD Simulation Occupancy', fontsize=12)
    ax.set_ylabel('Our Model Occupancy', fontsize=12)
    ax.set_title('State Occupancy Comparison: MD vs Our Model', fontsize=14)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置坐标轴范围
    padding = 0.05
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    # 计算并显示相关系数
    correlation = np.corrcoef(md_occupancy, our_occupancy)[0, 1]
    ax.text(0.05, 0.95, f'R = {correlation:.3f}',
            transform=ax.transAxes, fontsize=10)

    # 保持横纵比相等
    ax.set_aspect('equal')

    # 添加图例
    ax.legend()

    # 调整布局
    plt.tight_layout()

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax, md_occupancy, our_occupancy


def main():
    # 加载数据示例（实际使用时替换为您的数据）
    md_traj = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/uni_traj.npy')
    our_traj = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy')
    our_traj = our_traj[0]
    # 生成示例数据

    # 绘制占有率对比图
    fig, ax, md_occupancy, our_occupancy = plot_occupancy_comparison(
        md_traj, our_traj, n_clusters=20, save_path='/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/occu.png'
    )

    # 显示图片
    plt.show()

    # 打印占有率值
    print("\nMD Occupancy:", md_occupancy)
    print("Our Occupancy:", our_occupancy)


if __name__ == "__main__":
    main()