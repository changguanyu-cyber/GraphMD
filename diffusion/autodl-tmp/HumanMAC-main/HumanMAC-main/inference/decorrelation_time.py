import numpy as np
import pyemma
import os
from pyemma.util.contexts import settings
from sklearn.preprocessing import StandardScaler

class MSMAnalyzer:
    def __init__(self, trajectory_data, n_clusters=100, lag_time=10, save_dir="/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference"):
        """
        MSM 分析器初始化

        Args:
            trajectory_data: shape (n_frames, n_atoms, 3) 的轨迹数据
            n_clusters: 聚类中心数量
            lag_time: MSM 的滞后时间
            save_dir: 结果保存目录
        """
        self.data = trajectory_data
        self.n_clusters = n_clusters
        self.lag_time = lag_time
        self.save_dir = save_dir
        self.msm = None
        self.cluster_centers = None
        self.discrete_trajectories = None

        # 创建结果保存目录
        os.makedirs(self.save_dir, exist_ok=True)

    def preprocess_data(self):
        """数据预处理：展平 + 标准化"""
        n_frames = self.data.shape[0]
        self.features = self.data.reshape(n_frames, -1)

        # 标准化特征
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def cluster_data(self):
        """使用 k-means 进行聚类"""
        self.preprocess_data()

        # 使用 k-means 进行聚类
        self.kmeans = pyemma.coordinates.cluster_kmeans(
            self.features, k=self.n_clusters, max_iter=50, stride=1
        )

        self.cluster_centers = self.kmeans.clustercenters
        self.discrete_trajectories = self.kmeans.dtrajs

    def build_msm(self):
        """构建 MSM"""
        if self.discrete_trajectories is None:
            self.cluster_data()

        self.msm = pyemma.msm.estimate_markov_model(self.discrete_trajectories, lag=self.lag_time)

        print(f"MSM 构建完成，状态数: {self.msm.nstates}")

    def get_transition_matrix(self):
        """获取转移矩阵"""
        if self.msm is None:
            self.build_msm()
        return self.msm.transition_matrix

    def get_stationary_distribution(self):
        """计算平衡分布"""
        if self.msm is None:
            self.build_msm()
        return self.msm.stationary_distribution

    def get_mfpt_matrix(self):
        """计算均值首次通过时间 (MFPT) 矩阵"""
        if self.msm is None:
            self.build_msm()
        n_states = self.msm.nstates
        mfpt_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    mfpt_matrix[i, j] = self.msm.mfpt(i, j)
        return mfpt_matrix

    def save_results(self, filename="mar_result.txt"):
        """保存平衡分布、转移矩阵和 MFPT 到一个 npz 文件"""
        if self.msm is None:
            self.build_msm()

        results = {
            "transition_matrix": self.get_transition_matrix(),
            "stationary_distribution": self.get_stationary_distribution(),
            "mfpt_matrix": self.get_mfpt_matrix(),
        }

        np.savez(os.path.join(self.save_dir, filename), **results)
        print(f"结果已保存至 {os.path.join(self.save_dir, filename)}")

    def analyze(self):
        """运行 MSM 并保存结果"""
        self.build_msm()
        self.save_results()


def main():
    # 载入轨迹数据
    trajectory_data = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy')
    trajectory_data = trajectory_data[0]  # 选择第一个轨迹数据

    # 创建 MSM 分析器
    msm_analyzer = MSMAnalyzer(trajectory_data=trajectory_data, n_clusters=50, lag_time=10)

    # 运行分析并保存结果
    msm_analyzer.analyze()


if __name__ == "__main__":
    with settings(show_progress_bars=True):
        main()
