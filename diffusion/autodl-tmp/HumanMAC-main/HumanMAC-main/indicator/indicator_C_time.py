import numpy as np


def compute_ctime(trajectory):
    diff = trajectory[1:] - trajectory[:-1]
    dist_per_frame = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    scale = np.linalg.norm(trajectory.reshape(-1, trajectory.shape[-1]), axis=1).mean()
    return np.mean(dist_per_frame) / (scale + 1e-8)




# 示例用法
# trajectory = np.random.randn(1000, 21, 3)
# dist_ctime = euclidean_temporal_consistency(trajectory)
# print("基于欧式距离的时间一致性指标（均值）:", dist_ctime)


# 使用方式：
trajectory = np.load('/root/rmd17/extracted_data/aspirin_coord.npy')
trajectory = trajectory[:1000]
print(trajectory.shape)
ctime_score = compute_ctime(trajectory)
print("C_time (temporal consistency score):", ctime_score)
