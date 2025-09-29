import numpy as np

def linear_interpolate_frames(trajectory, num_insert):
    """
    对轨迹进行线性插帧。

    参数：
    - trajectory: numpy.ndarray，形状 (T, N, 3)，T 为帧数，N 为原子数
    - num_insert: int，每两帧之间插入的帧数

    返回：
    - 插帧后的轨迹，shape = (T + (T - 1) * num_insert, N, 3)
    """
    T, N, D = trajectory.shape
    new_T = T + (T - 1) * num_insert
    interpolated = []

    for i in range(T - 1):
        frame_start = trajectory[i]
        frame_end = trajectory[i + 1]
        # 添加原始帧
        interpolated.append(frame_start)

        # 插入 num_insert 帧
        for j in range(1, num_insert + 1):
            alpha = j / (num_insert + 1)  # 归一化插值系数
            interpolated_frame = (1 - alpha) * frame_start + alpha * frame_end
            interpolated.append(interpolated_frame)

    # 添加最后一帧
    interpolated.append(trajectory[-1])

    return np.stack(interpolated, axis=0)
# 假设你有一个轨迹文件 trajectory.npy, shape = (1000, 9, 3)
trajectory = np.load("/root/autodl-tmp/traj.npy")

# 每两帧之间插入 2 帧
new_trajectory1 = linear_interpolate_frames(trajectory, num_insert=1)
new_trajectory2 = linear_interpolate_frames(trajectory, num_insert=2)
new_trajectory3 = linear_interpolate_frames(trajectory, num_insert=3)
new_trajectory4 = linear_interpolate_frames(trajectory, num_insert=4)

# 输出形状


# 保存结果
np.save("/root/autodl-tmp/interpolated_trajectory1.npy", new_trajectory1)
np.save("/root/autodl-tmp/interpolated_trajectory2.npy", new_trajectory2)
np.save("/root/autodl-tmp/interpolated_trajectory3.npy", new_trajectory3)
np.save("/root/autodl-tmp/interpolated_trajectory4.npy", new_trajectory4)
