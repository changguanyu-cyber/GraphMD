import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# 设置主目录
root_dir = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/nap_matrix'

# 存储所有文件夹的统计结果
folder_stats = []

# 获取所有 sequence_xxx 目录
sequence_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith('sequence_')])

# 遍历所有序列
for seq_dir in sequence_dirs:
    seq_path = os.path.join(root_dir, seq_dir)

    # 存储每个序列帧的第一行第二列的元素
    first_row_second_col_values = []

    # 获取当前序列的 frame_*.txt 文件
    file_names = sorted(
        [f for f in os.listdir(seq_path) if f.startswith('frame_') and f.endswith('.txt')],
        key=lambda x: int(x.split('_')[1].split('.')[0])  # 按帧编号排序
    )

    # 遍历每个帧文件
    for file_name in file_names:
        file_path = os.path.join(seq_path, file_name)

        try:
            # 读取距离矩阵
            distance_matrix = np.loadtxt(file_path)
            first_row_second_col_value = distance_matrix[0, 1]
            first_row_second_col_values.append(first_row_second_col_value)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # 转换为 NumPy 数组
    first_row_second_col_values = np.array(first_row_second_col_values)

    # 计算每两帧之间的差异
    frame_differences = np.abs(np.diff(first_row_second_col_values))
    mean_diff = np.mean(frame_differences)
    variance_diff = np.var(frame_differences)

    # 打印结果
    print(f"Sequence {seq_dir} - 每两帧之间差异的绝对值的平均值: {mean_diff}")
    print(f"Sequence {seq_dir} - 每两帧之间差异的绝对值的方差: {variance_diff}")

    # 将结果保存到统计列表中
    folder_stats.append([seq_dir, mean_diff, variance_diff])

    # 绘制变化趋势图
    plt.plot(frame_differences, marker='o', linestyle='-', color='b')
    plt.title(f'Difference between Consecutive Frames ({seq_dir})')
    plt.xlabel('Frame Index')
    plt.ylabel('Difference in First Row Second Column')
    plt.grid(True)

    # 保存图像到对应文件夹下
    output_image_path = os.path.join(seq_path, 'diff2frames.png')
    plt.savefig(output_image_path)
    plt.close()  # 关闭当前图像，防止影响后续图像

    print(f"图像已保存到: {output_image_path}")

# 将统计结果保存到 CSV 文件
csv_file_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/folder_statistics.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sequence Folder', 'Mean Difference', 'Variance Difference'])  # 表头
    writer.writerows(folder_stats)  # 写入统计数据

print(f"所有文件夹的统计结果已保存到: {csv_file_path}")
