import os
import numpy as np
import matplotlib.pyplot as plt

# 设置输出文件夹路径
output_dir = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/newmatrix'

# 存储所有上三角部分的数值
upper_triangle_values = []

# 获取文件名，并确保它们以 'frame_' 开头并以 '.txt' 结尾
file_names = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.txt')]

# 按文件名中的数字部分排序，提取数字 'i' 并按其大小排序
file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))

# 遍历每个文件，读取距离矩阵并提取上三角元素
for file_name in file_names:
    # 构造文件路径
    distance_file_path = os.path.join(output_dir, file_name)

    # 读取距离矩阵
    try:
        distance_matrix = np.loadtxt(distance_file_path)  # 从文本文件中加载矩阵
    except Exception as e:
        print(f"Error reading {distance_file_path}: {e}")
        continue

    # 提取上三角部分的索引（不包括对角线）
    upper_triangular_indices = np.triu_indices(distance_matrix.shape[0], k=1)

    # 提取所有上三角元素的值
    upper_triangular = distance_matrix[upper_triangular_indices]

    # 将提取的上三角元素添加到列表中
    upper_triangle_values.append(upper_triangular)

# 将所有上三角部分的值转换为 NumPy 数组
upper_triangle_values = np.array(upper_triangle_values)

# 计算每个上三角元素的变化
changes = np.abs(np.diff(upper_triangle_values, axis=0))  # 计算相邻帧之间的变化

# 将所有变化数据展平
all_changes = changes.flatten()

# 设置区间范围（bins）为 [0, 2]，分为40个等间隔
bin_edges = np.linspace(0, 2, 41)

# 计算每个区间的频数
counts, _ = np.histogram(all_changes, bins=bin_edges)

# 总数据量
total_count = all_changes.size

# 打印每个区间的统计结果和所占百分比
print("区间统计结果：")
for i in range(len(bin_edges) - 1):
    percentage = (counts[i] / total_count) * 100  # 计算百分比
    print(f"区间 [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {counts[i]} 条数据，占比 {percentage:.2f}%")

# 绘制数据分布图（直方图）
plt.figure(figsize=(10, 6))
plt.hist(all_changes, bins=bin_edges, color='blue', edgecolor='black', alpha=0.7)

# 设置标题和标签
plt.title("Distribution of Distance Changes Between Frames")
plt.xlabel("Distance Change")
plt.ylabel("Frequency")
plt.grid(True)

# 保存并关闭图表
plt.savefig(os.path.join(output_dir, 'distance_change_distribution.png'))
plt.close()

print("已绘制并保存距离变化分布图。")

