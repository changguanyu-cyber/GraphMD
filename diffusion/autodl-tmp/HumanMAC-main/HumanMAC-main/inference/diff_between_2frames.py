import os
import numpy as np
import matplotlib.pyplot as plt

# 设置输出文件夹路径
output_dir = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/bene_matrix'

# 存储每个矩阵第一行第二列的元素
first_row_second_col_values = []

# 获取文件名，并确保它们以 'frame_' 开头并以 '.txt' 结尾
file_names = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.txt')]

# 按文件名中的数字部分排序，提取数字 'i' 并按其大小排序
file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))

# 遍历每个文件，读取距离矩阵并提取第一行第二列的元素
for file_name in file_names:
    # 构造文件路径
    distance_file_path = os.path.join(output_dir, file_name)

    # 读取距离矩阵
    try:
        distance_matrix = np.loadtxt(distance_file_path)  # 从文本文件中加载矩阵
    except Exception as e:
        print(f"Error reading {distance_file_path}: {e}")
        continue

    # 提取矩阵的第一行第二列的元素（即 [0, 1]）
    first_row_second_col_value = distance_matrix[0, 1]

    # 将该元素添加到列表中
    first_row_second_col_values.append(first_row_second_col_value)

# 将所有提取的值转换为 numpy 数组
first_row_second_col_values = np.array(first_row_second_col_values)

# 计算每两帧之间的差异
frame_differences = np.abs(np.diff(first_row_second_col_values))
mean_diff = np.mean(frame_differences)
variance_diff = np.var(frame_differences)

# 打印结果
print(f"每两帧之间差异的绝对值的平均值: {mean_diff}")
print(f"每两帧之间差异的绝对值的方差: {variance_diff}")
# 绘制每两帧之间差异的变化趋势图
plt.plot(frame_differences, marker='o', linestyle='-', color='b')
plt.title('Difference between Consecutive Frames')
plt.xlabel('Frame Index (1-based)')
plt.ylabel('Difference in First Row Second Column')
plt.grid(True)

# 保存图像到文件
#output_image_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/diff2frames.png'  # 指定保存路径
#plt.savefig(output_image_path)

# 显示图像（可选）
plt.show()

#print(f"图像已保存到: {output_image_path}")
