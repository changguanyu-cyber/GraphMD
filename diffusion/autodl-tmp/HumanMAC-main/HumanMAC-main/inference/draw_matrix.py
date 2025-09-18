import os
import numpy as np
import matplotlib.pyplot as plt

# 设置两个距离矩阵文件夹路径
matrix_dir1 = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/newmatrix'
matrix_dir2 = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/uni_matrix'


def extract_first_row_second_column(directory):
    """提取文件夹中所有矩阵第一行第二列元素的变化"""
    element_changes = []

    # 获取文件名，并确保它们以 'frame_' 开头并以 '.txt' 结尾
    file_names = [f for f in os.listdir(directory) if f.startswith('frame_') and f.endswith('.txt')]

    # 按文件名中的数字部分排序，提取数字 'i' 并按其大小排序
    file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 遍历每个文件，提取第一行第二列的元素
    for file_name in file_names:
        # 构造文件路径
        matrix_file_path = os.path.join(directory, file_name)

        # 读取距离矩阵
        try:
            distance_matrix = np.loadtxt(matrix_file_path)  # 从文本文件中加载矩阵
        except Exception as e:
            print(f"Error reading {matrix_file_path}: {e}")
            continue

        # 检查矩阵维度，确保有足够的行和列
        if distance_matrix.shape[0] >= 1 and distance_matrix.shape[1] >= 2:
            # 提取第一行第二列的元素
            element = distance_matrix[0, 1]
            element_changes.append(element)
        else:
            print(f"Matrix in {matrix_file_path} is too small, skipping.")

    return element_changes


# 提取两个文件夹中元素的变化
changes_dir1 = extract_first_row_second_column(matrix_dir1)
changes_dir2 = extract_first_row_second_column(matrix_dir2)

# 绘制变化曲线
plt.figure(figsize=(12, 6))

# 绘制文件夹1的变化
plt.plot(range(len(changes_dir1)), changes_dir1, marker='o', color='blue', label='Folder 1: First row, second column')

# 绘制文件夹2的变化
plt.plot(range(len(changes_dir2)), changes_dir2, marker='x', color='red', label='Folder 2: First row, second column')

# 设置标题和标签
plt.title("Change of First Row, Second Column Element Across Two Folders")
plt.xlabel("Frame Index")
plt.ylabel("Element Value")
plt.legend()
plt.grid(True)

# 保存图表
output_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/graph.png'
plt.savefig(output_path)
plt.close()

print(f"已绘制并保存第一行第二列元素变化曲线比较图：{output_path}")
