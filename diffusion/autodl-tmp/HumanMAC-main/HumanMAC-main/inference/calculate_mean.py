import csv

# 读取现有的 folder_statistics.csv 文件
csv_file_path = '/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/folder_statistics.csv'

# 存储所有的 Mean Difference 和 Variance Difference 值
mean_differences = []
variance_differences = []

# 打开 CSV 文件并读取数据
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)

    # 读取表头
    header = next(reader, None)  # 处理空文件的情况

    # 提取 Mean Difference 和 Variance Difference 列
    for row in reader:
        if len(row) < 3:  # 确保行至少有 3 列
            print(f"Skipping invalid row (not enough columns): {row}")
            continue
        try:
            mean_diff = float(row[1].strip())  # 去除可能的空格
            variance_diff = float(row[2].strip())
            mean_differences.append(mean_diff)
            variance_differences.append(variance_diff)
        except ValueError as e:
            print(f"Skipping row due to conversion error: {e} (row: {row})")

# 计算 Mean Difference 和 Variance Difference 的平均值
average_mean_diff = sum(mean_differences) / len(mean_differences) if mean_differences else 0
average_variance_diff = sum(variance_differences) / len(variance_differences) if variance_differences else 0

# 打印计算的平均值
print(f"Mean Difference 的平均值: {average_mean_diff}")
print(f"Variance Difference 的平均值: {average_variance_diff}")

# 将平均值追加到 CSV 文件中
with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # 将平均值写入文件
    writer.writerow(['Average', average_mean_diff, average_variance_diff])

print(f"Mean Difference 和 Variance Difference 的平均值已追加到 {csv_file_path}")
