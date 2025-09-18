import os

def create_empty_xyz_files(n, output_dir="/root/autodl-tmp/output_xyz_files"):
    num_files = 2 ** n
    print(f"Creating {num_files} empty .xyz files in folder '{output_dir}'...")

    # 创建目标文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建空白文件
    for i in range(num_files):
        file_path = os.path.join(output_dir, f"frame_{i:04d}.xyz")
        with open(file_path, 'w') as f:
            pass  # 不写任何内容，保持空白

# 示例：创建 2^3 = 8 个空白文件
create_empty_xyz_files(3)
