import os
folder="/data/liufengyi/data_gs/dtu/scan63/mask"

# 遍历文件夹中的所有 PNG 文件
for filename in os.listdir(folder):
    if filename.endswith('.png'):
        # 提取文件名（不包括扩展名）
        name, ext = os.path.splitext(filename)
        
        # 使用 zfill 添加前导零到四位数
        new_name = name.zfill(4) + ext
        
        # 构建原文件名和新文件名的完整路径
        old_file = os.path.join(folder, filename)
        new_file = os.path.join(folder, new_name)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')
