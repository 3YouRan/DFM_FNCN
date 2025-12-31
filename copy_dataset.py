import os
import shutil
import sys

# 源路径（从下载脚本中获取）
source_path = r"C:\Users\ROG\.cache\kagglehub\datasets\abdurrahmangulmez46\geometric-shapes-dataset\versions\1\geometric shapes"
target_path = r"./data/geometric_shapes"

# 形状映射（文件夹名称 -> 形状）
shape_mapping = {
    'circle': 'circle',
    'ellipse': 'ellipse',
    'octagonal': 'octagon',
    'parallelogram': 'parallelogram',
    'pentagon': 'pentagon',
    'rectangle': 'rectangle',
    'rhombus': 'rhombus',
    'square': 'square'
}

def copy_and_organize():
    if not os.path.exists(source_path):
        print(f"源路径不存在: {source_path}")
        return
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 为每个形状创建子目录
    for shape in shape_mapping.values():
        os.makedirs(os.path.join(target_path, shape), exist_ok=True)
    
    # 遍历源目录
    total_copied = 0
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 从父目录名称推断形状
                parent_dir = os.path.basename(root)
                # 格式如 "black circle"、"blue ellipse"
                parts = parent_dir.split()
                if len(parts) >= 2:
                    color = parts[0]
                    shape_key = parts[1]
                else:
                    continue
                # 映射形状
                if shape_key in shape_mapping:
                    shape_name = shape_mapping[shape_key]
                    dest_dir = os.path.join(target_path, shape_name)
                    # 复制文件
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    total_copied += 1
                    if total_copied % 100 == 0:
                        print(f"已复制 {total_copied} 个文件...")
    
    print(f"完成！共复制 {total_copied} 个图像到 {target_path}")
    
    # 统计每个类别的数量
    for shape in shape_mapping.values():
        shape_dir = os.path.join(target_path, shape)
        count = len([f for f in os.listdir(shape_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  类别 {shape}: {count} 张图像")

if __name__ == '__main__':
    copy_and_organize()