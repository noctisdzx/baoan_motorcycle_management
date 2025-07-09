import os
import shutil
from pathlib import Path

# 定义输入和输出路径
input_base = Path('datasets/zh')  # 替换为你的数据根文件夹路径（如 datasets）
output_base = Path('datasets/bn_zh')  # 替换为你希望保存整合后的数据的路径

# 确保输出文件夹存在
output_base.mkdir(parents=True, exist_ok=True)
(output_base / 'images').mkdir(parents=True, exist_ok=True)
(output_base / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(output_base / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(output_base / 'labels').mkdir(parents=True, exist_ok=True)
(output_base / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(output_base / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# 处理 classes.txt 文件
found_classes = False
for dataset_folder in input_base.glob('*'):
    if not dataset_folder.is_dir():
        continue
    current_labels_train = dataset_folder / 'labels'  / 'classes.txt'
    if current_labels_train.exists():
        # 复制到输出文件夹的 labels/train 文件夹中
        output_classes_path = output_base / 'labels'  / 'classes.txt'
        if not output_classes_path.exists():
            shutil.copy(current_labels_train, output_classes_path)
            found_classes = True
        break

if not found_classes:
    print("警告：未找到 classes.txt 文件")

# 遍历每个子数据文件夹（如 a、b、c）
for dataset_folder in input_base.glob('*'):
    if not dataset_folder.is_dir():
        continue
    dataset_name = dataset_folder.name  # 获取子文件夹名称，如 a、b、c
    
    # 遍历该子文件夹中的所有图像文件
    images_folder = dataset_folder / 'images'
    # 检查是否有 train 和 val 文件夹
    have_train = (images_folder / 'train').exists()
    have_val = (images_folder / 'val').exists()
    
    # 遍历需要处理的数据集类型（train 和 val）
    data_types = []
    if have_train:
        data_types.append('train')
    if have_val:
        data_types.append('val')
    
    for data_type in data_types:
        current_images = dataset_folder / 'images' / data_type
        current_labels = dataset_folder / 'labels' / data_type
        
        # 遍历当前路径下的所有图像文件
        for image_path in current_images.glob('*.*'):
            # 排除 .cache 文件
            if image_path.suffix == '.cache':
                continue
            
            # 检查是否是图像文件（假设支持jpg、png等格式）
            if image_path.suffix.lower() not in {'.jpg', '.png', '.bmp', '.jpeg'}:
                continue  # 跳过非图像文件
            
            # 获取文件名（不包含扩展名）
            image_name = image_path.name
            image_ext = image_path.suffix
            # 生成对应的标签文件名
            label_name = image_name.replace(image_ext, '.txt')
            label_path = current_labels / label_name
            
            if label_path.exists() and not label_path.name.startswith('.'):
                # 构造新的文件名和路径
                # 格式：子文件夹名 + 原文件名
                new_image_name = f"{dataset_name}_{image_name}"
                new_image_path = output_base / 'images' / data_type / new_image_name
                
                # 检查文件是否已存在，避免重复复制
                if not new_image_path.exists():
                    shutil.copy(image_path, new_image_path)
                
                # 处理标签文件
                new_label_name = f"{dataset_name}_{label_name}"
                new_label_path = output_base / 'labels' / data_type / new_label_name
                if not new_label_path.exists():
                    shutil.copy(label_path, new_label_path)
            else:
                print(f"警告：图像文件 {image_name} 在子文件夹 {dataset_name} 的 {data_type} 数据集中，没有对应的标签文件或标签文件不存在")

# 输出完成信息
print("数据整合完成！所有图像和对应的标签文件已整理到新的训练集中。")