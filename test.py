import os
from PIL import Image
import numpy as np

# 指定文件夹路径
folder_path = '/home/qwx/learning/classify_leaves/data/leaves/classify-leaves/images'

# 获取所有.jpg文件
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

# 限制为前30个文件
jpg_files = jpg_files[:1000]

print(f"找到 {len(jpg_files)} 个jpg文件")

# 遍历并显示每个图片的shape
for i, filename in enumerate(jpg_files):
    try:
        # 构建完整文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 打开图片并转换为numpy数组
        img = Image.open(file_path)
        img_array = np.array(img)
        
        # 打印shape
        print(f"{i+1:2d}. {filename}: {img_array.shape}")
        
    except Exception as e:
        print(f"{i+1:2d}. {filename}: 读取失败 - {e}")

# 如果使用OpenCV的替代方案
print("\n" + "="*50)
print("使用OpenCV的替代方案：")