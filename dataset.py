import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

class LeafDataset(Dataset):
    def __init__(self, csv_file, img_base_dir='', transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_base_dir = img_base_dir
        self.transform = transform

        # 创建标签映射
        self.unique_labels = sorted(self.data['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 添加编码后的标签列
        self.data['label_encoded'] = self.data['label'].map(self.label_to_idx)

        print(f"数据集大小: {len(self.data)}")
        print(f"类别数量: {len(self.unique_labels)}")
        print(f"前几个样本:\n{self.data[['image', 'label', 'label_encoded']].head()}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 直接从CSV获取图像路径和标签
        img_relative_path = self.data.iloc[idx]['image']  # 例如: "images/0.jpg"
        label = self.data.iloc[idx]['label_encoded']  # 使用编码后的数值标签
        
        # 构建完整的图像路径
        if self.img_base_dir:
            img_path = os.path.join(self.img_base_dir, img_relative_path)
        else:
            img_path = img_relative_path
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像: {img_path}, 错误: {e}")
            # 返回一个占位图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_num_classes(self):
        """获取类别数量"""
        return len(self.unique_labels)
    
    def get_label_mapping(self):
        """获取标签映射"""
        return self.label_to_idx, self.idx_to_label

# 定义数据变换
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 使用示例
if __name__ == "__main__":
    # 获取数据变换
    train_transform, val_transform = get_transforms()
    
    # 指定正确的路径
    data_dir = './data/leaves/classify-leaves'
    csv_file = os.path.join(data_dir, 'train.csv')
    
    # 创建数据集 - 指定图像基础目录
    train_dataset = LeafDataset(
        csv_file=csv_file,
        img_base_dir=data_dir,  # 添加这个参数
        transform=train_transform
    )
    
    # 获取类别数量
    num_classes = train_dataset.get_num_classes()
    print(f"类别数量: {num_classes}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=2
    )
    
    # 测试数据加载
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  图像维度: {images.shape}")
        print(f"  标签: {labels[:5]}")
        print(f"  标签类型: {type(labels[0])}")
        print(f"  标签数据类型: {labels.dtype}")
        
        if batch_idx == 2:
            break