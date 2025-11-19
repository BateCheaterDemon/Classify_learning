import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import argparse

# 导入您之前定义的模块
from dataset import LeafDataset, get_transforms
from resnet18 import ResNet18

class TestDataset:
    """测试数据集类，用于处理没有标签的测试数据"""
    def __init__(self, csv_file, img_base_dir='', transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_base_dir = img_base_dir
        self.transform = transform
        
        print(f"测试集大小: {len(self.data)}")
        print(f"前几个测试样本:\n{self.data.head()}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_relative_path = self.data.iloc[idx]['image']
        
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
            
        return image, img_relative_path

def load_model(model_path, num_classes, device='cuda'):
    """加载训练好的模型"""
    model = ResNet18(num_classes=num_classes, in_channels=3)
    
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("加载最佳模型检查点")
        else:
            model.load_state_dict(checkpoint)
            print("加载最终模型")
    else:
        raise ValueError(f"不支持的模型文件格式: {model_path}")
    
    model = model.to(device)
    model.eval()
    return model

def predict_test_set(model, test_loader, label_mapping, device='cuda'):
    """在测试集上进行预测"""
    predictions = []
    image_paths = []
    
    with torch.no_grad():
        for inputs, paths in tqdm(test_loader, desc='预测中'):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # 将预测结果转换为标签名称
            for i in range(len(predicted)):
                pred_idx = predicted[i].item()
                pred_label = label_mapping[pred_idx]
                predictions.append(pred_label)
                image_paths.append(paths[i])
    
    return image_paths, predictions

def main():
    parser = argparse.ArgumentParser(description='测试脚本')
    parser.add_argument('--model_path', type=str, default='best_resnet18_model.pth', help='模型路径')
    parser.add_argument('--data_dir', type=str, default='./data/leaves/classify-leaves', help='数据目录')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='测试集CSV文件')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='训练集CSV文件（用于获取标签映射）')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='输出结果CSV文件')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 完整的数据路径
    test_csv_path = os.path.join(args.data_dir, args.test_csv)
    train_csv_path = os.path.join(args.data_dir, args.train_csv)
    
    # 检查文件是否存在
    if not os.path.exists(test_csv_path):
        print(f"错误: 测试集文件不存在: {test_csv_path}")
        return
    
    if not os.path.exists(train_csv_path):
        print(f"错误: 训练集文件不存在: {train_csv_path}")
        return
    
    # 从训练集获取标签映射
    train_data = pd.read_csv(train_csv_path)
    unique_labels = sorted(train_data['label'].unique())
    label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
    
    print(f"类别数量: {len(unique_labels)}")
    print(f"前几个标签: {list(unique_labels)[:5]}")
    
    # 获取数据变换
    _, test_transform = get_transforms()
    
    # 创建测试数据集
    test_dataset = TestDataset(
        csv_file=test_csv_path,
        img_base_dir=args.data_dir,
        transform=test_transform
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    model = load_model(
        model_path=args.model_path,
        num_classes=len(unique_labels),
        device=device
    )
    
    # 进行预测
    print("开始预测...")
    image_paths, predictions = predict_test_set(
        model=model,
        test_loader=test_loader,
        label_mapping=label_mapping,
        device=device
    )
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'image': image_paths,
        'label': predictions
    })
    
    # 保存结果
    results_df.to_csv(args.output_csv, index=False)
    print(f"预测完成！结果已保存到: {args.output_csv}")
    print(f"结果文件前几行:")
    print(results_df.head())
    
    # 统计预测结果
    print(f"\n预测结果统计:")
    print(f"总预测样本数: {len(results_df)}")
    print(f"预测的类别分布:")
    label_counts = results_df['label'].value_counts()
    print(label_counts.head(10))  # 显示前10个最常见的预测类别

if __name__ == "__main__":
    # 需要导入PIL
    from PIL import Image
    main()