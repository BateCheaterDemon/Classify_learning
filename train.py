import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from tqdm import tqdm

# 导入您之前定义的模块
from dataset import LeafDataset, get_transforms
from resnet18 import ResNet18

def train_model(model: nn.Module, train_loader: DataLoader, criterion, optimizer, num_epochs=50, device='cuda'):
    """
    训练模型
    """
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_train_acc = 0.0
    model = model.to(device)
    
    print("开始训练...")
    print(f"使用设备: {device}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
        
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            _, preds = torch.max(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # 更新进度条
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # 更新学习率
        scheduler.step()
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 保存最佳模型
        if epoch_acc > best_train_acc:
            best_train_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': best_train_acc,
                'history': history
            }, 'best_resnet18_model.pth')
            print(f'保存最佳模型，训练准确率: {best_train_acc:.4f}')
    
    print(f'训练完成，最佳训练准确率: {best_train_acc:.4f}')
    return history, model

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = './data/leaves/classify-leaves'
    train_csv_file = os.path.join(data_dir, 'train.csv')
    
    # 检查文件是否存在
    if not os.path.exists(train_csv_file):
        print(f"错误: 训练集文件不存在: {train_csv_file}")
        return
    
    # 获取数据变换
    train_transform, _ = get_transforms()
    
    # 创建训练数据集
    train_dataset = LeafDataset(
        csv_file=train_csv_file,
        img_base_dir=data_dir,
        transform=train_transform
    )
    
    num_classes = train_dataset.data['label'].nunique()
    print(f"数据集类别数量: {num_classes}")
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建数据加载器
    batch_size = 128
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    
    # 创建模型
    model = ResNet18(num_classes=num_classes, in_channels=3)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练模型
    history, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
        device=device
    )
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_resnet18_model.pth')
    print("训练完成，模型已保存!")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()