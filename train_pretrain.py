import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from tqdm import tqdm
import torchvision.models as models

# 导入您之前定义的模块
from dataset import LeafDataset, get_transforms

def get_pretrained_resnet18(num_classes, in_channels=3):
    """
    获取预训练的ResNet18模型
    """
    # 加载预训练的ResNet18
    model = models.resnet18(pretrained=True)
    
    # 修改第一层卷积以适应不同的输入通道数（如果需要）
    if in_channels != 3:
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
    
    # 修改最后的全连接层以适应我们的类别数
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    print("使用预训练的ResNet18模型")
    print(f"修改全连接层: {num_features} -> {num_classes}")
    
    return model

def train_model(model: nn.Module, train_loader: DataLoader, criterion, optimizer, num_epochs=50, device='cuda'):
    """
    训练模型
    """
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 学习率调度器 - 对于预训练模型可以使用更小的学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
            _, preds = torch.max(outputs, 1)
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
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'当前学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if epoch_acc > best_train_acc:
            best_train_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': best_train_acc.item(),
                'history': history
            }, 'best_pretrained_resnet18_model.pth')
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
    
    # 获取数据变换 - 使用ImageNet的标准化参数
    train_transform, _ = get_transforms()
    
    # 创建训练数据集
    train_dataset = LeafDataset(
        csv_file=train_csv_file,
        img_base_dir=data_dir,
        transform=train_transform
    )
    
    num_classes = train_dataset.get_num_classes()
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
    
    # 创建预训练模型
    model = get_pretrained_resnet18(num_classes=num_classes, in_channels=3)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义损失函数和优化器 - 使用更小的学习率
    criterion = nn.CrossEntropyLoss()
    
    # 对于预训练模型，可以使用不同的学习率策略
    # 选项1: 统一学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 选项2: 不同层使用不同学习率（推荐）
    # optimizer = optim.Adam([
    #     {'params': model.conv1.parameters(), 'lr': 0.0001},
    #     {'params': model.bn1.parameters(), 'lr': 0.0001},
    #     {'params': model.layer1.parameters(), 'lr': 0.0001},
    #     {'params': model.layer2.parameters(), 'lr': 0.0001},
    #     {'params': model.layer3.parameters(), 'lr': 0.0001},
    #     {'params': model.layer4.parameters(), 'lr': 0.0001},
    #     {'params': model.fc.parameters(), 'lr': 0.001}  # 新加的全连接层用更大的学习率
    # ], weight_decay=1e-4)
    
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
    torch.save(trained_model.state_dict(), 'final_pretrained_resnet18_model.pth')
    print("训练完成，模型已保存!")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()