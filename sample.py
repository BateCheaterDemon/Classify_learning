import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import List, Optional

# 预训练权重URL
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
}

class Residual(nn.Module):
    """残差块"""
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, 
                              padding=1, stride=strides, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1,
                                  stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.conv3 = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return self.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """创建残差块序列"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet18(nn.Module):
    """基于Sequential块的ResNet18实现"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, 
                 pretrained: bool = False, pretrained_num_classes: int = 1000):
        super(ResNet18, self).__init__()
        
        # 构建网络模块
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
        # 预训练权重处理
        self.pretrained = pretrained
        self.pretrained_num_classes = pretrained_num_classes
        if pretrained:
            self.load_pretrained_weights(num_classes)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained_weights(self, num_classes: int):
        """加载预训练权重并处理分类头不匹配问题"""
        try:
            # 下载预训练权重
            state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
            
            # 处理输入通道不匹配（灰度图 vs RGB）
            if self.b1[0].in_channels == 1 and state_dict['conv1.weight'].shape[1] == 3:
                # 将RGB预训练权重转换为灰度图
                rgb_weights = state_dict['conv1.weight']
                gray_weights = rgb_weights.mean(dim=1, keepdim=True)
                state_dict['conv1.weight'] = gray_weights
            
            # 处理分类头不匹配
            if num_classes != self.pretrained_num_classes:
                # 移除最后的全连接层权重，让模型重新初始化
                if 'fc.weight' in state_dict:
                    del state_dict['fc.weight']
                if 'fc.bias' in state_dict:
                    del state_dict['fc.bias']
                print(f"分类头不匹配: 预训练{self.pretrained_num_classes}类 vs 当前{num_classes}类，已重新初始化分类头")
            
            # 加载权重（严格模式，忽略不匹配的键）
            load_result = self.load_state_dict(state_dict, strict=False)
            
            if load_result.missing_keys:
                print(f"缺失的键: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"意外的键: {load_result.unexpected_keys}")
                
            print("成功加载预训练权重！")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_feature_maps(self, x):
        """获取中间特征图（用于特征可视化）"""
        features = {}
        x = self.b1(x)
        features['b1'] = x
        x = self.b2(x)
        features['b2'] = x
        x = self.b3(x)
        features['b3'] = x
        x = self.b4(x)
        features['b4'] = x
        x = self.b5(x)
        features['b5'] = x
        return features

def create_resnet18(pretrained=False, num_classes=10, in_channels=1, **kwargs):
    """创建ResNet18模型的便捷函数"""
    return ResNet18(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        **kwargs
    )