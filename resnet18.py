import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import List,Optional

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=num_channels,
                               kernel_size=3,
                               padding=1,
                               stride=strides,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,
                               padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,
                                   stride=strides,bias=False)
        
            self.bn3 = nn.BatchNorm2d(num_features=num_channels)

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
    
def resnet_block(input_channels, num_channels, num_residuals,first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))

        else:
            blk.append(Residual(num_channels,num_channels))
    
    return blk
    
class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
    
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64,128,2))
        self.b4 = nn.Sequential(*resnet_block(128,256,2))
        self.b5 = nn.Sequential(*resnet_block(256,512,2))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512,num_classes)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

if __name__ == "__main__":
    net = ResNet18(num_classes=10, in_channels=3)  # 176个类别对应您的数据集

    X = torch.rand(size=(1, 3, 224, 224))
    print("=== 网络前向传播过程 ===")
    output = net(X)
    print(output.shape)


