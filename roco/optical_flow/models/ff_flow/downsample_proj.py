import torch
import torch.nn as nn

class DownSampleProj(nn.Module):
    def __init__(self,feature_channels:int):
        super().__init__()
        self.feature_channels=feature_channels
        self.conv1 = nn.Conv2d(feature_channels, feature_channels // 2 * 3, kernel_size=3, stride=2, padding=1) #128->192
        self.conv2 = nn.Conv2d(feature_channels // 2 * 3, feature_channels * 2, kernel_size=3, stride=2, padding=1)# 192->256  
        self.conv3 = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, stride=2, padding=1) #256->128

        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.relu(self.conv1(x))
        x2=self.conv2(x1)
        x3=self.conv3(self.relu(x2))

        return x2,x3    
