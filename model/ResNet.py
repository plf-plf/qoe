# ResNet的基本结构由多个残差块组成。
# 每个残差块包含两个卷积层，第一个卷积层用于提取特征，第二个卷积层用于生成残差。
# 残差块的输出是第二个卷积层的输出与第一个卷积层的输出的和。
# 残差块由两个3x3卷积层和一个1x1卷积层组成跳跃连接，其中1x1卷积层用于调整通道数，
# 使得输入和输出的通道数保持一致。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 定义了一个残差块，它包含两个卷积层和一个跳跃连接 如果输入和输出通道数不同，跳跃连接会包含一个额外的1x1卷积层来匹配维度
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, channel_size, num_classes):
        super(Model, self).__init__()
        self.block1 = ResidualBlock(in_channels=channel_size, out_channels=128, kernel_size=8, padding='same')
        self.block2 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.block3 = ResidualBlock(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) #在用激活函数之前,这些输出值通常被称为“logits”，它们可以是任何实数值，包括负数；使用BCEWithLogitsLoss
        # (batch_size, num_classes)
        return x