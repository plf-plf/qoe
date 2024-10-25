import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# InceptionModule模块通过多个不同大小的卷积核和池化操作来提取特征，并通过瓶颈层减少计算量
class InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters, bottleneck_size, kernel_size, use_bottleneck):
        super(InceptionModule, self).__init__()
        # nb_filters=32 bottleneck_size=32 kernel_size=41
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, padding='same', bias=False)
        else:
            self.bottleneck = None
        # ks=[41, 21, 11]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_size if self.bottleneck else in_channels, nb_filters, ks, padding='same', bias=False)
            for ks in kernel_size_s
            ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # bottleneck_size
        self.conv_1x1 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=1, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x):
        if self.bottleneck:
            x = self.bottleneck(x)
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.conv_1x1(self.maxpool(x))
        outputs = torch.cat(conv_outputs + [maxpool_output], dim=1)
        outputs = self.bn(outputs)
        return F.relu(outputs)

# ShortcutLayer主要用于实现残差连接
class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutLayer, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, residual):
        shortcut = self.conv_1x1(residual)
        shortcut = self.bn(shortcut)
        return F.relu(x + shortcut)

class Model(nn.Module):
    def __init__(self, dim_num, num_classes, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        super(Model, self).__init__()
        # dim_num, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        self.use_residual = use_residual
        self.depth = depth
        self.inception_modules = nn.ModuleList([
            InceptionModule(dim_num if i == 0 else nb_filters * 4, nb_filters, 32, kernel_size, use_bottleneck)
            for i in range(depth)
            ])
        self.shortcut_layers = nn.ModuleList([
            ShortcutLayer(dim_num if i == 0 else nb_filters * 4, nb_filters * 4)
            for i in range(depth // 3)
            ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, num_classes)

    def forward(self, x):
        residual = x
        for i, module in enumerate(self.inception_modules):
            x = module(x)
            if self.use_residual and i % 3 == 2:
                x = self.shortcut_layers[i // 3](x, residual)
                residual = x
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x