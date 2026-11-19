# DPA: 双池化注意力 (Dual Pooling Attention)
# 通过平均池化与最大池化的互补协作，动态调整通道权重
import torch
import torch.nn as nn


class DPA(nn.Module):
    """双池化注意力机制
    YAML: [-1, 1, DPA, []]
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        in_planes = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)
        z = self.max_pool(x).view(b, c)
        z = self.fc2(z).view(b, c, 1, 1)
        return x * y.expand_as(x) + x * z.expand_as(x) + x
