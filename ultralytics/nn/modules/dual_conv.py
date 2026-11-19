import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class DualConv(nn.Module):
    """双分支卷积，保证通道均分，兼容任意输入通道"""
    def __init__(self, in_channels, out_channels, k=3, s=1, g=2, bias=False):
        super().__init__()
        # 强制偶数通道，避免//2后通道丢失
        self.out = out_channels
        half = self.out // 2
        # 分支1：标准卷积
        self.conv1 = nn.Conv2d(in_channels, half, k, s, padding=k//2, bias=bias)
        # 分支2：分组卷积
        self.conv2 = nn.Conv2d(in_channels, half, k, s, padding=k//2, groups=g, bias=bias)
        # BN输入通道严格等于out_channels
        self.bn = nn.BatchNorm2d(self.out)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_cat = torch.cat([x1, x2], dim=1)
        return self.act(self.bn(x_cat))

class Bottleneck_Dual(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 严格匹配DualConv参数顺序：in_c, out_c, k, s
        self.cv1 = DualConv(c1, c_, k=3)
        self.cv2 = DualConv(c_, c2, k=3, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k2_DualConv(nn.Module):
    """YOLO11原生C3k2替换版，内部卷积全部改为DualConv"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.25):
        super().__init__()
        self.c = int(c2 * e)
        # 1×1卷积 k=1, s=1
        self.cv1 = DualConv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = DualConv(2 * self.c, c2, k=1, s=1)
        self.m = nn.Sequential(*[Bottleneck_Dual(self.c, self.c, shortcut=True, e=1.0) for _ in range(n)])
        self.c3k = c3k

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, dim=1)
        x = torch.cat((x1, self.m(x2)), dim=1)
        return self.cv2(x)