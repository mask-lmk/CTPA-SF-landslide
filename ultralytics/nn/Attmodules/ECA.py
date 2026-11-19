import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, *args, k_size=3, **kwargs):
        super().__init__()
        self.k_size = k_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        y = self.gap(x)                         # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)     # B,1,C

        # 动态创建权重，保证输出长度一定等于 C
        weight = torch.ones(
            1, 1, self.k_size,
            device=x.device,
            dtype=x.dtype
        ) / self.k_size

        y = F.conv1d(
            y,
            weight,
            padding=(self.k_size - 1) // 2
        )                                      # B,1,C

        if y.shape[-1] != c:
            y = F.interpolate(y, size=c, mode="nearest")

        y = y.transpose(-1, -2).unsqueeze(-1)   # B,C,1,1
        y = self.sigmoid(y)

        return x * y
