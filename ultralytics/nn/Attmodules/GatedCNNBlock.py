# GatedCNNBlock: 门控CNN块 (Simplified - no timm dep)
# 论文: https://arxiv.org/pdf/1612.08083
import torch
import torch.nn as nn


class GatedCNNBlock(nn.Module):
    """门控CNN块 - 抑制背景噪声、强化边缘特征
    YAML: [-1, 1, GatedCNNBlock, []]
    """

    def __init__(self, channels, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0):
        super().__init__()
        dim = channels
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # (B, C, H, W)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x + shortcut
