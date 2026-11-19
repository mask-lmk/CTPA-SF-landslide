# 方案1: GhostFusion — 幻影融合降参 (~35K省, -15% vs UltraLight)
# 核心改进:
#   1. Ghost风格融合: 一半通道由常规conv生成, 一半由廉价depthwise生成
#      fuse_reduce: 256*128常规 + 128*3*3廉价 = 32K (vs 65K) → 省33K
#   2. 共享SE: CNN和Transformer分支各自输出后先融合再共享SE → 省2K
#   3. GhostNet论文证明: 廉价操作生成的"幻影"特征与原始特征高度相关
# 精度保证: SE依然存在 + ECA精炼 + Ghost廉价特征补充细节
import torch
import torch.nn as nn


class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class GhostFusion(nn.Module):
    """Ghost风格融合: 一半常规conv + 一半廉价depthwise"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        half_out = out_channels // 2
        self.conv_primary = nn.Conv2d(in_channels, half_out, 1, bias=False)
        self.conv_cheap = nn.Conv2d(half_out, out_channels - half_out, 3,
                                     padding=1, groups=half_out, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        primary = self.conv_primary(x)
        cheap = self.conv_cheap(primary)
        return self.act(self.bn(torch.cat([primary, cheap], dim=1)))


class CNNTransformerParallel_V1_GhostFusion(nn.Module):
    """方案1: GhostFusion — 幻影融合版 (~198K params vs UltraLight ~233K)"""

    def __init__(self, c1, c2=None, num_heads=4, expansion=0.5):
        super().__init__()
        c2 = c2 or c1
        hidden_dim = int(c2 * expansion)
        while hidden_dim % num_heads != 0:
            hidden_dim += 1

        # 1) 输入投影 (保持)
        self.cv1 = nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU(inplace=True)

        # 2) CNN分支 (保持深度可分离)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

        # 3) Transformer分支 (保持1.5x FFN)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        ffn_hidden = int(hidden_dim * 1.5)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim)
        )

        # 4) GhostFusion: 替代65K的1x1 Conv (省33K)
        self.fuse = GhostFusion(hidden_dim * 2, c2)

        # 5) 共享SE精炼融合结果 (替代两个分支各自的SE, 省2K)
        self.shared_se = SEAttention(c2)

        # 6) ECA最终精炼
        self.fuse_eca = ECAAttention(c2)

        # Shortcut
        self.shortcut = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x0 = self.act1(self.bn1(self.cv1(x)))

        # CNN分支
        x_cnn = self.cnn_branch(x0)

        # Transformer分支
        bsz, c, h, w = x0.shape
        x_seq = x0.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_seq)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.ffn(self.norm(x_seq))
        x_trans = x_seq.transpose(1, 2).reshape(bsz, c, h, w)

        # Ghost融合 + 共享SE + ECA
        out = self.fuse(torch.cat([x_cnn, x_trans], dim=1))
        out = self.shared_se(out)
        out = self.fuse_eca(out)

        return out + identity
