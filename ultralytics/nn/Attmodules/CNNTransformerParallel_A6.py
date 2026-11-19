# CNNTransformerParallel_A6: 升级版 CNN+Transformer 并行模块
# CNN分支: GatedCNNBlock (抑制背景噪声 + 强化边缘)
# Transformer分支: 标准MultiheadAttention (全局上下文)
# Fusion: SCSA (空间+通道协同注意力融合)
import torch
import torch.nn as nn
from ultralytics.nn.Attmodules.GatedCNNBlock import GatedCNNBlock
from ultralytics.nn.Attmodules.SCSA_Att import SCSA_Att


class CNNTransformerParallel_A6(nn.Module):
    """升级版CNN-Transformer并行特征增强模块
    A6方案: GatedCNN + Transformer + SCSA融合
    用于YOLO P5层高层语义特征增强
    """

    def __init__(self, c1, c2=None, num_heads=4, expansion=0.5):
        super().__init__()
        c2 = c2 or c1
        hidden_dim = int(c2 * expansion)

        # 确保hidden_dim满足SCSA要求(被4整除)和Transformer要求(被num_heads整除)
        hidden_dim = max(hidden_dim, 64)
        while hidden_dim % 4 != 0 or hidden_dim % num_heads != 0:
            hidden_dim += 4

        self.hidden_dim = hidden_dim

        self.cv1 = nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU(inplace=True)

        # CNN分支: GatedCNNBlock (替代原来的简单depthwise conv)
        self.cnn_branch = GatedCNNBlock(hidden_dim)

        # Transformer分支
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Fusion: 使用SCSA替代简单的concat+1x1conv
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.scsa = SCSA_Att(c2)

        self.shortcut = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x0 = self.act1(self.bn1(self.cv1(x)))

        # CNN分支 (GatedCNN)
        x_cnn = self.cnn_branch(x0)

        # Transformer分支
        b, c, h, w = x0.shape
        x_seq = x0.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_seq)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.ffn(self.norm(x_seq))
        x_trans = x_seq.transpose(1, 2).reshape(b, c, h, w)

        # Fusion (concat + SCSA精炼)
        out = torch.cat([x_cnn, x_trans], dim=1)
        out = self.fuse(out)
        out = self.scsa(out)

        return out + identity
