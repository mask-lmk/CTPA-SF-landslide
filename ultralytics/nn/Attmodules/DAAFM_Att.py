# DAAFM: 双聚合注意力融合模块 (Simplified single-input version)
# 论文: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10918659
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAAFM_Att(nn.Module):
    """双聚合注意力融合模块 - 单输入版本
    内部自注意力 + 通道交叉注意力融合
    YAML: [-1, 1, DAAFM_Att, []]
    parse_model 自动注入 channels 作为第一参数
    """

    def __init__(self, channels, num_heads=8, ffn_expansion=2.66):
        super().__init__()
        dim = channels
        # 确保可被num_heads整除
        while dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1))

        # QKV生成
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

        # FFN
        hidden_dim = int(dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False)
        )

        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x

        # LayerNorm等效
        x_norm = self.norm1(x)

        # QKV
        qkv = self.qkv_dwconv(self.qkv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)

        # 多头重组
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 自注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)

        # 残差 + FFN
        x = shortcut + self.project_out(out)
        x = x + self.ffn(self.norm2(x))
        return x
