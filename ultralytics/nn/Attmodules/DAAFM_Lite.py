# DAAFM_Lite: 轻量版双聚合注意力融合 (降低FFN扩展比和头数)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAAFM_Lite(nn.Module):
    """轻量版DAAFM - FFN扩展1.5x, 4注意力头
    YAML: [-1, 1, DAAFM_Lite, []]
    """

    def __init__(self, channels):
        super().__init__()
        dim = channels
        num_heads = 4
        while dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1))

        # QKV (深度可分离降低参数量)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

        # FFN (降低扩展比: 2.66→1.5)
        hidden_dim = int(dim * 1.5)
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
        x_norm = self.norm1(x)

        qkv = self.qkv_dwconv(self.qkv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)

        x = shortcut + self.project_out(out)
        x = x + self.ffn(self.norm2(x))
        return x
