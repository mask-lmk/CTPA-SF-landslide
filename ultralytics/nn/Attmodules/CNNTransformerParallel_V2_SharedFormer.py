# CTPA-SF: reduced-dimensional Q/K/V attention, 1.25x FFN, three ECA gates.
# At hidden width 128: 128->120 projection, distinct Q/K/V width 40 each,
# four heads of width 10, and a 40->128 output projection.
# Attention: 20,480 weights vs 65,536 full-width weights (biases excluded).
# FFN: 40,960 vs 49,152 weights for 1.25x vs the 1.50x ablation.
# Base CTPA uses 2.0x FFN. Three width-3 ECA gates add nine weights.
# This is dimensionality reduction, not Q/K/V weight tying; no accuracy guarantee.
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SharedQKVAttention(nn.Module):
    """Reduced-dimensional attention with distinct Q, K and V outputs."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        # inner_dim: 能被3*num_heads整除的内部维度
        per_head = dim // (3 * num_heads)
        self.inner_dim = per_head * 3 * num_heads
        self.inner_per_head = self.inner_dim // (3 * num_heads)  # 每头每Q/K/V的通道
        self.scale = self.inner_per_head ** -0.5

        self.qkv_shared = nn.Linear(dim, self.inner_dim, bias=False)
        # 输出从inner_dim/3映射回dim
        self.out_proj = nn.Linear(self.inner_dim // 3, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv_shared(x)  # (B, N, inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, N, inner_dim/3)

        # 多头: (B, N, heads, per_head_dim)
        q = q.view(b, n, self.num_heads, self.inner_per_head).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.inner_per_head).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.inner_per_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, N, per_head_dim)
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim // 3)
        return self.out_proj(out)


class CNNTransformerParallel_V2_SharedFormer(nn.Module):
    """Parallel CNN and reduced-dimensional attention with independent ECA gates."""

    def __init__(self, c1, c2=None, num_heads=4, expansion=0.5):
        super().__init__()
        c2 = c2 or c1
        hidden_dim = int(c2 * expansion)
        while hidden_dim % num_heads != 0:
            hidden_dim += 1

        # 1) 输入投影
        self.cv1 = nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU(inplace=True)

        # 2) CNN分支 + ECA增强 (用省下的参数增强)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.cnn_eca = ECAAttention(hidden_dim)  # 新增: ECA增强CNN输出

        # 3) Reduced-dimensional Q/K/V attention and 1.25x FFN
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = SharedQKVAttention(hidden_dim, num_heads)  # reduced Q/K/V dimensions
        ffn_hidden = int(hidden_dim * 1.25)  # 1.5x→1.25x
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim)
        )
        self.trans_eca = ECAAttention(hidden_dim)  # ECA替代SE (更轻)

        # 4) 可学习分支权重
        self.cnn_weight = nn.Parameter(torch.ones(1))
        self.trans_weight = nn.Parameter(torch.ones(1))

        # 5) ECA融合 (轻量)
        self.fuse_reduce = nn.Conv2d(hidden_dim * 2, c2, 1, 1, 0, bias=False)
        self.fuse_bn = nn.BatchNorm2d(c2)
        self.fuse_act = nn.SiLU(inplace=True)
        self.fuse_eca = ECAAttention(c2)

        # Shortcut
        self.shortcut = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x0 = self.act1(self.bn1(self.cv1(x)))

        # CNN分支 + ECA
        x_cnn = self.cnn_eca(self.cnn_branch(x0))

        # SharedFormer分支
        bsz, c, h, w = x0.shape
        x_seq = x0.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_seq)
        x_attn = self.attn(x_norm)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.ffn(self.norm(x_seq))
        x_trans = x_seq.transpose(1, 2).reshape(bsz, c, h, w)
        x_trans = self.trans_eca(x_trans)

        # 加权融合 + ECA
        out = torch.cat([x_cnn * self.cnn_weight, x_trans * self.trans_weight], dim=1)
        out = self.fuse_act(self.fuse_bn(self.fuse_reduce(out)))
        out = self.fuse_eca(out)

        return out + identity
