# SCSA: 空间与通道协同注意力 (Simplified - no external deps)
# 论文: https://arxiv.org/pdf/2407.05128
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSA_Att(nn.Module):
    """SCSA 注意力模块 - 适配 YOLO parse_model 接口
    YAML: [-1, 1, SCSA_Att, []]
    parse_model 自动注入 channels 作为第一参数
    """

    def __init__(self, channels, head_num=None, window_size=8):
        super().__init__()
        dim = channels
        # 自动计算合理的 head_num
        if head_num is None:
            # 找到能整除 dim 的最大合理 head 数
            possible_heads = [h for h in [4, 8, 16] if dim % h == 0]
            head_num = possible_heads[-1] if possible_heads else 4

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size

        # 确保能被4整除 (SCSA 将通道分为4组)
        assert dim % 4 == 0, f'SCSA: dim ({dim}) must be divisible by 4'
        self.group_chans = dim // 4

        # 多尺度1D卷积 (H/W方向分别计算)
        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=3,
                                    padding=1, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=5,
                                       padding=2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=7,
                                       padding=3, groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=9,
                                       padding=4, groups=self.group_chans)

        self.sa_gate = nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        # 通道自注意力
        self.down_func = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.k = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.ca_gate = nn.Sigmoid()

    def forward(self, x):
        b, c, h_, w_ = x.shape

        # ---- 空间注意力 (H/W方向) ----
        x_h = x.mean(dim=3)  # (B, C, H)
        l_h, g_h_s, g_h_m, g_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)  # (B, C, W)
        l_w, g_w_s, g_w_m, g_w_l = torch.split(x_w, self.group_chans, dim=1)

        h_attn = self.sa_gate(self.norm_h(torch.cat([
            self.local_dwc(l_h), self.global_dwc_s(g_h_s),
            self.global_dwc_m(g_h_m), self.global_dwc_l(g_h_l)
        ], dim=1)))
        h_attn = h_attn.view(b, c, h_, 1)

        w_attn = self.sa_gate(self.norm_w(torch.cat([
            self.local_dwc(l_w), self.global_dwc_s(g_w_s),
            self.global_dwc_m(g_w_m), self.global_dwc_l(g_w_l)
        ], dim=1)))
        w_attn = w_attn.view(b, c, 1, w_)

        x = x * h_attn * w_attn

        # ---- 通道注意力 ----
        y = self.down_func(x)
        _, _, hd, wd = y.shape
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # 用原生reshape替代einops.rearrange
        q = q.reshape(b, self.head_num, self.head_dim, hd * wd)
        k = k.reshape(b, self.head_num, self.head_dim, hd * wd)
        v = v.reshape(b, self.head_num, self.head_dim, hd * wd)

        attn = (q.transpose(-2, -1) @ k) * self.scaler
        attn = attn.softmax(dim=-1)
        out = v @ attn.transpose(-2, -1)

        out = out.reshape(b, self.dim, hd, wd)
        ca_weight = self.ca_gate(out.mean((2, 3), keepdim=True))

        return ca_weight * x
