# PFAE: 金字塔频率注意力提取 (Simplified - no einops dep)
# 论文: https://arxiv.org/pdf/2503.11030
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFAE(nn.Module):
    """金字塔频率注意力提取模块
    YAML: [-1, 1, PFAE, []]
    parse_model 自动注入 channels 作为第一参数
    """

    def __init__(self, channels):
        super().__init__()
        dim = channels
        in_dim = max(dim // 2, 16)
        down_dim = max(in_dim // 2, 8)
        self.down_dim = down_dim
        self.in_dim = in_dim
        self.num_heads = min(8, down_dim)

        # 确保down_dim能被num_heads整除
        while down_dim % self.num_heads != 0:
            self.num_heads -= 1

        self.head_dim = down_dim // self.num_heads

        # 下采样
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.SiLU(inplace=True)
        )

        # 并行卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 1),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 3, padding=3, dilation=3),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, 3, padding=5, dilation=5),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, 3, padding=7, dilation=7),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, 3, padding=9, dilation=9),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )

        # 全局池化分支
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 1),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(6 * down_dim, down_dim, 1),
            nn.BatchNorm2d(down_dim),
            nn.SiLU(inplace=True)
        )

        # 输出
        self.out = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 2, 3, padding=1),
            nn.BatchNorm2d(down_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(down_dim // 2, dim, 1)
        )

        # 注意力参数
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, 1, bias=False)
        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 8, 1),
            nn.BatchNorm2d(down_dim // 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(down_dim // 8, down_dim, 1),
            nn.Sigmoid()
        )

    def frequency_attention(self, x):
        """频率域注意力"""
        b, c, h, w = x.shape
        # FFT
        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())
        tepqkv = torch.fft.fft2(x.float())

        # 多头分解 (替代 einops.rearrange)
        q_f = q_f.reshape(b, self.num_heads, self.head_dim, h * w)
        k_f = k_f.reshape(b, self.num_heads, self.head_dim, h * w)
        v_f = v_f.reshape(b, self.num_heads, self.head_dim, h * w)

        q_f = F.normalize(q_f.real, dim=-1) + 1j * F.normalize(q_f.imag, dim=-1)
        k_f = F.normalize(k_f.real, dim=-1) + 1j * F.normalize(k_f.imag, dim=-1)

        attn_f = (q_f @ k_f.conj().transpose(-2, -1)) * self.temperature
        attn_f_real = F.softmax(attn_f.real, dim=-1)
        attn_f_imag = F.softmax(attn_f.imag, dim=-1)
        attn_f = attn_f_real + 1j * attn_f_imag

        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = out_f.reshape(b, c, h, w)

        out_f_l = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out + x

    def forward(self, x):
        x_down = self.down_conv(x)
        conv1 = self.conv1(x_down)

        F_2 = self.frequency_attention(self.conv2(x_down))
        F_3 = self.frequency_attention(self.conv3(F_2))
        F_4 = self.frequency_attention(self.conv4(F_3))
        F_5 = self.frequency_attention(self.conv5(F_4))

        conv5 = F.interpolate(
            self.conv6(F.adaptive_avg_pool2d(x_down, 1)),
            size=x_down.size()[2:], mode='bilinear', align_corners=False
        )

        return self.out(self.fuse(torch.cat([conv1, F_2, F_3, F_4, F_5, conv5], dim=1)))
