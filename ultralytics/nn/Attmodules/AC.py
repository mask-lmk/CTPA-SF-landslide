import torch
import torch.nn as nn
import torch.nn.functional as F

class AC(nn.Module):
    def __init__(self, c_in, c_out=None, stride=1):
        super().__init__()
        # 参数容错处理，防止只传通道时报错
        if c_out is None:
            c_out = c_in
        self.c_out = c_out

        # 三支非对称卷积：3×3, 1×3, 3×1
        self.conv3x3 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1x3 = nn.Conv2d(c_in, c_out, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.conv3x1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)

        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x33 = self.conv3x3(x)
        x13 = self.conv1x3(x)
        x31 = self.conv3x1(x)
        out = x33 + x13 + x31
        out = self.bn(out)
        return out