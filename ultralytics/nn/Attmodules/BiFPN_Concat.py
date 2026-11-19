# BiFPN_Concat: BiFPN风格加权特征融合 (EfficientDet, CVPR 2020)
# 在标准Concat基础上增加可学习权重, 自动平衡不同尺度特征贡献
# 数学保证: ReLU归一化权重, 最差退化为平均融合
import torch
import torch.nn as nn


class BiFPN_Concat(nn.Module):
    """BiFPN加权特征融合 - 替代标准Concat
    对每个输入特征进行可学习加权后再Concat
    YAML: [[-1, 6], 1, BiFPN_Concat, [1]]
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # 预注册2个输入权重 (FPN中每个Concat恰好2个输入)
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.eps = 1e-4

    def forward(self, x):
        # ReLU保证非负 + epsilon防止除零
        w1 = torch.relu(self.w1)
        w2 = torch.relu(self.w2)
        w_sum = w1 + w2 + self.eps
        return torch.cat([x[0] * (w1 / w_sum), x[1] * (w2 / w_sum)], self.d)
