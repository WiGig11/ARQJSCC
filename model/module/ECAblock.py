import math
import torch
import torch.nn as nn
import numpy as np

class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        y = self.avg_pool(x)
        #y1 = y.squeeze(-1).transpose(-1, -2).contiguous()
        y = y.view(b,1,c)
        #print(torch.count_nonzero(y1-y2))
        y = self.conv1(y)
        #y1 = y.transpose(-1, -2).unsqueeze(-1).contiguous()
        y = y.view(b,c,1,1)
        #print(torch.count_nonzero(y1-y2))
        y = self.sigmoid(y)
        y = y.expand_as(x)
        return x*y
