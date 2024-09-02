import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from model.module.feature_extractor import Feature_Backbone_34,Feature_Backbone_50
#from model.module.pattern_vit import patternViT
from model.module.SEblock import SElayer
from model.module.gdn import GDN
import pdb
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 提升数值稳定性
    return e_x / e_x.sum(axis=0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True)  # affine=True to learn scale and shift
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        x = self.relu(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial = ConvBlock(3, 64, kernel_size=4, stride=2, padding=1)#16
        self.conv1 = ConvBlock(64, 128)#8
        self.conv2 = ConvBlock(128, 256)#4
        #self.final = ConvBlock(256, 1, kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.final(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminator = Discriminator()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x2 = self.downsample(x)
        x4 = self.downsample(x2)
        out = self.discriminator(x)
        out2 = self.discriminator(x2)
        out4 = self.discriminator(x4)
        return (out,out2,out4)
        #return (out,out4)
