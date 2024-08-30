import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.module.SEblock import SElayer,AFlayer
from model.module.gdn import GDN
from model.module.ECAblock import EfficientChannelAttention

import pdb
import math

#torch.autograd.set_detect_anomaly(True) 

class EncodingBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.activation = nn.PReLU()
        self.gdn = GDN(ch=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gdn(x)
        x = self.activation(x)
        return x
    
class EncodingResBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(EncodingResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        #self.ECAblock = self.build_ECA_block(dim)

    def build_ECA_block(self,dim):
        return EfficientChannelAttention(c=dim)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        gdn = GDN(ch=dim)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), gdn,nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        attention = self.build_ECA_block(dim=dim)
        conv_block+=[attention]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        gdn1 = GDN(ch=dim)
        #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)],gdn
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),gdn1]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def power_normalize(x):
    x = torch.mean(x, (-2, -1))
    b, c = x.shape  # Assuming x has shape [batch_size, 20]
    alpha = math.sqrt(c)
    energy = torch.norm(x, p=2, dim=1)# Calculate the L2 norm of each one-dimensional vector
    alpha = alpha / energy.unsqueeze(1)# Calculate the normalization factor alpha for each vector
    x_normalized = alpha * x# Apply alpha to each vector
    return x_normalized

class Encoder(nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()
        self.out_channels = out_channels
        self.channels = [3,256,256,256,256]
        self.block1 = EncodingBlock(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=9,stride=2,padding = 4)
        self.block2 = EncodingBlock(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=5,stride=2,padding = 2)
        self.block3 = EncodingBlock(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=5,stride=1,padding = 2)
        self.block4 = EncodingBlock(in_channels=self.channels[3],out_channels=self.channels[4],kernel_size=5,stride=1,padding = 2)
        self.block5 = EncodingBlock(in_channels=self.channels[4],out_channels=self.out_channels,kernel_size=5,stride=1,padding = 2)
        self.ResnetBlock1 = EncodingResBlock(dim=self.channels[1],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock2 = EncodingResBlock(dim=self.channels[2],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock3 = EncodingResBlock(dim=self.channels[3],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=True,use_bias=True)
        self.ResnetBlock4 = EncodingResBlock(dim=self.channels[4],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock5 = EncodingResBlock(dim=self.out_channels,padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.se1 = SElayer(channel=self.channels[1],reduction=16)
        self.se2 = SElayer(channel=self.channels[2],reduction=16)
        self.se3 = SElayer(channel=self.channels[3],reduction=16)
        self.se4 = SElayer(channel=self.channels[4],reduction=16)
        # original channels:3 16 32 32 32 400
        self.blocks = [self.block1,self.block2,self.block3,self.block4,self.block5]
        self.ResnetBlocks = [self.ResnetBlock1,self.ResnetBlock2,self.ResnetBlock3,self.ResnetBlock4,self.ResnetBlock5]
        self.ses = [self.se1,self.se2,self.se3,self.se4]

    def forward(self,x,snr):
        for i in range(4):
            x = self.blocks[i](x)
            x = self.ResnetBlocks[i](x)
            x = self.ses[i](x,snr)
        x = self.block5(x)
        x = self.ResnetBlock5(x)
        b,c,h,w=x.shape
        x = x.view(b,c*h*w,1,1)
        x = power_normalize(x)
        return x
        