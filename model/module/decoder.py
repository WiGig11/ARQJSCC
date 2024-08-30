import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.module.SEblock import SElayer,AFlayer
from model.module.ECAblock import EfficientChannelAttention
from model.module.gdn import GDN

import pdb
import math

torch.autograd.set_detect_anomaly(True)

class DecodingBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(DecodingBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,padding)
        self.igdn = GDN(ch=out_channels,inverse=True)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.igdn(x)
        x = self.activation(x)
        return x
    
class DecodingResBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(DecodingResBlock, self).__init__()
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
        igdn = GDN(ch=dim,inverse=True)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), igdn,nn.PReLU()]
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
        igdn1 = GDN(ch=dim,inverse=True)
        #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)], igdn
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),igdn1]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        #pdb.set_trace()
        out =  x + self.conv_block(x)  # add skip connections
        return out

class Decoder(nn.Module):
    def __init__(self,in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.channels = [256,256,256,256,3]
        self.block1 = DecodingBlock(in_channels=self.in_channels,out_channels=self.channels[0],kernel_size=5,stride=1,padding =2)
        self.block2 = DecodingBlock(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=5,stride=1,padding =2)
        self.block3 = DecodingBlock(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=5,stride=1,padding =2)
        self.block4 = DecodingBlock(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=5,stride=2,padding =2)
        self.finalgdn = GDN(ch=self.channels[4],inverse=True)
        self.block5 = nn.Sequential(nn.ConvTranspose2d(in_channels = self.channels[3], out_channels=self.channels[4], kernel_size=8,stride=2,padding=2),
                        self.finalgdn,
                        nn.PReLU())
        
        self.ResnetBlock1 = DecodingResBlock(dim=self.channels[0],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock2 = DecodingResBlock(dim=self.channels[1],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock3 = DecodingResBlock(dim=self.channels[2],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=True,use_bias=True)
        self.ResnetBlock4 = DecodingResBlock(dim=self.channels[3],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)
        self.ResnetBlock5 = DecodingResBlock(dim=self.channels[4],padding_type='zero',norm_layer=nn.BatchNorm2d,use_dropout=False,use_bias=True)

        self.se1 = SElayer(channel=self.channels[0],reduction=16)
        self.se2 = SElayer(channel=self.channels[1],reduction=16)
        self.se3 = SElayer(channel=self.channels[2],reduction=16)
        self.se4 = SElayer(channel=self.channels[3],reduction=16)
        # original channel: 400 32 32 32 16
        self.blocks = [self.block1,self.block2,self.block3,self.block4,self.block5]
        self.ResnetBlocks = [self.ResnetBlock1,self.ResnetBlock2,self.ResnetBlock3,self.ResnetBlock4,self.ResnetBlock5]
        self.ses = [self.se1,self.se2,self.se3,self.se4]
        self.final_acti = nn.Tanh()

    def forward(self,x,snr):
        x = x.unsqueeze(2).unsqueeze(3) #reshape to b,c,1,1
        b,c,_,_ = x.shape
        h = int(math.sqrt(c/self.in_channels))
        w = h
        c = self.in_channels
        x = x.view(b,c,h,w)
        for i in range(4):
            x = self.blocks[i](x)
            x = self.ResnetBlocks[i](x)
            x = self.ses[i](x,snr)
        x = self.block5(x)
        x = self.ResnetBlock5(x)
        x = self.final_acti(x)
        return x
