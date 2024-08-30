import torch
import torch.nn as nn
import torch.nn,functools as F
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import numpy as np

import pdb

torch.autograd.set_detect_anomaly(True)

def filesizeoftensor(file):
    pass
#TOO: 修改得到tensor大小的函数

def maskprecentage(heatmap,mask):
    # 计算像素数
    pixel_count = np.sum(mask)
    pixel_count_img = heatmap.size()[0]*heatmap.size()[1]
    return pixel_count/pixel_count_img
##################
class MixtureLossFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = torch.nn.MSELoss(reduction='none')

    def forward(self,heatmap,feature_received,mask,hyper_alpha=10):
        loss1 = self.loss_fn1(heatmap.float(), feature_received.float())
        #loss2 = filesizeoftensor(feature_received)
        loss2 = maskprecentage(feature_received,mask)
        return loss1+hyper_alpha*loss2
##############
class MSEImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = torch.nn.MSELoss(reduction='mean')

    def forward(self,target_image,received_image):
        loss1 = self.loss_fn1(target_image.float(), received_image.float())
        return loss1
#################
class SSIMLossImage(nn.Module):
    def __init__(self):
        super().__init__()    
    
    def forward(self,target_image,received_image):
        ssimer = SSIM(data_range=1.0, reduction='none').to(target_image.device)  # 计算每张图像的SSIM
        ssim = ssimer(target_image, received_image).clone()
        ssim = ssim.mean()
        return (1-ssim)

#################
class MixtureLossImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = torch.nn.MSELoss(reduction='mean')
        self.a = nn.Parameter(torch.tensor(0.5))
    
    def forward(self,target_image,received_image):
        loss1 = self.loss_fn1(target_image.float(), received_image.float())
        ssimer = SSIM(data_range=1.0, reduction='none').to(target_image.device)  # 计算每张图像的SSIM
        ssim = ssimer(target_image, received_image).clone()
        ssim = ssim.mean()
        #print(ssim)c
        #print(loss1)
        #pdb.set_trace()
        a_clamped = torch.clamp(self.a, 0, 1)
        return a_clamped * loss1 + (1 - a_clamped) * (1 - ssim)  # 注意SSIM通常是越大越好，所以使用1-ssim来计算损失
        
    
##############        
class CELossAck(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,ack,similarity):
        loss = self.loss_fn(ack, similarity)
        return loss
##################
class BCELossAck(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self,ack,similarity):
        """
        need [batch,2] size as input and target to calculate. 2 as one-hot, input as two logits representing the score
        """
        loss = self.loss_fn(ack.float(), similarity.float())
        return loss

class Least_Square_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()  # MSE Loss computes (x-y)^2 mean
    def forward(self,realout,fakeout):
        G_loss_all = 0
        D_loss_all = 0
        for real, fake in zip(realout, fakeout):
            G_loss = self.mse_loss(fake, torch.ones_like(fake))
            D_loss_real = self.mse_loss(real, torch.ones_like(real))
            D_loss_fake = self.mse_loss(fake, torch.zeros_like(fake))
            D_loss = D_loss_real + D_loss_fake
            D_loss_all = D_loss_all+D_loss
            G_loss_all = G_loss_all+G_loss
        return G_loss_all, D_loss_all