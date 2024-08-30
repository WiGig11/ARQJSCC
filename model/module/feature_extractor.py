import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Feature_Backbone_34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet34(pretrained=True)
        self.net.fc = torch.nn.Identity()
        # num_ftrs = self.net.fc.in_features 
        for param in self.net.parameters():
            param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
        self.net = nn.Sequential(*list(self.net.children())[:-2])
    def forward(self, x):
        return self.net(x)
    
class Feature_Backbone_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Identity()
        # num_ftrs = self.net.fc.in_features 
        for param in self.net.parameters():
            param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
        self.net = nn.Sequential(*list(self.net.children())[:-2])
    def forward(self, x):
        return self.net(x)