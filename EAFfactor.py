import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import pytorch_lightning as pl
from pytorch_lightning import Trainer

#from model.model import DeepJSCCARQ
from model.module.encoder import Encoder
from model.module.decoder import Decoder
from model.DeepjsccARQ import DeepJSCCARQ
from model.Deepjscc import DeepJSCC
from model.module.discriminator import Discriminator,MultiScaleDiscriminator
from channels.AWGN import AWGNChannel
from channels.Rayl import RayleighChannel
from loss.mixure_loss import MixtureLossImage,MixtureLossFeature,BCELossAck,MSEImageLoss,Least_Square_Loss

from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Set backend
import matplotlib.pyplot as plt

import numpy as np
import cv2
import pdb
import time
from tqdm import tqdm
import os

def imshow_andsave(img,title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #pdb.set_trace()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(title)
    plt.close()
    #plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_psnr(pre, img):
    #pdb.set_trace()
    # 假设pre和img都是单个图像的numpy数组，且已经缩放到0-255范围内且为uint8类型
    mse = np.mean((pre - img) ** 2)
    mse = max(mse, 1e-10)  # 避免除以0
    psnr = 10 * np.log10((1 ** 2) / mse)
    #print(psnr)
    return psnr, mse


def samechanneldifferentsnr():
    #model
    encoder = Encoder(out_channels=8)
    decoder = Decoder(in_channels=8)
    #channel = AWGNChannel()
    channel = RayleighChannel()
    #ckpt = 'logs/JSCC/RAYL/version_1/checkpoints/epoch=999-step=782000.ckpt'
    ckpt = 'test.ckpt'
    model = DeepJSCC.load_from_checkpoint(ckpt,
                    encoder=encoder,decoder=decoder,
                    loss_module_G=MSEImageLoss(),
                    channel=channel,
                    lr_scheduler_type = 'step',
                    lr_G = 1e-4
                )    
    addr = 'res/jscc_rayl/version_1'
    if not os.path.exists(addr):
        os.makesdir(addr)
    with open(addr+'/latest.txt', 'w') as f:
        print(model, file=f)
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #eval mode
    model = model.to(device)
    #model.eval()
    #data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,download=True, transform=transform)
    batch_size = 64  # 加载64张图像
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)  # 注意关闭shuffle
    ssimer = SSIM(data_range=1.0, reduction='none')  # 计算每张图像的SSIM
    snrs = [1,4,7,13,19]
    encoderse1 = []
    encoderse2 = []
    encoderse3 = []
    encoderse4 = []
    with torch.no_grad():
        for snr in snrs:
            print(snr)
            for i,data in tqdm(enumerate(val_loader,0)):
                images,_ = data
                images = images.to(device)
                pre = model(images, snr=snr)
                #y = self.fc2(torch.cat([snr,y],1)) y means the output scale factor
                encoderse1.append()
                encoderse1.append()
                encoderse1.append()
                encoderse1.append()

    all_outputs = torch.cat(all_outputs, dim=0)  # shape: (n, c, h, w)
    # mean and var
    mean = all_outputs.mean(dim=0)  # mean of all samples
    variance = all_outputs.var(dim=0)  # var of all samples

    print("Mean:", mean)
    print("Variance:", variance)

def main():
    samechanneldifferentsnr()

if __name__ =="__main__":
    main()
