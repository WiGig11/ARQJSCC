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

def get_ack(output):
    #output:tuple,check mean?
    res = 0
    return res

def test_validity():
    #model
    encoder = Encoder(out_channels=8)
    decoder = Decoder(in_channels=8)
    discriminator = MultiScaleDiscriminator()
    channel = AWGNChannel()
    ckpt = 'logs/deepjsccAF/version_0/checkpoints/epoch=909-step=355810.ckpt'
    model = DeepJSCCARQ.load_from_checkpoint(ckpt,
                    encoder=encoder,decoder=decoder,
                    loss_module_D=Least_Square_Loss(),loss_module_G=MSEImageLoss(),
                    channel=channel,
                    discriminator= discriminator,
                    hyperparameter=12,
                    lr_D = 5e-6,lr_G = 1e-4
                )    
    addr = 'res/res_AF_1b6'
    with open(addr+'/latest.txt', 'w') as f:
        print(model, file=f)
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #eval mode
    model = model.to(device)
    model.eval()

    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    classifier = classifier.to(device)
    classifier.eval()

    #data transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,download=True, transform=transform)
    batch_size = 64  # 加载64张图像
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)  # 注意关闭shuffle
    reapttime = 50
    discriminator = model.discriminator
    ack_acu = []
    noack_acu = []
    with torch.no_grad():
        for snr in range(-2, 25):
            acu1 = 0
            acu2 = 0
            for i in range(0,reapttime):
                correct_withack = 0
                correct_withoutack = 0
                total = 0
                for i,data in tqdm(enumerate(val_loader,0)):
                    images,label = data
                    images = images.to(device)
                    decoded,_,_ = model(images, snr=snr)
                    _, predicted = torch.max(decoded.data, 1)
                    total += label.size(0)
                    ack = get_ack(discriminator(decoded))
                    b,c,h,w = decoded.size()
                    for j in range(0,b):
                        if ack[j]==1:
                            # the class with the highest energy is what we choose as prediction
                            correct_withack += (predicted[j] == label[j]).sum().item()
                            #no count
                        else:
                            pass
                        correct_withoutack += (predicted == label).sum().item()
                print(f'Accuracy of the network on the 10000 test images with ACK: {100 * correct_withack // total} %')
                print(f'Accuracy of the network on the 10000 test images without ACK: {100 * correct_withoutack // total} %')
                acu1 += 100 * correct_withack // total
                acu2 += 100 * correct_withoutack // total
            ack_acu.append(acu1/reapttime)
            noack_acu.append(acu2/reapttime)
    
    print(ack_acu)
    print(noack_acu)
            

def main():
    test_validity()

if __name__ =="__main__":
    main()
