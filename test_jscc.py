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


def test_jscc():
   #model
    encoder = Encoder(out_channels=16)
    decoder = Decoder(in_channels=16)
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
        os.mkdir(addr)
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
    repeattimes = 50
    with torch.no_grad():
        all_psnrs = []
        all_mses = []
        all_ssims = []
        for snr in range(-2, 25):
            print(snr)
            psnrs = 0
            mses = 0
            ssims = 0
            for i,data in tqdm(enumerate(val_loader,0)):
                images,_ = data
                if i == 0:
                    imshow_andsave(torchvision.utils.make_grid(images.cpu()), addr+'/original_image.png')
                images = images.to(device)
                t = time.time()
                t1=time.time()
                for _ in range(repeattimes):  
                    pre = model(images, snr=snr)
                    #pdb.set_trace()
                    ssim = ssimer(torch.tensor(pre/2+0.5, dtype=torch.float32), torch.tensor(images/2+0.5, dtype=torch.float32))
                    mse = torch.mean(((pre/2 + 0.5) - (images/2 + 0.5)) ** 2, dim=[1, 2, 3])
                    psnr = 10*torch.log10(1/mse)
                    psnrs+=torch.mean(psnr)
                    ssims+=torch.mean(ssim)
                    mses+= torch.mean(mse)
                if i==0:
                    imshow_andsave(torchvision.utils.make_grid(pre.cpu()), addr+'/snr = {}.png'.format(snr))
                
            psnrs /= (repeattimes)*len(val_loader)
            mses /= (repeattimes)*len(val_loader)
            ssims /= (repeattimes)*len(val_loader)
            
            all_psnrs.append(torch.mean(psnrs).to('cpu'))
            print(torch.mean(psnrs).to('cpu'))
            all_mses.append(torch.mean(mses).to('cpu'))
            all_ssims.append(torch.mean(ssims).to('cpu'))
            t2 = time.time()
            #tqdm.write("30 test takes time {} when snr = {}".format(t2-t1,snr))
            tt = time.time()
            #tqdm.write("snr = {}, time = {}".format(snr,tt-t))
            

    # 绘制PSNR和MSE图表
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    #pdb.set_trace()
    plt.plot(range(-2, 25), all_psnrs,marker = 'o',markersize = 5)
    plt.title('Average PSNR for Different SNRs')
    plt.xlabel('SNR')
    plt.xlim(0,20)
    plt.ylabel('Average PSNR')

    plt.subplot(1, 3, 2)
    plt.plot(range(-2, 25), all_mses,marker = 'o',markersize = 5)
    plt.title('Average MSE for Different SNRs')
    plt.xlabel('SNR')
    plt.xlim(0,20)
    plt.ylabel('Average MSE')

    plt.subplot(1, 3, 3)
    plt.plot(range(-2, 25), all_ssims,marker = 'o',markersize = 5)
    plt.title('Average ssim for Different SNRs')
    plt.xlabel('SNR')
    plt.xlim(0,20)
    plt.ylabel('Average SSIM')

    plt.savefig(addr+'/curve.png')
    plt.close()
    np.savetxt(addr+'/psnrs.txt',all_psnrs)
    np.savetxt(addr+'/mses.txt',all_mses)
    np.savetxt(addr+'/ssims.txt',all_ssims)



def main():
    #test_feature_sc1()
    test_jscc()

if __name__ =="__main__":
    main()
