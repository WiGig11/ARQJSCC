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
import os
from tqdm import tqdm

def get_ack(ack_tuple):
    #pdb.set_trace()
    """
    输入：一个包含 n 个张量的 tuple，每个张量的形状为 [b, c, h, w]。
    输出：一个形状为 [b] 的张量，其中每个元素是 1 或 0，表示该样本是否被认为是真实的。
    """
    batch_size = ack_tuple[0].size(0)  # 获取 batch 大小
    
    # 初始化一个形状为 [b] 的张量用于存储各尺度结果的均值和
    combined_mean = torch.zeros(batch_size).to(ack_tuple[0].device)
    for ack in ack_tuple:
        # 对每个张量在 (c, h, w) 维度上取均值，得到 [b] 的结果
        mean_values = torch.mean(ack, dim=[1, 2, 3])
        
        # 将该尺度的均值累加到 combined_mean 中
        combined_mean += mean_values

    # 对累加后的均值取平均
    combined_mean /= len(ack_tuple)

    # 最后根据设定的阈值，判断每个样本是否通过判别
    threshold = 0.5
    ack_results = (combined_mean > threshold).float()
    
    return ack_results




def imshow_andsave(img,title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #pdb.set_trace()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(title)
    plt.close()
    #plt.show()

def test_validity():
    #model
    encoder = Encoder(out_channels=8)
    decoder = Decoder(in_channels=8)
    discriminator = MultiScaleDiscriminator()
    channel = AWGNChannel()
    ckpt = 'logs/ARQ/AWGN/version_6/checkpoints/epoch=499-step=782000.ckpt'
    model = DeepJSCCARQ.load_from_checkpoint(ckpt,
                    encoder=encoder,decoder=decoder,
                    loss_module_D=Least_Square_Loss(),loss_module_G=MSEImageLoss(),
                    channel=channel,
                    discriminator= discriminator,
                    hyperparameter=12,
                    lr_scheduler_type = 'step',
                    lr_D = 5e-6,lr_G = 1e-4
                )
    addr = 'res/gan_awgn/vadi_version6'
    if not os.path.exists(addr):
        os.mkdir(addr)
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
    valset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,download=True, transform=transform)
    batch_size = 64  # 加载64张图像
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)  # 注意关闭shuffle
    reapttime = 50
    ssimer = SSIM(data_range=1.0, reduction='none')  # 计算每张图像的SSIM
    discriminator = model.discriminator
    ack_acu = []
    noack_acu = []
    with torch.no_grad():
        all_psnrs = []
        all_mses = []
        all_ssims = []
        for snr in range(-2, 25):
            acu1 = 0
            acu2 = 0
            psnrs = 0
            mses = 0
            ssims = 0
            correct_withack = 0
            correct_withoutack = 0
            total = 0
            for i,data in tqdm(enumerate(val_loader,0)):
                images,label = data
                if i == 0:
                    imshow_andsave(torchvision.utils.make_grid(images.cpu()), addr+'/original_image.png')
                images = images.to(device)
                label = label.to(device)
                for _ in range(0,reapttime):
                    decoded,_,_ = model(images, snr=snr)
                    prediction = classifier(images)
                    ssim = ssimer(torch.tensor(decoded/2+0.5, dtype=torch.float32), torch.tensor(images/2+0.5, dtype=torch.float32))
                    mse = torch.mean(((decoded/2 + 0.5) - (images/2 + 0.5)) ** 2, dim=[1, 2, 3])
                    psnr = 10*torch.log10(1/mse)
                    psnrs+=torch.mean(psnr)
                    ssims+=torch.mean(ssim)
                    mses+= torch.mean(mse)
                    _, predicted = torch.max(prediction.data, 1)
                    total += label.size(0)
                    pdb.set_trace()
                    ack = get_ack(discriminator(decoded))
                    b,c,h,w = decoded.size()
                    for j in range(0,b):
                        if ack[j]==1:
                            # if ack , we think it passed the check of dis, so we calculate the prob of right
                            # classification of those dis thinks is the right
                            correct_withack += (predicted[j] == label[j]).sum().item()
                            #no count
                        else:
                            pass
                    correct_withoutack += (predicted == label).sum().item()
                    
                if i==0:
                    imshow_andsave(torchvision.utils.make_grid(decoded.cpu()), addr+'/snr = {}.png'.format(snr))
            print(f'Accuracy of the network on the 10000 test images with ACK: {100 * correct_withack // total} %')
            print(f'Accuracy of the network on the 10000 test images without ACK: {100 * correct_withoutack // total} %')
            acu1 += 100 * correct_withack // total
            acu2 += 100 * correct_withoutack // total
            psnrs /= (reapttime)*len(val_loader)
            mses /= (reapttime)*len(val_loader)
            ssims /= (reapttime)*len(val_loader)
            
            all_psnrs.append(torch.mean(psnrs).to('cpu'))
            print(torch.mean(psnrs).to('cpu'))
            all_mses.append(torch.mean(mses).to('cpu'))
            all_ssims.append(torch.mean(ssims).to('cpu'))         
            ack_acu.append(acu1)
            noack_acu.append(acu2)
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

    print(ack_acu)
    print(noack_acu)
            

def main():
    test_validity()

if __name__ =="__main__":
    main()
