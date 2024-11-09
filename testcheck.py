import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import pytorch_lightning as pl
from pytorch_lightning import Trainer

# from model.model import DeepJSCCARQ
from model.module.encoder import Encoder
from model.module.decoder import Decoder
from model.DeepjsccARQ import DeepJSCCARQ
from model.module.discriminator import Discriminator, MultiScaleDiscriminator
from channels.AWGN import AWGNChannel
from channels.Rayl import RayleighChannel
from loss.mixure_loss import (
    MixtureLossImage,
    MixtureLossFeature,
    BCELossAck,
    MSEImageLoss,
    Least_Square_Loss,
)

from PIL import Image
import matplotlib

matplotlib.use("Agg")  # Set backend
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import numpy as np
import cv2
import pdb
import time
import tqdm
import os


def imshow_andsave(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # pdb.set_trace()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(title)
    plt.close()
    # plt.show()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_psnr(pre, img):
    # pdb.set_trace()
    # 假设pre和img都是单个图像的numpy数组，且已经缩放到0-255范围内且为uint8类型
    mse = np.mean((pre - img) ** 2)
    mse = max(mse, 1e-10)  # 避免除以0
    psnr = 10 * np.log10((1**2) / mse)
    # print(psnr)
    return psnr, mse

def test(addr,val_loader,device,repeattimes,model,ssimer):
    with torch.no_grad():
        all_psnrs = []
        all_mses = []
        all_ssims = []
        for snr in range(-2, 25):
            print(snr)
            psnrs = 0
            mses = 0
            ssims = 0
            for i, data in tqdm(enumerate(val_loader, 0)):
                images, _ = data
                if i == 0:
                    imshow_andsave(
                        torchvision.utils.make_grid(images.cpu()),
                        addr + "/original_image.png",
                    )
                images = images.to(device)
                t = time.time()
                t1 = time.time()
                for _ in range(repeattimes):
                    pre, _, _ = model(images, snr=snr)
                    # pdb.set_trace()
                    ssim = ssimer(
                        torch.tensor(pre / 2 + 0.5, dtype=torch.float32),
                        torch.tensor(images / 2 + 0.5, dtype=torch.float32),
                    )
                    mse = torch.mean(
                        ((pre / 2 + 0.5) - (images / 2 + 0.5)) ** 2, dim=[1, 2, 3]
                    )
                    psnr = 10 * torch.log10(1 / mse)
                    psnrs += torch.mean(psnr)
                    ssims += torch.mean(ssim)
                    mses += torch.mean(mse)
                if i == 0:
                    imshow_andsave(
                        torchvision.utils.make_grid(pre.cpu()),
                        addr + "/snr = {}.png".format(snr),
                    )

            psnrs /= (repeattimes) * len(val_loader)
            mses /= (repeattimes) * len(val_loader)
            ssims /= (repeattimes) * len(val_loader)

            all_psnrs.append(torch.mean(psnrs).to("cpu"))
            print(torch.mean(psnrs).to("cpu"))
            all_mses.append(torch.mean(mses).to("cpu"))
            all_ssims.append(torch.mean(ssims).to("cpu"))
            t2 = time.time()
            # tqdm.write("30 test takes time {} when snr = {}".format(t2-t1,snr))
            tt = time.time()
            # tqdm.write("snr = {}, time = {}".format(snr,tt-t))

    # 绘制PSNR和MSE图表
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    # pdb.set_trace()
    plt.plot(range(-2, 25), all_psnrs, marker="o", markersize=5)
    plt.title("Average PSNR for Different SNRs")
    plt.xlabel("SNR")
    plt.xlim(0, 20)
    plt.ylabel("Average PSNR")

    plt.subplot(1, 3, 2)
    plt.plot(range(-2, 25), all_mses, marker="o", markersize=5)
    plt.title("Average MSE for Different SNRs")
    plt.xlabel("SNR")
    plt.xlim(0, 20)
    plt.ylabel("Average MSE")

    plt.subplot(1, 3, 3)
    plt.plot(range(-2, 25), all_ssims, marker="o", markersize=5)
    plt.title("Average ssim for Different SNRs")
    plt.xlabel("SNR")
    plt.xlim(0, 20)
    plt.ylabel("Average SSIM")

    plt.savefig(addr + "/curve.png")
    plt.close()
    np.savetxt(addr + "/psnrs.txt", all_psnrs)
    np.savetxt(addr + "/mses.txt", all_mses)
    np.savetxt(addr + "/ssims.txt", all_ssims)

def test_check(val_loader,device,repeattimes,model,dis_thre):
    #test_check(addr,val_loader,device,repeattimes,discriminator)
    #define discriminator
    discriminator = model.discriminator
    #load classifier model
    net =  torch.hub.load("chenyaofo/pytorch-cifar-models","cifar10_resnet56",pretrained=True)
    net = net.to(device)
    with torch.no_grad():
        accu_transes = []
        accues = []
        accu_dis_transes = [] 
        for snr in tqdm.tqdm(range(-2, 22)):
            total = 0
            correct_trans =  0
            correct = 0
            correct_trans_dis = 0
            dis_num = 0
            for data in val_loader:
                images, labels = data[0].to(device),data[1].to(device)
                for _ in range(repeattimes):
                    #1.get transmission
                    decoded,_,_ = model(images, snr=snr)
                    #2.get pd decoder result
                    decoder_classifcation = discriminator(decoded)
                    #3.get classification result for the origin and the trans
                    classifcation_trans = net(decoded)
                    classifcation = net(images)
                    _,preds_trans= torch.max(classifcation_trans.data, 1)
                    _,preds= torch.max(classifcation.data, 1)
                    total += labels.size(0)
                    correct_trans += (preds_trans == labels).sum().item()
                    correct += (preds == labels).sum().item()
                    #4.discard the wrong images            
                    #traverse batch
                    preds_dis_trans= preds_trans
                    pdb.set_trace()
                    for j in range(images.size()[0]):
                        #pdb.set_trace()
                        if torch.mean(decoder_classifcation[0][j]+decoder_classifcation[1][j]+decoder_classifcation[2][j]) < dis_thre:
                            pass
                        elif preds_dis_trans[j]==labels[j]:
                            correct_trans_dis = correct_trans_dis+1                           
                            dis_num = dis_num+1
            #statics!
            accu_trans = correct_trans/total
            accu = correct/total
            accu_dis_trans = correct_trans_dis/dis_num
            accu_transes.append(accu_trans)
            accues.append(accu)
            accu_dis_transes.append(accu_dis_trans)
        return accu_transes,accues,accu_dis_transes

def find_right_results(preds, labels, decoder_classification, threshold):
    # 计算每个样本的均值，结果为形状 (batch_size,)
    mean_values = decoder_classification.mean(dim=[1, 2, 3])
    
    # 根据阈值获取鉴定结果，结果为形状 (batch_size,)
    decoder_results = (mean_values >= threshold).long()
    
    # 计算正确分类和错误分类，结果为形状 (batch_size,)
    correct_classification = (preds == labels).long()
    incorrect_classification = (preds != labels).long()
    
    # 计算 TP, FP, TN, FN
    # 真阳性 (TP): 正确分类且被判断为有效图像
    TP = torch.sum(correct_classification * decoder_results).item()
    # 真阴性 (TN): 错误分类且被判断为无效图像
    TN = torch.sum(incorrect_classification * (1 - decoder_results)).item()
    # 假阳性 (FP): 错误分类但被判断为有效图像
    FP = torch.sum(incorrect_classification * decoder_results).item()
    # 假阴性 (FN): 正确分类但被判断为无效图像
    FN = torch.sum(correct_classification * (1 - decoder_results)).item()
    
    return TP, FP, TN, FN

def evaluate_preformance_singleSNR(TP, FP, TN, FN):
    #Accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    #f1 = 2*precision*recall/(precision+recall)
    ##Specificity = TN/(TN+FP)
    #FPR = FP/(FP+TN)
    #pdb.set_trace()
    #return Accuracy, precision, recall, FPR,f1, Specificity
    return precision, recall

def PRcurve(recalls,precision):
    plt.figure()
    plt.plot(recalls, precision, marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], 'r--')  # 添加对角线
    plt.xlabel('Recall')
    plt.ylabel('Precisoins')
    plt.title('R-P Curve')
    title = 'PRCurve'+'.png'
    plt.savefig(title)
    plt.show()

def ROCcurve(FPRS,recalls):
    plt.figure()
    plt.plot(FPRS, recalls, marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], 'r--')  # 添加对角线
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    title = 'ROCCurve'+'.png'
    plt.savefig(title)
    plt.show()

def charts(device,repeattimes,val_loader,model,threshold):
    """
    addr:place the result
    device:device
    repeattimes:repeattimes
    val_loader:val_loader
    """
    discriminator = model.discriminator
    #load classifier model
    net =  torch.hub.load("chenyaofo/pytorch-cifar-models","cifar10_resnet56",pretrained=True)
    net = net.to(device)
    #define transform and valset
    with torch.no_grad():
        recalls = []
        precisions = []
        TPs = []
        FPs = []
        TNs = []
        FNs = []
        for snr in tqdm.tqdm(range(-2, 22)):
            TPsnr = 0
            TNsnr = 0
            FPsnr = 0
            FNsnr = 0
            for data in val_loader:
                images, labels = data[0].to(device),data[1].to(device)
                #1.get transmission
                for _ in range(repeattimes):
                    #1.get transmission
                    decoded,_,_ = model(images, snr=snr)
                    #2.get pd decoder result
                    decoder_classifcation = discriminator(decoded)
                    #3.get classification result
                    classifcation = net(decoded)
                    _,preds= torch.max(classifcation.data, 1)
                    #4.get confusion result
                    TP, FP, TN, FN = find_right_results(preds.cpu().numpy(),labels.cpu().numpy(),decoder_classifcation.cpu().numpy(),threshold)
                    TPsnr = TPsnr + TP
                    FPsnr = FPsnr + FP
                    TNsnr = TNsnr + TN
                    FNsnr = FNsnr + FN
            P,R = evaluate_preformance_singleSNR(TP, FP, TN, FN)
            TPs.append(TPsnr)
            FPs.append(FPsnr)
            TNs.append(TNsnr)
            FNs.append(FNsnr)
            recalls.append(R)
            precisions.append(P)
    return TPs,FPs,TNs,FNs,recalls,precisions

def test_dis(hparams):
    # define para
    addr = hparams.addr
    name = hparams.name
    if "awgn" in hparams.communication_channel: #TODO: this is not a fault but a design to simulate channel mismatch
        channel = RayleighChannel()
        addr = os.path.join(addr, "awgn")
        addr = os.path.join(addr, name)
        if not os.path.exists(addr):
            os.makedirs(addr)
    elif "rayl" in hparams.communication_channel:
        channel = AWGNChannel()
        addr = os.path.join(addr, "rayl")
        addr = os.path.join(addr, name)
        if not os.path.exists(addr):
            os.makedirs(addr)
    ckpt = hparams.ckpt_addr
    if torch.cuda.is_available() and "1" in hparams.device:
        device = torch.device("cuda:1")
    elif torch.cuda.is_available() and "0" in hparams.device:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)
    #deine model
    in_channel = hparams.channel
    encoder = Encoder(out_channels=in_channel)
    decoder = Decoder(in_channels=in_channel)
    discriminator = MultiScaleDiscriminator()
    hp = hparams.hyperparameter
    model = DeepJSCCARQ.load_from_checkpoint(ckpt,
        map_location=device,  # cuda:0
        encoder=encoder,decoder=decoder,
        loss_module_D=Least_Square_Loss(),loss_module_G=MSEImageLoss(),
        channel=channel,discriminator=discriminator,hyperparameter=hp,
        lr_scheduler_type="step",lr_D=5e-6,lr_G=1e-4,
    )
    with open(addr + "/latest.txt", "w") as f:
        print(model, file=f)

    # eval mode
    model = model.to(device)
    # define data
    batch_size = hparams.batch_size  # 加载64张图像
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    valset = torchvision.datasets.CIFAR10(root="./cifar_data", train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=15)  # 注意关闭shuffle
    #define metric
    ssimer = SSIM(data_range=1.0, reduction="none")  # 计算每张图像的SSIM
    '''
    test!!!
    # '''
    repeattimes = 50
    #test(addr,val_loader,device,repeattimes,model,ssimer)
    #define discriminator
    '''
    checkpoint = torch.load(ckpt, map_location='cpu')
    state_dict = checkpoint['state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'discriminator' in k}
    discriminator.load_state_dict(filtered_state_dict, strict=False)
    discriminator.to(device)
    '''
    accu_transes,accues,accu_dis_transes = test_check(val_loader,device,repeattimes,model,0.3)
    print('accu_transes\n')
    for i in range(0,len(accu_transes)):
        print(accu_transes[i])
    print('\n')
    print('accues\n')
    for i in range(0,len(accues)):
        print(accues[i])
    print('\n')
    print('accu_dis_transes\n')
    for i in range(0,len(accu_dis_transes)):
        print(accu_dis_transes[i])
    print('\n')

    TPs,FPs,TNs,FNs,recalls,precisions = charts(device,repeattimes,val_loader,model,0.3)
    print('TPs\n')
    for i in range(0,len(TPs)):
        print(TPs[i])
    print('\n')
    print('FPs\n')
    for i in range(0,len(FPs)):
        print(FPs[i])
    print('\n')
    print('TNs\n')
    for i in range(0,len(TNs)):
        print(TNs[i])
    print('\n')
    print('FNs\n')
    for i in range(0,len(FNs)):
        print(FNs[i])
    print('\n')
    print('recalls\n')
    for i in range(0,len(recalls)):
        print(recalls[i])
    print('\n')
    print('precisions\n')
    for i in range(0,len(precisions)):
        print(precisions[i])
    print('\n')



def main(hparams):
    # test_feature_sc1()
    test_dis(hparams)

# ckpt = "ARQ/AWGN/version_0/checkpoints/epoch=499-step=782000.ckpt"
# ckpt = 'logs/ARQ/AWGN/version_7/checkpoints/epoch=499-step=782000.ckpt'
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_addr", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--channel", type=int, default=8)
    parser.add_argument("--communication_channel", type=str, default=None)
    parser.add_argument("--hyperparameter", type=int, default=5)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--addr", type=str, default=None)
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()
    main(args)
#python test.py --ckpt_addr 'ARQ/AWGN/version_0/checkpoints/epoch=499-step=782000.ckpt' --batch_size 64 --device 0 --channel 8 --communication_channel 'awgn' --hyperparameter 5 --name 'version_0' --addr 'res/awgn/'
#python test.py --ckpt_addr 'ARQ/AWGN/version_0/checkpoints/epoch=499-step=782000.ckpt' --batch_size 64 --device 0 --channel 8 --communication_channel 'awgn' --hyperparameter 5 --name 'version_0' --addr 'res/rayl/'

#python testcheck.py --ckpt_addr 'logs/RAYL/v2.ckpt' --batch_size 64 --device 0 --channel 8 --communication_channel 'rayl' --hyperparameter 5 --name 'version_0' --addr 'res/rayl/'
