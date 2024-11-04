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
from tqdm import tqdm
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


def test_jscc(hparams):
    # model
    in_channel = hparams.channel
    encoder = Encoder(out_channels=in_channel)
    decoder = Decoder(in_channels=in_channel)
    discriminator = MultiScaleDiscriminator()
    addr = hparams.addr
    name = hparams.name
    if "awgn" in hparams.communication_channel:
        channel = AWGNChannel()
        addr = os.path.join(addr, "awgn")
        addr = os.path.join(addr, name)
        if not os.path.exists(addr):
            os.makedirs(addr)
    elif "rayl" in hparams.communication_channel:
        channel = RayleighChannel()
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
    hp = hparams.hyperparameter
    model = DeepJSCCARQ.load_from_checkpoint(
        ckpt,
        map_location=device,  # cuda:0
        encoder=encoder,
        decoder=decoder,
        loss_module_D=Least_Square_Loss(),
        loss_module_G=MSEImageLoss(),
        channel=channel,
        discriminator=discriminator,
        hyperparameter=hp,
        lr_scheduler_type="step",
        lr_D=5e-6,
        lr_G=1e-4,
    )

    with open(addr + "/latest.txt", "w") as f:
        print(model, file=f)

    # eval mode
    model = model.to(device)
    # data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    valset = torchvision.datasets.CIFAR10(
        root="./cifar_data", train=False, download=True, transform=transform
    )
    batch_size = hparams.batch_size  # 加载64张图像
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=15
    )  # 注意关闭shuffle
    ssimer = SSIM(data_range=1.0, reduction="none")  # 计算每张图像的SSIM
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


def main(hparams):
    # test_feature_sc1()
    test_jscc(hparams)

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