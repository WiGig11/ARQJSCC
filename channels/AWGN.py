import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class AWGNChannel:
    def __init__(self):
        """
        创建一个 AWGN 信道实例。
        """

    def forward(self, signal, snr_dB):
        """
        向信号添加 AWGN 噪声（实部是白噪声）。
        :param signal: 输入信号（PyTorch张量，位于GPU上）
        :param snr_dB: 信噪比（以分贝为单位）
        :return: 带噪声的信号（PyTorch张量，位于GPU上）
        """
        #print(signal.shape)
        device = signal.device
        snr_linear = 10 ** (snr_dB / 10)
        signal_power = torch.mean(torch.abs(signal) ** 2)    # 计算信号功率
        noise_power = torch.sqrt(signal_power / snr_linear)# 计算噪声功率
        noise_power = noise_power.to(signal.device)
        noise_signal = torch.randn(*signal.shape)# 生成实部是白噪声的噪声
        #noise_signal = noise_signal.to('cuda:1')
        noise_signal = noise_signal.to(device)
        noise = noise_power * noise_signal
        noisy_signal = signal + noise# 将噪声添加到信号上
        return noisy_signal# 返回带噪声的信号，并将其移到GPU上
