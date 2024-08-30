import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class RayleighChannel:
    def __init__(self):
        """
        创建一个瑞利衰落信道实例。
        """
        pass

    def forward(self, signal, snr_dB):
        """
        通过瑞利衰落信道传播信号，并添加相应的噪声。
        :param signal: 输入信号（PyTorch张量，位于GPU上）
        :param snr_dB: 信噪比（以分贝为单位）
        :return: 经过瑞利衰落处理的带噪声信号（PyTorch张量，位于GPU上）
        """
        device = signal.device
        snr_linear = 10 ** (snr_dB / 10)
        
        # 生成瑞利衰落系数
        real_part = torch.randn(signal.shape, device=device)
        imag_part = torch.randn(signal.shape, device=device)
        rayleigh_fading = torch.sqrt(real_part ** 2 + imag_part ** 2) / torch.sqrt(torch.tensor(2.0, device=device))
        
        # 应用衰落系数到信号
        faded_signal = signal * rayleigh_fading
        
        # 计算信号功率和噪声功率
        signal_power = torch.mean(faded_signal.abs() ** 2)
        noise_power = torch.sqrt(signal_power / snr_linear)
        noise = noise_power * torch.randn(signal.shape, device=device)
        
        # 添加噪声到衰落后的信号
        noisy_signal = faded_signal + noise
        return noisy_signal