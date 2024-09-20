import torch
from torch import nn
from torch.autograd import Function

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        # 确保 b 在 inputs 的设备上
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
    def __init__(self,
                ch,
                inverse=False,
                beta_min=1e-6,
                gamma_init=.1,
                reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = nn.Parameter(torch.tensor([reparam_offset]))

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # 创建 beta 参数
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # 创建 gamma 参数
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        device = inputs.device  # 动态获取输入张量的设备

        # 确保所有相关的张量都在相同的设备上
        beta = LowerBound.apply(self.beta, self.beta_bound).to(device)
        beta = beta ** 2 - self.pedestal.to(device)

        gamma = LowerBound.apply(self.gamma, self.gamma_bound).to(device)
        gamma = gamma ** 2 - self.pedestal.to(device)
        gamma = gamma.view(inputs.size(1), inputs.size(1), 1, 1).to(device)

        # 归一化计算
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)
        
        # 应用归一化
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        return outputs