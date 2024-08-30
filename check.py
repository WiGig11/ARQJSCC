# test_torch_cuda.py
import torch

if torch.cuda.is_available():
    print("CUDA is available! :)")
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available. :(")

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 创建一个随机的 tensor
    tensor = torch.randn(3, 3)  # 创建一个3x3的随机张量

    # 将 tensor 移到 CUDA 设备上
    tensor_cuda = tensor.to('cuda')

    print("Tensor on CUDA:", tensor_cuda)
else:
    print("CUDA is not available. Please check your installation.")
