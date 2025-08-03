import torch
from compile_dev import cuda_ops

x = torch.randn(100, 200, device='cuda')
w = torch.randn(200, 300, device='cuda')
y = cuda_ops['matmul'].matmul_forward(x, w)
print(y.shape)  # torch.Size([100, 300])

