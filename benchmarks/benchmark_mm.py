import torch
import time
from compile_dev import cuda_ops

def benchmark():
    x = torch.randn(2048, 2048, device='cuda')
    w = torch.randn(2048, 2048, device='cuda')

    # Warm up
    for _ in range(5):
        cuda_ops['matmul'].matmul_forward(x, w)
    torch.cuda.synchronize()

    # Test speed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        y = cuda_ops['matmul'].matmul_forward(x, w)
    end.record()
    torch.cuda.synchronize()

    print(f"Operator matmul consumes: {start.elapsed_time(end)/10:.4f} ms")

if __name__ == "__main__":
    benchmark()
