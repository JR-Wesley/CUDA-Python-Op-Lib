try:
    from compile_dev import cuda_ops
except ImportError:
    print("Auto building...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "compile_dev.py"])

import torch

def test_matmul():
    x = torch.randn(10, 20, device='cuda')
    w = torch.randn(20, 30, device='cuda')
    y = cuda_ops['matmul'].matmul_forward(x, w)
    y_torch = x @ w
    assert torch.allclose(y, y_torch, rtol=1e-4), "Test failed!"
    print("Matmul test passed")

if __name__ == "__main__":
    test_matmul()
