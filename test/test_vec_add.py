import torch
from torch.utils.cpp_extension import load

# 动态编译并加载 CUDA 扩展
cuda_vec_add = load(
    name="vec_add",
    sources=["../csrc/bindings/binding.cpp"],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

# 创建测试数据
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = torch.empty_like(a)

# 直接调用你的 CUDA kernel
cuda_vec_add.ec_add(a, b, c)  # 假设 kernel 名为 ec_add

# 验证
print("Result (first 5):", c[:5])
print("Correct:", torch.allclose(c, a + b))
