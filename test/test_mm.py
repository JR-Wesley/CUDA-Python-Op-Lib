import torch
from torch.utils.cpp_extension import load

# 动态编译并加载CUDA代码
matmul_ops = load(
    name="matmul_ops",  # 扩展名称
    sources=[
        "../csrc/binding/binding.cpp",  # 绑定代码
        "../csrc/kernel/mat_mul.cu"           # CUDA kernel
    ],
    # include_dirs=["../csrc/binding"],
                  # "../csrc/kernel"],  # 指定头文件目录（关键！）
    extra_cuda_cflags=["-O3"],  # 编译优化
    verbose=True  # 显示编译过程（可选）
)

# 测试代码
def test_matmul():
    M, K, N = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # 调用自定义kernel
    C_custom = matmul_ops.matmul(A, B)
    # 调用PyTorch内置函数
    C_torch = torch.matmul(A, B)
    
    # 验证结果
    error = torch.norm(C_custom - C_torch)
    print(f"误差: {error.item()}")
    print(f"相对误差: {error / torch.norm(C_torch).item()}")
    assert error < 1e-3, "结果不匹配！"

if __name__ == "__main__":
    test_matmul()
