// src/bindings/mat_mul_cub.cpp

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// 假设你在 .cu 文件中定义了这个 kernel
void launch_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream);

torch::Tensor matmul_cuda_forward(torch::Tensor input, torch::Tensor weight) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "Weight must be float32");

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(1);

    auto output = torch::empty({M, N}, input.options());

    // ✅ 正确获取当前 stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_matmul_kernel(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_cuda_forward, "CUDA matmul forward");
}
