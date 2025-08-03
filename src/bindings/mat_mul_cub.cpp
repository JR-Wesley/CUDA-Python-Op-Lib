#include <torch/extension.h>
#include <vector>

void launch_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
);

torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B) {
    // Input check
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "Invalid matrix dimensions");

    auto C = torch::empty({M, N}, A.options());

    launch_gemm_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_cuda_forward, "CUDA Matrix Multiplication (MxK @ KxN)");
}
