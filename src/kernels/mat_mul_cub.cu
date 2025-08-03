#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" void launch_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,     // 注意：PyTorch 是 row-major，cublas 是 column-major
        A, K,
        &beta,
        C, N
    );

    cublasDestroy(handle);
}
