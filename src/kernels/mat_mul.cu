#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

// (N, N) * (N, N)
__global__ void matMulKernel(const float* A, const float* B, float* C, unsigned int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < N) && (col < N)) {
		float sum = 0.0f;
		for (int i = 0; i < N; ++i) {
			sum += A[row * N + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}

