#include "./configs.cuh"

namespace Ops {

namespace Elemwise {

// run on CPU and GPU
__host__ __device__ __forceinline__ float add(float a, float b) {
	return a + b;
}

// Compute vector sum C_h = A_h + B_h
void vecAddHost(const float* A_h, const float* B_h, float* C_h, const int N) {
	for (int i = 0; i < N; ++i) {
		C_h[i] = add(A_h[i], B_h[i]);
	}
}

__global__ void vecAddKernel(const float* A, const float* B, float* C, const unsigned int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = add(A[i], B[i]);
	}
}

__device__ __forceinline__ float4 getFloat4(float* ptr, const int idx) {
    return *reinterpret_cast<float4*>(&ptr[idx]);
}

__global__ void vecAddKernelFloat4(float* A, float* B, float* C, const unsigned int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 4;

  if (idx < N) {
    float4 a = getFloat4(A, idx);
    float4 b = getFloat4(B, idx);
    float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

    if (idx + 0 < N) C[idx + 0] = c.x;
    if (idx + 1 < N) C[idx + 1] = c.y;
    if (idx + 2 < N) C[idx + 2] = c.z;
    if (idx + 3 < N) C[idx + 3] = c.w;
  }
}

void vecAddDevice(const float* A_h, const float* B_h, float* C_h, const int N, const bool if_float4=false) {
	int size = N * sizeof(float);
	float* A_d, *B_d, *C_d;

	// Allocate GPU memory
	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	// Copy to the GPU
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// launch the kernel function (a grid of threads)
	const unsigned int numThreadsPerBlock = 512;
  unsigned int numBlocks;
	// ceil(N/512)
  if (if_float4) {
    numBlocks = (N + 4 * numThreadsPerBlock - 1) / (4 * numThreadsPerBlock);
  } else {
    numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
  }
	vecAddKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);

	// Copy to the CPU
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	// Deallocate GPU memory
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

void launch_vec_add(const float* A, const float* B, float* C, unsigned int N) {
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;

  vecAddKernel<<<gridSize, blockSize>>>(A, B, C, N);
}

} // namespace Elemwise
} // namespace Ops
