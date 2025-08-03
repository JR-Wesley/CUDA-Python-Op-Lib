#include <ctime>
#include <cuda_runtime.h>
#include <stdio.h>

// run on CPU and GPU
__host__ __device__ float add(float a, float b) {
	return a + b;
}

// Compute vector sum C_h = A_h + B_h
void vecAddHost(float* A_h, float* B_h, float* C_h, int N) {
	for (int i = 0; i < N; ++i) {
		C_h[i] = add(A_h[i], B_h[i]);
	}
}

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = add(A[i], B[i]);
	}
}

void vecAddDevice(float* A_h, float* B_h, float* C_h, int N) {
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
	// ceil(N/512)
	const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
	vecAddKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);

	// Copy to the CPU
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	// Deallocate GPU memory
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}


int main(int argc, char** argv) {
	
	cudaDeviceSynchronize();

    // Allocate host memory
	int N = (1 << 25);
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));
	for (unsigned i = 0; i < N; ++i) {
		A[i] = rand();
		B[i] = rand();
	}

	clock_t start_host = clock();
	vecAddHost(A, B, C, N);
	clock_t end_host = clock();
	double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC;

	clock_t start_device = clock();
	vecAddDevice(A, B, C, N);
	clock_t end_device = clock();
	double time_device = (double)(end_device - start_device) / CLOCKS_PER_SEC;

	printf("Time taken for host: %f seconds\n", time_host);
	printf("Time taken for device: %f seconds", time_device);

	return 0;
}
