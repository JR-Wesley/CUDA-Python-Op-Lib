#include "./configs.cuh"

namespace Kernel {
namespace Matmul {

// (N, N) * (N, N)
__global__ void matMulNaive(const float* M, const float* N, float* P, const int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < Width) && (col < Width)) {
		float sum = 0.0f;
		for (int k = 0; k < Width; ++k) {
			sum += M[row * Width + k] * N[k * Width + col];
		}
		P[row * Width + col] = sum;
	}
}

#define TILE_WIDTH 16
__global__ void matMulTile(const float *M, const float *N, float *P, const int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // int bx
}

void launch_mat_mm(const float *M, const float *N, float *P, const int Width) {
  SETUP_KERNEL_CONFIG(16, 16, Width, Width);
  LAUNCH_KERNEL(matMulNaive, M, N, P, Width);
}

} //namespace Matmul
} // namespace Kernel
