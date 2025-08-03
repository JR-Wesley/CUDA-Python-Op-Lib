#include <cuda_runtime.h>


__global__ void rgb2gray_kernel ( unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* gray, unsigned int width, unsigned int height) {
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	if ( row < height && col < width) {
		unsigned int i = row*width + col;
		gray[i] = r[i]*3/10 + g[i]*6/10 + b[i]*1/10;
	}
}


int main() {
// const unsigned int width = 32;
// const unsigned int height = 32;

}

