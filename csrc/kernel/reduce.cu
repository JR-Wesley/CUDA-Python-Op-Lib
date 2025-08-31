#include "./configs.cuh"
 // dim3 block_size(BLOCK_SIZE);  // BLOCK_SIZE 是通过宏定义的某个数字
 // dim3 grid_size(CIEL(N, BLOCK_SIZE));
 // reduce_v1<<<grid_size, block_size>>>(d_x, d_y, N);

namespace Reduce {

__global__ void sum_naive(const float* input, float* output, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(output, input[idx]);
  }
}

// __global__ void sum_shmem(const float* input, float* output, int N) {
//     int tid = threadIdx.x;
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ float input_s[BLOCK_SIZE];
//
//     // 1. 搬运和线程数量(blockDim.x)相等的数据，到当前block的共享内存中
//     input_s[tid] = (idx < N) ? input[idx] : 0.0f;
//     __syncthreads();
//
//     // 2. 用1/2, 1/4, 1/8...的线程进行折半归约
//     for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//         if (tid < offset) {  // 2.折半归约
//             input_s[tid] += input_s[tid + offset];
//         }
//         __syncthreads();
//     }
//
//     // 3. 每个block的第一个线程将计算结果累加到输出中
//     if (tid == 0) atomicAdd(output, input_s[0]);
// }
//
//  __global__ void reduce_v3(float* d_x, float* d_y, const int N) {
//      __shared__ float s_y[32];  // 仅需要32个，因为一个block最多1024个线程，最多1024/32=32个warp
//
//      int idx = blockDim.x * blockIdx.x + threadIdx.x;
//      int warpId = threadIdx.x / warpSize;  // 当前线程属于哪个warp
//      int laneId = threadIdx.x % warpSize;  // 当前线程是warp中的第几个线程
//
//      float val = (idx < N) ? d_x[idx] : 0.0f;  // 搬运d_x[idx]到当前线程的寄存器中
//      #pragma unroll
//      for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
//          val += __shfl_down_sync(0xFFFFFFFF, val, offset);   // 在一个warp里折半归约
//      }
//
//      if (laneId == 0) s_y[warpId] = val;  // 每个warp里的第一个线程，负责将数据存储到shared mem中
//      __syncthreads();
//
//      if (warpId == 0) {  // 使用每个block中的第一个warp对s_y进行最后的归约
//          int warpNum = blockDim.x / warpSize;  // 每个block中的warp数量
//          val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
//          for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
//              val += __shfl_down_sync(0xFFFFFFFF, val, offset);
//          }
//          if (laneId == 0) atomicAdd(d_y, val);  // 使用此warp中的第一个线程，将结果累加到输出
//      }
//  }

}

