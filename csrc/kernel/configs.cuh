#pragma once

#include "./exception.cuh"
#include <cuda_runtime.h>

// 配置核函数启动参数（blockDim和gridDim）
// 参数：
// - block_x, block_y: 线程块的x和y维度大小
// - width, height: 问题维度（用于计算grid大小）
#define SETUP_KERNEL_CONFIG(block_x, block_y, width, height) \
    dim3 blockDim((block_x), (block_y)); \
    dim3 gridDim( \
        (width + blockDim.x - 1) / blockDim.x, \
        (height + blockDim.y - 1) / blockDim.y \
    )

// 启动核函数并检查错误
// 参数：
// - kernel: 要启动的CUDA核函数名
// - ...: 核函数的输入参数（可变参数）
#define LAUNCH_KERNEL(kernel, ...) \
    do { \
        kernel<<<gridDim, blockDim>>>(__VA_ARGS__); \
        cudaError_t e = cudaGetLastError(); \
        if (e != cudaSuccess) { \
            EPException cuda_exception("CUDA Error", __FILE__, __LINE__, cudaGetErrorString(e)); \
            fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, cuda_exception.what()); \
            throw cuda_exception; \
        } \
    } while (0)

// 带CUDA流的扩展版本（支持异步执行）
#define SETUP_KERNEL_CONFIG_WITH_STREAM(block_x, block_y, width, height, stream) \
    dim3 blockDim((block_x), (block_y)); \
    dim3 gridDim( \
        (width + blockDim.x - 1) / blockDim.x, \
        (height + blockDim.y - 1) / blockDim.y \
    ); \
    cudaStream_t __launch_stream = (stream)

#define LAUNCH_KERNEL_WITH_STREAM(kernel, ...) \
    do { \
        kernel<<<gridDim, blockDim, 0, __launch_stream>>>(__VA_ARGS__); \
        cudaError_t e = cudaGetLastError(); \
        if (e != cudaSuccess) { \
            EPException cuda_exception("CUDA Error", __FILE__, __LINE__, cudaGetErrorString(e)); \
            fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, cuda_exception.what()); \
            throw cuda_exception; \
        } \
    } while (0)

