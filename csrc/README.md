# 算子列表

## Elemwise 向量加

作为对比，实现了 CPU 形式和 CUDA 形式的向量加。可以看出 CUDA 可以把循环的算子通过 SIMT 并行化加速。

### Profiling

> Warning: nvprof is not supported on devices with compute capability 8.0 and higher.


## Softmax 


# Build

Makefile is for naive version.
CMakeLists is for torch
