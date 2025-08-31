mkdir -p build && cd build
# 如果是通过Pytorch
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') -DTORCH_CUDA_ARCH_LIST="8.9" -L | grep Python
#下载的单独Libtorch
# cmake -DCMAKE_PREFIX_PATH=<LIBTORCH_ROOT> ..
make -j8
