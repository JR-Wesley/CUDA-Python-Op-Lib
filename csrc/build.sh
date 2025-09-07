mkdir -p build && cd build
# 如果是通过Pytorch
# cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') -DTORCH_CUDA_ARCH_LIST="8.9" -L | grep Python
cmake .. -DCMAKE_PREFIX_PATH="/home/mikasa/.venvs/cuda_ops/lib/python3.13/site-packages/torch/share/cmake"
#下载的单独Libtorch
# cmake -DCMAKE_PREFIX_PATH=<LIBTORCH_ROOT> ..
make -j8
