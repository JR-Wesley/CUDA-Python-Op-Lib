# # CMakeLists.txt
#
# cmake_minimum_required(VERSION 3.18)  # Ensure CUDA support
# project(VecAddTest LANGUAGES CXX CUDA)
# # Optional: Set C++ standard
# set_property(TARGET test_vec_add PROPERTY CXX_STANDARD 17)
#
# # Find LibTorch
# # You need to set TORCH_ROOT to your LibTorch installation path
# set(TORCH_ROOT "/home/yourname/libtorch" CACHE PATH "Path to LibTorch")
#
# # Find LibTorch package
# find_package(Torch REQUIRED PATHS "${TORCH_ROOT}" NO_DEFAULT_PATH)
# if(NOT Torch_FOUND)
#     message(FATAL_ERROR "Torch not found. Set TORCH_ROOT correctly.")
# endif()
# include_directories(include ${TORCH_INCLUDE_DIRS})
#
# # 🧱 创建一个静态库，包含所有 src/ 下的 .cu 文件
# file(GLOB CUDA_SOURCES "src/kernel/*.cu")
# add_library(cuda_kernels STATIC ${CUDA_SOURCES})
#
# # 设置 CUDA 标准
# set_property(TARGET cuda_kernels PROPERTY CUDA_STANDARD 14)
# # 确保链接 CUDA runtime
# target_link_libraries(cuda_kernels PRIVATE CUDA::cudart)
#
# enable_testing()
#
# # 使用宏避免重复代码
# macro(add_test_executable test_name source_file)
#     add_executable(${test_name} ${source_file})
#     target_link_libraries(${test_name} PRIVATE cuda_kernels ${TORCH_LIBRARIES})
#     set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
#     add_test(NAME ${test_name} COMMAND ${test_name})
# endmacro()
#
# # 添加每个测试
# add_test_executable(test_vec_add test/cpp/test_vec_add.cpp)
# add_test_executable(test_vec_add_torch test/cpp/test_vec_add_torch.cpp)
# # add_test_executable(test_vec_mul test/test_vec_mul.cpp)
# # add_test_executable(test_all test/test_all.cpp)
#
