cmake_minimum_required(VERSION 3.18)
project(cuda_vec_add LANGUAGES CXX CUDA)

# 设置标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# # 🔧 设置 LibTorch 路径（修改为你自己的路径）
# set(LIBTORCH ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch)
# list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH}")
find_package(Torch REQUIRED)

file(GLOB BIND_SOURCES "binding/binding.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cu")

set(ALL_SOURCES
    ${BIND_SOURCES}
    ${KERNEL_SOURCES}
    test/test_vec_add_torch.cpp  # 主文件也包含进来
)

set(TARGET_NAME "vec_add_test")

add_executable(${TARGET_NAME} ${ALL_SOURCES})

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")

# 编译选项
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_89>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

#  输出信息（调试用）
message(STATUS "Target: ${TARGET_NAME}")
message(STATUS "Sources: ${ALL_SOURCES}")
