cmake_minimum_required(VERSION 3.18)
project(cuda_vec_add LANGUAGES CXX CUDA)

# 设置标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

# 🔧 设置 LibTorch 路径（修改为你自己的路径）
set(LIBTORCH ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch)
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH}")
find_package(Torch REQUIRED)

# Find files
file(GLOB BIND_SOURCES "binding/binding.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cu")

set(TEST_FILES
    test/test_vec_add_torch.cpp  # 主文件也包含进来
    test/test_mm.cpp
)

# 遍历所有测试文件，为每个文件创建可执行目标
foreach(TEST_FILE ${TEST_FILES})
# 从文件名提取目标名称（例如 test_mm.cpp -> test_mm）
  get_filename_component(TARGET_NAME ${TEST_FILE} NAME_WE)

# 创建可执行目标
  add_executable(${TARGET_NAME}
      ${TEST_FILE}
      ${BIND_SOURCES}
      ${KERNEL_SOURCES}
    )

# Link torch lib
  target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")

# 编译选项
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_89>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )

#  输出信息（调试用）
  message(STATUS "Target: ${TARGET_NAME}")
  message(STATUS "Sources: ${ALL_SOURCES}")
endforeach()
