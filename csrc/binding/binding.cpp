#include "./binding.h"

namespace Ops {

namespace Elemwise {

torch::Tensor vec_add(torch::Tensor A, torch::Tensor B) {
    // 1. 输入检查
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same size");
    TORCH_CHECK(A.dtype() == torch::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat, "B must be float32");

    auto N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
    auto C = torch::empty({N}, options);

    // 2. 获取原始指针
    const float* ptr_A = A.data_ptr<float>();
    const float* ptr_B = B.data_ptr<float>();
    float* ptr_C = C.data_ptr<float>();

    // 3. 调用 CUDA 函数
    Elemwise::launch_vec_add(ptr_A, ptr_B, ptr_C, N);

    return C;
}

}


}

// 绑定模块
// PYBIND11_MODULE(mylib, m) {
//     m.doc() = "PyTorch extension for vec_add";
//     m.def("vec_add", &vec_add, "Element-wise add using CUDA kernel");
// }
