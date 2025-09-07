#include "./binding.h"
#include "../kernel/api.cuh"
#include "../kernel/configs.cuh"

namespace Ops {

namespace Elemwise {

torch::Tensor vec_add(torch::Tensor A, torch::Tensor B) {
  // 1. Input check
  TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
  TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
  TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same size");
  TORCH_CHECK(A.dtype() == torch::kFloat, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat, "B must be float32");

  // 2. Prepare
  const int N = A.size(0);
  auto options = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
  auto C = torch::empty({N}, options);

  Kernel::Elemwise::launch_vec_add(A.data_ptr<float>(), B.data_ptr<float>(),
                                   C.data_ptr<float>(), N);

  return C;
}

} // namespace Elemwise

namespace Matmul {
torch::Tensor matmul(torch::Tensor M, torch::Tensor N) {
  // 1. Input check
  TORCH_CHECK(M.device().is_cuda(), "M must be on CUDA");
  TORCH_CHECK(N.device().is_cuda(), "N must be on CUDA");
  TORCH_CHECK(M.dtype() == torch::kFloat32, "M must be float32");
  TORCH_CHECK(N.dtype() == torch::kFloat32, "N must be float32");
  TORCH_CHECK(M.ndimension() == 2, "Tensor M must be 2-dim");
  TORCH_CHECK(N.ndimension() == 2, "Tensor M must be 2-dim");

  // 假设是方阵乘法（Width x Width）
  const int Width = M.size(0);
  TORCH_CHECK(M.size(1) == Width && N.size(0) == Width && N.size(1) == Width,
              "输入必须是 Width x Width 的方阵");

  torch::Tensor P = torch::empty({Width, Width}, M.options());

  // 3. Launch kernel
  Kernel::Matmul::launch_mat_mm(M.data_ptr<float>(), N.data_ptr<float>(),
                                P.data_ptr<float>(), Width);

  return P;
}
} // namespace Matmul

} // namespace Ops

// Binding for Python
// PYBIND11_MODULE(mylib, m) {
// m.doc() = "PyTorch extension for vec_add";
// m.def("vec_add", &vec_add, "Element-wise add using CUDA kernel");
//   m.def("matmul", &matmul, "CUDA matmul")
// }
