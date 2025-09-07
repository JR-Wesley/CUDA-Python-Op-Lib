#include "../binding/binding.h"
#include "./utils.h"
#include <print>

int main() {
  const int N = (1 << 24);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  auto A = torch::randn({N}, options);
  auto B = torch::randn({N}, options);

  std::println("Testing vec_add binding on {} elements...", N);

  // 调用 binding 中的 vec_add 函数（和 Python 调用的是同一个）
  auto C = Ops::Elemwise::vec_add(A, B);

  torch::cuda::synchronize();

  std::println("Testing vec_add binding on {} elements...", N);
  std::println("Result (first 5): {}", to_string(C.slice(0, 0, 5)));

  // 验证
  auto expected = A + B;
  bool correct = torch::allclose(C, expected);
  std::println("Correct: {}", (correct ? "✅ Yes" : "❌ No"));

  return 0;
}
