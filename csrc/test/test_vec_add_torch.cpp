#include <iostream>
#include "../binding/binding.h"

int main() {
    const int N = 1000;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, 0);

    auto A = torch::randn({N}, options);
    auto B = torch::randn({N}, options);

    std::cout << "Testing vec_add binding on " << N << " elements...\n";

    // 调用 binding 中的 vec_add 函数（和 Python 调用的是同一个）
    auto C = Ops::Elemwise::vec_add(A, B);

    torch::cuda::synchronize();

    std::cout << "Result (first 5): " << C.slice(0, 0, 5) << "\n";

    // 验证
    auto expected = A + B;
    bool correct = torch::allclose(C, expected);
    std::cout << "Correct: " << (correct ? "✅ Yes" : "❌ No") << "\n";

    return 0;
}
