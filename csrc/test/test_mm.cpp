#include "../binding/binding.h"
#include "./utils.h"
#include <print>

int main() {
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available! Exiting..." << std::endl;
    return 1;
  }

  // 设置随机种子，确保结果可复现
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);

  // 测试不同大小的矩阵
  std::vector<std::pair<int, int>> sizes = {
      {64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}};
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  // 每个测试运行的迭代次数
  constexpr int iterations = 100;

  for (auto &size : sizes) {
    int m = size.first;
    int n = size.second;

    auto a = torch::randn({m, n}, options);
    auto b = torch::randn({n, m}, options);

    // Warmup
    auto custom_result = Ops::Matmul::matmul(a, b);
    auto torch_result = torch::matmul(a, b);

    // Veri
    bool is_close = torch::allclose(custom_result, torch_result, 1e-5, 1e-3);
    std::println("Matrix size: {} x {}. Verify {}", m, n,
                 (is_close ? "pass" : "failed"));

    if (!is_close) {
      std::println("  Custom result: {} vs Torch result: {}",
                   to_string(custom_result.slice(0, 0, 5)),
                   to_string(torch_result.slice(0, 0, 5)));
    }

    // 测试自定义CUDA算子
    double custom_total_time = measure_time([&]() {
      for (int i = 0; i < iterations; ++i) {
        Ops::Matmul::matmul(a, b);
      }
    });
    double custom_avg_time = custom_total_time / iterations;

    // 测试PyTorch内置算子
    double torch_total_time = measure_time([&]() {
      for (int i = 0; i < iterations; ++i) {
        torch::matmul(a, b);
      }
    });
    double torch_avg_time = torch_total_time / iterations;

    std::println("  Matrix size: {} x {}", m, n);
    std::println("  CUDA kernel: {} ms/times", custom_avg_time * 1000);
    std::println("  Torch kernel: {} ms/times", torch_avg_time * 1000);
    std::println("Acceleration: {:.4f}\n", torch_avg_time / custom_avg_time);
  }

  return 0;
}
