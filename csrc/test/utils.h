#pragma once

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

template <typename Clock = std::chrono::steady_clock> class Timer {
public:
  void start() { t_start = Clock::now(); }
  double elapsed() const {
    return std::chrono::duration<double>(Clock::now() - t_start).count();
  }

private:
  typename Clock::time_point t_start;
};

// 计时工具函数
template <typename Func, typename... Args>
double measure_time(Func func, Args &&...args) {
  auto start = std::chrono::high_resolution_clock::now();

  // 执行函数
  func(std::forward<Args>(args)...);

  // 等待CUDA操作完成
  torch::cuda::synchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

// 初始化随机数据
void initialize_data(std::vector<float> &A, std::vector<float> &B) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 10.0f);

  std::for_each(A.begin(), A.end(), [&](float &x) { x = dis(gen); });
  std::for_each(B.begin(), B.end(), [&](float &x) { x = dis(gen); });
}

// 使用 Tensor 自带的 operator<< 转换为字符串
std::string to_string(const torch::Tensor &tensor) {
  std::ostringstream oss;
  oss << tensor; // ← 这是 torch::Tensor 自带的能力！
  return oss.str();
}
