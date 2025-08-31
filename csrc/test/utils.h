#pragma once

#include <algorithm>
#include <random>
#include <chrono>
#include <vector>

torch::Tensor vec_add(torch::Tensor A, torch::Tensor B);

template<typename Clock = std::chrono::steady_clock>
class Timer {
public:
    void start() { t_start = Clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t_start).count();
    }
private:
    typename Clock::time_point t_start;
};

// 初始化随机数据
void initialize_data(std::vector<float>& A, std::vector<float>& B) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    std::for_each(A.begin(), A.end(), [&](float& x) { x = dis(gen); });
    std::for_each(B.begin(), B.end(), [&](float& x) { x = dis(gen); });
}

