// #include <iostream>
// #include <cassert>
// #include <algorithm>
// #include <numeric>
//
// //  LibTorch 实现
// torch::Tensor softmax_torch(const torch::Tensor& input) {
//     // 确保输入是 1D
//     TORCH_CHECK(input.dim() == 1, "Input must be 1D");
//     return torch::softmax(input, /*dim=*/0);
// }
//
// //  纯 C++ 实现
// std::vector<float> softmax_cpp(const std::vector<float>& input) {
//     std::vector<float> output = input;
//
//     // Step 1: 减去最大值（数值稳定）
//     float max_val = *std::max_element(output.begin(), output.end());
//     std::transform(output.begin(), output.end(), output.begin(),
//                    [max_val](float x) { return std::exp(x - max_val); });
//
//     // Step 2: 归一化
//     float sum_val = std::accumulate(output.begin(), output.end(), 0.0f);
//     std::transform(output.begin(), output.end(), output.begin(),
//                    [sum_val](float x) { return x / sum_val; });
//
//     return output;
// }
//
// int main() {
//     const int N = 5;
//     // 生成测试数据
//     std::vector<float> host_input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
//
//     // 🔹 LibTorch 实现
//     auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//     auto input_tensor = torch::from_blob(host_input.data(), {N}, torch::kFloat32).to(options);
//
//     auto output_torch = softmax_torch(input_tensor);
//
//     // 复制回 CPU 以便比较
//     auto output_torch_cpu = output_torch.cpu();
//
//     // 🔹 纯 C++ 实现
//     auto output_cpp = softmax_cpp(host_input);
//
//     // 🔍 对比结果
//     std::cout << "Input: ";
//     for (float x : host_input) std::cout << x << " ";
//     std::cout << "\n";
//
//     std::cout << "Torch: ";
//     for (int i = 0; i < N; ++i) {
//         std::cout << output_torch_cpu[i].item<float>() << " ";
//     }
//     std::cout << "\n";
//
//     std::cout << "C++  : ";
//     for (float x : output_cpp) std::cout << x << " ";
//     std::cout << "\n";
//
//     // ✅ 验证是否一致
//     bool all_close = true;
//     for (int i = 0; i < N; ++i) {
//         float diff = std::abs(output_torch_cpu[i].item<float>() - output_cpp[i]);
//         if (diff > 1e-5) {
//             all_close = false;
//             break;
//         }
//     }
//
//     std::cout << "Result: " << (all_close ? "✅ Pass" : "❌ Fail") << "\n";
//
//     return 0;
// }
int main() {
    return 0;
}
