#pragma once

namespace Kernel {

namespace Elemwise {

void launch_vec_add(const float* A, const float* B, float* C, unsigned int N);

} // namespace Elemwise

namespace Reduce {
void softmax();

}

namespace Matmul {
void launch_mat_mm(const float* M, const float* N, float* P, const int Width);

} //namespace Matmul

} // namespace Kernel
