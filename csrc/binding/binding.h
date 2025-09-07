#pragma once

// used for python extension
// #include <torch/extension.h>
#include <torch/torch.h>

namespace Ops {

namespace Elemwise {
torch::Tensor vec_add(torch::Tensor A, torch::Tensor B);
}

namespace Matmul {
torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
} // namespace Matmul

} // namespace Ops
