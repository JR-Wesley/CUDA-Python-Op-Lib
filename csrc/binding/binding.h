// used for python extension
// #include <torch/extension.h>
#include <torch/torch.h>
#include "../kernel/api.cuh"

namespace Ops {
namespace Elemwise {
    torch::Tensor vec_add(torch::Tensor A, torch::Tensor B);
}
}

