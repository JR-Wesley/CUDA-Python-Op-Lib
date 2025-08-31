#pragma once

namespace Ops {

namespace Elemwise {

void vecAddHost(const float* A_h, const float* B_h, float* C_h, const int N);
void vecAddDevice(const float* A_h, const float* B_h, float* C_h, const int N, const bool if_float4=false);
void launch_vec_add(const float* A, const float* B, float* C, unsigned int N);

}

namespace Reduce {
void softmax();

}

namespace Matmul {

}

}
