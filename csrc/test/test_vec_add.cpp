#include "../kernel/api.cuh"
#include "../kernel/configs.cuh"
// #include "./utils.h"

int main(int argc, char **argv) {
  cudaDeviceSynchronize();

  // Allocate host memory
  const unsigned int N = (1 << 26);
  std::vector<float> A(N), B(N), C(N);
  initialize_data(A, B);
  printf("Vector size: %d, floats (%.2f MB)\n", N,
         N * sizeof(float) / (1024.0 * 1024.0));

  Timer<> timer;

  // --- Host Test ---
  timer.start();
  Ops::Elemwise::vecAddHost(A.data(), B.data(), C.data(), N);
  double time_host = timer.elapsed();

  // --- Device Test (default) ---
  timer.start();
  Ops::Elemwise::vecAddDevice(A.data(), B.data(), C.data(), N);
  double time_device = timer.elapsed();

  // --- Device Test (float4) ---
  timer.start();
  Ops::Elemwise::vecAddDevice(A.data(), B.data(), C.data(), N, true);
  double time_device_float4 = timer.elapsed();

  // Note that the time also counts memory copy.
  printf("\n📊 Performance Results:\n");
  printf("Host (CPU) execution time:      %.4f seconds\n", time_host);
  printf("Device (GPU) execution time:    %.4f seconds\n", time_device);
  printf("Device (GPU, float4) time:      %.4f seconds\n", time_device_float4);

  if (time_host > 1e-6) {
    printf("Speedup (vs host):              %.4f\n", time_device / time_host);
    printf("Speedup (float4 vs default):    %.4f\n",
           time_device_float4 / time_device);
  }

  return 0;
}
