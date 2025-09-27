#include "../utility/hip_utility.hpp"
#include "../utility/ops_gemm.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>

#define NAIVE_IMPLEMENTATION 1
// Naive implementation of softmax kernel
__global__ void online_softmax_kernel_naive(float *input, float *output, int M,
                                            int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M) {

    // Pass one: Find max value in the row
    float max_val = -INFINITY;
    for (int col = 0; col < N; col++) {
      float element = input[row * N + col];
      max_val = fmaxf(max_val, element);
    }

    // Pass two: Compute sum of exp(x - max_val)
    float sum_exp = 0.0f;
    for (int col = 0; col < N; col++) {
      sum_exp += expf(input[row * N + col] - max_val);
    }

    // Pass three: Compute softmax values
    for (int col = 0; col < N; col++) {
      output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
    }
  }
}

__global__ void online_softmax_kernel_optimized(float *input, float *output,
                                                int M, int N) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < M) {
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Pass one and two combined - find max and compute sum of exponentials
    for (int col = 0; col < N; col++) {
      float element = input[row * N + col];
      if (element > max_val) {
        // When we find a new maximum, we need to rescale our accumulated sum.
        // exp(x - new_max) = exp(x - old_max) * exp(old_max - new_max)
        // So we multiply our current sum by exp(old_max - new_max) to adjust
        // for the new baseline.
        // source: https://dev.to/lewis_won/online-softmax-by-hand-4h13#safe
        float old_max = max_val;
        max_val = element;
        sum_exp = sum_exp * expf(old_max - max_val);
      }
      sum_exp += expf(element - max_val);
    }

    // Pass three - compute softmax values
    for (int col = 0; col < N; col++) {
      output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
    }
  }
}

void cpu_softmax(float *input, float *output, int M, int N);

int main() {

  int M = 2048;
  int N = 2048;

  float *h_input = new float[M * N];
  float *h_output_gpu = new float[M * N];
  float *h_output_cpu = new float[M * N];

  std::cout << "Initializing input matrix (" << M << "x" << N << ")...\n";
  initialize_matrix(h_input, M, N);

  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, M * N * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_input, h_input, M * N * sizeof(float),
                      hipMemcpyHostToDevice));

  /*
  Kernel One - Naive Implementation - Small Matrices
  For naive implementation, we can assume
  1. Three passes are okay
  2. Granularity of work is one row per thread
  */

#if NAIVE_IMPLEMENTATION
  int threadsPerBlock = 256;
  int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(online_softmax_kernel_naive, dim3(blocksPerGrid),
                     dim3(threadsPerBlock), 0, 0, d_input, d_output, M, N);
#else
  /*
  Kernel Two
  For optimized implementation, we can assume
  1. Reduce three passes to two passes
  2. Granularity of work is one row per threadBlock
  */
  int threadsPerBlock = 1024;
  int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(online_softmax_kernel_optimized, dim3(blocksPerGrid),
                     dim3(threadsPerBlock), 0, 0, d_input, d_output, M, N);
#endif

  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_output_gpu, d_output, M * N * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Run CPU validation
  std::cout << "Running CPU validation...\n";
  cpu_softmax(h_input, h_output_cpu, M, N);

  // Compare results
  std::cout << "\n";
  compare_float_matrices(h_output_gpu, h_output_cpu, M, N);

  // sanity check
  std::cout << "\nVerifying row sums (should be ~1.0):\n";
  for (int i = 0; i < std::min(20, M); i++) {
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
      sum += h_output_gpu[i * N + j];
    }
    std::cout << "Row " << i << " sum: " << sum << std::endl;
  }

  // Cleanup
  delete[] h_input;
  delete[] h_output_gpu;
  delete[] h_output_cpu;
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  return 0;
}

void cpu_softmax(float *input, float *output, int M, int N) {
  for (int row = 0; row < M; row++) {
    // Pass one: Find max value in the row
    float max_val = -INFINITY;
    for (int col = 0; col < N; col++) {
      float element = input[row * N + col];
      if (element > max_val) {
        max_val = element;
      }
    }

    // Pass two: Compute sum of exp(x - max_val)
    float sum_exp = 0.0f;
    for (int col = 0; col < N; col++) {
      sum_exp += expf(input[row * N + col] - max_val);
    }

    // Pass three: Compute softmax values
    for (int col = 0; col < N; col++) {
      output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
    }
  }
}
