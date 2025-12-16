#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono>

#include <hip/hip_runtime.h>
#include "../include/HIP/HIP_Utils.hpp"

void matrixMultiplicationOnCPU(float *A, float *B, float *C, int N);
__global__ void matrixMultiplicationKernel(float *A, float *B, float *C, int N);
void initialize(float *A, float *B, float *C, int N);
void verifyResults(float *h_C, float *C, int N);

int main() {
  int N = 2048;

  // Allocate host memory
  float *A = new float[N * N]; // 2048 * 2048 * 4 = 16MB
  float *B = new float[N * N]; // 2048 * 2048 * 4 = 16MB
  float *C = new float[N * N]; // 2048 * 2048 * 4 = 16MB
  float *h_C = new float[N * N]; // 2048 * 2048 * 4 = 16MB

  // Initialize matrices with random values
  initialize(A, B, C, N);
  
  // CPU matrix multiplication
  matrixMultiplicationOnCPU(A, B, C, N);

  //Allocate device memory
  float *d_A; float *d_B; float *d_C;
  //HIP_CHECK(hipMallocManaged(&d_A, N * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_A, N * N * sizeof(float))); // 2048 * 2048 * 4 = 16MB
  HIP_CHECK(hipMalloc(&d_B, N * N * sizeof(float))); // 2048 * 2048 * 4 = 16MB
  HIP_CHECK(hipMalloc(&d_C, N * N * sizeof(float))); // 2048 * 2048 * 4 = 16MB 

  // Copy data to device
  HIP_CHECK(hipMemcpy(d_A, A, N * N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, B, N * N * sizeof(float), hipMemcpyHostToDevice));

  // Kernel launch configuration
  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  // GPU matrix multiplication
  // measure time using hipEvent_t
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, 0));
  hipLaunchKernelGGL(matrixMultiplicationKernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N);
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));

  HIP_CHECK(hipMemcpy(h_C, d_C, N * N * sizeof(float), hipMemcpyDeviceToHost));

  float time;
  HIP_CHECK(hipEventElapsedTime(&time, start, stop));
  std::cout << "GPU time: " << time << " ms" << std::endl;

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  // Verify results
  verifyResults(h_C, C, N);

  // Cleanup
  HIP_CHECK(hipFree(d_A));   HIP_CHECK(hipFree(d_B));   HIP_CHECK(hipFree(d_C));
  delete[] A;   delete[] B;   delete[] C;   delete[] h_C;
  return 0;
}

void initialize(float *A, float *B, float *C, int N) {
  
  srand(time(0));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      B[i * N + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      C[i * N + j] = 0.0f;
    }
  }
}
void matrixMultiplicationOnCPU(float *A, float *B, float *C, int N) {
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0f;
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "CPU time: " << elapsed.count() * 1000 << " ms" << std::endl;
}
__global__ void matrixMultiplicationKernel(float *A, float *B, float *C,
                                           int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < N && y < N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += A[y * N + i] * B[i * N + x];
    }
    C[y * N + x] = sum;
  }
}
void verifyResults(float *h_C, float *C, int N) {
  float epsilon = 1e-3;
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(h_C[i * N + j] - C[i * N + j]) > epsilon) {
        std::cerr << std::fixed << std::setprecision(5)
            << "Results do not match at index (" << i << ", " << j
            << ")! "
            << "Host: " << C[i * N + j] << " Device: " << h_C[i * N + j]
            << std::endl;
        return;
      }
    }
  }
  std::cout << "Results match!" << std::endl;
}
