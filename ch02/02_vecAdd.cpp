#include <cstdlib> // rand()
#include <cmath>   // fabs()
#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
    }                                                                          \
  }

void initializeAB(float *A, float *B, int n);
void addOnHost(float *A, float *B, float *C, int n);
void verifyResults(float *C_h, float *C_d, int n);

__global__ void addOnDevice(float *A, float *B, float *C, int n) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < n) {
    C[globalThreadId] = A[globalThreadId] + B[globalThreadId];
  }
}

int main() {
  int N = 1000000; // one million numbers

  float *A_h, *B_h, *C_h;
  A_h = new float[N];
  B_h = new float[N];
  C_h = new float[N];

  initializeAB(A_h, B_h, N);

  // std::cout << "Initialization finished\n";

  float *A_d, *B_d, *C_d;
  HIP_CHECK(hipMalloc(&A_d, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&B_d, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&C_d, N * sizeof(float)));

  HIP_CHECK(hipMemcpy(A_d, A_h, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, N * sizeof(float), hipMemcpyHostToDevice));

  // std::cout << "Memory copy finished\n";

  int blockSize = 256; // number of threads per block
  int numBlocks = (N + blockSize - 1) / blockSize; // number of blocks per grid

  // compute on GPU
  hipLaunchKernelGGL(addOnDevice, dim3(numBlocks), dim3(blockSize), 0, 0, A_d,
                     B_d, C_d, N);
  // compute on CPU
  addOnHost(A_h, B_h, C_h, N);

  HIP_CHECK(hipDeviceSynchronize()); // making sure kernel execution is finished
  HIP_CHECK(hipMemcpy(C_h, C_d, N * sizeof(float), hipMemcpyDeviceToHost));

  verifyResults(C_h, C_d, N);

  // free memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  delete[] A_h;
  delete[] B_h;
  delete[] C_h;
}

void initializeAB(float *A, float *B, int n) {
  for (int i = 0; i < n; i++) {
    A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}
void addOnHost(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    C[i] = A[i] + B[i];
  }
}

void verifyResults(float *C_h, float *C_d, int n) {
  for (int i = 0; i < n; i++) {
    float epsilon = 1e-5;
    if (fabs(C_h[i] - C_d[i]) > epsilon) {
      std::cerr << "Results do not match at index " << i << "! Host: " << C_h[i]
                << " Device: " << C_d[i] << std::endl;
      return;
    }
  }
  std::cout << "Results match!" << std::endl;
}