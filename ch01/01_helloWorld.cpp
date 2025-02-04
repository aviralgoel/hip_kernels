#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(-1);                                                                \
    }                                                                          \
  }

__global__ void helloWorldKernel() {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  int blockId = blockIdx.x;
  int localThreadId = threadIdx.x;

  printf("Hello World from GPU! Global Thread ID: %d, Block ID: %d, Local "
         "Thread ID: %d\n",
         globalThreadId, blockId, localThreadId);
}

int main() {

  int numBlocks = 2;       // number of blocks per grid
  int threadsPerBlock = 4; // number of threads per block

  //hipLaunchKernelGGL(helloWorldKernel, dim3(numBlocks), dim3(threadsPerBlock),
                     0, 0);

  // Launch the kernel on GPU
  HIP_CHECK(hipLaunchKernelGGL(helloWorldKernel, dim3(numBlocks), dim3(threadsPerBlock),
                     0, 0));

  // wait for GPU to finish the work
  HIP_CHECK(hipDeviceSynchronize());

  return 0;
}
