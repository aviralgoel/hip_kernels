#include <hip/hip_runtime.h>
#include <iostream>

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

  // Launch the kernel on GPU
  hipLaunchKernelGGL(helloWorldKernel, dim3(numBlocks), dim3(threadsPerBlock),
                     0, 0);

  // wait for GPU to finish the work
  hipDeviceSynchronize();

  return 0;
}
