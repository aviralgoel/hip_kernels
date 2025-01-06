#include <iostream>
#include <hip/hip_runtime.h>

__global__ void helloWorldKernel() {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x;
    int localThreadId = threadIdx.x;

    printf("Hello World from GPU! Global Thread ID: %d, Block ID: %d, Local Thread ID: %d\n", globalThreadId, blockId, localThreadId);
}

int main() {
    int numBlocks = 2;
    int threadsPerBlock = 4;

    hipLaunchKernelGGL(helloWorldKernel, dim3(numBlocks), dim3(threadsPerBlock), 0, 0);

    hipDeviceSynchronize();

    return 0;
}