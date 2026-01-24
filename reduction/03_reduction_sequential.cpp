#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../utility/hip_utility.hpp"

// Solution 3: Shared Memory Reduction with Sequential Addressing
// Approach: Same as solution 2 but with better addressing pattern to reduce warp divergence

__global__ void reduction_kernel(const float* input, float* output, int N)
{
    extern __shared__ float sdata[];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    sdata[threadIdx.x] = (tid < N) ? input[tid] : 0.0f;

    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        int index = 2 * i * threadIdx.x;
        if (index + i < blockDim.x)
        {
            sdata[index] += sdata[index + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

double cpu_reduce(const float *input, int N) {
    double sum = 0.0;
    for(int i = 0; i < N; i++) {
        sum += (double)input[i];
    }
    return sum;
}

void solve(const float* input, float* output, int N)
{
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* temp;
    HIP_CHECK(hipMalloc(&temp, blocksPerGrid * sizeof(float)));
    size_t sharedMemSize = sizeof(float) * threadsPerBlock;
    hipLaunchKernelGGL(reduction_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), sharedMemSize, 0, input, temp, N);
    HIP_CHECK(hipDeviceSynchronize());
    int currentN = blocksPerGrid;
    
    while(currentN > 1)
    {
        int nextThreadsPerBlock = min(currentN, 1024);
        int nextBlockPerGrid = (currentN + nextThreadsPerBlock - 1) / nextThreadsPerBlock;
        size_t nextSharedMemSize = sizeof(float) * nextThreadsPerBlock;
        hipLaunchKernelGGL(reduction_kernel, dim3(nextBlockPerGrid), dim3(nextThreadsPerBlock), nextSharedMemSize, 0, temp, temp, currentN);
        currentN = nextBlockPerGrid;
    }
    HIP_CHECK(hipMemcpy(output, temp, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(temp));
}

int main(int argc, char* argv[]) {

    int N = 1000000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    float *h_input = new float[N];
    srand(42);
    for(int i = 0; i < N; i++) {
        h_input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = cpu_reduce(h_input, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    solve(d_input, d_output, N);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));

    float gpu_result;
    HIP_CHECK(hipMemcpy(&gpu_result, d_output, sizeof(float), hipMemcpyDeviceToHost));

    double abs_error = fabs((double)gpu_result - cpu_result);
    double rel_error = abs_error / (fabs(cpu_result) + 1e-10);
    bool passed = (rel_error < 1e-4) || (abs_error < 1e-3);
    
    printf("N=%d, CPU=%.3fms, GPU=%.3fms, Speedup=%.2fx, Result=%.6f, Error=%.2e, %s\n",
           N, cpu_time_ms, gpu_time_ms, cpu_time_ms / gpu_time_ms, 
           gpu_result, rel_error, passed ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    delete[] h_input;

    return passed ? 0 : 1;
}
