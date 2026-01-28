#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../utility/hip_utility.hpp"

// Solution 5: Warp Shuffle Reduction
// Approach: Use shared memory reduction until 32 threads, then use warp shuffle intrinsics

__global__ void reduction_kernel(const float* input, float* output, int N)
{
    extern __shared__ float sdata[];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    sdata[threadIdx.x] = (tid < N) ? input[tid] : 0.0f;
    __syncthreads();

    for (int i = blockDim.x / 2 ; i >= 32 ; i /= 2)
    {
        if (threadIdx.x < i)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }

    float val = sdata[threadIdx.x];

    if (threadIdx.x < 32)
    {
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            val += __shfl_down(val, offset);
        }
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = val;
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* ping;
    float* pong;
    HIP_CHECK(hipMalloc(&ping, blocksPerGrid * sizeof(float)));
    HIP_CHECK(hipMalloc(&pong, blocksPerGrid * sizeof(float)));
    size_t sharedMemSize = sizeof(float) * threadsPerBlock;
    hipLaunchKernelGGL(reduction_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), sharedMemSize, 0, input, ping, N);
    int currentN = blocksPerGrid;

    bool isPing = true;
    while(currentN > 1)
    {
        int nextThreadsPerBlock = 1;
        while (nextThreadsPerBlock * 2 <= min(currentN, 1024))
        {
            nextThreadsPerBlock *= 2;
        }
        int nextBlockPerGrid = (currentN + nextThreadsPerBlock - 1) / nextThreadsPerBlock;
        size_t nextSharedMemSize = sizeof(float) * nextThreadsPerBlock;
        if(isPing)
        {
            hipLaunchKernelGGL(reduction_kernel, dim3(nextBlockPerGrid), dim3(nextThreadsPerBlock), nextSharedMemSize, 0, ping, pong, currentN);
        }
        else
        {
            hipLaunchKernelGGL(reduction_kernel, dim3(nextBlockPerGrid), dim3(nextThreadsPerBlock), nextSharedMemSize, 0, pong, ping, currentN);
        }
        isPing = !isPing;
        currentN = nextBlockPerGrid;
    }
    if(isPing)
    {
        HIP_CHECK(hipMemcpy(output, ping, sizeof(float), hipMemcpyDeviceToHost));
    }
    else
    {
        HIP_CHECK(hipMemcpy(output, pong, sizeof(float), hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipFree(ping));
    HIP_CHECK(hipFree(pong));
}

int main(int argc, char* argv[]) {

    int N = 1000000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    float *h_input = new float[N];
    srand(42);
    for(int i = 0; i < N; i++) {
        // h_input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_input[i] = 1.0f;
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
