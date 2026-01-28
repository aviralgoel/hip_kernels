#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../utility/hip_utility.hpp"

// Solution 6: Grid-Stride Reduction with Multiple Elements Per Thread
// Approach: Each thread processes multiple elements, reducing grid size and atomic contention

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 8
#define THREADS_PER_WARP 32
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD)
#define ceil_div(a, b) (((a) + (b) - 1) / (b))

__global__ void reduction_kernel(const float* input, float* output, int N)
{
    __shared__ float sdata[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * ELEMENTS_PER_BLOCK;

    float sum = 0.0f;

    int gid0 = block_start + tid;
    if (gid0 < N)
        sum = input[gid0];

    #pragma unroll
    for (int i = 1; i < ELEMENTS_PER_THREAD; ++i) {
        int gid = block_start + i * THREADS_PER_BLOCK + tid;
        if (gid < N)
            sum += input[gid];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > THREADS_PER_WARP; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    float val = 0.0f;
    if (tid < THREADS_PER_WARP) {
        val = sdata[tid] + sdata[threadIdx.x + THREADS_PER_WARP];
        val += __shfl_down(val, 16);
        val += __shfl_down(val, 8);
        val += __shfl_down(val, 4);
        val += __shfl_down(val, 2);
        val += __shfl_down(val, 1);
    }

    if (tid == 0) {
        atomicAdd(output, val);
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
    int blocksPerGrid = ceil_div(N, ELEMENTS_PER_BLOCK);
    hipLaunchKernelGGL(reduction_kernel, dim3(blocksPerGrid), dim3(THREADS_PER_BLOCK), 0, 0, input, output, N);
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
    HIP_CHECK(hipMemset(d_output, 0, sizeof(float)));

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
