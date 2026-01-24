#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../utility/hip_utility.hpp"

// Solution 1: Naive Atomic Add Reduction
// Approach: Each thread atomically adds its element to a global sum

__global__ void reduce_add(const float *input, double *sum, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N) {
        atomicAdd(sum, (double)input[tid]);
    }
}

double cpu_reduce(const float *input, int N) {
    double sum = 0.0;
    for(int i = 0; i < N; i++) {
        sum += (double)input[i];
    }
    return sum;
}

void solve(const float* input, float* output, int N) {  
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;

    double *d_sum;
    HIP_CHECK(hipMalloc((void **)&d_sum, sizeof(double)));
    HIP_CHECK(hipMemset(d_sum, 0, sizeof(double)));

    reduce_add<<<grid_size, block_size>>>(input, d_sum, N);

    double h_sum = 0.0;
    HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(double), hipMemcpyDeviceToHost));
    
    float sum = (float)h_sum;
    HIP_CHECK(hipMemcpy(output, &sum, sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipFree(d_sum));
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
