#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../utility/hip_utility.hpp"

#define THREADS_PER_BLOCK 256
#define div_up(a, b) (((a) + (b) - 1) / (b))

// YOUR KERNEL - Debug it yourself!
__global__ void softmax_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int elements_per_thread = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int thread_start = tid * elements_per_thread;

    __shared__ float local_max[THREADS_PER_BLOCK];
    __shared__ float local_sum[THREADS_PER_BLOCK];
    __shared__ float global_max;
    __shared__ float global_sum;

    local_max[tid] = -100000.0f;
    local_sum[tid] = 0.0f;
    __syncthreads();

    // Find local maximum
    for (int i = 0; i < elements_per_thread; i++) {
        if (thread_start + i < N) {
            local_max[tid] = max(local_max[tid], input[thread_start + i]);
        }
        __syncthreads();
    }

    // Reduce to find global maximum
    for (int offset = THREADS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local_max[tid] = max(local_max[tid], local_max[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        global_max = local_max[0];
    }
    __syncthreads();

    // Compute exp(x - max) and accumulate local sum
    for (int i = 0; i < elements_per_thread; i++) {
        if (thread_start + i < N) {
            local_sum[tid] += __expf(input[thread_start + i] - global_max);
        }
        __syncthreads();
    }

    // Reduce to find global sum
    for (int offset = THREADS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local_sum[tid] += local_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        global_sum = local_sum[0];
    }
    __syncthreads();

    // Normalize and write output
    for (int i = 0; i < elements_per_thread; i++) {
        if (thread_start + i < N) {
            output[tid * elements_per_thread + i] = 
                __expf(input[tid * elements_per_thread + i] - global_max) / global_sum;
        }
    }
}

// CPU reference implementation for validation
void cpu_softmax(const float* input, float* output, int N) {
    // Find max (for numerical stability)
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    printf("Max val: %f\n", max_val);

    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += (double)output[i];
    }
    printf("Sum: %f\n", sum);

    // Normalize
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

// The solve function - this is what leetgpu calls
void solve(const float* input, float* output, int N) {
    int blocksPerGrid = 1;
    softmax_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, output, N);
    HIP_CHECK(hipDeviceSynchronize());
}

int main(int argc, char* argv[]) {
    int N = 3;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Allocate host memory
    float* h_input = new float[N];
    float* h_output_gpu = new float[N];
    float* h_output_cpu = new float[N];

    // Initialize input with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = ((float)rand() / (float)RAND_MAX) * 40.0f - 20.0f;  // Range [-20, 20]
    }

    // Alternative: initialize N = 3 with 1.0, 2.0, 3.0
    // for (int i = 0; i < N; i++) {
    //     h_input[i] = 1.0f + i;
    // }

    // CPU reference computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_softmax(h_input, h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Allocate device memory
    float* d_input;
    float* d_output;
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, N * sizeof(float)));

    // GPU timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    solve(d_input, d_output, N);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back
    HIP_CHECK(hipMemcpy(h_output_gpu, d_output, N * sizeof(float), hipMemcpyDeviceToHost));

    // Validate results
    double max_error = 0.0;
    double sum_gpu = 0.0;
    double sum_cpu = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs((double)h_output_gpu[i] - (double)h_output_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
        sum_gpu += (double)h_output_gpu[i];
        sum_cpu += (double)h_output_cpu[i];
    }

    bool passed = (max_error < 1e-4);

    printf("N=%d, CPU=%.3fms, GPU=%.3fms, MaxError=%.2e, SumGPU=%.6f, SumCPU=%.6f, %s\n",
           N, cpu_time_ms, gpu_time_ms, max_error, sum_gpu, sum_cpu,
           passed ? "PASS" : "FAIL");

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;

    return passed ? 0 : 1;
}
